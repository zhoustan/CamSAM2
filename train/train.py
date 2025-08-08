import argparse
import sys
from pathlib import Path

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# Get the current file path
current_path = Path(__file__).resolve()

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import random
from typing import List, Optional, Tuple
from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam.transformer import TwoWayTransformer, Attention
from sam2.modeling.sam2_utils import LayerNorm2d, MLP
from sam2.build_sam import build_sam2
from utils import misc
from utils.MoCAMaskPseudoDataset import MoCAMaskMemDataset
from utils.loss_mask import loss_masks, loss_prototype
import numpy as np
import logging
from sam2.modeling.sam2_utils import sample_box_points, get_next_point
from fast_pytorch_kmeans import KMeans

from flash_attn import flash_attn_func
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter

NO_OBJ_SCORE = -1024.0


class DecamouflagedMaskDecoder(MaskDecoder):
    def __init__(self, model_type):
        super().__init__(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=True,
            iou_prediction_use_sigmoid=True,
            pred_obj_scores=True,
            pred_obj_scores_mlp=True,
            dynamic_multimask_via_stability=True,
        )

        assert model_type in ['hiera_large', 'hiera_base_plus', 'hiera_small', 'hiera_tiny']

        checkpoint_dict = {
            'hiera_large': '../checkpoints/sam2_hiera_large.pt',
            'hiera_base_plus': '../checkpoints/sam2_hiera_base_plus.pt',
            'hiera_small': '../checkpoints/sam2_hiera_small.pt',
            'hiera_tiny': '../checkpoints/sam2_hiera_tiny.pt',
        }
        checkpoint_path = checkpoint_dict[model_type]
        checkpoint = torch.load(checkpoint_path)
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint  # In case the checkpoint is directly the state_dict
        sam_mask_decoder_dict = {}
        for n, p in state_dict.items():  # 'state_dict' is a dictionary
            if n.split('.')[0] == 'sam_mask_decoder':
                new_key = n.replace('sam_mask_decoder.', '')
                sam_mask_decoder_dict[new_key] = p

        self.load_state_dict(sam_mask_decoder_dict, strict=False)
        print("Decamouflaged Decoder init from SAM2 MaskDecoder")
        for n, p in self.named_parameters():
            p.requires_grad = False

        transformer_dim = 256
        self.transformer_dim = transformer_dim
        self.decamouflaged_token = nn.Embedding(1, transformer_dim)
        self.decamouflaged_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        self.num_mask_tokens = self.num_mask_tokens + 1

        self.compress_hiera_feat_0 = nn.Sequential(
            nn.Conv2d(transformer_dim // 8, transformer_dim, kernel_size=3, stride=1, padding=1),
            LayerNorm2d(transformer_dim),
            nn.GELU(),
            nn.Conv2d(transformer_dim, transformer_dim // 8, kernel_size=3, stride=1, padding=1)
        )

        self.compress_hiera_feat_1 = nn.Sequential(
            nn.Conv2d(transformer_dim // 4, transformer_dim // 2, kernel_size=3, stride=1, padding=1),
            LayerNorm2d(transformer_dim // 2),
            nn.GELU(),
            nn.Conv2d(transformer_dim // 2, transformer_dim // 8, kernel_size=3, stride=1, padding=1),
            # Reduce channels to 32
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # Upsample to [1, 32, 256, 256]
        )

        self.embedding_encoder = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
        )

        self.embedding_mask_feature = nn.Sequential(
            nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1)
        )

        self.proj = nn.Conv2d(33, 32, kernel_size=1)
        self.mlp_q = nn.Linear(32, 32)
        self.mlp_k = nn.Linear(32, 32)
        self.mlp_v = nn.Linear(32, 32)

    def forward(self,
                image_embeddings: torch.Tensor,
                image_pe: torch.Tensor,
                sparse_prompt_embeddings: torch.Tensor,
                dense_prompt_embeddings: torch.Tensor,
                multimask_output: bool,
                repeat_image: bool,
                hiera_feature: List[torch.Tensor],
                prototypes: List[torch.Tensor],
                high_res_features: Optional[List[torch.Tensor]] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        masks, iou_pred, mask_tokens_out, object_score_logits, upscaled_embedding = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features
        )

        if multimask_output:
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        elif self.dynamic_multimask_via_stability and not self.training:
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            masks = masks[:, 0:1, :, :]
            iou_pred = iou_pred[:, 0:1]

        if multimask_output and self.use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out[:, 1:]  # [b, 3, c] shape
        else:
            # Take the mask output token. Here we *always* use the token for single mask output.
            # At test time, even if we track after 1-click (and using multimask_output=True),
            # we still take the single mask token here. The rationale is that we always track
            # after multiple clicks during training, so the past tokens seen during training
            # are always the single mask token (and we'll let it be the object-memory token).
            sam_tokens_out = mask_tokens_out[:, 0:1]  # [b, 1, c] shape

        hiera_feature_0 = self.compress_hiera_feat_0(hiera_feature[0])
        hiera_feature_1 = self.compress_hiera_feat_1(hiera_feature[1])
        vis_feat = self.embedding_encoder(image_embeddings)
        decamouflaged_feature = vis_feat + hiera_feature_0 + hiera_feature_1

        b, c, h, w = upscaled_embedding.shape
        device = upscaled_embedding.device
        singlemask_logits = self._get_single_mask_logits(masks, iou_pred, multimask_output).expand(-1, 1, -1, -1)
        assert singlemask_logits.shape == (1, 1, 256, 256)
        concatenated_decamouflaged_feature = torch.cat((decamouflaged_feature, singlemask_logits), dim=1)
        assert concatenated_decamouflaged_feature.shape == (1, 33, 256, 256)
        decamouflaged_feature = self.proj(concatenated_decamouflaged_feature)  # [1, 32, 256, 256]
        decamouflaged_feature_for_prototype = decamouflaged_feature
        proj_decamouflaged_feature = decamouflaged_feature.permute(0, 2, 3, 1).view(b, h * w, c)  # [1, 65536, 32]
        assert proj_decamouflaged_feature.shape == (1, 65536, 32)

        upscaled_embedding_embedded = self.embedding_mask_feature(upscaled_embedding)

        point_num = 5
        if prototypes is not None and len(prototypes) != 0 and not all(i is None for i in prototypes):
            filtered_prototypes = []
            for i in prototypes:
                if i is not None:
                    filtered_prototypes.append(i)

            prototypes = torch.stack(filtered_prototypes)  # [num_prototypes, point_num, 32]

            num_prototypes = prototypes.size(0)
            seqlen = h * w
            nheads = 1
            headdim = 32
            proj_decamouflaged_feature = proj_decamouflaged_feature.view(b, seqlen, nheads,
                                                                         headdim)  # [1, 65536, 1, 32]
            prototypes = prototypes.view(b, num_prototypes * point_num, nheads, headdim)  # [1, k * point_num, 1, 32]

            att_out = flash_attn_func(
                self.mlp_q(proj_decamouflaged_feature).half(),  # query: [1, 65536, 1, 32]
                self.mlp_k(prototypes).half(),  # key: [1, k, 1, 32]
                self.mlp_v(prototypes).half(),  # value: [1, k, 1, 32]
                dropout_p=0.0
            )  # Output shape: [1, 65536, 1, 32]

            att_out = att_out.squeeze(2).permute(0, 2, 1).contiguous().view(b, c, h, w)  # [1, 32, 256, 256]
            decamouflaged_feature = upscaled_embedding_embedded + att_out
        else:
            decamouflaged_feature = upscaled_embedding_embedded + decamouflaged_feature

        hyper_in = self.decamouflaged_mlp(mask_tokens_out[:, -1, :])
        mask_decamouflaged = (hyper_in @ decamouflaged_feature.view(b, c, h * w)).view(b, -1, h, w)

        sample_points = self._mask_slic(mask_decamouflaged.squeeze() > 0, point_num)
        mask_indices = (mask_decamouflaged.squeeze() > 0).nonzero(as_tuple=False)  # [num_points, 2]

        if mask_indices.size(0) < point_num:
            prototype = torch.zeros((point_num, c), device=device)
        else:
            h_indices, w_indices = mask_indices[:, 0], mask_indices[:, 1]
            features = decamouflaged_feature_for_prototype[0, :, h_indices, w_indices].T.detach()
            h_indices_centers, w_indices_centers = sample_points[:, 0], sample_points[:, 1]
            initial_centers = decamouflaged_feature_for_prototype[0, :, h_indices_centers, w_indices_centers].T.detach()

            kmeans = KMeans(n_clusters=point_num, mode='cosine', max_iter=1)
            labels = kmeans.fit_predict(features, initial_centers)

            labels = torch.tensor(labels, device=features.device)

            prototype = torch.zeros((point_num, features.shape[1]), device=features.device)

            for i in range(point_num):
                mask = (labels == i)
                cluster_points = features[mask]
                if cluster_points.size(0) > 0:
                    prototype[i] = cluster_points.mean(dim=0)

        if multimask_output:
            mask_decamouflaged = mask_decamouflaged.repeat(1, 3, 1, 1)

        return masks, iou_pred, sam_tokens_out, object_score_logits, mask_decamouflaged, prototype

    def predict_masks(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            repeat_image: bool,
            high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        s = 0
        if self.pred_obj_scores:
            output_tokens = torch.cat(
                [
                    self.obj_score_token.weight,
                    self.iou_token.weight,
                    self.mask_tokens.weight,
                    self.decamouflaged_token.weight,
                ],
                dim=0,
            )
            s = 1
        else:
            output_tokens = torch.cat(
                [self.iou_token.weight, self.mask_tokens.weight, self.decamouflaged_token.weight], dim=0
            )

        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings
        src = src + dense_prompt_embeddings
        assert (
                image_pe.size(0) == 1
        ), "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, (s + 1): (s + 1 + self.num_mask_tokens), :]
        src = src.transpose(1, 2).view(b, c, h, w)

        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            if i < self.num_mask_tokens - 1:
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in[:, :self.num_mask_tokens - 1] @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        iou_pred = self.iou_prediction_head(iou_token_out)

        if self.pred_obj_scores:
            assert s == 1
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            # Obj scores logits - default to 10.0, i.e. assuming the object is present, sigmoid(10)=1
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

        return masks, iou_pred, mask_tokens_out, object_score_logits, upscaled_embedding

    def _get_single_mask_logits(self, all_mask_logits, all_iou_scores, multimask_output):
        if multimask_output:
            multimask_logits = all_mask_logits[:, 1:, :, :]
            multimask_iou_scores = all_iou_scores[:, 1:]
            best_scores_inds = torch.argmax(multimask_iou_scores, dim=-1)
            batch_inds = torch.arange(
                multimask_iou_scores.size(0), device=all_iou_scores.device
            )
            best_multimask_logits = multimask_logits[batch_inds, best_scores_inds]
            best_multimask_logits = best_multimask_logits.unsqueeze(1)
            best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds]
            best_multimask_iou_scores = best_multimask_iou_scores.unsqueeze(1)
            return best_multimask_logits
        else:
            # The mask from singlemask output token 0 and its stability score
            singlemask_logits = all_mask_logits[:, 0:1, :, :]
            singlemask_iou_scores = all_iou_scores[:, 0:1]
            return singlemask_logits

    def _get_stability_scores(self, mask_logits):
        """
        Compute stability scores of the mask logits based on the IoU between upper and
        lower thresholds, similar to https://github.com/fairinternal/onevision/pull/568.
        """
        mask_logits = mask_logits.flatten(-2)
        stability_delta = self.dynamic_multimask_stability_delta
        area_i = torch.sum(mask_logits > stability_delta, dim=-1).float()
        area_u = torch.sum(mask_logits > -stability_delta, dim=-1).float()
        stability_scores = torch.where(area_u > 0, area_i / area_u, 1.0)
        return stability_scores

    def _dynamic_multimask_via_stability(self, all_mask_logits, all_iou_scores):
        """
        When outputting a single mask, if the stability score from the current single-mask
        output (based on output token 0) falls below a threshold, we instead select from
        multi-mask outputs (based on output token 1~3) the mask with the highest predicted
        IoU score. This is intended to ensure a valid mask for both clicking and tracking.
        """
        # The best mask from multimask output tokens (1~3)
        multimask_logits = all_mask_logits[:, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, 1:]
        best_scores_inds = torch.argmax(multimask_iou_scores, dim=-1)
        batch_inds = torch.arange(
            multimask_iou_scores.size(0), device=all_iou_scores.device
        )
        best_multimask_logits = multimask_logits[batch_inds, best_scores_inds]
        best_multimask_logits = best_multimask_logits.unsqueeze(1)
        best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds]
        best_multimask_iou_scores = best_multimask_iou_scores.unsqueeze(1)

        # The mask from singlemask output token 0 and its stability score
        singlemask_logits = all_mask_logits[:, 0:1, :, :]
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        stability_scores = self._get_stability_scores(singlemask_logits)
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh

        # Dynamically fall back to best multimask output upon low stability scores.
        mask_logits_out = torch.where(
            is_stable[..., None, None].expand_as(singlemask_logits),
            singlemask_logits,
            best_multimask_logits,
        )
        iou_scores_out = torch.where(
            is_stable.expand_as(singlemask_iou_scores),
            singlemask_iou_scores,
            best_multimask_iou_scores,
        )
        return mask_logits_out, iou_scores_out

    def _mask_slic(self, pred, point_num, avg_sp_area=100):
        '''
        :param mask: the RoI region to do clustering, torch tensor: H x W
        :param down_stride: downsampled stride for RoI region
        :param max_num_sp: the maximum number of superpixels
        :return: segments: the coordinates of the initial seed, max_num_sp x 2
        '''
        assert point_num >= 0
        mask = pred  # Binary mask
        h, w = mask.shape
        max_num_sp = point_num

        segments_x = np.zeros(max_num_sp, dtype=np.int64)
        segments_y = np.zeros(max_num_sp, dtype=np.int64)

        m_np = mask.cpu().numpy()
        m_np_down = m_np

        nz = np.nonzero(m_np_down)

        if len(nz[0]) != 0:

            p = [np.min(nz[0]), np.min(nz[1])]
            pend = [np.max(nz[0]), np.max(nz[1])]

            # cropping to bounding box around ROI
            m_np_roi = np.copy(m_np_down)[p[0]:pend[0] + 1, p[1]:pend[1] + 1]
            num_sp = max_num_sp

        else:
            num_sp = 0

        if (num_sp != 0) and (num_sp != 1):
            for i in range(num_sp):
                # n seeds are placed as far as possible from every other seed and the edge.

                # STEP 1:  conduct Distance Transform and choose the maximum point
                dtrans = distance_transform_edt(m_np_roi)  #
                dtrans = gaussian_filter(dtrans, sigma=0.1)

                coords1 = np.nonzero(dtrans == np.max(dtrans))
                segments_x[i] = coords1[0][0]
                segments_y[i] = coords1[1][0]

                # STEP 2:  set the point to False and repeat Step 1
                m_np_roi[segments_x[i], segments_y[i]] = False
                segments_x[i] += p[0]
                segments_y[i] += p[1]

        segments = np.concatenate([segments_x[..., np.newaxis], segments_y[..., np.newaxis]], axis=1)  # max_num_sp x 2
        segments = torch.from_numpy(segments)
        return segments


def get_args_parser():
    parser = argparse.ArgumentParser('CamSAM2', add_help=False)

    parser.add_argument("--data_path", type=str, default="/scratch/username/MoCA-Video-Train/")
    parser.add_argument("--output", type=str, default="../work_dir/work_dir_tiny/",
                        help="Path to the directory where masks and checkpoints will be output")
    parser.add_argument("--model_type", type=str, default="hiera_tiny",
                        help="The type of model to load, in ['hiera_large', 'hiera_base_plus', 'hiera_small', 'hiera_tiny']")
    parser.add_argument("--device", type=str, default="cuda",
                        help="The device to run generation on.")

    parser.add_argument('--num_frames', default=8, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--lr_drop_epoch', default=10, type=int)
    parser.add_argument('--max_epoch_num', default=15, type=int)
    parser.add_argument('--input_size', default=[1024, 1024], type=list)
    parser.add_argument('--batch_size_train', default=8, type=int)
    parser.add_argument('--batch_size_valid', default=1, type=int)
    parser.add_argument('--model_save_fre', default=1, type=int)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local-rank', type=int, help='local rank for dist')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_data_path', type=str, default=None)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument("--restore-model", type=str,
                        help="The path to the hq_decoder training checkpoint for evaluation")

    return parser.parse_args()


def main(net, args):
    misc.init_distributed_mode(args)
    print('world size: {}'.format(args.world_size))
    print('rank: {}'.format(args.rank))
    print('local_rank: {}'.format(args.local_rank))
    print("args: " + str(args) + '\n')
    args.sam2_model_cfg = None
    args.sam2_ckpt_path = None

    base_path = current_path.parent.parent
    sys.path.append(base_path)
    if args.model_type == "hiera_large":
        args.sam2_ckpt_path = "../checkpoints/sam2_hiera_large.pt"
        args.sam2_model_cfg = "sam2_hiera_l.yaml"
    elif args.model_type == "hiera_base_plus":
        args.sam2_ckpt_path = "../checkpoints/sam2_hiera_base_plus.pt"
        args.sam2_model_cfg = "sam2_hiera_b+.yaml"
    elif args.model_type == "hiera_small":
        args.sam2_ckpt_path = "../checkpoints/sam2_hiera_small.pt"
        args.sam2_model_cfg = "sam2_hiera_s.yaml"
    elif args.model_type == "hiera_tiny":
        args.sam2_ckpt_path = "../checkpoints/sam2_hiera_tiny.pt"
        args.sam2_model_cfg = "sam2_hiera_t.yaml"
    else:
        args.sam2_ckpt_path = "../checkpoints/sam2_hiera_tiny.pt"
        args.sam2_model_cfg = "sam2_hiera_t.yaml"

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if not args.eval:
        print("--- create training dataloader ---")
        train_dataset = MoCAMaskMemDataset(data_root=args.data_path, memory_length=args.num_frames)

        sampler = DistributedSampler(train_dataset)
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler, args.batch_size_train, drop_last=True)
        train_dataloaders = DataLoader(train_dataset, batch_sampler=batch_sampler_train, num_workers=0)

        print(len(train_dataloaders), " train dataloaders created")

    valid_dataloaders = None

    if torch.cuda.is_available():
        net.cuda()
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu],
                                                    find_unused_parameters=args.find_unused_params)
    net_without_ddp = net.module

    if not args.eval:
        print("--- define optimizer ---")
        optimizer = optim.Adam(net_without_ddp.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08,
                               weight_decay=0)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop_epoch)
        lr_scheduler.last_epoch = args.start_epoch

        train(args, net, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler)
    else:
        print(1)


def setup_logger(log_file="log.txt", logger_name=__name__):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Check if the logger already has handlers (to avoid adding duplicate handlers)
    if not logger.handlers:
        # File handler for logging to file
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Console handler for logging to the console
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Create formatter and add it to both handlers
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


def get_prototypes(output_dict):
    """
        output_dict = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        only consider prompting on the first frame here.
    """

    total_len = len(output_dict["cond_frame_outputs"].keys()) + len(output_dict["non_cond_frame_outputs"].keys())
    prototypes = [None] * total_len
    for frame_idx, out in output_dict["cond_frame_outputs"].items():
        prototypes[frame_idx] = out["prototype"]
    for frame_idx, out in output_dict["non_cond_frame_outputs"].items():
        prototypes[frame_idx] = out["prototype"]
    return prototypes


def train(args, net, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler):
    if misc.is_main_process():
        os.makedirs(args.output, exist_ok=True)
    rng = np.random.default_rng(seed=42)
    net.train()
    _ = net.to(device=args.device)

    sam2 = build_sam2(args.sam2_model_cfg, ckpt_path=args.sam2_ckpt_path, device=args.device, mode="eval", )

    for param in sam2.parameters():
        param.requires_grad = False

    # create logger
    log_file = args.output + 'log.txt'
    logger = setup_logger(log_file)
    if misc.is_main_process():
        logger.info(args)
    for epoch in range(args.start_epoch, args.max_epoch_num):
        msg = ''.join(["epoch:   ", str(epoch), "  learning rate:  ", str(optimizer.param_groups[0]["lr"])])
        if misc.is_main_process():
            logger.info(msg)
        metric_logger = misc.MetricLogger(delimiter="  ", logger=logger)

        if hasattr(train_dataloaders.batch_sampler.sampler, 'set_epoch'):
            train_dataloaders.batch_sampler.sampler.set_epoch(epoch)

        for data in metric_logger.log_every(train_dataloaders, 10):
            batch_size = len(data['current_image'])

            total_loss = 0
            total_loss_mask_sam = 0
            total_loss_dice_sam = 0
            total_loss_mask_decamouflaged = 0
            total_loss_dice_decamouflaged = 0
            for i_batch in range(batch_size):

                memory_images = data['memory_images'][i_batch].to(args.device)
                memory_labels = data['memory_labels'][i_batch].to(args.device)

                output_dict = {
                    "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                    "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                }
                num_frames = memory_images.size(0)

                for frame_idx in range(num_frames):
                    current_out = {"point_inputs": None, "mask_inputs": None, "prototype": None}
                    # print("memory_images[frame_idx].shape", memory_images[frame_idx].shape)  # [3, 1024, 1024]
                    with torch.no_grad():
                        backbone_out = sam2.forward_image(memory_images[frame_idx].unsqueeze(0))

                        (
                            backbone_out,
                            vision_feats,  # [[65536, 1, 32], [16384, 1, 64], [4096, 1, 256]]
                            vision_pos_embeds,
                            feat_sizes
                        ) = sam2._prepare_backbone_features(backbone_out)

                    current_vision_feats = vision_feats
                    current_vision_pos_embeds = vision_pos_embeds
                    if len(current_vision_feats) > 1:
                        high_res_features = [
                            x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                            for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
                        ]
                    else:
                        high_res_features = None

                    hiera_feature_0 = current_vision_feats[0].permute(1, 2, 0)
                    hiera_feature_0 = hiera_feature_0.view(-1, 32, *feat_sizes[0])
                    hiera_feature_1 = current_vision_feats[1].permute(1, 2, 0)
                    hiera_feature_1 = hiera_feature_1.view(-1, 64, *feat_sizes[1])
                    hiera_features = [hiera_feature_0, hiera_feature_1]
                    pix_feat = current_vision_feats[-1].permute(1, 2, 0)  # [1, 256, 4096]
                    pix_feat = pix_feat.view(-1, sam2.hidden_dim, *feat_sizes[-1])  # [1, 256, 64, 64]
                    B = pix_feat.size(0)
                    device = pix_feat.device

                    if frame_idx == 0:
                        # random generate mask, box, or click prompt with prob 0.5, 0.25, 0.25 respectively
                        prob_to_use_pt_input = 0.5
                        prob_to_use_box_input = 0.5

                        use_pt_input = rng.random() < prob_to_use_pt_input

                        mask = torch.tensor(memory_labels[frame_idx][0], dtype=torch.bool)
                        mask_H, mask_W = mask.shape
                        mask_inputs_orig = mask[None, None]  # add batch and channel dimension
                        mask_inputs_orig = mask_inputs_orig.float().to(args.device)
                        if mask_H != sam2.image_size or mask_W != sam2.image_size:
                            mask_inputs = torch.nn.functional.interpolate(
                                mask_inputs_orig,
                                size=(sam2.image_size, sam2.image_size),
                                align_corners=False,
                                mode="bilinear",
                                antialias=True,  # use antialias for downsampling
                            )
                            mask_inputs = (mask_inputs >= 0.5).float()
                        else:
                            mask_inputs = mask_inputs_orig

                        sam_mask_prompt = None
                        if not use_pt_input:
                            multimask_output = True
                            sam_mask_prompt = torch.nn.functional.interpolate(
                                mask_inputs.float(),
                                size=sam2.sam_prompt_encoder.mask_input_size,
                                align_corners=False,
                                mode="bilinear",
                                antialias=True,  # use antialias for downsampling
                            )
                            sam_point_coords = torch.zeros(B, 1, 2, device=device)
                            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)
                        else:
                            use_box_input = rng.random() < prob_to_use_box_input
                            mask_inputs_orig = torch.tensor(mask_inputs, dtype=torch.bool)
                            if use_box_input:
                                multimask_output = False
                                sam_point_coords, sam_point_labels = sample_box_points(mask_inputs_orig)
                            else:
                                multimask_output = True
                                sam_point_coords, sam_point_labels = get_next_point(
                                    gt_masks=mask_inputs_orig,
                                    pred_masks=None,
                                    method=("uniform"),
                                )

                        with torch.no_grad():
                            pix_feat = sam2._prepare_memory_conditioned_features(
                                frame_idx=frame_idx,
                                is_init_cond_frame=True,
                                current_vision_feats=current_vision_feats[-1:],  # [[4096, 1, 256]]
                                current_vision_pos_embeds=current_vision_pos_embeds[-1:],  # [[4096, 1, 256]]
                                feat_sizes=feat_sizes[-1:],  # [(64, 64)]
                                output_dict=output_dict,
                                num_frames=num_frames,
                                track_in_reverse=False,
                            )

                            sparse_embeddings, dense_embeddings = sam2.sam_prompt_encoder(
                                points=(sam_point_coords, sam_point_labels),
                                boxes=None,
                                masks=sam_mask_prompt,
                            )

                        (
                            low_res_multimasks,  # [1, 3, 256, 256]
                            ious,  # [1, 3]
                            sam_output_tokens,  # [1, 1, 256]
                            object_score_logits,  # [1, 1]
                            decamouflaged_mask,  # [1, 3, 256, 256]
                            prototype
                        ) = net(
                            image_embeddings=pix_feat,  # [1, 256, 64, 64]
                            image_pe=sam2.sam_prompt_encoder.get_dense_pe(),  # [1, 256, 64, 64]
                            sparse_prompt_embeddings=sparse_embeddings,  # [1, 2, 256]
                            dense_prompt_embeddings=dense_embeddings,  # [1, 256, 64, 64]
                            multimask_output=multimask_output,
                            repeat_image=True,
                            hiera_feature=hiera_features,  # [1, 32, 256, 256]
                            high_res_features=high_res_features,  # [[1, 32, 256, 256], [1, 64, 128, 128]]
                            prototypes=get_prototypes(output_dict),
                        )

                        low_res_multimasks_for_loss = low_res_multimasks
                        low_res_multimasks = decamouflaged_mask

                        object_score_logits = torch.tensor([[10.]], device=device)
                        if sam2.pred_obj_scores:
                            is_obj_appearing = object_score_logits > 0
                            # Mask used for spatial memories is always a *hard* choice between obj and no obj,
                            # consistent with the actual mask prediction
                            low_res_multimasks = torch.where(
                                is_obj_appearing[:, None, None],
                                low_res_multimasks,
                                NO_OBJ_SCORE,
                            )

                        low_res_multimasks = low_res_multimasks.float()

                        high_res_multimasks = F.interpolate(
                            low_res_multimasks,
                            size=(sam2.image_size, sam2.image_size),
                            mode="bilinear",
                            align_corners=False,
                        )  # [1, 3, 1024, 1024]

                        sam_output_token = sam_output_tokens[:, 0]  # [1, 256]
                        if multimask_output:
                            best_iou_inds = torch.argmax(ious, dim=-1)
                            batch_inds = torch.arange(B, device=device)
                            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                            decamouflaged_mask = decamouflaged_mask[batch_inds, best_iou_inds].unsqueeze(1)
                            low_res_multimasks_for_loss = low_res_multimasks_for_loss[
                                batch_inds, best_iou_inds].unsqueeze(1)
                            if sam_output_tokens.size(1) > 1:
                                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
                        else:
                            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

                        with torch.no_grad():
                            obj_ptr = sam2.obj_ptr_proj(sam_output_token)
                            if sam2.pred_obj_scores:
                                if sam2.soft_no_obj_ptr:
                                    lambda_is_obj_appearing = object_score_logits.sigmoid()
                                else:
                                    lambda_is_obj_appearing = is_obj_appearing.float()

                                if sam2.fixed_no_obj_ptr:
                                    obj_ptr = lambda_is_obj_appearing * obj_ptr
                                obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * sam2.no_obj_ptr
                    else:
                        with torch.no_grad():
                            pix_feat = sam2._prepare_memory_conditioned_features(
                                frame_idx=frame_idx,
                                is_init_cond_frame=False,
                                current_vision_feats=current_vision_feats[-1:],
                                current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                                feat_sizes=feat_sizes[-1:],
                                output_dict=output_dict,
                                num_frames=num_frames,
                                track_in_reverse=False,
                            )

                        B = pix_feat.size(0)
                        device = pix_feat.device
                        sam_point_coords = torch.zeros(B, 1, 2, device=args.device)
                        sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=args.device)
                        sam_mask_prompt = None
                        multimask_output = True
                        with torch.no_grad():
                            sparse_embeddings, dense_embeddings = sam2.sam_prompt_encoder(
                                points=(sam_point_coords, sam_point_labels),
                                boxes=None,
                                masks=sam_mask_prompt,
                            )

                        (
                            low_res_multimasks,
                            ious,
                            sam_output_tokens,
                            object_score_logits,
                            decamouflaged_mask,
                            prototype
                        ) = net(
                            image_embeddings=pix_feat,
                            image_pe=sam2.sam_prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                            multimask_output=multimask_output,
                            repeat_image=True,
                            hiera_feature=hiera_features,
                            high_res_features=high_res_features,
                            prototypes=get_prototypes(output_dict),
                        )
                        low_res_multimasks_for_loss = low_res_multimasks
                        low_res_multimasks = decamouflaged_mask

                        object_score_logits = torch.tensor([[10.]], device=device)
                        if sam2.pred_obj_scores:
                            is_obj_appearing = object_score_logits > 0

                            # Mask used for spatial memories is always a *hard* choice between obj and no obj,
                            # consistent with the actual mask prediction
                            low_res_multimasks = torch.where(
                                is_obj_appearing[:, None, None],
                                low_res_multimasks,
                                NO_OBJ_SCORE,
                            )

                        low_res_multimasks = low_res_multimasks.float()
                        high_res_multimasks = F.interpolate(
                            low_res_multimasks,
                            size=(sam2.image_size, sam2.image_size),
                            mode="bilinear",
                            align_corners=False,
                        )

                        sam_output_token = sam_output_tokens[:, 0]
                        if multimask_output:
                            best_iou_inds = torch.argmax(ious, dim=-1)
                            batch_inds = torch.arange(B, device=device)
                            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
                            decamouflaged_mask = decamouflaged_mask[batch_inds, best_iou_inds].unsqueeze(1)
                            low_res_multimasks_for_loss = low_res_multimasks_for_loss[
                                batch_inds, best_iou_inds].unsqueeze(1)
                            if sam_output_tokens.size(1) > 1:
                                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
                        else:
                            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

                        with torch.no_grad():
                            obj_ptr = sam2.obj_ptr_proj(sam_output_token)
                            if sam2.pred_obj_scores:
                                if sam2.soft_no_obj_ptr:
                                    lambda_is_obj_appearing = object_score_logits.sigmoid()
                                else:
                                    lambda_is_obj_appearing = is_obj_appearing.float()

                                if sam2.fixed_no_obj_ptr:
                                    obj_ptr = lambda_is_obj_appearing * obj_ptr
                                obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * sam2.no_obj_ptr

                    current_out["prototype"] = prototype
                    current_out["pred_masks"] = low_res_masks  # [1, 1, 256, 256]
                    current_out["pred_masks_high_res"] = high_res_masks  # [1, 1, 1024, 1024]
                    current_out["obj_ptr"] = obj_ptr  # [1, 256]

                    high_res_masks_for_mem_enc = high_res_masks
                    with torch.no_grad():
                        maskmem_features, maskmem_pos_enc = sam2._encode_new_memory(
                            current_vision_feats=current_vision_feats,
                            feat_sizes=feat_sizes,
                            pred_masks_high_res=high_res_masks_for_mem_enc,
                            is_mask_from_pts=False,
                        )
                    current_out["maskmem_features"] = maskmem_features  # [1, 64, 64, 64]
                    current_out["maskmem_pos_enc"] = maskmem_pos_enc  # [1, 1, 64, 64, 64]

                    if frame_idx == 0:
                        output_dict["cond_frame_outputs"][frame_idx] = current_out
                    else:
                        output_dict["non_cond_frame_outputs"][frame_idx] = current_out

                    loss_mask_sam, loss_dice_sam = loss_masks(low_res_multimasks_for_loss.float(),
                                                              memory_labels[frame_idx].unsqueeze(0).float(),
                                                              len(low_res_multimasks_for_loss))
                    loss_mask_decamouflaged, loss_dice_decamouflaged = loss_masks(decamouflaged_mask.float(),
                                                                                  memory_labels[frame_idx].unsqueeze(
                                                                                      0).float(),
                                                                                  len(decamouflaged_mask))

                    total_loss_mask_sam += loss_mask_sam
                    total_loss_mask_decamouflaged += loss_mask_decamouflaged
                    total_loss_dice_sam += loss_dice_sam
                    total_loss_dice_decamouflaged += loss_dice_decamouflaged
                    total_loss += total_loss_mask_sam + total_loss_mask_decamouflaged + total_loss_dice_sam + total_loss_dice_decamouflaged

            loss_dict = {
                "loss_mask_sam": total_loss_mask_sam / batch_size / num_frames,
                "loss_mask_decamouflaged": total_loss_mask_decamouflaged / batch_size / num_frames,
                "loss_dice_sam": total_loss_dice_sam / batch_size / num_frames,
                "loss_dice_decamouflaged": total_loss_dice_decamouflaged / batch_size / num_frames,
            }

            total_loss = total_loss / batch_size / (num_frames * 2)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            losses_reduced_scaled = sum(loss_dict_reduced.values())
            loss_value = losses_reduced_scaled.item()
            metric_logger.update(training_loss=loss_value, **loss_dict_reduced)
            
        metric_logger.synchronize_between_processes()
        if misc.is_main_process():
            logger.info("Finished epoch:      " + str(epoch))
            logger.info("Averaged stats:" + str(metric_logger))

        lr_scheduler.step()

        net.train()

        if misc.is_main_process():
            # merge weights
            new_checkpoint = {"model": {}}
            for k, v in sam2.state_dict().items():
                new_checkpoint["model"][k] = sam2.state_dict()[k]
            for k, v in net.state_dict().items():
                new_key = k.replace("module.", 'sam_mask_decoder.')
                new_checkpoint["model"][new_key] = v

            save_path = "%s/CamSAM2_epoch_%s_merged.pth" % (args.output, epoch)
            torch.save(new_checkpoint, save_path)

    if misc.is_main_process():
        logger.info("Training Reaches The Maximum Epoch Number")


if __name__ == "__main__":
    args = get_args_parser()
    net = DecamouflagedMaskDecoder(args.model_type)
    main(net, args)
