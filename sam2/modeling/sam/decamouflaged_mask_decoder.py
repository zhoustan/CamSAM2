import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Type
from sam2.modeling.sam2_utils import LayerNorm2d, MLP
from flash_attn import flash_attn_func
import numpy as np
from fast_pytorch_kmeans import KMeans
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter


class DecamouflagedMaskDecoder(nn.Module):
    def __init__(
            self,
            *,
            transformer_dim: int,
            transformer: nn.Module,
            num_multimask_outputs: int = 3,
            activation: Type[nn.Module] = nn.GELU,
            iou_head_depth: int = 3,
            iou_head_hidden_dim: int = 256,
            use_high_res_features: bool = False,
            iou_prediction_use_sigmoid=False,
            dynamic_multimask_via_stability=False,
            dynamic_multimask_stability_delta=0.05,
            dynamic_multimask_stability_thresh=0.98,
            pred_obj_scores: bool = False,
            pred_obj_scores_mlp: bool = False,
            use_multimask_token_for_obj_ptr: bool = False,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.pred_obj_scores = pred_obj_scores
        if self.pred_obj_scores:
            self.obj_score_token = nn.Embedding(1, transformer_dim)
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr

        # 2x conv. trans.
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )

        # stride 4, 8 feats. from img. enc.
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = nn.Conv2d(
                transformer_dim, transformer_dim // 8, kernel_size=1, stride=1
            )
            self.conv_s1 = nn.Conv2d(
                transformer_dim, transformer_dim // 4, kernel_size=1, stride=1
            )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
            sigmoid_output=iou_prediction_use_sigmoid,
        )
        if self.pred_obj_scores:
            self.pred_obj_score_head = nn.Linear(transformer_dim, 1)
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)

        # When outputting a single mask, optionally we can dynamically fall back to the best
        # multimask output token if the single mask output token gives low stability scores.
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

        # decamouflaged parameters
        self.decamouflaged_token = nn.Embedding(1, transformer_dim)
        self.decamouflaged_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        self.num_mask_tokens = self.num_mask_tokens + 1

        # IOF
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
                hiera_feature: List[torch.Tensor] = None,
                prototypes: List[torch.Tensor] = None,
                high_res_features: Optional[List[torch.Tensor]] = None,
                output_mode: str = "original_sam2_mask",
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # Continue with the rest of the forward pass, using combined_embeddings

        masks, iou_pred, mask_tokens_out, object_score_logits, upscaled_embedding = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
            output_mode=output_mode,
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

        if output_mode == 'original_sam2_mask':
            return masks, iou_pred, sam_tokens_out, object_score_logits, None, None

        # IOF
        hiera_feature_0 = self.compress_hiera_feat_0(hiera_feature[0])
        hiera_feature_1 = self.compress_hiera_feat_1(hiera_feature[1])
        vis_feat = self.embedding_encoder(image_embeddings)
        decamouflaged_feature = vis_feat + hiera_feature_0 + hiera_feature_1

        # EOF
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
            prototypes = torch.stack(filtered_prototypes)
            # (batch_size, seqlen, nheads, headdim)
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

        # OPG
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
                if cluster_points.size(0) > 0:  # Check to avoid division by zero
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
            output_mode: str,
            high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        if output_mode == 'original_sam2_mask':
            s = 0
            if self.pred_obj_scores:
                output_tokens = torch.cat(
                    [
                        self.obj_score_token.weight,
                        self.iou_token.weight,
                        self.mask_tokens.weight,
                    ],
                    dim=0,
                )
                s = 1
            else:
                output_tokens = torch.cat(
                    [self.iou_token.weight, self.mask_tokens.weight], dim=0
                )
            output_tokens = output_tokens.unsqueeze(0).expand(
                sparse_prompt_embeddings.size(0), -1, -1
            )
            tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

            # Expand per-image data in batch direction to be per-mask
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

            # Run the transformer
            hs, src = self.transformer(src, pos_src, tokens)
            iou_token_out = hs[:, s, :]
            mask_tokens_out = hs[:, s + 1: (s + 1 + self.num_mask_tokens), :]

            # Upscale mask embeddings and predict masks using the mask tokens
            src = src.transpose(1, 2).view(b, c, h, w)
            if not self.use_high_res_features:
                upscaled_embedding = self.output_upscaling(src)
            else:
                dc1, ln1, act1, dc2, act2 = self.output_upscaling
                feat_s0, feat_s1 = high_res_features
                upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
                upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

            hyper_in_list: List[torch.Tensor] = []
            for i in range(self.num_mask_tokens - 1):
                hyper_in_list.append(
                    self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
                )
            hyper_in = torch.stack(hyper_in_list, dim=1)
            b, c, h, w = upscaled_embedding.shape
            masks = (hyper_in[:, :self.num_mask_tokens - 1] @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

            # Generate mask quality predictions
            iou_pred = self.iou_prediction_head(iou_token_out)
            if self.pred_obj_scores:
                assert s == 1
                object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
            else:
                # Obj scores logits - default to 10.0, i.e. assuming the object is present, sigmoid(10)=1
                object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

            return masks, iou_pred, mask_tokens_out, object_score_logits, upscaled_embedding
        else:
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
                    [
                        self.iou_token.weight,
                        self.mask_tokens.weight,
                        self.decamouflaged_token.weight
                    ], dim=0
                )
            output_tokens = output_tokens.unsqueeze(0).expand(
                sparse_prompt_embeddings.size(0), -1, -1
            )
            tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

            if repeat_image:
                src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
                # print(torch.sum(src-image_embeddings))
            else:
                assert image_embeddings.shape[0] == tokens.shape[0]
                src = image_embeddings
            # print("src.shape", src.shape)
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

            # upscaling
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

        nz = np.nonzero(m_np_down)  # find all non-zero
        # After transform, there may be no nonzero in the label
        if len(nz[0]) != 0:  # if mask exist

            p = [np.min(nz[0]), np.min(nz[1])]  # left top
            pend = [np.max(nz[0]), np.max(nz[1])]  # right down

            # cropping to bounding box around ROI
            m_np_roi = np.copy(m_np_down)[p[0]:pend[0] + 1, p[1]:pend[1] + 1]  # roi
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
