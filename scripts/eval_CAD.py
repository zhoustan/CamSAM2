import os
import argparse
import numpy as np
import torch
from PIL import Image
import json
import random
import re
import pandas as pd
from new_eval import *
from sam2.build_sam import build_camsam2_video_predictor
import cv2


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.float32).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print("\nNote: MPS support is preliminary. Expect different or degraded performance.")

    return device


def eval_cls_with_click(cls, output_mode, test_data_folder, gt_dir, predictor, output_path):
    video_dir = os.path.join(test_data_folder, cls, 'new_frames')
    gt_dir = os.path.join(gt_dir, cls, 'groundtruth')

    # process gt
    gt_files = [
        p for p in os.listdir(gt_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    ]
    gt_files_sorted = sorted(gt_files, key=lambda x: int(re.findall(r'\d+', x)[0]))

    gt_to_eval = []
    for i in gt_files_sorted:
        img_np = np.array(Image.open(os.path.join(gt_dir, i)))
        gt_to_eval.append(img_np)
    gt_to_eval = np.array(gt_to_eval)

    real_ann_idx = 0  # prompt on first frame
    foreground_pixels = np.argwhere(gt_to_eval[real_ann_idx])
    k = 1  # run with 1-click prompt
    positive_point = []
    if len(foreground_pixels) > 0:
        random.seed(42)
        for i in range(k):
            positive_point.append(random.choice(foreground_pixels))
    positive_point = [i[::-1] for i in positive_point]

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    inference_state = predictor.init_state(video_path=video_dir, output_mode=output_mode)
    predictor.reset_state(inference_state)

    ann_frame_idx = real_ann_idx  # the frame index we interact with
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

    # Let's add a positive click at (x, y) = (210, 350) to get started
    points = np.array(positive_point, dtype=np.float32)
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array([1] * k, np.int32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    prediction_to_eval = []
    for out_frame_idx in range(0, len(frame_names)):
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            prediction_to_eval.append(out_mask[0])
    prediction_to_eval = np.array(prediction_to_eval)

    save_dir = os.path.join(output_path, "click_prompt", cls)
    os.makedirs(save_dir, exist_ok=True)
    for i in range(len(gt_files_sorted)):
        cv2.imwrite(os.path.join(save_dir, gt_files_sorted[i]), prediction_to_eval[i]*255)

    metrics = calculate_metrics(prediction_to_eval, gt_to_eval)
    return metrics


def eval_cls_with_bbox(cls, output_mode, test_data_folder, gt_dir, predictor, output_path):
    video_dir = os.path.join(test_data_folder, cls, 'new_frames')
    gt_dir = os.path.join(gt_dir, cls, 'groundtruth')

    # process gt
    gt_files = [
        p for p in os.listdir(gt_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    ]
    gt_files_sorted = sorted(gt_files, key=lambda x: int(re.findall(r'\d+', x)[0]))

    gt_to_eval = []
    for i in gt_files_sorted:
        img_np = np.array(Image.open(os.path.join(gt_dir, i)))
        gt_to_eval.append(img_np)
    gt_to_eval = np.array(gt_to_eval)

    real_ann_idx = 0  # prompt on first frame
    y_indices, x_indices = np.where(gt_to_eval[real_ann_idx])

    x_min = np.min(x_indices)
    y_min = np.min(y_indices)
    x_max = np.max(x_indices)
    y_max = np.max(y_indices)

    # Return the bounding box in XYXY format
    bbox = np.array([x_min, y_min, x_max, y_max])

    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    inference_state = predictor.init_state(video_path=video_dir, output_mode=output_mode)
    predictor.reset_state(inference_state)

    ann_frame_idx = real_ann_idx  # the frame index we interact with
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        box=bbox
    )

    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    prediction_to_eval = []
    for out_frame_idx in range(0, len(frame_names)):
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            prediction_to_eval.append(out_mask[0])

    prediction_to_eval = np.array(prediction_to_eval)

    save_dir = os.path.join(output_path, "box_prompt", cls)
    os.makedirs(save_dir, exist_ok=True)
    for i in range(len(gt_files_sorted)):
        cv2.imwrite(os.path.join(save_dir, gt_files_sorted[i]), prediction_to_eval[i]*255)

    metrics = calculate_metrics(prediction_to_eval, gt_to_eval)
    return metrics


def eval_cls_with_mask(cls, output_mode, test_data_folder, gt_dir, predictor, output_path):
    video_dir = os.path.join(test_data_folder, cls, 'new_frames')
    gt_dir = os.path.join(gt_dir, cls, 'groundtruth')

    # process gt
    gt_files = [
        p for p in os.listdir(gt_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    ]
    gt_files_sorted = sorted(gt_files, key=lambda x: int(re.findall(r'\d+', x)[0]))

    gt_to_eval = []
    for i in gt_files_sorted:
        img_np = np.array(Image.open(os.path.join(gt_dir, i)))
        gt_to_eval.append(img_np)
    gt_to_eval = np.array(gt_to_eval)

    real_ann_idx = 0  # prompt on first frame
    full_mask = gt_to_eval[real_ann_idx]

    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    inference_state = predictor.init_state(video_path=video_dir, output_mode=output_mode)
    predictor.reset_state(inference_state)

    ann_frame_idx = real_ann_idx  # the frame index we interact with
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

    _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        mask=full_mask
    )

    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    prediction_to_eval = []
    for out_frame_idx in range(0, len(frame_names)):
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            prediction_to_eval.append(out_mask[0])
    prediction_to_eval = np.array(prediction_to_eval)

    save_dir = os.path.join(output_path, "mask_prompt", cls)
    os.makedirs(save_dir, exist_ok=True)
    for i in range(len(gt_files_sorted)):
        cv2.imwrite(os.path.join(save_dir, gt_files_sorted[i]), prediction_to_eval[i] * 255)

    metrics = calculate_metrics(prediction_to_eval, gt_to_eval)
    return metrics


def evaluate_class(cls, predictor, output_mode, prompt_type, test_data_folder, gt_dir, output_path):
    if prompt_type == 'mask':
        return eval_cls_with_mask(cls, output_mode, test_data_folder, gt_dir, predictor, output_path)
    elif prompt_type == 'box':
        return eval_cls_with_bbox(cls, output_mode, test_data_folder, gt_dir, predictor, output_path)
    else:
        return eval_cls_with_click(cls, output_mode, test_data_folder, gt_dir, predictor, output_path)


def parse_args():
    parser = argparse.ArgumentParser("CamSAM2 Evaluation", add_help=True)
    parser.add_argument("--model_cfg", type=str, default="sam2_hiera_t.yaml")
    # parser.add_argument("--ckpt_path", type=str, default="work_dir/CamSAM2_tiny.pth")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/sam2_hiera_tiny.pt")
    parser.add_argument("--output_mode", type=str, default="combined_mask",
                        choices=["original_sam2_mask", "combined_mask"])
    parser.add_argument("--prompt_types", type=str, default="mask,box,point")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--camsam2_extra", type=str, required=False)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    if args.camsam2_extra is not None:
        predictor = build_camsam2_video_predictor(args.model_cfg, args.ckpt_path, device=device, camsam2_extra=args.camsam2_extra)
    else:
        predictor = build_camsam2_video_predictor(args.model_cfg, args.ckpt_path, device=device)
    prompt_types = args.prompt_types.split(',') if "," in args.prompt_types else [args.prompt_types]
    classes = [cls for cls in os.listdir(args.data_path) if not cls.startswith('.')]

    for prompt_type in prompt_types:
        result_path = os.path.join(args.output_path, f"CAD_{prompt_type}.json")
        results = {}
        for cls in classes:
            metric = evaluate_class(cls, predictor, args.output_mode, prompt_type, args.data_path, args.gt_dir, args.output_path)
            results[cls] = convert_ndarray_to_list(metric)
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=4)

    # Summarize results
    results_list = []
    for prompt_type in prompt_types:
        result_path = os.path.join(args.output_path, f"CAD_{prompt_type}.json")
        results = parse_result_json(result_path)
        results['prompt_type'] = prompt_type
        results_list.append(results)

    df_results = pd.DataFrame(results_list)
    columns_order = ['prompt_type'] + [col for col in df_results.columns if col != 'prompt_type']
    df_results = df_results[columns_order]
    df_results = df_results.drop(columns=['BIoU', 'TIoU', 'Boundary Accuracy'], errors='ignore')
    df_results = df_results.round(3)

    output_csv = os.path.join(args.output_path, "result.csv")
    df_results.to_csv(output_csv, sep='\t', encoding='utf-8')
    print(df_results)


if __name__ == "__main__":
    main()
