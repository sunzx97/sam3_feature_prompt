import torch
import os
import json
from pathlib import Path
from PIL import Image
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import numpy as np

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.box_ops import box_xywh_to_cxcywh


def normalize_bbox(bbox, width, height):
    """归一化 bbox 到 [0, 1]"""
    normalized = bbox.clone()
    normalized[:, 0] /= width
    normalized[:, 1] /= height
    normalized[:, 2] /= width
    normalized[:, 3] /= height
    return normalized


class VOCRefineEvaluator:
    """VOC 数据集重新评估器，通过添加提示框优化检测"""

    def __init__(
            self,
            ckpt_path: str,
            voc_root: str,
            per_image_results_path: str,
            image_set: str = "val",
            confidence_threshold: float = 0.5,
            iou_threshold: float = 0.5,
            device: str = "cuda"
    ):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        self.VOC_CLASSES = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
            'dog', 'horse', 'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor'
        ]

        print("Loading SAM3 model...")
        self.model = build_sam3_image_model(
            checkpoint_path=ckpt_path,
            load_from_HF=False,
            device=device,
            eval_mode=True,
        )

        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

        self.voc_root = voc_root
        self.image_set = image_set

        with open(per_image_results_path, 'r') as f:
            self.per_image_results = json.load(f)

        print(f"Loaded {len(self.per_image_results)} images for refinement")

    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        inter_w = max(0, inter_x_max - inter_x_min)
        inter_h = max(0, inter_y_max - inter_y_min)
        inter_area = inter_w * inter_h

        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area if union_area > 0 else 0.0
        return iou

    # def run_refinement_evaluation(self, output_dir: str) -> Dict:
    #     os.makedirs(output_dir, exist_ok=True)
    #
    #     refined_results = {
    #         'per_image': [],
    #         'summary': {},
    #     }
    #
    #     total_gt_before = 0
    #     total_missed_before = 0
    #     total_fp_before = 0
    #     total_correct_after = 0
    #     total_missed_after = 0
    #     total_fp_after = 0
    #
    #     print(f"Starting refinement evaluation on {len(self.per_image_results)} images...")
    #
    #     i = 0
    #
    #     for img_idx, img_result in enumerate(self.per_image_results):
    #         i = i + 1
    #         if i > 6:
    #             break
    #         if (img_idx + 1) % 10 == 0:
    #             print(f"Processing image {img_idx + 1}/{len(self.per_image_results)}")
    #
    #         img_id = img_result['image_id']
    #         missed_detections = img_result.get('missed_detections', [])
    #         false_positives = img_result.get('false_positives', [])
    #         correct_detections = img_result.get('correct_detections', [])
    #
    #         total_gt_before += len(missed_detections) + len(correct_detections)
    #         total_missed_before += len(missed_detections)
    #         total_fp_before += len(false_positives)
    #
    #         try:
    #             img_path = os.path.join(
    #                 self.voc_root,
    #                 "JPEGImages", f"{img_id}.jpg"
    #             )
    #             image = Image.open(img_path).convert('RGB')
    #         except Exception as e:
    #             print(f"Error loading image {img_id}: {e}")
    #             continue
    #
    #         width, height = image.size
    #
    #         processor = Sam3Processor(self.model, confidence_threshold=self.confidence_threshold)
    #         inference_state = processor.set_image(image)
    #
    #         prompts_added = []
    #
    #         for missed in missed_detections:
    #             category = missed['category']
    #             bbox_xyxy = missed['bbox']
    #
    #             bbox_xyxy_tensor = torch.tensor(bbox_xyxy).view(-1, 4)
    #             bbox_cxcywh = box_xywh_to_cxcywh(bbox_xyxy_tensor)
    #             norm_box = normalize_bbox(bbox_cxcywh, width, height).flatten().tolist()
    #
    #             processor.reset_all_prompts(inference_state)
    #             inference_state = processor.set_text_prompt(
    #                 state=inference_state,
    #                 prompt=category
    #             )
    #             inference_state = processor.add_geometric_prompt(
    #                 state=inference_state,
    #                 box=norm_box,
    #                 label=True
    #             )
    #
    #             prompts_added.append({
    #                 'type': 'missed',
    #                 'category': category,
    #                 'bbox': bbox_xyxy,
    #                 'label': True
    #             })
    #
    #         for fp in false_positives:
    #             category = fp['category']
    #             bbox_xyxy = fp['bbox']
    #
    #             bbox_xyxy_tensor = torch.tensor(bbox_xyxy).view(-1, 4)
    #             bbox_cxcywh = box_xywh_to_cxcywh(bbox_xyxy_tensor)
    #             norm_box = normalize_bbox(bbox_cxcywh, width, height).flatten().tolist()
    #
    #             processor.reset_all_prompts(inference_state)
    #             inference_state = processor.set_text_prompt(
    #                 state=inference_state,
    #                 prompt=category
    #             )
    #             inference_state = processor.add_geometric_prompt(
    #                 state=inference_state,
    #                 box=norm_box,
    #                 label=False
    #             )
    #
    #             prompts_added.append({
    #                 'type': 'false_positive',
    #                 'category': category,
    #                 'bbox': bbox_xyxy,
    #                 'label': False
    #             })
    #
    #         boxes = inference_state.get('boxes', [])
    #         scores = inference_state.get('scores', [])
    #
    #         refined_preds = []
    #         if len(scores) > 0:
    #             for box, score in zip(boxes, scores):
    #                 if score > self.confidence_threshold:
    #                     refined_preds.append({
    #                         'bbox': box.tolist(),
    #                         'score': float(score),
    #                         'category': self._get_category_from_state(inference_state, box),
    #                     })
    #
    #         gt_objects = []
    #         for correct in correct_detections:
    #             gt_objects.append({
    #                 'category': correct['gt_category'],
    #                 'bbox': correct['gt_bbox']
    #             })
    #         for missed in missed_detections:
    #             gt_objects.append({
    #                 'category': missed['category'],
    #                 'bbox': missed['bbox']
    #             })
    #
    #         image_result = self._analyze_image_predictions(
    #             img_id=img_id,
    #             gt_objects=gt_objects,
    #             predictions=refined_preds,
    #             width=width,
    #             height=height,
    #         )
    #
    #         image_result['prompts_added'] = prompts_added
    #         image_result['before_stats'] = {
    #             'missed': len(missed_detections),
    #             'fp': len(false_positives),
    #             'correct': len(correct_detections)
    #         }
    #         image_result['after_stats'] = {
    #             'missed': len(image_result['missed_detections']),
    #             'fp': len(image_result['false_positives']),
    #             'correct': len(image_result['correct_detections'])
    #         }
    #
    #         refined_results['per_image'].append(image_result)
    #
    #         total_correct_after += len(image_result['correct_detections'])
    #         total_missed_after += len(image_result['missed_detections'])
    #         total_fp_after += len(image_result['false_positives'])
    #
    #     refined_results['summary'] = {
    #         'total_images': len(refined_results['per_image']),
    #         'before': {
    #             'total_gt': total_gt_before,
    #             'missed': total_missed_before,
    #             'fp': total_fp_before,
    #         },
    #         'after': {
    #             'total_correct': total_correct_after,
    #             'missed': total_missed_after,
    #             'fp': total_fp_after,
    #         },
    #         'improvement': {
    #             'missed_reduction': total_missed_before - total_missed_after,
    #             'fp_reduction': total_fp_before - total_fp_after,
    #         }
    #     }
    #
    #     self._save_results(refined_results, output_dir)
    #     self._print_summary(refined_results['summary'])
    #
    #     return refined_results
    # def run_refinement_evaluation(self, output_dir: str) -> Dict:
    #     os.makedirs(output_dir, exist_ok=True)
    #
    #     refined_results = {
    #         'per_image': [],
    #         'summary': {},
    #     }
    #
    #     total_gt_before = 0
    #     total_missed_before = 0
    #     total_fp_before = 0
    #     total_correct_after = 0
    #     total_missed_after = 0
    #     total_fp_after = 0
    #
    #     print(f"Starting refinement evaluation on {len(self.per_image_results)} images...")
    #
    #     i = 0
    #
    #     for img_idx, img_result in enumerate(self.per_image_results):
    #         i = i + 1
    #         if i > 6:
    #             break
    #         if (img_idx + 1) % 10 == 0:
    #             print(f"Processing image {img_idx + 1}/{len(self.per_image_results)}")
    #
    #         img_id = img_result['image_id']
    #         missed_detections = img_result.get('missed_detections', [])
    #         false_positives = img_result.get('false_positives', [])
    #         correct_detections = img_result.get('correct_detections', [])
    #
    #         total_gt_before += len(missed_detections) + len(correct_detections)
    #         total_missed_before += len(missed_detections)
    #         total_fp_before += len(false_positives)
    #
    #         try:
    #             img_path = os.path.join(
    #                 self.voc_root,
    #                 "JPEGImages", f"{img_id}.jpg"
    #             )
    #             image = Image.open(img_path).convert('RGB')
    #         except Exception as e:
    #             print(f"Error loading image {img_id}: {e}")
    #             continue
    #
    #         width, height = image.size
    #
    #         processor = Sam3Processor(self.model, confidence_threshold=self.confidence_threshold)
    #         inference_state = processor.set_image(image)
    #
    #         all_refined_preds = []
    #         prompts_added = []
    #
    #         for missed in missed_detections:
    #             category = missed['category']
    #             bbox_xyxy = missed['bbox']
    #
    #             bbox_xyxy_tensor = torch.tensor(bbox_xyxy).view(-1, 4)
    #             bbox_cxcywh = box_xywh_to_cxcywh(bbox_xyxy_tensor)
    #             norm_box = normalize_bbox(bbox_cxcywh, width, height).flatten().tolist()
    #
    #             processor.reset_all_prompts(inference_state)
    #             inference_state = processor.set_text_prompt(
    #                 state=inference_state,
    #                 prompt=category
    #             )
    #             inference_state = processor.add_geometric_prompt(
    #                 state=inference_state,
    #                 box=norm_box,
    #                 label=True
    #             )
    #
    #             boxes = inference_state.get('boxes', [])
    #             scores = inference_state.get('scores', [])
    #
    #             if len(scores) > 0:
    #                 for box, score in zip(boxes, scores):
    #                     if score > self.confidence_threshold:
    #                         all_refined_preds.append({
    #                             'bbox': box.tolist(),
    #                             'score': float(score),
    #                             'category': category,
    #                         })
    #
    #             prompts_added.append({
    #                 'type': 'missed',
    #                 'category': category,
    #                 'bbox': bbox_xyxy,
    #                 'label': True
    #             })
    #
    #         for fp in false_positives:
    #             category = fp['category']
    #             bbox_xyxy = fp['bbox']
    #
    #             bbox_xyxy_tensor = torch.tensor(bbox_xyxy).view(-1, 4)
    #             bbox_cxcywh = box_xywh_to_cxcywh(bbox_xyxy_tensor)
    #             norm_box = normalize_bbox(bbox_cxcywh, width, height).flatten().tolist()
    #
    #             processor.reset_all_prompts(inference_state)
    #             inference_state = processor.set_text_prompt(
    #                 state=inference_state,
    #                 prompt=category
    #             )
    #             inference_state = processor.add_geometric_prompt(
    #                 state=inference_state,
    #                 box=norm_box,
    #                 label=False
    #             )
    #
    #             prompts_added.append({
    #                 'type': 'false_positive',
    #                 'category': category,
    #                 'bbox': bbox_xyxy,
    #                 'label': False
    #             })
    #
    #         gt_objects = []
    #         for correct in correct_detections:
    #             gt_objects.append({
    #                 'category': correct['gt_category'],
    #                 'bbox': correct['gt_bbox']
    #             })
    #         for missed in missed_detections:
    #             gt_objects.append({
    #                 'category': missed['category'],
    #                 'bbox': missed['bbox']
    #             })
    #
    #         image_result = self._analyze_image_predictions(
    #             img_id=img_id,
    #             gt_objects=gt_objects,
    #             predictions=all_refined_preds,
    #             width=width,
    #             height=height,
    #         )
    #
    #         image_result['prompts_added'] = prompts_added
    #         image_result['before_stats'] = {
    #             'missed': len(missed_detections),
    #             'fp': len(false_positives),
    #             'correct': len(correct_detections)
    #         }
    #         image_result['after_stats'] = {
    #             'missed': len(image_result['missed_detections']),
    #             'fp': len(image_result['false_positives']),
    #             'correct': len(image_result['correct_detections'])
    #         }
    #
    #         refined_results['per_image'].append(image_result)
    #
    #         total_correct_after += len(image_result['correct_detections'])
    #         total_missed_after += len(image_result['missed_detections'])
    #         total_fp_after += len(image_result['false_positives'])
    #
    #     refined_results['summary'] = {
    #         'total_images': len(refined_results['per_image']),
    #         'before': {
    #             'total_gt': total_gt_before,
    #             'missed': total_missed_before,
    #             'fp': total_fp_before,
    #         },
    #         'after': {
    #             'total_correct': total_correct_after,
    #             'missed': total_missed_after,
    #             'fp': total_fp_after,
    #         },
    #         'improvement': {
    #             'missed_reduction': total_missed_before - total_missed_after,
    #             'fp_reduction': total_fp_before - total_fp_after,
    #         }
    #     }
    #
    #     self._save_results(refined_results, output_dir)
    #     self._print_summary(refined_results['summary'])
    #
    #     return refined_results

    def run_refinement_evaluation(self, output_dir: str) -> Dict:
        os.makedirs(output_dir, exist_ok=True)

        refined_results = {
            'per_image': [],
            'summary': {},
        }

        total_gt_before = 0
        total_missed_before = 0
        total_fp_before = 0
        total_correct_after = 0
        total_missed_after = 0
        total_fp_after = 0

        print(f"Starting refinement evaluation on {len(self.per_image_results)} images...")

        # i = 0

        for img_idx, img_result in enumerate(self.per_image_results):
            # i = i + 1
            # if i > 6:
            #     break
            if (img_idx + 1) % 10 == 0:
                print(f"Processing image {img_idx + 1}/{len(self.per_image_results)}")

            img_id = img_result['image_id']
            missed_detections = img_result.get('missed_detections', [])
            false_positives = img_result.get('false_positives', [])
            correct_detections = img_result.get('correct_detections', [])

            total_gt_before += len(missed_detections) + len(correct_detections)
            total_missed_before += len(missed_detections)
            total_fp_before += len(false_positives)

            try:
                img_path = os.path.join(
                    self.voc_root,
                    "JPEGImages", f"{img_id}.jpg"
                )
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {img_id}: {e}")
                continue

            width, height = image.size

            all_refined_preds = []
            prompts_added = []

            for category in self.VOC_CLASSES:
                category_missed = [m for m in missed_detections if m['category'] == category]
                category_fp = [fp for fp in false_positives if fp['category'] == category]

                # if not category_missed and not category_fp:
                #     continue

                processor = Sam3Processor(self.model, confidence_threshold=self.confidence_threshold)
                inference_state = processor.set_image(image)

                processor.reset_all_prompts(inference_state)
                inference_state = processor.set_text_prompt(
                    state=inference_state,
                    prompt=category
                )

                for missed in category_missed:
                    bbox_xyxy = missed['bbox']
                    # bbox_xyxy_tensor = torch.tensor(bbox_xyxy).view(-1, 4)
                    # bbox_cxcywh = box_xywh_to_cxcywh(bbox_xyxy_tensor)
                    # norm_box = normalize_bbox(bbox_cxcywh, width, height).flatten().tolist()
                    x_min, y_min, x_max, y_max = bbox_xyxy
                    center_x = (x_min + x_max) / 2.0
                    center_y = (y_min + y_max) / 2.0
                    width_box = x_max - x_min
                    height_box = y_max - y_min
                    bbox_cxcywh = [center_x, center_y, width_box, height_box]
                    norm_box = normalize_bbox(torch.tensor(bbox_cxcywh).view(-1, 4), width, height).flatten().tolist()

                    inference_state = processor.add_geometric_prompt(
                        state=inference_state,
                        box=norm_box,
                        label=True
                    )

                    prompts_added.append({
                        'type': 'missed',
                        'category': category,
                        'bbox': bbox_xyxy,
                        'label': True
                    })

                for fp in category_fp:
                    bbox_xyxy = fp['bbox']
                    # bbox_xyxy_tensor = torch.tensor(bbox_xyxy).view(-1, 4)
                    # bbox_cxcywh = box_xywh_to_cxcywh(bbox_xyxy_tensor)
                    # norm_box = normalize_bbox(bbox_cxcywh, width, height).flatten().tolist()
                    x_min, y_min, x_max, y_max = bbox_xyxy
                    center_x = (x_min + x_max) / 2.0
                    center_y = (y_min + y_max) / 2.0
                    width_box = x_max - x_min
                    height_box = y_max - y_min
                    bbox_cxcywh = [center_x, center_y, width_box, height_box]
                    norm_box = normalize_bbox(torch.tensor(bbox_cxcywh).view(-1, 4), width, height).flatten().tolist()

                    inference_state = processor.add_geometric_prompt(
                        state=inference_state,
                        box=norm_box,
                        label=False
                    )

                    prompts_added.append({
                        'type': 'false_positive',
                        'category': category,
                        'bbox': bbox_xyxy,
                        'label': False
                    })

                boxes = inference_state.get('boxes', [])
                scores = inference_state.get('scores', [])

                if len(scores) > 0:
                    for box, score in zip(boxes, scores):
                        if score > self.confidence_threshold:
                            all_refined_preds.append({
                                'bbox': box.tolist(),
                                'score': float(score),
                                'category': category,
                            })

            gt_objects = []
            for correct in correct_detections:
                gt_objects.append({
                    'category': correct['gt_category'],
                    'bbox': correct['gt_bbox']
                })
            for missed in missed_detections:
                gt_objects.append({
                    'category': missed['category'],
                    'bbox': missed['bbox']
                })

            image_result = self._analyze_image_predictions(
                img_id=img_id,
                gt_objects=gt_objects,
                predictions=all_refined_preds,
                width=width,
                height=height,
            )

            image_result['prompts_added'] = prompts_added
            image_result['before_stats'] = {
                'missed': len(missed_detections),
                'fp': len(false_positives),
                'correct': len(correct_detections)
            }
            image_result['after_stats'] = {
                'missed': len(image_result['missed_detections']),
                'fp': len(image_result['false_positives']),
                'correct': len(image_result['correct_detections'])
            }

            refined_results['per_image'].append(image_result)

            total_correct_after += len(image_result['correct_detections'])
            total_missed_after += len(image_result['missed_detections'])
            total_fp_after += len(image_result['false_positives'])

        refined_results['summary'] = {
            'total_images': len(refined_results['per_image']),
            'before': {
                'total_gt': total_gt_before,
                'missed': total_missed_before,
                'fp': total_fp_before,
            },
            'after': {
                'total_correct': total_correct_after,
                'missed': total_missed_after,
                'fp': total_fp_after,
            },
            'improvement': {
                'missed_reduction': total_missed_before - total_missed_after,
                'fp_reduction': total_fp_before - total_fp_after,
            }
        }

        self._save_results(refined_results, output_dir)
        self._print_summary(refined_results['summary'])

        return refined_results


    def _get_category_from_state(self, inference_state, box):
        boxes = inference_state.get('boxes', [])
        for i, b in enumerate(boxes):
            if torch.allclose(b, box, atol=1e-4):
                return inference_state.get('categories', [None])[i]
        return None

    def _analyze_image_predictions(
            self,
            img_id: str,
            gt_objects: List[Dict],
            predictions: Dict[str, List[Dict]],
            width: int,
            height: int,
    ) -> Dict:
        result = {
            'image_id': img_id,
            'missed_detections': [],
            'false_positives': [],
            'correct_detections': [],
        }

        all_preds = []
        if isinstance(predictions, dict):
            for category, preds in predictions.items():
                if isinstance(preds, list):
                    for pred in preds:
                        all_preds.append(pred)
        elif isinstance(predictions, list):
            all_preds = predictions

        gt_matched = [False] * len(gt_objects)
        pred_matched = [False] * len(all_preds)

        for pred_idx, pred in enumerate(all_preds):
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt_obj in enumerate(gt_objects):
                if gt_matched[gt_idx]:
                    continue

                if pred['category'] != gt_obj['category']:
                    continue

                iou = self._compute_iou(pred['bbox'], gt_obj['bbox'])

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= self.iou_threshold and best_gt_idx >= 0:
                gt_matched[best_gt_idx] = True
                pred_matched[pred_idx] = True

                matched_gt = gt_objects[best_gt_idx]

                result['correct_detections'].append({
                    'gt_category': matched_gt['category'],
                    'pred_category': pred['category'],
                    'iou': best_iou,
                    'score': pred['score'],
                    'gt_bbox': matched_gt['bbox'],
                    'pred_bbox': pred['bbox'],
                })

        for gt_idx, gt_obj in enumerate(gt_objects):
            if not gt_matched[gt_idx]:
                result['missed_detections'].append({
                    'category': gt_obj['category'],
                    'bbox': gt_obj['bbox'],
                })

        for pred_idx, pred in enumerate(all_preds):
            if not pred_matched[pred_idx]:
                result['false_positives'].append({
                    'category': pred['category'],
                    'bbox': pred['bbox'],
                    'score': pred['score'],
                })

        return result

    def _save_results(self, results: Dict, output_dir: str):
        per_image_file = os.path.join(output_dir, 'refined_per_image_results.json')
        with open(per_image_file, 'w') as f:
            json.dump(results['per_image'], f, indent=2)
        print(f"Saved refined per-image results to {per_image_file}")

        summary_file = os.path.join(output_dir, 'refinement_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(results['summary'], f, indent=2)
        print(f"Saved refinement summary to {summary_file}")

        report_file = os.path.join(output_dir, 'refinement_report.txt')
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SAM3 VOC Dataset Refinement Evaluation Report\n")
            f.write("=" * 80 + "\n\n")

            summary = results['summary']
            f.write("BEFORE REFINEMENT:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total GT: {summary['before']['total_gt']}\n")
            f.write(f"Missed Detections: {summary['before']['missed']}\n")
            f.write(f"False Positives: {summary['before']['fp']}\n\n")

            f.write("AFTER REFINEMENT:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Correct Detections: {summary['after']['total_correct']}\n")
            f.write(f"Missed Detections: {summary['after']['missed']}\n")
            f.write(f"False Positives: {summary['after']['fp']}\n\n")

            f.write("IMPROVEMENT:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Missed Reduction: {summary['improvement']['missed_reduction']}\n")
            f.write(f"FP Reduction: {summary['improvement']['fp_reduction']}\n")

            if summary['before']['missed'] > 0:
                missed_improve = (summary['improvement']['missed_reduction'] / summary['before']['missed']) * 100
                f.write(f"Missed Improvement Rate: {missed_improve:.2f}%\n")

            if summary['before']['fp'] > 0:
                fp_improve = (summary['improvement']['fp_reduction'] / summary['before']['fp']) * 100
                f.write(f"FP Improvement Rate: {fp_improve:.2f}%\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"Saved text report to {report_file}")

    def _print_summary(self, summary: Dict):
        print("\n" + "=" * 80)
        print("REFINEMENT EVALUATION SUMMARY")
        print("=" * 80)
        print("\nBEFORE REFINEMENT:")
        print(f"  Total GT: {summary['before']['total_gt']}")
        print(f"  Missed: {summary['before']['missed']}")
        print(f"  False Positives: {summary['before']['fp']}")
        print("\nAFTER REFINEMENT:")
        print(f"  Correct: {summary['after']['total_correct']}")
        print(f"  Missed: {summary['after']['missed']}")
        print(f"  False Positives: {summary['after']['fp']}")
        print("\nIMPROVEMENT:")
        print(f"  Missed Reduction: {summary['improvement']['missed_reduction']}")
        print(f"  FP Reduction: {summary['improvement']['fp_reduction']}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Refine SAM3 VOC evaluation with geometric prompts')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=r'C:\Users\win10\.cache\modelscope\hub\models\facebook\sam3\sam3.pt',
        # default=r'/home/sun/.cache/modelscope/hub/models/facebook/sam3/sam3.pt',
        help='Path to SAM3 checkpoint'
    )
    parser.add_argument(
        '--voc-root',
        type=str,
        # default=r'/home/sun/data/coding/object_detection_based_on_context_learning/SAM_based/VOC2012_train_val',
        default=r'E:\user\szx\sam\VOC2012_train_val',
        help='Root directory of VOC dataset'
    )
    parser.add_argument(
        '--image-set',
        type=str,
        default='val',
        choices=['train', 'val', 'test'],
        help='Dataset split'
    )
    parser.add_argument(
        '--per-image-results',
        type=str,
        default='./voc_eval_results_v2/per_image_results.json',
        help='Path to per-image results JSON from initial evaluation'
    )
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.5,
        help='Confidence threshold for detections'
    )
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.5,
        help='IoU threshold for matching'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./voc_refinement_results_v2',
        help='Output directory for refinement results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to run evaluation on'
    )

    args = parser.parse_args()

    evaluator = VOCRefineEvaluator(
        ckpt_path=args.checkpoint,
        voc_root=args.voc_root,
        per_image_results_path=args.per_image_results,
        image_set=args.image_set,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
        device=args.device,
    )

    results = evaluator.run_refinement_evaluation(output_dir=args.output_dir)
