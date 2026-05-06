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
from sam3.visualization_utils import draw_box_on_image


class VOCEvaluator:
    """VOC 数据集评估器，遵循 COCO 评估标准"""

    def __init__(
            self,
            ckpt_path: str,
            voc_root: str,
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

        if self.image_set == "test":
            raise ValueError(
                f"Unsupported image_set: {self.image_set}. "
                "VOC2012 test set does not have public annotations. "
                "Please use 'train', 'val', or 'trainval' instead."
            )

        self.annotations = self._load_voc_annotations()

        print(f"Loaded {len(self.annotations)} images for evaluation")

    def _load_voc_annotations(self) -> Dict[str, Dict]:
        annotations = {}

        imageset_file = os.path.join(
            self.voc_root,
            "ImageSets", "Main", f"{self.image_set}.txt"
        )

        with open(imageset_file, 'r') as f:
            image_ids = [line.strip() for line in f.readlines()]

        for img_id in image_ids:
            ann_file = os.path.join(
                self.voc_root,
                "Annotations", f"{img_id}.xml"
            )
            if not os.path.exists(ann_file):
                continue

            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(ann_file)
                root = tree.getroot()

                size = root.find('size')
                width = int(size.find('width').text)
                height = int(size.find('height').text)

                objects = []
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    bbox_elem = obj.find('bndbox')
                    bbox = [
                        float(bbox_elem.find('xmin').text),
                        float(bbox_elem.find('ymin').text),
                        float(bbox_elem.find('xmax').text),
                        float(bbox_elem.find('ymax').text),
                    ]
                    objects.append({
                        'category': name,
                        'bbox': bbox,
                    })

                img_path = os.path.join(
                    self.voc_root,
                    "JPEGImages", f"{img_id}.jpg"
                )

                annotations[img_id] = {
                    'image_path': img_path,
                    'width': width,
                    'height': height,
                    'objects': objects,
                }

            except Exception as e:
                print(f"Error loading annotation for {img_id}: {e}")
                continue

        return annotations

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

    def run_evaluation(self, output_dir: str) -> Dict:
        os.makedirs(output_dir, exist_ok=True)

        all_results = {
            'per_image': [],
            'per_category': defaultdict(list),
            'summary': {},
        }

        total_gt = 0
        total_pred = 0

        print(f"Starting evaluation on {len(self.annotations)} images...")

        for img_idx, (img_id, ann_info) in enumerate(self.annotations.items()):
            if (img_idx + 1) % 100 == 0:
                print(f"Processing image {img_idx + 1}/{len(self.annotations)}")

            try:
                image = Image.open(ann_info['image_path']).convert('RGB')
            except Exception as e:
                print(f"Error loading image {img_id}: {e}")
                continue

            width, height = image.size
            gt_objects = ann_info['objects']

            total_gt += len(gt_objects)

            pred_by_category = {}

            for category in self.VOC_CLASSES:
                processor = Sam3Processor(self.model, confidence_threshold=self.confidence_threshold)
                inference_state = processor.set_image(image)

                processor.reset_all_prompts(inference_state)
                inference_state = processor.set_text_prompt(
                    state=inference_state,
                    prompt=category
                )

                boxes = inference_state.get('boxes', [])
                scores = inference_state.get('scores', [])

                if len(scores) > 0:
                    for box, score in zip(boxes, scores):
                        if score > self.confidence_threshold:
                            if category not in pred_by_category:
                                pred_by_category[category] = []
                            pred_by_category[category].append({
                                'bbox': box.tolist(),
                                'score': float(score),
                                'category': category,
                            })
                            total_pred += 1

            image_result = self._analyze_image_predictions(
                img_id=img_id,
                gt_objects=gt_objects,
                predictions=pred_by_category,
                width=width,
                height=height,
            )

            all_results['per_image'].append(image_result)

            for category in self.VOC_CLASSES:
                cat_preds = pred_by_category.get(category, [])
                for pred in cat_preds:
                    all_results['per_category'][category].append(pred)

        all_results['summary'] = self._compute_summary(
            all_results['per_image'],
            total_gt,
            total_pred,
        )

        self._save_results(all_results, output_dir)
        self._print_summary(all_results['summary'])

        return all_results

    def _convert_cxcywh_to_xyxy(
            self,
            boxes: torch.Tensor,
            width: int,
            height: int
    ) -> torch.Tensor:
        boxes_xyxy = boxes.clone()
        boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * width
        boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * height
        boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * width
        boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * height
        return boxes_xyxy

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
        for category, preds in predictions.items():
            for pred in preds:
                all_preds.append(pred)

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

    def _compute_summary(
            self,
            per_image_results: List[Dict],
            total_gt: int,
            total_pred: int,
    ) -> Dict:
        total_correct = 0
        total_missed = 0
        total_fp = 0

        category_stats = defaultdict(lambda: {
            'correct': 0,
            'missed': 0,
            'fp': 0,
        })

        for img_result in per_image_results:
            total_correct += len(img_result['correct_detections'])
            total_missed += len(img_result['missed_detections'])
            total_fp += len(img_result['false_positives'])

            for correct_det in img_result['correct_detections']:
                category_stats[correct_det['gt_category']]['correct'] += 1

            for missed in img_result['missed_detections']:
                category_stats[missed['category']]['missed'] += 1

            for fp in img_result['false_positives']:
                category_stats[fp['category']]['fp'] += 1

        precision = total_correct / (total_correct + total_fp) if (total_correct + total_fp) > 0 else 0
        recall = total_correct / (total_correct + total_missed) if (total_correct + total_missed) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        summary = {
            'total_images': len(per_image_results),
            'total_ground_truth': total_gt,
            'total_predictions': total_pred,
            'total_correct': total_correct,
            'total_missed': total_missed,
            'total_false_positives': total_fp,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'per_category': dict(category_stats),
        }

        return summary

    def _save_results(self, results: Dict, output_dir: str):
        per_image_file = os.path.join(output_dir, 'per_image_results.json')
        with open(per_image_file, 'w') as f:
            json.dump(results['per_image'], f, indent=2)
        print(f"Saved per-image results to {per_image_file}")

        per_category_file = os.path.join(output_dir, 'per_category_results.json')
        with open(per_category_file, 'w') as f:
            json.dump(dict(results['per_category']), f, indent=2)
        print(f"Saved per-category results to {per_category_file}")

        summary_file = os.path.join(output_dir, 'summary.json')
        with open(summary_file, 'w') as f:
            json.dump(results['summary'], f, indent=2)
        print(f"Saved summary to {summary_file}")

        report_file = os.path.join(output_dir, 'evaluation_report.txt')
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SAM3 VOC Dataset Evaluation Report (COCO Standard)\n")
            f.write("=" * 80 + "\n\n")

            summary = results['summary']
            f.write("OVERALL STATISTICS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Images: {summary['total_images']}\n")
            f.write(f"Total Ground Truth Objects: {summary['total_ground_truth']}\n")
            f.write(f"Total Predictions: {summary['total_predictions']}\n")
            f.write(f"Correct Detections: {summary['total_correct']}\n")
            f.write(f"Missed Detections (False Negatives): {summary['total_missed']}\n")
            f.write(f"False Positives: {summary['total_false_positives']}\n\n")

            f.write("METRICS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Precision: {summary['precision']:.4f}\n")
            f.write(f"Recall: {summary['recall']:.4f}\n")
            f.write(f"F1 Score: {summary['f1_score']:.4f}\n\n")

            f.write("PER-CATEGORY BREAKDOWN:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Category':<15} {'Correct':>10} {'Missed':>10} {'FP':>10}\n")
            f.write("-" * 80 + "\n")

            for category in sorted(self.VOC_CLASSES):
                stats = summary['per_category'].get(category, {
                    'correct': 0, 'missed': 0, 'fp': 0
                })
                f.write(f"{category:<15} {stats['correct']:>10} {stats['missed']:>10} {stats['fp']:>10}\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"Saved text report to {report_file}")

    def _print_summary(self, summary: Dict):
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY (COCO Standard)")
        print("=" * 80)
        print(f"Total Images: {summary['total_images']}")
        print(f"Total Ground Truth: {summary['total_ground_truth']}")
        print(f"Total Predictions: {summary['total_predictions']}")
        print("-" * 80)
        print(f"Correct Detections: {summary['total_correct']}")
        print(f"Missed Detections: {summary['total_missed']}")
        print(f"False Positives: {summary['total_false_positives']}")
        print("-" * 80)
        print(f"Precision: {summary['precision']:.4f}")
        print(f"Recall: {summary['recall']:.4f}")
        print(f"F1 Score: {summary['f1_score']:.4f}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate SAM3 on VOC dataset (COCO Standard)')
    parser.add_argument(
        '--checkpoint',
        type=str,
        # default='/home/sun/.cache/modelscope/hub/models/facebook/sam3/sam3.pt',
        default=r'C:\Users\win10\.cache\modelscope\hub\models\facebook\sam3\sam3.pt',

        help='Path to SAM3 checkpoint'
    )
    parser.add_argument(
        '--voc-root',
        type=str,
        # default='/home/sun/data/coding/object_detection_based_on_context_learning/SAM_based/VOC2012_train_val',
        default=r'E:\user\szx\sam\VOC2012_train_val',

        help='Root directory of VOC dataset'
    )
    parser.add_argument(
        '--image-set',
        type=str,
        default='val',
        choices=['train', 'val', 'test'],
        help='Dataset split to evaluate'
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
        help='IoU threshold for matching predictions to ground truth'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./voc_eval_results_v2',
        help='Output directory for evaluation results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to run evaluation on'
    )

    args = parser.parse_args()

    evaluator = VOCEvaluator(
        ckpt_path=args.checkpoint,
        voc_root=args.voc_root,
        image_set=args.image_set,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
        device=args.device,
    )

    results = evaluator.run_evaluation(output_dir=args.output_dir)
