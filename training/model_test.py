import os
import json
import time
import argparse
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


def _list_images(images_dir: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    out: List[str] = []
    for root, _dirs, files in os.walk(images_dir):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in exts:
                out.append(os.path.join(root, fn))
    out.sort()
    return out


def _default_labels_dir(images_dir: str) -> str:
    """
    Try to follow YOLO dataset layout:
    .../images  -> .../labels
    """
    parent = os.path.dirname(os.path.normpath(images_dir))
    base = os.path.basename(os.path.normpath(images_dir)).lower()
    if base == "images":
        return os.path.join(parent, "labels")
    # fallback: sibling labels
    return os.path.join(parent, "labels")


def _label_path_for_image(image_path: str, images_dir: str, labels_dir: str) -> str:
    rel = os.path.relpath(image_path, images_dir)
    rel_noext = os.path.splitext(rel)[0]
    return os.path.join(labels_dir, rel_noext + ".txt")


def _load_image_wh(image_path: str) -> Tuple[int, int]:
    # pillow is already a dependency via YOLOPredictor; keep it lightweight here
    from PIL import Image

    with Image.open(image_path) as im:
        w, h = im.size
    return w, h


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    cls: int
    conf: Optional[float] = None


def _yolo_to_xyxy(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    x1 = (xc - w / 2.0) * img_w
    y1 = (yc - h / 2.0) * img_h
    x2 = (xc + w / 2.0) * img_w
    y2 = (yc + h / 2.0) * img_h
    return x1, y1, x2, y2


def _read_yolo_labels(label_path: str, img_w: int, img_h: int) -> List[Box]:
    if not os.path.exists(label_path):
        return []
    out: List[Box] = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                xc = float(parts[1])
                yc = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except Exception:
                continue
            x1, y1, x2, y2 = _yolo_to_xyxy(xc, yc, w, h, img_w, img_h)
            out.append(Box(x1=x1, y1=y1, x2=x2, y2=y2, cls=cls))
    return out


def _iou(a: Box, b: Box) -> float:
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _run_predict_ultralytics(model_path: str, image_path: str, device: str, conf: float, iou_thr: float, max_det: int) -> List[Box]:
    # 如果是TensorRT模型，先设置PATH
    if model_path.lower().endswith('.engine'):
        try:
            from utils.model_converter.tensorrt_converter import TensorRTConverter
            TensorRTConverter._add_tensorrt_to_path()
        except ImportError:
            pass  # 如果导入失败，继续执行
    
    from ultralytics import YOLO

    model = YOLO(model_path)
    results = model.predict(
        source=image_path,
        conf=conf,
        iou=iou_thr,
        max_det=max_det,
        device=device,
        verbose=False,
    )
    out: List[Box] = []
    if not results:
        return out
    r0 = results[0]
    boxes = getattr(r0, "boxes", None)
    if boxes is None:
        return out
    n = len(boxes)
    for i in range(n):
        b = boxes[i]
        xyxy = b.xyxy[0].cpu().numpy().tolist()
        cls = int(b.cls[0].cpu().numpy().item())
        c = float(b.conf[0].cpu().numpy().item())
        out.append(Box(x1=float(xyxy[0]), y1=float(xyxy[1]), x2=float(xyxy[2]), y2=float(xyxy[3]), cls=cls, conf=c))
    return out


def _run_predict_onnx(model_path: str, image_path: str, device: str, conf: float, _iou_thr: float, _max_det: int) -> List[Box]:
    # Minimal ONNX path: use YOLOPredictor's onnx branch logic without importing app UI.
    # Here we do a conservative parse: assumes output is [x1,y1,x2,y2,conf,cls] normalized or absolute depending on model.
    import numpy as np
    import onnxruntime as ort
    from PIL import Image

    providers = ["CPUExecutionProvider"]
    if device in ("cuda", "gpu"):
        avail = ort.get_available_providers()
        if "CUDAExecutionProvider" in avail:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    sess = ort.InferenceSession(model_path, providers=providers)
    input_name = sess.get_inputs()[0].name

    image = Image.open(image_path).convert("RGB")
    img = np.array(image)
    img_h, img_w = img.shape[0], img.shape[1]

    x = img.transpose(2, 0, 1).astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)
    outputs = sess.run(None, {input_name: x})
    if not outputs or len(outputs[0]) == 0:
        return []
    det = outputs[0]
    out: List[Box] = []

    # Handle common shapes: (N,6) or (1,N,6)
    if isinstance(det, list):
        det = np.array(det)
    det = np.array(det)
    if det.ndim == 3 and det.shape[0] == 1:
        det = det[0]
    if det.ndim != 2 or det.shape[1] < 6:
        return []

    for row in det:
        try:
            x1, y1, x2, y2, c, cls = row[:6]
            c = float(c)
            if c < conf:
                continue
            # If coords look normalized, upscale
            if 0.0 <= float(x2) <= 1.5 and 0.0 <= float(y2) <= 1.5:
                x1, y1, x2, y2 = float(x1) * img_w, float(y1) * img_h, float(x2) * img_w, float(y2) * img_h
            out.append(Box(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2), cls=int(cls), conf=c))
        except Exception:
            continue
    return out


def _match_tp_fp_fn(pred: List[Box], gt: List[Box], iou_thr: float) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
    """
    Greedy matching per class by best IoU.
    Returns per-class tp/fp/fn dicts.
    """
    tp: Dict[int, int] = {}
    fp: Dict[int, int] = {}
    fn: Dict[int, int] = {}

    gt_by_cls: Dict[int, List[Box]] = {}
    for g in gt:
        gt_by_cls.setdefault(g.cls, []).append(g)

    pred_by_cls: Dict[int, List[Box]] = {}
    for p in pred:
        pred_by_cls.setdefault(p.cls, []).append(p)

    classes = set(gt_by_cls.keys()) | set(pred_by_cls.keys())
    for cls in classes:
        gts = gt_by_cls.get(cls, [])
        preds = pred_by_cls.get(cls, [])
        used = [False] * len(gts)
        cls_tp = 0
        cls_fp = 0

        # sort by confidence so "stronger" predictions match first
        preds_sorted = sorted(preds, key=lambda b: (b.conf if b.conf is not None else 1.0), reverse=True)
        for p in preds_sorted:
            best_iou = 0.0
            best_j = -1
            for j, g in enumerate(gts):
                if used[j]:
                    continue
                v = _iou(p, g)
                if v > best_iou:
                    best_iou = v
                    best_j = j
            if best_j >= 0 and best_iou >= iou_thr:
                used[best_j] = True
                cls_tp += 1
            else:
                cls_fp += 1

        cls_fn = used.count(False)
        if cls_tp:
            tp[cls] = cls_tp
        if cls_fp:
            fp[cls] = cls_fp
        if cls_fn:
            fn[cls] = cls_fn

    return tp, fp, fn


def main():
    ap = argparse.ArgumentParser(description="Model test: run inference on a dataset images folder and summarize stats.")
    ap.add_argument("--model", required=True, help="Model path (.pt/.pth/.onnx/.engine)")
    ap.add_argument("--images", required=True, help="Images directory (will be scanned recursively)")
    ap.add_argument("--labels", default="", help="Labels directory (optional). If empty, infer from images dir.")
    ap.add_argument("--device", default="cpu", help="Device for inference (cpu/cuda/0 etc, depending on backend). Note: TensorRT models require CUDA.")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU threshold for matching & (ultralytics) NMS")
    ap.add_argument("--max-det", type=int, default=300, help="Max detections per image")
    ap.add_argument("--out", default="", help="Output json path. Default: logs/model_test_result.json")
    args = ap.parse_args()

    model_path = os.path.abspath(args.model)
    images_dir = os.path.abspath(args.images)
    labels_dir = os.path.abspath(args.labels) if args.labels else _default_labels_dir(images_dir)

    if not os.path.exists(model_path):
        raise SystemExit(f"Model not found: {model_path}")
    if not os.path.isdir(images_dir):
        raise SystemExit(f"Images dir not found: {images_dir}")

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # allow importing project modules when executed via `python training/model_test.py`
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    logs_dir = os.path.join(root_dir, "logs")
    _ensure_dir(logs_dir)
    out_path = os.path.abspath(args.out) if args.out else os.path.join(logs_dir, "model_test_result.json")

    images = _list_images(images_dir)
    if not images:
        raise SystemExit(f"No images found under: {images_dir}")

    # Use the same predictor as the main app (auto label), to keep behavior consistent.
    from utils.yolo_predictor import YOLOPredictor

    predictor = YOLOPredictor()
    predictor.set_params(
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        max_detections=args.max_det,
        device=args.device,
    )
    if not predictor.load_model(model_path):
        raise SystemExit("Failed to load model in YOLOPredictor.")
    backend = predictor.model_type or "unknown"

    total_pred = 0
    total_gt = 0
    images_with_labels = 0
    total_time_s = 0.0

    tp_all: Dict[int, int] = {}
    fp_all: Dict[int, int] = {}
    fn_all: Dict[int, int] = {}

    cls_pred_count: Dict[int, int] = {}
    cls_gt_count: Dict[int, int] = {}

    for idx, img_path in enumerate(images):
        t0 = time.perf_counter()
        raw_preds = predictor.predict(img_path) or []
        pred_boxes: List[Box] = []
        # YOLOPredictor returns BoundingBox objects for ultralytics and dicts for onnx path (current implementation).
        for p in raw_preds:
            if isinstance(p, dict):
                pred_boxes.append(
                    Box(
                        x1=float(p.get("x1", 0.0)),
                        y1=float(p.get("y1", 0.0)),
                        x2=float(p.get("x2", 0.0)),
                        y2=float(p.get("y2", 0.0)),
                        cls=int(p.get("class_id", 0)),
                        conf=float(p.get("confidence", 0.0)),
                    )
                )
            else:
                # BoundingBox-like
                pred_boxes.append(
                    Box(
                        x1=float(getattr(p, "x1")),
                        y1=float(getattr(p, "y1")),
                        x2=float(getattr(p, "x2")),
                        y2=float(getattr(p, "y2")),
                        cls=int(getattr(p, "class_id")),
                        conf=float(getattr(p, "confidence", 0.0)) if getattr(p, "confidence", None) is not None else None,
                    )
                )
        dt = time.perf_counter() - t0
        total_time_s += dt

        total_pred += len(pred_boxes)
        for p in pred_boxes:
            cls_pred_count[p.cls] = cls_pred_count.get(p.cls, 0) + 1

        # labels (optional)
        lab_path = _label_path_for_image(img_path, images_dir, labels_dir)
        try:
            img_w, img_h = _load_image_wh(img_path)
        except Exception:
            img_w, img_h = 0, 0
        gt_boxes = _read_yolo_labels(lab_path, img_w, img_h) if (img_w and img_h) else []
        if gt_boxes:
            images_with_labels += 1
            total_gt += len(gt_boxes)
            for g in gt_boxes:
                cls_gt_count[g.cls] = cls_gt_count.get(g.cls, 0) + 1
            tp, fp, fn = _match_tp_fp_fn(pred_boxes, gt_boxes, args.iou)
            for k, v in tp.items():
                tp_all[k] = tp_all.get(k, 0) + v
            for k, v in fp.items():
                fp_all[k] = fp_all.get(k, 0) + v
            for k, v in fn.items():
                fn_all[k] = fn_all.get(k, 0) + v

        # lightweight progress for CLI
        if (idx + 1) % 25 == 0 or (idx + 1) == len(images):
            print(f"[{idx+1}/{len(images)}] processed, avg {total_time_s/(idx+1):.4f}s/img")

    tp_sum = sum(tp_all.values())
    fp_sum = sum(fp_all.values())
    fn_sum = sum(fn_all.values())

    precision = (tp_sum / (tp_sum + fp_sum)) if (tp_sum + fp_sum) > 0 else None
    recall = (tp_sum / (tp_sum + fn_sum)) if (tp_sum + fn_sum) > 0 else None
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)

    per_class: Dict[str, dict] = {}
    all_classes = sorted(set(cls_pred_count.keys()) | set(cls_gt_count.keys()) | set(tp_all.keys()) | set(fp_all.keys()) | set(fn_all.keys()))
    for cls in all_classes:
        tp = tp_all.get(cls, 0)
        fp = fp_all.get(cls, 0)
        fn = fn_all.get(cls, 0)
        p = (tp / (tp + fp)) if (tp + fp) > 0 else None
        r = (tp / (tp + fn)) if (tp + fn) > 0 else None
        f = None
        if p is not None and r is not None and (p + r) > 0:
            f = 2 * p * r / (p + r)
        per_class[str(cls)] = {
            "gt": cls_gt_count.get(cls, 0),
            "pred": cls_pred_count.get(cls, 0),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": p,
            "recall": r,
            "f1": f,
        }

    result = {
        "backend": backend,
        "model": model_path,
        "images_dir": images_dir,
        "labels_dir": labels_dir,
        "device": args.device,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "total_images": len(images),
        "images_with_labels": images_with_labels,
        "total_pred": total_pred,
        "total_gt": total_gt,
        "avg_time_s_per_image": total_time_s / max(1, len(images)),
        "total_time_s": total_time_s,
        "tp": tp_sum,
        "fp": fp_sum,
        "fn": fn_sum,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_class": per_class,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"RESULT_JSON={out_path}")


if __name__ == "__main__":
    main()


