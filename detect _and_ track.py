"""
detect_and_track.py — Real-time object detection + tracking.

Pipeline: OpenCV video capture (webcam or file) -> YOLOv8 detection (Ultralytics)
-> SORT tracking (Kalman filter + Hungarian algorithm, see sort_tracker.py)
-> bounding boxes with class label + track ID drawn on each frame.

Usage:
    python detect_and_track.py                      # webcam (default camera)
    python detect_and_track.py --source video.mp4    # video file
    python detect_and_track.py --source 0 --save out.mp4   # save annotated output
    python detect_and_track.py --classes person car  # only track these classes
    python detect_and_track.py --conf 0.4             # detection confidence threshold

Press 'q' to quit the display window.

Requirements:
    pip install ultralytics opencv-python scipy numpy
    (first run auto-downloads the yolov8n.pt weights, ~6MB)
"""

import argparse
import time

import cv2
import numpy as np
from ultralytics import YOLO

from sort_tracker import SortTracker

# A distinct, stable color per track ID (derived deterministically from the ID
# itself, so the same object keeps the same color across frames).
def color_for_id(track_id: int):
    np.random.seed(track_id * 37 + 3)
    return tuple(int(c) for c in np.random.randint(60, 255, size=3))


def draw_annotations(frame, tracks):
    for track_id, box, label, confidence in tracks:
        x1, y1, x2, y2 = [int(v) for v in box]
        color = color_for_id(track_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text = f"#{track_id} {label} {confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, text, (x1 + 3, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    return frame


def draw_hud(frame, fps, track_count):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 34), (20, 20, 20), -1)
    cv2.putText(frame, f"FPS: {fps:.1f}   Tracked objects: {track_count}",
                (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def run(source, model_path="yolov8n.pt", conf_threshold=0.35, classes=None,
        save_path=None, max_age=15, min_hits=3, iou_threshold=0.3, display=True,
        max_frames=None):

    model = YOLO(model_path)
    class_names = model.names  # {id: name} from the model's training config

    # Resolve which class IDs to keep, if the user restricted by name.
    allowed_class_ids = None
    if classes:
        wanted = {c.lower() for c in classes}
        allowed_class_ids = {i for i, name in class_names.items() if name.lower() in wanted}
        unknown = wanted - {class_names[i].lower() for i in allowed_class_ids}
        if unknown:
            print(f"Warning: unknown class names ignored: {sorted(unknown)}")

    # Video source: an integer string means a webcam index.
    cap_source = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps_in, (width, height))

    tracker = SortTracker(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)

    prev_time = time.time()
    fps_smooth = 0.0
    frame_count = 0

    if display:
        print("Press 'q' in the display window to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break  # end of video file, or camera disconnected

        # --- Detection ---
        results = model(frame, conf=conf_threshold, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if allowed_class_ids is not None and cls_id not in allowed_class_ids:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            label = class_names[cls_id]
            detections.append(([x1, y1, x2, y2], label, conf))

        # --- Tracking ---
        tracks = tracker.update(detections)

        # --- Draw + display ---
        frame = draw_annotations(frame, tracks)

        now = time.time()
        inst_fps = 1.0 / max(now - prev_time, 1e-6)
        fps_smooth = inst_fps if fps_smooth == 0 else (0.9 * fps_smooth + 0.1 * inst_fps)
        prev_time = now
        frame = draw_hud(frame, fps_smooth, len(tracks))

        if display:
            cv2.imshow("Object Detection + Tracking (press 'q' to quit)", frame)
        if writer:
            writer.write(frame)

        frame_count += 1
        if display:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if max_frames is not None and frame_count >= max_frames:
            break

    cap.release()
    if writer:
        writer.release()
    if display:
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time object detection + tracking (YOLO + SORT).")
    parser.add_argument("--source", default="0",
                         help="Webcam index (e.g. 0) or path to a video file. Default: 0 (default webcam).")
    parser.add_argument("--model", default="yolov8n.pt",
                         help="Ultralytics YOLO model to use. Default: yolov8n.pt (fast, good for real-time/CPU).")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold (0-1).")
    parser.add_argument("--classes", nargs="*", default=None,
                         help="Restrict tracking to these class names, e.g. --classes person car. Default: all 80 COCO classes.")
    parser.add_argument("--save", default=None, help="Path to save the annotated output video (e.g. output.mp4).")
    parser.add_argument("--max-age", type=int, default=15,
                         help="Frames a track survives without a matching detection before being dropped.")
    parser.add_argument("--min-hits", type=int, default=3,
                         help="Consecutive detections needed before a track is confirmed/displayed.")
    parser.add_argument("--iou-threshold", type=float, default=0.3,
                         help="Minimum IoU for a detection to be matched to an existing track.")
    parser.add_argument("--no-display", action="store_true",
                         help="Run without opening a display window (e.g. on a headless server) — use with --save.")
    parser.add_argument("--max-frames", type=int, default=None,
                         help="Stop after this many frames (mainly for testing).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        source=args.source,
        model_path=args.model,
        conf_threshold=args.conf,
        classes=args.classes,
        save_path=args.save,
        max_age=args.max_age,
        min_hits=args.min_hits,
        iou_threshold=args.iou_threshold,
        display=not args.no_display,
        max_frames=args.max_frames,
    )
