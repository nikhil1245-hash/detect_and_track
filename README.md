# Real-Time Object Detection + Tracking

Webcam or video file → YOLOv8 object detection → SORT tracking → bounding boxes with class labels and persistent track IDs, displayed live.

![example output](<img width="768" height="576" alt="image" src="https://github.com/user-attachments/assets/50660c41-3cb6-45f9-996f-b9710bba3205" />


## Setup

```bash
pip install -r requirements.txt
```

First run auto-downloads `yolov8n.pt` (~6MB) — Ultralytics' smallest, fastest YOLOv8 variant, good for real-time use on CPU.

## Run it

```bash
# Default webcam
python detect_and_track.py

# A video file instead
python detect_and_track.py --source path/to/video.mp4

# Only track certain classes
python detect_and_track.py --classes person car

# Save the annotated output
python detect_and_track.py --source video.mp4 --save output.mp4

# Headless (no display window) — e.g. on a server, combine with --save
python detect_and_track.py --source video.mp4 --no-display --save output.mp4
```

Press **`q`** in the display window to quit.

Full options: `python detect_and_track.py --help`

## How it works

```
Frame (OpenCV) → YOLOv8 detection → [boxes, labels, confidences]
                                            │
                                            ▼
                              SORT tracker (Kalman filter +
                              Hungarian algorithm IoU matching)
                                            │
                                            ▼
                         [track_id, box, label, confidence] per object
                                            │
                                            ▼
                              Draw boxes + labels + IDs, display
```

**Detection — `detect_and_track.py`:** Each frame is run through a pre-trained YOLOv8 model (`ultralytics` package), trained on the 80-class COCO dataset (person, car, dog, bus, etc.). Returns bounding boxes, class labels, and confidence scores.

**Tracking — `sort_tracker.py`:** A from-scratch implementation of SORT (Simple Online and Realtime Tracking):
- Each tracked object gets a **Kalman filter** modeling its position, size, and velocity, so its next location can be *predicted* even between detections.
- Every frame, predicted track positions are matched against the new detections using **IoU (Intersection-over-Union)** as the similarity score, solved optimally with the **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`) — this is what keeps the same ID on the same object frame-to-frame, including through brief occlusions.
- A track is only confirmed (shown) after `min_hits` consecutive detections, and is dropped after `max_age` frames with no matching detection — this filters out flickery false-positive detections while tolerating short gaps (e.g. an object briefly hidden behind another).

No `sort` or `deep_sort_realtime` package dependency — the tracker is plain numpy + scipy, so it's easy to read and modify.

## Tuning

| Flag | Effect |
|---|---|
| `--conf` | Higher = fewer, more confident detections. Lower = catches more but noisier. |
| `--min-hits` | Higher = tracks take longer to "confirm" but are more reliable. |
| `--max-age` | Higher = tracks survive longer occlusions, but may drift during them. |
| `--iou-threshold` | Higher = stricter matching (more ID switches on fast motion). Lower = looser (more risk of merging nearby objects). |

## Files

| File | Purpose |
|---|---|
| `detect_and_track.py` | Main pipeline — video I/O, YOLO inference, drawing, CLI |
| `sort_tracker.py` | SORT implementation (Kalman filter + Hungarian algorithm) |
| `requirements.txt` | Dependencies |

## Notes

- `yolov8n.pt` (the "nano" model) is tuned for speed over accuracy — swap `--model yolov8s.pt` / `yolov8m.pt` for better accuracy at the cost of FPS, especially without a GPU.
- This can't run as a GitHub Pages static site (it needs a Python runtime and OpenCV's native display), so it's meant to be cloned and run locally, or in a Colab/Kaggle notebook with `--no-display --save`.
