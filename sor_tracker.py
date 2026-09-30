"""
sort_tracker.py — A from-scratch implementation of SORT
(Simple Online and Realtime Tracking, Bewley et al. 2016).

Each tracked object is modeled with a constant-velocity Kalman filter over
[center_x, center_y, area, aspect_ratio] (+ their velocities). Frame-to-frame
association between existing tracks and new detections is done via IoU
(Intersection-over-Union) cost and solved optimally with the Hungarian
algorithm (scipy.optimize.linear_sum_assignment).

No external tracking library required — only numpy + scipy.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Vectorized IoU between two sets of boxes in [x1, y1, x2, y2] format."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))

    boxes_a = np.expand_dims(boxes_a, 1)  # (A, 1, 4)
    boxes_b = np.expand_dims(boxes_b, 0)  # (1, B, 4)

    xx1 = np.maximum(boxes_a[..., 0], boxes_b[..., 0])
    yy1 = np.maximum(boxes_a[..., 1], boxes_b[..., 1])
    xx2 = np.minimum(boxes_a[..., 2], boxes_b[..., 2])
    yy2 = np.minimum(boxes_a[..., 3], boxes_b[..., 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    intersection = w * h

    area_a = (boxes_a[..., 2] - boxes_a[..., 0]) * (boxes_a[..., 3] - boxes_a[..., 1])
    area_b = (boxes_b[..., 2] - boxes_b[..., 0]) * (boxes_b[..., 3] - boxes_b[..., 1])
    union = area_a + area_b - intersection

    return np.where(union > 0, intersection / union, 0.0)


def xyxy_to_z(box):
    """[x1,y1,x2,y2] -> [cx, cy, area, aspect_ratio] (SORT's state parametrization)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2.0, y1 + h / 2.0
    area = max(w * h, 1e-6)
    aspect = w / max(h, 1e-6)
    return np.array([cx, cy, area, aspect])


def z_to_xyxy(z):
    """[cx, cy, area, aspect_ratio] -> [x1, y1, x2, y2]."""
    cx, cy, area, aspect = z
    area = max(area, 1e-6)
    w = np.sqrt(area * aspect)
    h = area / max(w, 1e-6)
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0])


class KalmanBoxTracker:
    """A constant-velocity Kalman filter tracking a single object's box."""

    count = 0  # class-level counter -> unique, ever-increasing track IDs

    def __init__(self, box, label, confidence):
        z = xyxy_to_z(box)  # [cx, cy, area, aspect]

        # State: [cx, cy, area, aspect, vcx, vcy, varea] (aspect assumed ~constant)
        self.x = np.zeros(7)
        self.x[:4] = z

        self.F = np.eye(7)  # state transition
        for i in range(3):
            self.F[i, i + 4] = 1.0  # position += velocity

        self.H = np.zeros((4, 7))  # measurement function (we observe cx,cy,area,aspect)
        self.H[:4, :4] = np.eye(4)

        self.P = np.eye(7) * 10.0          # state covariance (start uncertain)
        self.P[4:, 4:] *= 1000.0            # velocities start very uncertain
        self.Q = np.eye(7) * 0.01           # process noise
        self.Q[4:, 4:] *= 0.01
        self.R = np.eye(4) * 1.0            # measurement noise

        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.label = label
        self.confidence = confidence

        self.hits = 1
        self.age = 0
        self.time_since_update = 0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        if self.x[2] <= 0:  # area must stay positive
            self.x[2] = 1.0
        return z_to_xyxy(self.x[:4])

    def update(self, box, label, confidence):
        z = xyxy_to_z(box)
        y = z - self.H @ self.x  # innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

        self.time_since_update = 0
        self.hits += 1
        self.label = label
        self.confidence = confidence

    def get_state(self):
        return z_to_xyxy(self.x[:4])


class SortTracker:
    """
    Multi-object tracker. Call `update(detections)` once per frame with
    detections = list of (box[x1,y1,x2,y2], label, confidence).
    Returns a list of (track_id, box, label, confidence) for currently
    confirmed tracks.
    """

    def __init__(self, max_age=15, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age          # frames a track survives with no matching detection
        self.min_hits = min_hits        # detections needed before a track is "confirmed"
        self.iou_threshold = iou_threshold
        self.trackers = []

    def update(self, detections):
        # 1. Predict new locations for all existing trackers.
        predicted_boxes = np.array([t.predict() for t in self.trackers]) if self.trackers else np.empty((0, 4))

        det_boxes = np.array([d[0] for d in detections]) if detections else np.empty((0, 4))

        # 2. Associate detections to trackers via IoU + Hungarian algorithm.
        matched, unmatched_dets, unmatched_trks = self._associate(det_boxes, predicted_boxes)

        # 3. Update matched trackers with their assigned detection.
        for det_idx, trk_idx in matched:
            box, label, conf = detections[det_idx]
            self.trackers[trk_idx].update(box, label, conf)

        # 4. Create new trackers for unmatched detections.
        for det_idx in unmatched_dets:
            box, label, conf = detections[det_idx]
            self.trackers.append(KalmanBoxTracker(box, label, conf))

        # 5. Remove dead trackers (no update for too many frames).
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        # 6. Return confirmed tracks (enough hits, or still fresh at the start).
        results = []
        for t in self.trackers:
            if t.time_since_update == 0 and (t.hits >= self.min_hits or t.age <= self.min_hits):
                results.append((t.id, t.get_state(), t.label, t.confidence))
        return results

    def _associate(self, det_boxes, trk_boxes):
        if len(trk_boxes) == 0:
            return [], list(range(len(det_boxes))), []
        if len(det_boxes) == 0:
            return [], [], list(range(len(trk_boxes)))

        iou_matrix = iou_batch(det_boxes, trk_boxes)
        cost_matrix = 1.0 - iou_matrix  # Hungarian algorithm minimizes cost
        det_indices, trk_indices = linear_sum_assignment(cost_matrix)

        matched = []
        unmatched_dets = list(range(len(det_boxes)))
        unmatched_trks = list(range(len(trk_boxes)))

        for d, t in zip(det_indices, trk_indices):
            if iou_matrix[d, t] >= self.iou_threshold:
                matched.append((d, t))
                unmatched_dets.remove(d)
                unmatched_trks.remove(t)

        return matched, unmatched_dets, unmatched_trks
