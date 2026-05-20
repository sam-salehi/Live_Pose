# Live_Pose — Project Context

## What this repo does
Real-time and batch 3D boxing pose estimation pipeline using YOLOv8 (2D keypoints) + MotionBERT-Lite (3D lifting). Key scripts:

- `live_pose3d.py` — live webcam inference, body-frame normalization, arm metrics
- `analyze_video.py` — batch analysis: sends video to server, receives 3D poses, detects and classifies punches, plots wrist velocity/elbow/elevation signals

## Companion workspace
`~/workspace/pose/` — more complete ML pipeline (training, evaluation, dataset management).
Key files there:
- `preprocess.py` — `_load_annotations()` / `_normalize_label()`: authoritative annotation parser for all video formats
- `analyse_pose_metrics.py` — full biomechanical analysis (all 6 classes, V4–V10), outputs `analysis_output/metrics.csv` and violin/scatter plots

## Dataset
- `MotionBERT_3d/V{4-10}/X3D.npy` — shape `(N, 17, 3)` H36M 3D poses (entire video)
- `Annotation_files/V{4-10}.xlsx` — punch windows with **1-indexed** frame numbers; use `start - 1` for numpy slicing
- xlsx schema differs per video; use `_load_annotations()` from `preprocess.py` to handle it

### Annotation frame-index convention
All annotation `start`/`end` values are **1-indexed raw video frame numbers** that index directly into the `X3D.npy` array.  
Convert: `clip = frames[start - 1 : end]`

### Known FPS
- V7: 29.97 fps (confirmed from companion mp4, frame count matches npy)
- V9: 23.98 fps (companion mp4 is a short clip — fps may be approximate)
- V10: 25.0 fps (same caveat)
- V4, V5, V6, V8: FPS unknown; assume 30 or skip speed-magnitude features

## Punch taxonomy (6 classes)
| Label | Hand | Type |
|---|---|---|
| Jab | Lead (left, orthodox) | Straight |
| Cross | Rear (right, orthodox) | Straight |
| Lead Hook | Left | Hook |
| Rear Hook | Right | Hook |
| Lead Uppercut | Left | Uppercut |
| Rear Uppercut | Right | Uppercut |

V6 contains no Jabs (only hooks and uppercuts).

## H36M joint indices (17 joints)
```
0: Pelvis  1: RHip   2: RKnee   3: RAnkle
4: LHip    5: LKnee  6: LAnkle  7: Spine
8: Thorax  9: Neck  10: Head
11: LShoulder  12: LElbow  13: LWrist
14: RShoulder  15: RElbow  16: RWrist
```

## Body frame (from `normalize_to_body_frame` in `live_pose3d.py`)
- **X** = lateral (left hip → right hip)
- **Y** = forward (out of chest), Z × X
- **Z** = vertical (hip → neck, orthogonalized)
- Origin: pelvis (joint 0), scaled by torso length

## Key analysis findings (from notes.md)
- `elev_at_ext` — arm elevation at extension: mean ~89.7°, CV 0.056 for jabs (very consistent)
- `reach_peak` — peak wrist-shoulder distance: mean ~0.957 for jabs, CV 0.060
- `vel_X_frac` — lateral velocity fraction: mean 0.857 for jabs (surprisingly high — body frame
  may misalign with camera angle for monocular footage; treat with caution)
- `elbow_peak` — mean 162.6° for jabs; better than `elbow_at_ext` (162.6° vs 138.7°, lower CV)

## Punch detection in analyze_video.py
State machine: speed crosses adaptive threshold → confirm with elbow angle peak > 120° and reach increase.
Classifier: elevation peak < 70° → uppercut; reach peak ≥ 0.85 → jab; else → hook.

## Current goal
Find which features best separate **Jab vs Lead Hook vs Lead Uppercut** across V4–V10.
Script: `punch_feature_analysis.py`
