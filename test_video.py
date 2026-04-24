import os
import json
import cv2
import torch
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import OrderedDict, Counter

from model.agcn import Model

# =========================
# SETTINGS
# =========================
VIDEO_PATH = "test_videos/lunges.mp4"
MODEL_PATH = "pose_landmarker_heavy.task"
WEIGHTS_PATH = "work_dir/custom/joint/model-39-600.pt"
LABEL_MAP_PATH = "dataset_legacy/label_map.json"

OUTPUT_VIDEO_PATH = "test_videos/lunges_out.mp4"

NUM_POINT = 33
NUM_PERSON = 1
FIXED_FRAMES = 30
SAFE_TIMESTAMP_STEP_MS = 33

# This script is specifically for squat testing
CLASS_NAME_FOR_SEGMENT = "squat"

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

POSE_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26),
    (25, 27), (26, 28),
    (27, 29), (28, 30),
    (29, 31), (30, 32)
]

SEGMENT_CONFIG = {
    "pushup": {
        "signal": "shoulder",
        "start_state": "up",
        "motion_threshold": 0.03,
    },
    "bench_dips": {
        "signal": "shoulder",
        "start_state": "up",
        "motion_threshold": 0.03,
    },
    "pullup": {
        "signal": "shoulder",
        "start_state": "down",
        "motion_threshold": 0.03,
    },
    "squat": {
        "signal": "hip",
        "start_state": "up",
        "motion_threshold": 0.03,
    },
    "squat_wrong_form": {
        "signal": "hip",
        "start_state": "up",
        "motion_threshold": 0.03,
    },
    "lunges": {
        "signal": "hip",
        "start_state": "up",
        "motion_threshold": 0.035,
    },
}

DEFAULT_CONFIG = {
    "signal": "shoulder",
    "start_state": "up",
    "motion_threshold": 0.03,
}


# =========================
# MEDIAPIPE
# =========================
def create_detector():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO
    )
    return vision.PoseLandmarker.create_from_options(options)


# =========================
# HELPERS
# =========================
def get_motion_signal(landmarks, signal_type):
    if signal_type == "shoulder":
        return (landmarks[LEFT_SHOULDER].y + landmarks[RIGHT_SHOULDER].y) / 2.0
    elif signal_type == "hip":
        return (landmarks[LEFT_HIP].y + landmarks[RIGHT_HIP].y) / 2.0
    else:
        raise ValueError(f"Unknown signal_type: {signal_type}")


def resample_sequence_interp(sequence, target_frames):
    T, V, C = sequence.shape

    if T == 0:
        raise ValueError("Sequence is empty.")

    if T == target_frames:
        return sequence.astype(np.float32)

    old_positions = np.arange(T, dtype=np.float32)
    new_positions = np.linspace(0, T - 1, target_frames, dtype=np.float32)

    resampled = np.zeros((target_frames, V, C), dtype=np.float32)

    for v in range(V):
        for c in range(C):
            resampled[:, v, c] = np.interp(
                new_positions,
                old_positions,
                sequence[:, v, c]
            )

    return resampled


def normalize_pose_sequence(sequence):
    seq = sequence.copy()
    hip_center = (seq[:, LEFT_HIP, :] + seq[:, RIGHT_HIP, :]) / 2.0
    seq = seq - hip_center[:, np.newaxis, :]
    return seq.astype(np.float32)


def to_2sagcn_shape(sequence):
    sequence = np.transpose(sequence, (2, 0, 1))   # (C, T, V)
    sequence = np.expand_dims(sequence, axis=-1)   # (C, T, V, 1)
    return sequence.astype(np.float32)


def draw_pose(frame, landmarks, width, height):
    for lm in landmarks:
        x = int(lm.x * width)
        y = int(lm.y * height)
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    for start, end in POSE_CONNECTIONS:
        x1 = int(landmarks[start].x * width)
        y1 = int(landmarks[start].y * height)
        x2 = int(landmarks[end].x * width)
        y2 = int(landmarks[end].y * height)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)


def load_label_map(label_map_path):
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    idx_to_class = {v: k for k, v in label_map.items()}
    return label_map, idx_to_class


def clean_state_dict(state_dict):
    cleaned = OrderedDict()
    for k, v in state_dict.items():
        new_key = k
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]
        cleaned[new_key] = v
    return cleaned


def load_weights_file(weights_path, device):
    checkpoint = torch.load(weights_path, map_location=device)

    if isinstance(checkpoint, OrderedDict):
        return clean_state_dict(checkpoint)

    if isinstance(checkpoint, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return clean_state_dict(checkpoint[key])

    raise ValueError("Unsupported checkpoint format.")


def load_model(weights_path, num_class):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Model(
        num_class=num_class,
        num_point=NUM_POINT,
        num_person=NUM_PERSON,
        graph="graph.blazepose.Graph",
        graph_args={"labeling_mode": "spatial"},
    ).to(device)

    state_dict = load_weights_file(weights_path, device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model, device


# =========================
# PREDICTION
# =========================
def predict_rep(model, device, seq):
    seq = resample_sequence_interp(seq, FIXED_FRAMES)
    seq = normalize_pose_sequence(seq)
    seq = to_2sagcn_shape(seq)
    seq = np.expand_dims(seq, axis=0)

    x = torch.tensor(seq, dtype=torch.float32).to(device)

    with torch.no_grad():
        out = model(x)
        logits = out[0].detach().cpu().numpy()
        prob = torch.softmax(out, dim=1)
        pred = torch.argmax(prob, dim=1).item()
        conf = prob[0, pred].item()
        conf_logits = logits[pred]

    return pred, conf, prob[0].cpu().numpy(), logits


# =========================
# MAIN VIDEO TEST
# =========================
def test_squat_video():
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Pose model not found: {MODEL_PATH}")
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Weights not found: {WEIGHTS_PATH}")
    if not os.path.exists(LABEL_MAP_PATH):
        raise FileNotFoundError(f"Label map not found: {LABEL_MAP_PATH}")

    label_map, idx_to_class = load_label_map(LABEL_MAP_PATH)
    num_class = len(label_map)

    print("Classes:")
    for name, idx in label_map.items():
        print(f"  {idx} -> {name}")
    print()

    model, device = load_model(WEIGHTS_PATH, num_class=num_class)
    detector = create_detector()

    cfg = SEGMENT_CONFIG.get(CLASS_NAME_FOR_SEGMENT.lower(), DEFAULT_CONFIG)
    signal_type = cfg["signal"]
    initial_state = cfg["start_state"]
    motion_threshold = float(cfg["motion_threshold"])

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        detector.close()
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    cooldown_frames = max(1, int(fps * 0.3))
    min_rep_frames = max(5, int(fps * 0.25))
    max_rep_frames = max(20, int(fps * 4.0))

    frame_id = 0
    timestamp_ms = 0

    smoothed_signal = None
    alpha = 0.2

    state = initial_state
    last_rep_frame = -9999
    min_seen = None
    max_seen = None
    candidate_rep_start = None

    frame_keypoints = {}
    rep_count = 0
    rep_results = []

    active_label = "Waiting..."
    active_conf = 0.0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            output_frame = frame.copy()
            timestamp_ms += SAFE_TIMESTAMP_STEP_MS

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            result = detector.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]

                joints = []
                for lm in landmarks:
                    joints.append([lm.x, lm.y, lm.z])

                if len(joints) == NUM_POINT:
                    frame_keypoints[frame_id] = np.array(joints, dtype=np.float32)

                draw_pose(output_frame, landmarks, width, height)

                raw_signal = get_motion_signal(landmarks, signal_type)

                if smoothed_signal is None:
                    smoothed_signal = raw_signal
                else:
                    smoothed_signal = alpha * raw_signal + (1 - alpha) * smoothed_signal

                if min_seen is None or smoothed_signal < min_seen:
                    min_seen = smoothed_signal
                if max_seen is None or smoothed_signal > max_seen:
                    max_seen = smoothed_signal

                motion_range = (max_seen - min_seen) if min_seen is not None and max_seen is not None else 0.0

                if motion_range > motion_threshold:
                    up_threshold = min_seen + motion_range * 0.35
                    down_threshold = min_seen + motion_range * 0.70

                    if initial_state == "up":
                        if state == "up" and smoothed_signal > down_threshold:
                            state = "down"
                            if candidate_rep_start is None:
                                candidate_rep_start = frame_id

                        elif (
                            state == "down"
                            and smoothed_signal < up_threshold
                            and (frame_id - last_rep_frame) > cooldown_frames
                        ):
                            state = "up"
                            rep_end = frame_id

                            if candidate_rep_start is None:
                                candidate_rep_start = max(0, frame_id - int(fps))

                            rep_len = rep_end - candidate_rep_start + 1

                            if min_rep_frames <= rep_len <= max_rep_frames:
                                seq = []
                                for fid in range(candidate_rep_start, rep_end + 1):
                                    if fid in frame_keypoints:
                                        seq.append(frame_keypoints[fid])

                                if len(seq) > 0:
                                    seq = np.array(seq, dtype=np.float32)
                                    pred_idx, conf, probs, logits = predict_rep(model, device, seq)
                                    conf_logits = logits[pred_idx]
                                    pred_name = idx_to_class.get(pred_idx, str(pred_idx))

                                    rep_count += 1
                                    rep_results.append({
                                        "rep": rep_count,
                                        "start_frame": candidate_rep_start,
                                        "end_frame": rep_end,
                                        "pred_idx": pred_idx,
                                        "pred_name": pred_name,
                                        "confidence": conf,
                                        "probs": probs,
                                        "logits": logits
                                    })

                                    active_label = pred_name
                                    active_conf = conf
                                    last_rep_frame = frame_id

                            candidate_rep_start = frame_id + 1

                    else:
                        if state == "down" and smoothed_signal < up_threshold:
                            state = "up"
                            if candidate_rep_start is None:
                                candidate_rep_start = frame_id

                        elif (
                            state == "up"
                            and smoothed_signal > down_threshold
                            and (frame_id - last_rep_frame) > cooldown_frames
                        ):
                            state = "down"
                            rep_end = frame_id

                            if candidate_rep_start is None:
                                candidate_rep_start = max(0, frame_id - int(fps))

                            rep_len = rep_end - candidate_rep_start + 1

                            if min_rep_frames <= rep_len <= max_rep_frames:
                                seq = []
                                for fid in range(candidate_rep_start, rep_end + 1):
                                    if fid in frame_keypoints:
                                        seq.append(frame_keypoints[fid])

                                if len(seq) > 0:
                                    seq = np.array(seq, dtype=np.float32)
                                    pred_idx, conf, probs, logits = predict_rep(model, device, seq)
                                    conf_logits = logits[pred_idx]
                                    pred_name = idx_to_class.get(pred_idx, str(pred_idx))

                                    rep_count += 1
                                    rep_results.append({
                                        "rep": rep_count,
                                        "start_frame": candidate_rep_start,
                                        "end_frame": rep_end,
                                        "pred_idx": pred_idx,
                                        "pred_name": pred_name,
                                        "confidence": conf,
                                        "probs": probs
                                    })

                                    active_label = pred_name
                                    active_conf = conf
                                    last_rep_frame = frame_id

                            candidate_rep_start = frame_id + 1

            cv2.putText(output_frame, f"Exercise: {CLASS_NAME_FOR_SEGMENT}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(output_frame, f"Reps: {rep_count}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(output_frame, f"State: {state}", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(output_frame, f"Prediction: {active_label}", (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            cv2.putText(output_frame, f"Confidence: {active_conf:.3f}", (20, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

            if smoothed_signal is not None:
                cv2.putText(output_frame, f"{signal_type.title()}Y: {smoothed_signal:.3f}", (20, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            writer.write(output_frame)
            frame_id += 1

    finally:
        cap.release()
        writer.release()
        detector.close()

    if not rep_results:
        print("No reps found.")
        print(f"Annotated video saved to: {OUTPUT_VIDEO_PATH}")
        return

    print(f"Total reps found: {len(rep_results)}\n")

    for item in rep_results:
        print(
            f"Rep {item['rep']}: "
            f"{item['pred_name']} "
            f"(confidence={item['confidence']:.4f}, "
            f"frames={item['start_frame']}->{item['end_frame']})"
        )

        top_indices = np.argsort(-item["probs"])[:3]
        print("  Top predictions:")
        for idx in top_indices:
            cls_name = idx_to_class.get(int(idx), str(idx))
            print(f"    {cls_name}: {item['probs'][idx]:.4f}")

        print("  === LOGITS ===")
        for i, val in enumerate(item["logits"]):
            print(f"    {idx_to_class[i]}: {val:.4f}")

        print()

    majority = Counter([r["pred_name"] for r in rep_results]).most_common(1)[0][0]
    avg_conf = float(np.mean([r["confidence"] for r in rep_results]))

    print("Final video summary:")
    print(f"  Majority prediction: {majority}")
    print(f"  Average confidence: {avg_conf:.4f}")
    print(f"  Annotated video saved to: {OUTPUT_VIDEO_PATH}")


if __name__ == "__main__":
    test_squat_video()