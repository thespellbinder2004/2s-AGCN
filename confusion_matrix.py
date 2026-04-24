import os
import json
import cv2
import torch
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import OrderedDict

from model.agcn import Model

# =========================
# SETTINGS
# =========================
TEST_ROOT = "test_dataset"   # root folder, subfolders = class names
MODEL_PATH = "pose_landmarker_heavy.task"
WEIGHTS_PATH = "work_dir/custom/joint/model-39-800.pt"
LABEL_MAP_PATH = "dataset_legacy/label_map.json"

NUM_POINT = 33
NUM_PERSON = 1
FIXED_FRAMES = 30
SAFE_TIMESTAMP_STEP_MS = 33

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24

SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

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
    "pushup_elbows_flared": {
        "signal": "shoulder",
        "start_state": "up",
        "motion_threshold": 0.03,
    },

    "bench_dips": {
        "signal": "shoulder",
        "start_state": "up",
        "motion_threshold": 0.03,
    },
    "bench_dips_elbows_flared": {
        "signal": "shoulder",
        "start_state": "up",
        "motion_threshold": 0.03,
    },

    "squat": {
        "signal": "hip",
        "start_state": "up",
        "motion_threshold": 0.03,
    },
    "squat_body_leaning_forward": {
        "signal": "hip",
        "start_state": "up",
        "motion_threshold": 0.03,
    },
    "squat_legs_too_narrow": {
        "signal": "hip",
        "start_state": "up",
        "motion_threshold": 0.03,
    },
    "squat_legs_too_wide": {
        "signal": "hip",
        "start_state": "up",
        "motion_threshold": 0.03,
    },

    "lunges": {
        "signal": "hip",
        "start_state": "up",
        "motion_threshold": 0.035,
    },
    "lunges_body_leaning_forward": {
        "signal": "hip",
        "start_state": "up",
        "motion_threshold": 0.035,
    },

    "bicep_curls": {
        "signal": "curl_distance",
        "start_state": "down",
        "motion_threshold": 0.06,
    },
    "bicep_curls_elbows_moving": {
        "signal": "curl_distance",
        "start_state": "down",
        "motion_threshold": 0.06,
    },

    "burpees": {
        "signal": "hip",
        "start_state": "up",
        "motion_threshold": 0.05,
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
    elif signal_type == "curl_distance":
        left_dx = landmarks[LEFT_WRIST].x - landmarks[LEFT_SHOULDER].x
        left_dy = landmarks[LEFT_WRIST].y - landmarks[LEFT_SHOULDER].y
        right_dx = landmarks[RIGHT_WRIST].x - landmarks[RIGHT_SHOULDER].x
        right_dy = landmarks[RIGHT_WRIST].y - landmarks[RIGHT_SHOULDER].y

        left_dist = (left_dx ** 2 + left_dy ** 2) ** 0.5
        right_dist = (right_dx ** 2 + right_dy ** 2) ** 0.5

        return (left_dist + right_dist) / 2.0
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

    return pred, conf, prob[0].cpu().numpy(), logits


def find_videos_in_test_root(test_root):
    items = []

    if not os.path.isdir(test_root):
        raise FileNotFoundError(f"Test root folder not found: {test_root}")

    for class_name in sorted(os.listdir(test_root)):
        class_dir = os.path.join(test_root, class_name)
        if not os.path.isdir(class_dir):
            continue

        for file_name in sorted(os.listdir(class_dir)):
            if file_name.lower().endswith(SUPPORTED_VIDEO_EXTENSIONS):
                video_path = os.path.join(class_dir, file_name)
                items.append({
                    "actual_class": class_name,
                    "video_path": video_path,
                    "file_name": file_name,
                })

    return items


def get_base_exercise_name(class_name: str) -> str:
    name = class_name.lower().strip()

    if name.startswith("pushup"):
        return "pushup"
    if name.startswith("bench_dips"):
        return "bench_dips"
    if name.startswith("squat"):
        return "squat"
    if name.startswith("lunges") or name.startswith("lunge"):
        return "lunges"
    if name.startswith("bicep_curls") or name.startswith("bicep_curl"):
        return "bicep_curls"
    if name.startswith("burpees") or name.startswith("burpee"):
        return "burpees"

    return name


def choose_segmentation_config(class_name):
    base_name = get_base_exercise_name(class_name)
    return SEGMENT_CONFIG.get(base_name, DEFAULT_CONFIG)


def classify_video_by_majority(video_path, actual_class, model, device, idx_to_class):
    cfg = choose_segmentation_config(actual_class)
    signal_type = cfg["signal"]
    initial_state = cfg["start_state"]
    motion_threshold = float(cfg["motion_threshold"])

    detector = create_detector()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        detector.close()
        print(f"[ERROR] Could not open video: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

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
    rep_predictions = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

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
                                    pred_name = idx_to_class.get(pred_idx, str(pred_idx))
                                    rep_predictions.append(pred_name)
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
                                    pred_name = idx_to_class.get(pred_idx, str(pred_idx))
                                    rep_predictions.append(pred_name)
                                    last_rep_frame = frame_id

                            candidate_rep_start = frame_id + 1

            frame_id += 1

    finally:
        cap.release()
        detector.close()

    if not rep_predictions:
        return None

    majority_prediction = max(set(rep_predictions), key=rep_predictions.count)
    return majority_prediction

def classify_video_per_rep(video_path, actual_class, model, device, idx_to_class):
    cfg = choose_segmentation_config(actual_class)
    signal_type = cfg["signal"]
    initial_state = cfg["start_state"]
    motion_threshold = float(cfg["motion_threshold"])

    detector = create_detector()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        detector.close()
        print(f"[ERROR] Could not open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

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
    rep_predictions = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

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
                                    pred_name = idx_to_class.get(pred_idx, str(pred_idx))

                                    rep_predictions.append({
                                        "rep_index": len(rep_predictions) + 1,
                                        "predicted": pred_name,
                                        "confidence": conf,
                                        "start_frame": candidate_rep_start,
                                        "end_frame": rep_end,
                                    })

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
                                    pred_name = idx_to_class.get(pred_idx, str(pred_idx))

                                    rep_predictions.append({
                                        "rep_index": len(rep_predictions) + 1,
                                        "predicted": pred_name,
                                        "confidence": conf,
                                        "start_frame": candidate_rep_start,
                                        "end_frame": rep_end,
                                    })

                                    last_rep_frame = frame_id

                            candidate_rep_start = frame_id + 1

            frame_id += 1

    finally:
        cap.release()
        detector.close()

    return rep_predictions

def compute_one_vs_rest_metrics(results, class_names):
    metrics = {}

    for cls in class_names:
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for item in results:
            actual = item["actual"]
            predicted = item["predicted"]

            if actual == cls and predicted == cls:
                tp += 1
            elif actual != cls and predicted != cls:
                tn += 1
            elif actual != cls and predicted == cls:
                fp += 1
            elif actual == cls and predicted != cls:
                fn += 1

        metrics[cls] = {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
        }

    return metrics


def safe_div(a, b):
    return a / b if b != 0 else 0.0


def print_metrics_table(metrics, total_tests):
    print()
    print(f"Number of tests made: {total_tests}")
    print()

    headers = [
        "Exercise", "TP", "TN", "FP", "FN",
        "Accuracy", "Precision", "Specificity", "Recall"
    ]

    rows = []
    for cls, m in metrics.items():
        tp = m["TP"]
        tn = m["TN"]
        fp = m["FP"]
        fn = m["FN"]

        accuracy = safe_div(tp + tn, tp + tn + fp + fn)
        precision = safe_div(tp, tp + fp)
        specificity = safe_div(tn, tn + fp)
        recall = safe_div(tp, tp + fn)

        rows.append([
            cls,
            tp, tn, fp, fn,
            f"{accuracy:.4f}",
            f"{precision:.4f}",
            f"{specificity:.4f}",
            f"{recall:.4f}",
        ])

    col_widths = [max(len(str(row[i])) for row in ([headers] + rows)) + 2 for i in range(len(headers))]

    header_line = "".join(str(headers[i]).ljust(col_widths[i]) for i in range(len(headers)))
    print(header_line)
    print("-" * len(header_line))

    for row in rows:
        print("".join(str(row[i]).ljust(col_widths[i]) for i in range(len(row))))


def main():
    if not os.path.exists(TEST_ROOT):
        raise FileNotFoundError(f"Test root not found: {TEST_ROOT}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Pose model not found: {MODEL_PATH}")
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Weights not found: {WEIGHTS_PATH}")
    if not os.path.exists(LABEL_MAP_PATH):
        raise FileNotFoundError(f"Label map not found: {LABEL_MAP_PATH}")

    # ✅ LOAD EVERYTHING
    label_map, idx_to_class = load_label_map(LABEL_MAP_PATH)
    class_names = [name for name, _ in sorted(label_map.items(), key=lambda x: x[1])]
    model, device = load_model(WEIGHTS_PATH, num_class=len(label_map))
    test_items = find_videos_in_test_root(TEST_ROOT)

    if not test_items:
        print("No videos found.")
        return

    results = []
    skipped = 0

    try:
        for i, item in enumerate(test_items, start=1):
            actual_class = item["actual_class"]
            video_path = item["video_path"]
            file_name = item["file_name"]

            print(f"[{i}/{len(test_items)}] Testing: {video_path}")

            if actual_class not in label_map:
                print(f"  -> SKIPPED: {actual_class} not in label_map")
                skipped += 1
                continue

            rep_results = classify_video_per_rep(
                video_path,
                actual_class,
                model,
                device,
                idx_to_class
            )

            if not rep_results:
                print("  -> SKIPPED: no reps detected")
                skipped += 1
                continue

            for rep in rep_results:
                pred = rep["predicted"]

                print(f"  -> REP {rep['rep_index']} | actual: {actual_class} | predicted: {pred}")

                results.append({
                    "file_name": file_name,
                    "video_path": video_path,
                    "actual": actual_class,
                    "predicted": pred,
                })

    except Exception as e:
        print(f"[ERROR] {e}")

    # ✅ METRICS (VERY IMPORTANT)
    if not results:
        print("\nNo valid results.")
        print(f"Skipped: {skipped}")
        return

    metrics = compute_one_vs_rest_metrics(results, class_names)
    print_metrics_table(metrics, total_tests=len(results))

    print()
    print("Valid reps:", len(results))
    print("Skipped:", skipped)


if __name__ == "__main__":
    main()

