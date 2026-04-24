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
TEST_ROOT = "test_dataset_segmented"   # root folder, subfolders = class names
MODEL_PATH = "pose_landmarker_heavy.task"
WEIGHTS_PATH = "work_dir/custom/joint/model-39-920.pt"
LABEL_MAP_PATH = "dataset_legacy/label_map.json"

NUM_POINT = 33
NUM_PERSON = 1
FIXED_FRAMES = 30
SAFE_TIMESTAMP_STEP_MS = 33

SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

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

    LEFT_HIP = 23
    RIGHT_HIP = 24

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

def get_prediction_group(class_name: str):
    name = class_name.strip().lower()

    if name.startswith("squat"):
        return [
            "squat",
            "squat_body_leaning_forward",
            "squat_legs_too_narrow",
            "squat_legs_too_wide",
        ]

    if name.startswith("pushup"):
        return [
            "pushup",
            "pushup_elbows_flared",
        ]

    if name.startswith("bench_dips"):
        return [
            "bench_dips",
            "bench_dips_elbows_flared",
        ]

    if name.startswith("bicep_curls"):
        return [
            "bicep_curls",
            "bicep_curls_elbows_moving",
        ]

    if name.startswith("lunges") or name.startswith("lunge"):
        return [
            "lunges",
            "lunges_body_leaning_forward",
        ]

    if name.startswith("burpees") or name.startswith("burpee"):
        return [
            "burpees",
        ]

    return [name]

def predict_sequence_restricted(model, device, seq, actual_class, label_map, idx_to_class):
    seq = resample_sequence_interp(seq, FIXED_FRAMES)
    seq = normalize_pose_sequence(seq)
    seq = to_2sagcn_shape(seq)
    seq = np.expand_dims(seq, axis=0)

    x = torch.tensor(seq, dtype=torch.float32).to(device)

    with torch.no_grad():
        out = model(x)   # shape: [1, num_class]
        logits = out[0].detach().cpu().numpy()

    allowed_class_names = get_prediction_group(actual_class)
    allowed_indices = [
        label_map[name]
        for name in allowed_class_names
        if name in label_map
    ]

    if not allowed_indices:
        raise ValueError(f"No allowed prediction classes found for: {actual_class}")

    restricted_logits = logits[allowed_indices]
    restricted_logits_tensor = torch.tensor(restricted_logits, dtype=torch.float32)
    restricted_probs = torch.softmax(restricted_logits_tensor, dim=0).numpy()

    best_local_idx = int(np.argmax(restricted_probs))
    pred_idx = allowed_indices[best_local_idx]
    pred_name = idx_to_class[pred_idx]
    conf = float(restricted_probs[best_local_idx])

    return pred_name, conf, allowed_class_names, restricted_probs

def predict_sequence(model, device, seq):
    seq = resample_sequence_interp(seq, FIXED_FRAMES)
    seq = normalize_pose_sequence(seq)
    seq = to_2sagcn_shape(seq)
    seq = np.expand_dims(seq, axis=0)

    x = torch.tensor(seq, dtype=torch.float32).to(device)

    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, dim=1)
        pred = torch.argmax(prob, dim=1).item()
        conf = prob[0, pred].item()

    return pred, conf

def normalize_class_name(folder_name: str) -> str:
    name = folder_name.lower()

    if "bench dip" in name:
        return "bench_dips"
    if "push" in name:
        return "pushup"
    if "squat" in name:
        return "squat"
    if "lunge" in name:
        return "lunges"
    if "bicep" in name:
        return "bicep_curls"
    if "burpee" in name:
        return "burpees"

    return name.replace(" ", "_")

def find_videos_in_test_root(test_root):
    items = []

    for class_name in sorted(os.listdir(test_root)):
        class_dir = os.path.join(test_root, class_name)

        if not os.path.isdir(class_dir):
            continue

        actual_class = class_name.strip().lower()

        for video_folder in sorted(os.listdir(class_dir)):
            video_dir = os.path.join(class_dir, video_folder)

            if not os.path.isdir(video_dir):
                continue

            # ✅ ONLY rep_clips
            rep_dir = os.path.join(video_dir, "rep_clips")

            if not os.path.isdir(rep_dir):
                continue

            for file_name in sorted(os.listdir(rep_dir)):
                file_path = os.path.join(rep_dir, file_name)

                # ignore folders / junk
                if not os.path.isfile(file_path):
                    continue

                if file_name.lower().endswith(SUPPORTED_VIDEO_EXTENSIONS):
                    items.append({
                        "actual_class": actual_class,
                        "video_path": file_path,
                        "file_name": file_name,
                    })

    return items

def extract_keypoints_from_video(video_path):
    detector = create_detector()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        detector.close()
        print(f"[ERROR] Could not open video: {video_path}")
        return None

    frame_keypoints = []
    timestamp_ms = 0

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
                    frame_keypoints.append(joints)

    finally:
        cap.release()
        detector.close()

    if not frame_keypoints:
        return None

    return np.array(frame_keypoints, dtype=np.float32)

def classify_segmented_video(video_path, actual_class, model, device, label_map, idx_to_class):
    seq = extract_keypoints_from_video(video_path)

    if seq is None or len(seq) == 0:
        return None

    pred_name, conf, allowed_classes, restricted_probs = predict_sequence_restricted(
        model=model,
        device=device,
        seq=seq,
        actual_class=actual_class,
        label_map=label_map,
        idx_to_class=idx_to_class
    )

    return {
        "predicted": pred_name,
        "confidence": conf,
        "num_frames_used": len(seq),
        "allowed_classes": allowed_classes,
    }

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

            pred_result = classify_segmented_video(
                video_path,
                actual_class,
                model,
                device,
                label_map,
                idx_to_class
            )

            if not pred_result:
                print("  -> SKIPPED: no pose detected")
                skipped += 1
                continue

            pred = pred_result["predicted"]
            conf = pred_result["confidence"]
            frames_used = pred_result["num_frames_used"]

            print(f"  -> actual: {actual_class}")
            print(f"  -> predicted: {pred}")
            print(f"  -> confidence: {conf:.4f}")
            print(f"  -> frames used: {frames_used}")

            results.append({
                "file_name": file_name,
                "video_path": video_path,
                "actual": actual_class,
                "predicted": pred,
            })

    except Exception as e:
        print(f"[ERROR] {e}")

    if not results:
        print("\nNo valid results.")
        print(f"Skipped: {skipped}")
        return

    metrics = compute_one_vs_rest_metrics(results, class_names)
    print_metrics_table(metrics, total_tests=len(results))

    print()
    print("Valid tests:", len(results))
    print("Skipped:", skipped)


if __name__ == "__main__":
    main()