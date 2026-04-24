import os
import time
import json
import traceback
import logging
from typing import Optional, Dict, Any

import pymysql
from pymysql.cursors import DictCursor


# =========================
# CONFIG
# =========================
DB_HOST = "localhost"
DB_NAME = "bettergym_db"
DB_USER = "root"
DB_PASS = ""
DB_PORT = 3306

# How often to check for new pending videos when idle
POLL_INTERVAL_SECONDS = 5

# Optional: if your PHP stores relative paths like uploads/ai_videos/...
# and your worker runs from a different folder, set this to your PHP root.
# Example:
#   APP_ROOT = r"C:\xampp\htdocs\bettergym_api"
#   APP_ROOT = "/var/www/html/bettergym_api"

APP_ROOT = r"D:\xampp\htdocs"
VIDEO_BASE_DIR = r"D:\xampp\.uploads\uploads\ai_videos"

# Log file
LOG_FILE = os.path.join(APP_ROOT, "worker.log")


# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)


# =========================
# DB HELPERS
# =========================
def get_connection():
    return pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        port=DB_PORT,
        charset="utf8mb4",
        cursorclass=DictCursor,
        autocommit=False
    )


def make_absolute_path(file_path: str) -> str:
    if os.path.isabs(file_path):
        return file_path
    return os.path.normpath(os.path.join(VIDEO_BASE_DIR, file_path))


# =========================
# QUEUE FUNCTIONS
# =========================
def fetch_next_pending_job(conn) -> Optional[Dict[str, Any]]:
    """
    Gets the oldest pending video.
    """
    sql = """
        SELECT
            id,
            user_id,
            workout_session_id,
            exercise_name,
            file_name,
            file_path,
            uploaded_at,
            processing_status
        FROM exercise_videos
        WHERE processing_status = 'pending'
        ORDER BY uploaded_at ASC
        LIMIT 1
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        row = cur.fetchone()
    return row


def claim_job(conn, video_id: int) -> bool:
    """
    Tries to atomically mark a job as processing.
    Returns True if claim succeeded.
    """
    sql = """
        UPDATE exercise_videos
        SET processing_status = 'processing',
            error_message = NULL
        WHERE id = %s
          AND processing_status = 'pending'
    """
    with conn.cursor() as cur:
        affected = cur.execute(sql, (video_id,))
    conn.commit()
    return affected == 1


def mark_done(conn, video_id: int):
    sql = """
        UPDATE exercise_videos
        SET processing_status = 'done',
            error_message = NULL
        WHERE id = %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (video_id,))
    conn.commit()

def mark_session_completed_if_all_done(conn, session_id: str):
    check_sql = """
        SELECT COUNT(*) AS remaining
        FROM exercise_videos
        WHERE workout_session_id = %s
          AND processing_status IN ('pending', 'processing')
    """
    with conn.cursor() as cur:
        cur.execute(check_sql, (session_id,))
        row = cur.fetchone()

    if row and int(row["remaining"]) == 0:
        update_sql = """
            UPDATE workout_sessions
            SET status = 'COMPLETED'
            WHERE id = %s
        """
        with conn.cursor() as cur:
            cur.execute(update_sql, (session_id,))
        conn.commit()

def mark_failed(conn, video_id: int, error_message: str):
    sql = """
        UPDATE exercise_videos
        SET processing_status = 'failed',
            error_message = %s
        WHERE id = %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (error_message[:5000], video_id))
    conn.commit()

def insert_processed_result(
    conn,
    user_id: int,
    session_id: str,
    exercise_name: str,
    result_json: Dict[str, Any],
    model_status: str = "done"
):
    score = int(round(result_json.get("score", 0) or 0))

    file_path = result_json.get("annotated_video_path") or None
    file_name = os.path.basename(str(file_path).replace("\\", "/")) if file_path else None

    sql = """
        INSERT INTO processed_videos (
            user_id,
            session_id,
            exercise_name,
            result_json,
            file_path,
            file_name,
            model_status,
            score,
            created_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
    """

    with conn.cursor() as cur:
        cur.execute(
            sql,
            (
                user_id,
                session_id,
                exercise_name,
                json.dumps(result_json, ensure_ascii=False),
                file_path,
                file_name,
                model_status,
                score
            )
        )
    conn.commit()

# =========================
# AI PIPELINE
# =========================
def run_ai_pipeline(video_abs_path: str, exercise_name: str):
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

    MODEL_PATH = "pose_landmarker_heavy.task"
    WEIGHTS_PATH = "work_dir/custom/joint/model-89-2430.pt"
    LABEL_MAP_PATH = "dataset_legacy/label_map.json"

    NUM_POINT = 33
    NUM_PERSON = 1
    FIXED_FRAMES = 30
    SAFE_TIMESTAMP_STEP_MS = 33

    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24

    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16

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
        "bicep_curls": {"signal": "curl_distance", "start_state": "down", "motion_threshold": 0.06},
        "bicep_curl": {"signal": "curl_distance", "start_state": "down", "motion_threshold": 0.06},
        "pushup": {"signal": "shoulder", "start_state": "up", "motion_threshold": 0.03},
        "push_up": {"signal": "shoulder", "start_state": "up", "motion_threshold": 0.03},
        "bench_dips": {"signal": "shoulder", "start_state": "up", "motion_threshold": 0.03},
        "squat": {"signal": "hip", "start_state": "up", "motion_threshold": 0.03},
        "squat_wrong_form": {"signal": "hip", "start_state": "up", "motion_threshold": 0.03},
        "lunges": {"signal": "hip", "start_state": "up", "motion_threshold": 0.035},
        "lunge": {"signal": "hip", "start_state": "up", "motion_threshold": 0.035},
    }

    DEFAULT_CONFIG = {
        "signal": "shoulder",
        "start_state": "up",
        "motion_threshold": 0.03,
    }

    def normalize_exercise_name(name: str) -> str:
        name = name.strip().lower().replace(" ", "_")

        aliases = {
            "push_up": "pushup",

            "bicep_curl": "bicep_curls",
            "bicep_curls": "bicep_curls",

            "bench_dip": "bench_dips",
            "bench_dips": "bench_dips",
        }

        return aliases.get(name, name)

    def get_motion_signal(landmarks, signal_type):
        if signal_type == "shoulder":
            return (landmarks[LEFT_SHOULDER].y + landmarks[RIGHT_SHOULDER].y) / 2.0
        elif signal_type == "hip":
            return (landmarks[LEFT_HIP].y + landmarks[RIGHT_HIP].y) / 2.0
        elif signal_type == "wrist":
            return (landmarks[LEFT_WRIST].y + landmarks[RIGHT_WRIST].y) / 2.0
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
                resampled[:, v, c] = np.interp(new_positions, old_positions, sequence[:, v, c])

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

    def clean_state_dict(state_dict):
        cleaned = OrderedDict()
        for k, v in state_dict.items():
            new_key = k[7:] if k.startswith("module.") else k
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

    def predict_rep(model, device, seq, allowed_class_names=None):
        seq = resample_sequence_interp(seq, FIXED_FRAMES)
        seq = normalize_pose_sequence(seq)
        seq = to_2sagcn_shape(seq)
        seq = np.expand_dims(seq, axis=0)

        x = torch.tensor(seq, dtype=torch.float32).to(device)

        with torch.no_grad():
            out = model(x)
            logits_tensor = out[0]
            logits = logits_tensor.detach().cpu().numpy()

            full_prob = torch.softmax(out, dim=1)[0].cpu().numpy()

            if allowed_class_names:
                allowed_indices = [
                    idx for idx, name in idx_to_class.items()
                    if normalize_exercise_name(name) in allowed_class_names
                ]

                if allowed_indices:
                    allowed_logits = logits_tensor[allowed_indices]
                    allowed_probs = torch.softmax(allowed_logits, dim=0).cpu().numpy()

                    best_allowed_pos = int(np.argmax(allowed_probs))
                    pred = int(allowed_indices[best_allowed_pos])
                    conf = float(allowed_probs[best_allowed_pos])

                    prob = np.zeros_like(full_prob)
                    for pos, class_idx in enumerate(allowed_indices):
                        prob[class_idx] = allowed_probs[pos]
                else:
                    prob = full_prob
                    pred = int(np.argmax(prob))
                    conf = float(prob[pred])
            else:
                prob = full_prob
                pred = int(np.argmax(prob))
                conf = float(prob[pred])

        return pred, conf, prob, logits

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

    def make_relative_to_app_root(abs_path):
        try:
            return os.path.relpath(abs_path, VIDEO_BASE_DIR).replace("\\", "/")
        except Exception:
            return abs_path.replace("\\", "/")

    if not os.path.exists(video_abs_path):
        raise FileNotFoundError(f"Video not found: {video_abs_path}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Pose model not found: {MODEL_PATH}")
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Weights not found: {WEIGHTS_PATH}")
    if not os.path.exists(LABEL_MAP_PATH):
        raise FileNotFoundError(f"Label map not found: {LABEL_MAP_PATH}")

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    idx_to_class = {v: k for k, v in label_map.items()}
    num_class = len(label_map)

    ALLOWED_CLASSES_BY_EXERCISE = {
        "lunges": ["lunges", "lunges_body_leaning_forward"],
        "lunge": ["lunges", "lunges_body_leaning_forward"],

        "squat": [
            "squat",
            "squat_body_leaning_forward",
            "squat_legs_too_narrow",
            "squat_legs_too_wide",
        ],

        "pushup": ["pushup", "pushup_elbows_flared"],
        "push_up": ["pushup", "pushup_elbows_flared"],

        "bicep_curls": ["bicep_curls", "bicep_curls_elbows_moving"],
        "bicep_curl": ["bicep_curls", "bicep_curls_elbows_moving"],

        "bench_dips": ["bench_dips", "bench_dips_elbows_flared"],
    }

    model, device = load_model(WEIGHTS_PATH, num_class=num_class)

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    normalized_name = normalize_exercise_name(exercise_name)
    cfg = SEGMENT_CONFIG.get(normalized_name, DEFAULT_CONFIG)

    allowed_class_names = ALLOWED_CLASSES_BY_EXERCISE.get(normalized_name)

    signal_type = cfg["signal"]
    initial_state = cfg["start_state"]
    motion_threshold = float(cfg["motion_threshold"])

    video_dir = os.path.dirname(video_abs_path)
    video_name = os.path.splitext(os.path.basename(video_abs_path))[0]

    annotated_video_abs = os.path.join(video_dir, f"{video_name}_out.mp4")
    segments_dir_abs = os.path.join(video_dir, f"{video_name}_segments")
    os.makedirs(segments_dir_abs, exist_ok=True)

    cap = cv2.VideoCapture(video_abs_path)
    if not cap.isOpened():
        detector.close()
        raise RuntimeError(f"Could not open video: {video_abs_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    annotated_writer = cv2.VideoWriter(annotated_video_abs, fourcc, fps, (width, height))

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
    stored_frames = {}
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
            stored_frames[frame_id] = frame.copy()
            timestamp_ms += SAFE_TIMESTAMP_STEP_MS

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            result = detector.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]

                joints = [[lm.x, lm.y, lm.z] for lm in landmarks]
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
                                    pred_idx, conf, probs, logits = predict_rep(
                                        model,
                                        device,
                                        seq,
                                        allowed_class_names=allowed_class_names
                                    )
                                    pred_name = idx_to_class.get(pred_idx, str(pred_idx))

                                    if allowed_class_names:
                                        num_top = len(allowed_class_names)
                                    else:
                                        num_top = 3  # fallback

                                    if allowed_class_names:
                                        allowed_indices = [
                                            idx for idx, name in idx_to_class.items()
                                            if normalize_exercise_name(name) in allowed_class_names
                                        ]

                                        allowed_probs = probs[allowed_indices]

                                        sorted_idx = np.argsort(-allowed_probs)

                                        top_predictions = [
                                            {
                                                "label": idx_to_class.get(int(allowed_indices[i]),
                                                                          str(allowed_indices[i])),
                                                "prob": float(allowed_probs[i])
                                            }
                                            for i in sorted_idx[:len(allowed_indices)]
                                        ]
                                    else:
                                        top_indices = np.argsort(-probs)[:3]
                                        top_predictions = [
                                            {
                                                "label": idx_to_class.get(int(idx), str(idx)),
                                                "prob": float(probs[idx])
                                            }
                                            for idx in top_indices
                                        ]
                                    logit_map = {
                                        idx_to_class.get(i, str(i)): float(val)
                                        for i, val in enumerate(logits)
                                    }

                                    rep_count += 1
                                    active_label = pred_name
                                    active_conf = float(conf)
                                    last_rep_frame = frame_id

                                    # save segmented rep video
                                    rep_file_name = f"rep_{rep_count}_{pred_name}.mp4"
                                    rep_abs_path = os.path.join(segments_dir_abs, rep_file_name)
                                    rep_writer = cv2.VideoWriter(rep_abs_path, fourcc, fps, (width, height))

                                    for fid in range(candidate_rep_start, rep_end + 1):
                                        if fid in stored_frames:
                                            rep_writer.write(stored_frames[fid])
                                    rep_writer.release()

                                    expected_label = normalize_exercise_name(exercise_name)
                                    predicted_label = normalize_exercise_name(pred_name)
                                    is_good_form = predicted_label == expected_label

                                    rep_results.append({
                                        "rep": rep_count,
                                        "start_frame": int(candidate_rep_start),
                                        "end_frame": int(rep_end),
                                        "pred_idx": int(pred_idx),
                                        "pred_name": pred_name,
                                        "confidence": float(conf),
                                        "is_good_form": is_good_form,
                                        "form_label": "good" if is_good_form else "bad",
                                        "top_predictions": top_predictions,
                                        "logits": logit_map,
                                        "segment_video_path": make_relative_to_app_root(rep_abs_path)
                                    })

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
                                    pred_idx, conf, probs, logits = predict_rep(
                                        model,
                                        device,
                                        seq,
                                        allowed_class_names=allowed_class_names
                                    )
                                    pred_name = idx_to_class.get(pred_idx, str(pred_idx))

                                    if allowed_class_names:
                                        num_top = len(allowed_class_names)
                                    else:
                                        num_top = 3  # fallback

                                    if allowed_class_names:
                                        allowed_indices = [
                                            idx for idx, name in idx_to_class.items()
                                            if normalize_exercise_name(name) in allowed_class_names
                                        ]

                                        allowed_probs = probs[allowed_indices]

                                        sorted_idx = np.argsort(-allowed_probs)

                                        top_predictions = [
                                            {
                                                "label": idx_to_class.get(int(allowed_indices[i]),
                                                                          str(allowed_indices[i])),
                                                "prob": float(allowed_probs[i])
                                            }
                                            for i in sorted_idx[:len(allowed_indices)]
                                        ]
                                    else:
                                        top_indices = np.argsort(-probs)[:3]
                                        top_predictions = [
                                            {
                                                "label": idx_to_class.get(int(idx), str(idx)),
                                                "prob": float(probs[idx])
                                            }
                                            for idx in top_indices
                                        ]

                                    logit_map = {
                                        idx_to_class.get(i, str(i)): float(val)
                                        for i, val in enumerate(logits)
                                    }

                                    rep_count += 1
                                    active_label = pred_name
                                    active_conf = float(conf)
                                    last_rep_frame = frame_id

                                    rep_file_name = f"rep_{rep_count}_{pred_name}.mp4"
                                    rep_abs_path = os.path.join(segments_dir_abs, rep_file_name)
                                    rep_writer = cv2.VideoWriter(rep_abs_path, fourcc, fps, (width, height))

                                    for fid in range(candidate_rep_start, rep_end + 1):
                                        if fid in stored_frames:
                                            rep_writer.write(stored_frames[fid])
                                    rep_writer.release()

                                    expected_label = normalize_exercise_name(exercise_name)
                                    predicted_label = normalize_exercise_name(pred_name)
                                    is_good_form = predicted_label == expected_label

                                    rep_results.append({
                                        "rep": rep_count,
                                        "start_frame": int(candidate_rep_start),
                                        "end_frame": int(rep_end),
                                        "pred_idx": int(pred_idx),
                                        "pred_name": pred_name,
                                        "confidence": float(conf),
                                        "is_good_form": is_good_form,
                                        "form_label": "good" if is_good_form else "bad",
                                        "top_predictions": top_predictions,
                                        "logits": logit_map,
                                        "segment_video_path": make_relative_to_app_root(rep_abs_path)
                                    })

                            candidate_rep_start = frame_id + 1

            cv2.putText(output_frame, f"Exercise: {exercise_name}", (20, 40),
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

            annotated_writer.write(output_frame)
            frame_id += 1

    finally:
        cap.release()
        annotated_writer.release()
        detector.close()

    if not rep_results:
        return {
            "exercise": exercise_name,
            "segment_config_used": cfg,
            "annotated_video_path": make_relative_to_app_root(annotated_video_abs),
            "segments_folder_path": make_relative_to_app_root(segments_dir_abs),
            "good_reps": 0,
            "bad_reps": 0,
            "score": 0,
            "total_reps": 0,
            "majority_prediction": None,
            "average_confidence": 0.0,
            "reps": [],
            "message": "No reps found."
        }

    majority = Counter([r["pred_name"] for r in rep_results]).most_common(1)[0][0]
    avg_conf = float(np.mean([r["confidence"] for r in rep_results]))

    good_reps = sum(1 for r in rep_results if r["is_good_form"])
    bad_reps = len(rep_results) - good_reps
    score = int(round((good_reps / len(rep_results)) * 100)) if rep_results else 0.0

    return {
        "exercise": exercise_name,
        "segment_config_used": cfg,
        "annotated_video_path": make_relative_to_app_root(annotated_video_abs),
        "segments_folder_path": make_relative_to_app_root(segments_dir_abs),
        "total_reps": len(rep_results),
        "good_reps": good_reps,
        "bad_reps": bad_reps,
        "score": score,
        "majority_prediction": majority,
        "average_confidence": avg_conf,
        "reps": rep_results
    }

# Helper functions
def normalize_exercise_name(name: str) -> str:
    name = name.strip().lower().replace(" ", "_")

    aliases = {
        "push_up": "pushup",

        "bicep_curl": "bicep_curls",
        "bicep_curls": "bicep_curls",

        "bench_dip": "bench_dips",
        "bench_dips": "bench_dips",
    }

    return aliases.get(name, name)

def update_session_global_score(conn, session_id: str):
    """
    Updates workout_sessions.global_score as the rounded average
    of all processed_videos.score values for the session.
    """
    avg_sql = """
        SELECT AVG(score) AS avg_score
        FROM processed_videos
        WHERE session_id = %s
          AND model_status = 'done'
    """

    with conn.cursor() as cur:
        cur.execute(avg_sql, (session_id,))
        row = cur.fetchone()

    avg_score = 0
    if row and row["avg_score"] is not None:
        avg_score = int(round(float(row["avg_score"])))

    update_sql = """
        UPDATE workout_sessions
        SET global_score = %s
        WHERE id = %s
    """

    with conn.cursor() as cur:
        cur.execute(update_sql, (avg_score, session_id))

    conn.commit()

# =========================
# SINGLE JOB PROCESSOR
# =========================
def process_one_job():
    conn = None
    try:
        conn = get_connection()

        job = fetch_next_pending_job(conn)
        if not job:
            logging.info("No pending videos found.")
            return False

        video_id = job["id"]
        user_id = job["user_id"]
        session_id = job["workout_session_id"]
        exercise_name = job["exercise_name"]
        file_path = job["file_path"]

        logging.info(
            f"Found pending job | video_id={video_id} | user_id={user_id} | "
            f"session_id={session_id} | exercise={exercise_name}"
        )

        claimed = claim_job(conn, video_id)
        if not claimed:
            logging.info(f"Job already claimed by another worker | video_id={video_id}")
            return True

        abs_video_path = make_absolute_path(file_path)
        logging.info(f"Processing file: {abs_video_path}")

        result = run_ai_pipeline(abs_video_path, exercise_name)

        insert_processed_result(
            conn=conn,
            user_id=user_id,
            session_id=session_id,
            exercise_name=exercise_name,
            result_json=result,
            model_status="done"
        )

        create_notifications_from_result(
            conn=conn,
            user_id=user_id,
            session_id=session_id,
            exercise_name=exercise_name,
            result_json=result
        )

        update_session_global_score(conn, session_id)

        mark_done(conn, video_id)
        mark_session_completed_if_all_done(conn, session_id)
        logging.info(f"Job completed successfully | video_id={video_id}")
        return True

    except Exception as e:
        error_text = f"{str(e)}\n{traceback.format_exc()}"
        logging.error(error_text)

        try:
            if conn is not None:
                # If video_id exists in local scope, mark failed
                if "video_id" in locals():
                    mark_failed(conn, video_id, error_text)
                    logging.info(f"Marked failed | video_id={video_id}")
        except Exception as inner_e:
            logging.error(f"Failed to mark job as failed: {inner_e}")

        return True

    finally:
        if conn is not None:
            conn.close()

# FOR NOTIFICATIONS
def prettify_name(name: str) -> str:
    return name.strip().replace("_", " ").title()

def get_bad_form_feedback(predicted_name: str):
    key = normalize_exercise_name(predicted_name)

    feedbacks = {
        'bench_dips_elbows_flared': {
            'label': 'elbows flared',
            'risk': 'Anterior Shoulder Capsular Strain',
        },
        'bicep_curls_elbows_moving': {
            'label': 'elbows moving',
            'risk': 'Anterior Shoulder Strain',
        },
        'lunges_body_leaning_forward': {
            'label': 'body leaning forward',
            'risk': 'Lumbar Strain',
        },
        'pushup_elbows_flared': {
            'label': 'elbows flared',
            'risk': 'Shoulder Impingement',
        },
        'squat_body_leaning_forward': {
            'label': 'body leaning forward',
            'risk': 'Erector Spinae Strain',
        },
        'squat_legs_too_narrow': {
            'label': 'legs too narrow',
            'risk': 'Meniscus Tear',
        },
        'squat_legs_too_wide': {
            'label': 'legs too wide',
            'risk': 'Hip Adductor Strain',
        },
    }

    return feedbacks.get(key)


def insert_notification(conn, user_id: int, session_id: str, title: str, message: str):
    sql = """
        INSERT INTO notifications (
            user_id,
            session_id,
            title,
            message,
            is_read,
            created_at
        ) VALUES (%s, %s, %s, %s, 0, NOW())
    """
    with conn.cursor() as cur:
        cur.execute(sql, (user_id, session_id, title, message))
    conn.commit()


def notification_exists(conn, user_id: int, session_id: str, title: str, message: str) -> bool:
    sql = """
        SELECT id
        FROM notifications
        WHERE user_id = %s
          AND session_id = %s
          AND title = %s
          AND message = %s
        LIMIT 1
    """
    with conn.cursor() as cur:
        cur.execute(sql, (user_id, session_id, title, message))
        row = cur.fetchone()
    return row is not None






def create_notifications_from_result(conn, user_id: int, session_id: str, exercise_name: str, result_json: dict):
    reps = result_json.get("reps", []) or []
    if not reps:
        return

    grouped_bad_forms = {}

    for rep in reps:
        if rep.get("is_good_form") is True:
            continue

        pred_name = str(rep.get("pred_name", "")).strip()
        if not pred_name:
            continue

        grouped_bad_forms[pred_name] = grouped_bad_forms.get(pred_name, 0) + 1

    for pred_name, count in grouped_bad_forms.items():
        if count < 3:
            continue

        feedback = get_bad_form_feedback(pred_name)
        if not feedback:
            continue

        exercise_display = prettify_name(exercise_name)
        bad_form_label = feedback["label"]
        risk = feedback["risk"]

        title = f"{exercise_display} Wrong Form Detected"

        if count == 1:
            message = (
                f"1 wrong-form {exercise_display.lower()} rep detected: "
                f"{bad_form_label}. This could lead to {risk}."
            )
        else:
            message = (
                f"{count} wrong-form {exercise_display.lower()} reps detected: "
                f"{bad_form_label}. This could lead to {risk}."
            )

        if not notification_exists(conn, user_id, session_id, title, message):
            insert_notification(conn, user_id, session_id, title, message)

# =========================
# MAIN LOOP
# =========================
def main():
    logging.info("Worker started.")
    while True:
        had_activity = process_one_job()

        if had_activity:
            # If it processed something (or failed something), try next quickly
            time.sleep(1)
        else:
            # If no pending jobs, wait before polling again
            time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()