import os
import json
import pickle
import random
import gc
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================
# SETTINGS
# =========================
INPUT_ROOT = "WORKOUT_VIDEOS"   # folder with class folders inside
DATASET_ROOT = "dataset"
MODEL_PATH = "pose_landmarker_heavy.task"
VIDEO_OUTPUT_ROOT = "segmented_output"

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".wmv")

FIXED_FRAMES = 30
NUM_JOINTS = 33
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42

SAVE_VIDEO_OUTPUTS = True
SAFE_TIMESTAMP_STEP_MS = 33   # avoids timestamp issues

# =========================
# BLAZEPOSE LANDMARK IDS
# =========================
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

# =========================
# MOVEMENT-BASED CONFIG
# =========================
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
# MEDIAPIPE DETECTOR
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
def list_video_files(folder_path):
    return [
        os.path.join(folder_path, f)
        for f in sorted(os.listdir(folder_path))
        if f.lower().endswith(VIDEO_EXTENSIONS)
    ]


def get_motion_signal(landmarks, signal_type):
    if signal_type == "shoulder":
        return (landmarks[LEFT_SHOULDER].y + landmarks[RIGHT_SHOULDER].y) / 2.0
    elif signal_type == "hip":
        return (landmarks[LEFT_HIP].y + landmarks[RIGHT_HIP].y) / 2.0
    else:
        raise ValueError(f"Unknown signal_type: {signal_type}")


def resample_sequence_interp(sequence, target_frames):
    """
    Input:  (T, V, C)
    Output: (target_frames, V, C)
    """
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
    """
    Normalize by hip center.
    Input:  (T, V, C)
    Output: (T, V, C)
    """
    seq = sequence.copy()
    hip_center = (seq[:, LEFT_HIP, :] + seq[:, RIGHT_HIP, :]) / 2.0
    seq = seq - hip_center[:, np.newaxis, :]
    return seq.astype(np.float32)


def to_2sagcn_shape(sequence):
    """
    Input:  (T, V, C)
    Output: (C, T, V, M)
    """
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


def save_rep_clips_from_skeleton_video(skeleton_video_path, rep_segments, rep_clips_folder, fps, total_frames):
    """
    Saves rep clips by RE-READING the already-saved skeleton video,
    instead of storing all frames in RAM.
    """
    if not os.path.exists(skeleton_video_path):
        return

    cap = cv2.VideoCapture(skeleton_video_path)
    if not cap.isOpened():
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    for rep_num, (start_f, end_f) in enumerate(rep_segments, start=1):
        pad_before = int(0.2 * fps)
        pad_after = int(0.2 * fps)

        clip_start = max(0, start_f - pad_before)
        clip_end = min(total_frames - 1, end_f + pad_after)

        clip_path = os.path.join(rep_clips_folder, f"rep_{rep_num}.mp4")
        clip_writer = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
        current_frame = clip_start

        while current_frame <= clip_end:
            ret, frame = cap.read()
            if not ret:
                break
            clip_writer.write(frame)
            current_frame += 1

        clip_writer.release()

    cap.release()

# =========================
# REP EXTRACTION + VIDEO OUTPUT
# =========================
def extract_reps_from_video(video_path, class_name, detector, save_video_outputs=True):
    """
    Returns:
        rep_sequences: list of np.array with shape (T, 33, 3)
    """
    cfg = SEGMENT_CONFIG.get(class_name.lower(), DEFAULT_CONFIG)

    signal_type = cfg["signal"]
    initial_state = cfg["start_state"]
    motion_threshold = float(cfg["motion_threshold"])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    Could not open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

    # --- squat-specific stable range tracking ---
    baseline_up = None
    rep_bottom = None

    candidate_rep_start = None

    frame_keypoints = {}
    rep_segments = []

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    rep_count = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    skeleton_writer = None
    skeleton_video_path = None
    rep_clips_folder = None

    if save_video_outputs:
        video_output_folder = os.path.join(VIDEO_OUTPUT_ROOT, class_name, video_name)
        rep_clips_folder = os.path.join(video_output_folder, "rep_clips")
        os.makedirs(rep_clips_folder, exist_ok=True)

        skeleton_video_path = os.path.join(video_output_folder, "output_skeleton.mp4")
        skeleton_writer = cv2.VideoWriter(skeleton_video_path, fourcc, fps, (width, height))

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

            if len(joints) == NUM_JOINTS:
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

            # ============================================
            # SPECIAL FIX FOR SQUAT / SQUAT_WRONG_FORM
            # ============================================
            if class_name.lower() in ["squat", "squat_wrong_form"]:
                # update standing baseline only while in UP
                if state == "up":
                    if baseline_up is None:
                        baseline_up = smoothed_signal
                    else:
                        baseline_up = 0.9 * baseline_up + 0.1 * smoothed_signal

                # UP -> DOWN
                if state == "up" and baseline_up is not None:
                    down_trigger = baseline_up + motion_threshold

                    if smoothed_signal > down_trigger:
                        state = "down"
                        rep_bottom = smoothed_signal

                        if candidate_rep_start is None:
                            candidate_rep_start = frame_id

                # while DOWN, keep tracking deepest point
                elif state == "down":
                    if rep_bottom is None or smoothed_signal > rep_bottom:
                        rep_bottom = smoothed_signal

                    # DOWN -> UP
                    if baseline_up is not None and rep_bottom is not None:
                        rep_range = rep_bottom - baseline_up

                        if rep_range > motion_threshold:
                            up_trigger = baseline_up + rep_range * 0.5

                            if (
                                smoothed_signal < up_trigger
                                and (frame_id - last_rep_frame) > cooldown_frames
                            ):
                                state = "up"
                                rep_end = frame_id

                                if candidate_rep_start is None:
                                    candidate_rep_start = max(0, frame_id - int(fps))

                                rep_len = rep_end - candidate_rep_start + 1
                                if min_rep_frames <= rep_len <= max_rep_frames:
                                    rep_segments.append((candidate_rep_start, rep_end))
                                    last_rep_frame = frame_id
                                    rep_count += 1

                                candidate_rep_start = frame_id + 1
                                rep_bottom = None

                                # reset global motion range so old video history
                                # does not keep affecting future squat reps
                                min_seen = smoothed_signal
                                max_seen = smoothed_signal

            # ============================================
            # DEFAULT LOGIC FOR OTHER EXERCISES
            # ============================================
            else:
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
                                rep_segments.append((candidate_rep_start, rep_end))
                                last_rep_frame = frame_id
                                rep_count += 1

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
                                rep_segments.append((candidate_rep_start, rep_end))
                                last_rep_frame = frame_id
                                rep_count += 1

                            candidate_rep_start = frame_id + 1

            cv2.putText(
                output_frame,
                f"Rep count: {rep_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                output_frame,
                f"State: {state}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            if smoothed_signal is not None:
                cv2.putText(
                    output_frame,
                    f"{signal_type.title()}Y: {smoothed_signal:.3f}",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

            if class_name.lower() in ["squat", "squat_wrong_form"]:
                if baseline_up is not None:
                    cv2.putText(
                        output_frame,
                        f"BaselineUp: {baseline_up:.3f}",
                        (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 200, 255),
                        2
                    )

                if rep_bottom is not None:
                    cv2.putText(
                        output_frame,
                        f"RepBottom: {rep_bottom:.3f}",
                        (20, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 150, 255),
                        2
                    )

        if save_video_outputs and skeleton_writer is not None:
            skeleton_writer.write(output_frame)

        frame_id += 1

    cap.release()

    if skeleton_writer is not None:
        skeleton_writer.release()

    rep_sequences = []

    for start_f, end_f in rep_segments:
        seq = []

        for fid in range(start_f, end_f + 1):
            if fid in frame_keypoints:
                seq.append(frame_keypoints[fid])

        if len(seq) == 0:
            continue

        seq = np.array(seq, dtype=np.float32)
        rep_sequences.append(seq)

    if save_video_outputs and rep_clips_folder is not None and skeleton_video_path is not None:
        save_rep_clips_from_skeleton_video(
            skeleton_video_path=skeleton_video_path,
            rep_segments=rep_segments,
            rep_clips_folder=rep_clips_folder,
            fps=fps,
            total_frames=frame_id
        )

    # free memory from this video before returning
    del frame_keypoints
    del rep_segments
    gc.collect()

    return rep_sequences


# =========================
# SPLIT DATASET
# =========================
def split_dataset(data, labels, sample_names, train_split=0.8, seed=42):
    N = len(data)
    indices = list(range(N))

    random.seed(seed)
    random.shuffle(indices)

    split_idx = int(N * train_split)

    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    train_data = data[train_idx]
    val_data = data[val_idx]

    train_labels = labels[train_idx]
    val_labels = labels[val_idx]

    train_names = [sample_names[i] for i in train_idx]
    val_names = [sample_names[i] for i in val_idx]

    return train_data, train_labels, train_names, val_data, val_labels, val_names


# =========================
# MAIN
# =========================
def main():
    if not os.path.exists(INPUT_ROOT):
        raise RuntimeError(f"Input root folder not found: {INPUT_ROOT}")

    os.makedirs(DATASET_ROOT, exist_ok=True)
    os.makedirs(VIDEO_OUTPUT_ROOT, exist_ok=True)

    class_folders = [
        d for d in sorted(os.listdir(INPUT_ROOT))
        if os.path.isdir(os.path.join(INPUT_ROOT, d))
    ]

    if not class_folders:
        raise RuntimeError(f"No class folders found inside: {INPUT_ROOT}")

    label_map = {class_name: idx for idx, class_name in enumerate(class_folders)}

    print("Classes found:")
    for class_name, idx in label_map.items():
        print(f"  {idx} -> {class_name}")

    all_samples = []
    all_labels = []
    sample_names = []

    for class_name in class_folders:
        class_path = os.path.join(INPUT_ROOT, class_name)
        video_files = list_video_files(class_path)

        if not video_files:
            print(f"\nSkipping empty class folder: {class_name}")
            continue

        print(f"\nProcessing class: {class_name} ({len(video_files)} videos)")
        class_label = label_map[class_name]

        for video_path in video_files:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            print(f"  Video: {video_name}")

            detector = None
            try:
                detector = create_detector()

                rep_sequences = extract_reps_from_video(
                    video_path=video_path,
                    class_name=class_name,
                    detector=detector,
                    save_video_outputs=SAVE_VIDEO_OUTPUTS
                )

                if not rep_sequences:
                    print("    No reps found.")
                    continue

                print(f"    Reps found: {len(rep_sequences)}")

                for rep_idx, seq in enumerate(rep_sequences, start=1):
                    seq = resample_sequence_interp(seq, FIXED_FRAMES)
                    seq = normalize_pose_sequence(seq)
                    seq = to_2sagcn_shape(seq)

                    all_samples.append(seq)
                    all_labels.append(class_label)
                    sample_names.append(f"{class_name}/{video_name}_rep_{rep_idx}")

                del rep_sequences
                gc.collect()

            except Exception as e:
                print(f"    Error processing {video_name}: {e}")

            finally:
                if detector is not None:
                    detector.close()
                gc.collect()

    if not all_samples:
        raise RuntimeError("No samples were created. Check videos or segmentation thresholds.")

    data = np.stack(all_samples, axis=0)
    labels = np.array(all_labels, dtype=np.int64)

    # optional cleanup before split
    del all_samples
    gc.collect()

    print("\nFull dataset created.")
    print(f"data.shape   = {data.shape}")
    print(f"labels.shape = {labels.shape}")

    train_data, train_labels, train_names, val_data, val_labels, val_names = split_dataset(
        data=data,
        labels=labels,
        sample_names=sample_names,
        train_split=TRAIN_SPLIT,
        seed=RANDOM_SEED
    )

    train_data_path = os.path.join(DATASET_ROOT, "train_data.npy")
    val_data_path = os.path.join(DATASET_ROOT, "val_data.npy")
    train_label_path = os.path.join(DATASET_ROOT, "train_label.pkl")
    val_label_path = os.path.join(DATASET_ROOT, "val_label.pkl")
    label_map_path = os.path.join(DATASET_ROOT, "label_map.json")

    np.save(train_data_path, train_data)
    np.save(val_data_path, val_data)

    with open(train_label_path, "wb") as f:
        pickle.dump((train_names, train_labels.tolist()), f)

    with open(val_label_path, "wb") as f:
        pickle.dump((val_names, val_labels.tolist()), f)

    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=4)

    print("\nDone.")
    print(f"Saved: {train_data_path}")
    print(f"Saved: {train_label_path}")
    print(f"Saved: {val_data_path}")
    print(f"Saved: {val_label_path}")
    print(f"Saved: {label_map_path}")
    print(f"Saved videos under: {VIDEO_OUTPUT_ROOT}")

    print("\nTrain set:")
    print(f"  train_data.shape = {train_data.shape}")
    print(f"  train_labels     = {len(train_labels)}")

    print("\nValidation set:")
    print(f"  val_data.shape   = {val_data.shape}")
    print(f"  val_labels       = {len(val_labels)}")


if __name__ == "__main__":
    main()