
import numpy as np
import cv2

# Memory for sustained state checking
state_memory = {}
memory_size = 15

# Memory for hand motion tracking
hand_history = {}

# Motion smoothing history
motion_history = {}
motion_window_size = 10

def smooth_state(worker_id, new_state):
    if worker_id not in state_memory:
        state_memory[worker_id] = []
    state_memory[worker_id].append(new_state)
    if len(state_memory[worker_id]) > memory_size:
        state_memory[worker_id].pop(0)
    return "Working" if state_memory[worker_id].count("Working") > (memory_size // 2) else "Idle"

def get_torso_angle(kpts):
    ls, rs = kpts[5][:2], kpts[6][:2]
    lh, rh = kpts[11][:2], kpts[12][:2]
    neck = (ls + rs) / 2
    mid_hip = (lh + rh) / 2
    vec = neck - mid_hip
    return abs(np.arctan2(vec[1], vec[0]) * 180 / np.pi)

def head_motion(prev_kpts, curr_kpts):
    head_points = [0, 1, 2]
    prev = np.array([prev_kpts[i][:2] for i in head_points])
    curr = np.array([curr_kpts[i][:2] for i in head_points])
    return np.linalg.norm(curr - prev, axis=1).mean()

def update_hand_trajectory(worker_id, lw, rw):
    if worker_id not in hand_history:
        hand_history[worker_id] = {'lw': [], 'rw': []}
    for hand, val in zip(['lw', 'rw'], [lw, rw]):
        hand_history[worker_id][hand].append(val)
        if len(hand_history[worker_id][hand]) > 20:
            hand_history[worker_id][hand].pop(0)
    lw_motion = np.mean([
        np.linalg.norm(np.array(hand_history[worker_id]['lw'][i]) - np.array(hand_history[worker_id]['lw'][i-1]))
        for i in range(1, len(hand_history[worker_id]['lw']))
    ]) if len(hand_history[worker_id]['lw']) > 1 else 0
    rw_motion = np.mean([
        np.linalg.norm(np.array(hand_history[worker_id]['rw'][i]) - np.array(hand_history[worker_id]['rw'][i-1]))
        for i in range(1, len(hand_history[worker_id]['rw']))
    ]) if len(hand_history[worker_id]['rw']) > 1 else 0
    return max(lw_motion, rw_motion)

def get_optical_flow_motion(prev_frame_gray, curr_frame_gray, roi):
    if roi is None:
        return 0
    x, y, w, h = roi
    prev_roi = prev_frame_gray[y:y+h, x:x+w]
    curr_roi = curr_frame_gray[y:y+h, x:x+w]
    flow = cv2.calcOpticalFlowFarneback(prev_roi, curr_roi, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.mean(mag)

def keypoint_motion(prev_kpts, curr_kpts):
    conf_mask = (prev_kpts[:, 2] > 0.5) & (curr_kpts[:, 2] > 0.5)
    if np.sum(conf_mask) < 5:
        return 0
    return np.linalg.norm(curr_kpts[conf_mask, :2] - prev_kpts[conf_mask, :2], axis=1).mean()

def update_motion_history(worker_id, motion):
    if worker_id not in motion_history:
        motion_history[worker_id] = []
    motion_history[worker_id].append(motion)
    if len(motion_history[worker_id]) > motion_window_size:
        motion_history[worker_id].pop(0)
    return np.mean(motion_history[worker_id])

def check_activity(kpts, prev_kpts=None, prev_frame=None, curr_frame=None, roi=None, worker_id=None):
    if kpts.shape[0] != 17:
        return "Idle"

    # Key body points
    lw, rw = kpts[9][:2], kpts[10][:2]
    le, re = kpts[7][:2], kpts[8][:2]
    ls, rs = kpts[5][:2], kpts[6][:2]

    # Arm distances
    active_pose = (
        np.linalg.norm(lw - ls) > 60 or
        np.linalg.norm(rw - rs) > 60 or
        np.linalg.norm(le - ls) > 50 or
        np.linalg.norm(re - rs) > 50
    )

    # Heuristics
    torso_angle = get_torso_angle(kpts)
    head_mov = head_motion(prev_kpts, kpts) if prev_kpts is not None else 0
    hand_mov = update_hand_trajectory(worker_id, lw, rw) if worker_id is not None else 0
    flow_mov = get_optical_flow_motion(prev_frame, curr_frame, roi) if prev_frame is not None and curr_frame is not None else 0
    motion = keypoint_motion(prev_kpts, kpts) if prev_kpts is not None else 0
    avg_motion = update_motion_history(worker_id, motion) if worker_id is not None and prev_kpts is not None else 0

    # Final classification
    engaged = (
        active_pose or
        (40 < torso_angle < 70) or
        hand_mov > 2.5 or
        flow_mov > 0.5 or
        avg_motion > 5 or
        (head_mov < 2 and prev_kpts is not None)
    )

    return smooth_state(worker_id, "Working" if engaged else "Idle")
