"""
ขั้นที่ 1: สกัด landmark ใหม่จากวิดีโอที่ปรับขนาดแล้ว (VDO 960x720/) ด้วย MediaPipe

⚠️ จุดสำคัญ: Label รายเฟรมที่คนตรวจไว้แล้ว (ใน CSV Student.csv / CSV Prototype.csv) ยังใช้ได้อยู่
ไม่ต้องให้คนมา label ใหม่ — เพราะเฟรมที่ 00_resize_videos.py สร้างขึ้นมาจากวิดีโอ 'เวลาเดียวกัน'
กับที่เคย label ไว้ (สุ่มทุก 0.1 วิ = 10fps เหมือนตอนสร้าง JPG Student640x480 เดิม) แค่พิกเซล
เปลี่ยนไปเพราะย่อขนาด+letterbox — จับคู่ผ่านชื่อไฟล์เดิม (clip + frame_idx) แล้วแทนที่แค่พิกัด
พิกเซล ส่วน Label ดึงของเดิมมาใช้เป๊ะ

output: landmarks_raw.csv (right-side 6 จุด, พิกัดพิกเซลในเฟรม 960x720, join กับ Label เดิม)
"""

import argparse
import os
import re
import time

import cv2
import mediapipe as mp
import pandas as pd
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "pose_landmarker.task"
SAMPLE_FPS = 10  # ต้องตรงกับตอนสร้าง JPG Student640x480 เดิม (ทุก 0.1 วิ) ถึงจะจับคู่ label ได้ถูก

# ⚠️ บั๊กที่เจอ: ผู้ใช้รายงานว่าคลิปที่มีคนอื่นเดิน/ยืนอยู่ไกลๆ ด้านหลัง (เล็กกว่านักวิ่งหลักชัดเจน ไม่ทับซ้อน
# กัน) ทำให้โมเดลทายผิด — สาเหตุคือเดิม num_poses=1 บอก MediaPipe ให้คืนคนที่มันมั่นใจที่สุดแค่คนเดียว
# โดยที่โค้ดเราไม่มีทางรู้/ควบคุมได้เลยว่ามันเลือกใคร (pose_landmarks[0] เป็นแค่ index ตรงๆ ไม่ใช่การ
# "เลือก" จากผู้สมัครหลายคน) แก้โดยเพิ่ม num_poses เป็น 2 แล้วเลือกเองด้วย select_largest_person()
# ไม่ใช้ 3+ เพราะไม่มีหลักฐานว่าต้องรองรับคนเกิน 2 คนพร้อมกันในเฟรม และยิ่งมากยิ่งช้าลง
NUM_POSES = 2

RIGHT_SIDE_LANDMARKS = {
    12: "right_shoulder", 24: "right_hip", 26: "right_knee",
    28: "right_ankle", 30: "right_heel", 32: "right_foot_index",
}


def select_largest_person(pose_landmarks_list, width, height):
    """เมื่อ MediaPipe ตรวจเจอคนมากกว่า 1 คนในเฟรม (num_poses=2) เลือกคนที่ 'กรอบสี่เหลี่ยมครอบ 6 จุด
    ฝั่งขวาที่ติดตามอยู่แล้วมีพื้นที่ใหญ่ที่สุด' คือคนที่ตัวใหญ่สุด/ใกล้กล้องสุด — ไม่เชื่อลำดับที่ MediaPipe
    คืนมาตรงๆ เพราะ Tasks API ไม่รับประกันว่า index 0 คือคนที่มั่นใจที่สุดหรือคนหลักเสมอไป

    ใช้พื้นที่อย่างเดียว ไม่ถ่วงน้ำหนักตำแหน่งกึ่งกลางเฟรมด้วย เพราะยืนยันจากผู้ใช้แล้วว่าคนพื้นหลังที่ทำให้
    เกิดปัญหาอยู่ไกลกว่า/ตัวเล็กกว่าอย่างชัดเจน ไม่ได้อยู่ใกล้กึ่งกลางกว่านักวิ่งหลัก — ไม่จำเป็นต้องเพิ่ม
    พารามิเตอร์ที่ยังไม่มีหลักฐานว่าต้องใช้ (ข้อมูลมีแค่ 22 คลิป ยิ่งมีตัวแปรน้อยยิ่งดี)

    คืนค่า pose (list ของ 33 landmark) ของคนที่เลือก หรือ None ถ้าไม่มีใครเลย"""
    if not pose_landmarks_list:
        return None
    if len(pose_landmarks_list) == 1:
        return pose_landmarks_list[0]  # กรณีปกติ (คนเดียวในเฟรม) — no-op ไม่เปลี่ยนพฤติกรรมเดิมเลย

    best_pose, best_area = None, -1.0
    for pose in pose_landmarks_list:
        xs = [pose[idx].x * width for idx in RIGHT_SIDE_LANDMARKS]
        ys = [pose[idx].y * height for idx in RIGHT_SIDE_LANDMARKS]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if area > best_area:
            best_area, best_pose = area, pose
    return best_pose


def create_detector(model_path=MODEL_PATH):
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options, num_poses=NUM_POSES, running_mode=vision.RunningMode.VIDEO,
    )
    return vision.PoseLandmarker.create_from_options(options)


def load_existing_labels(student_csv="CSV Student.csv", prototype_csv="CSV Prototype.csv"):
    """โหลด Label เดิมที่คนตรวจไว้แล้ว คืน dict {filename: Label}"""
    labels = {}
    for path in (student_csv, prototype_csv):
        df = pd.read_csv(path)
        labels.update(dict(zip(df["filename"], df["Label"])))
    return labels


def extract_clip(detector, video_path, clip_name, label_lookup, is_prototype, debug_log=None):
    """สุ่มเฟรมทุก 1/SAMPLE_FPS วินาที (เหมือนตอนสร้าง JPG เดิม) สกัด 6 จุดฝั่งขวา จับคู่กับ Label เดิม

    debug_log: ถ้าส่ง list เข้ามา จะบันทึกทุกเฟรมที่ MediaPipe เจอคนมากกว่า 1 คน (clip, filename,
    จำนวนผู้สมัคร, พื้นที่ของคนที่เลือกเทียบกับอันดับรอง) ไว้ตรวจสอบว่าการเลือกคนทำงานถูกต้องไหม"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"เปิดวิดีโอไม่ได้: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, round(src_fps / SAMPLE_FPS))

    rows = []
    frame_idx_src = 0
    out_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx_src % frame_interval == 0:
            t_sec = out_idx / SAMPLE_FPS
            # ชื่อไฟล์ต้องตรงกับที่คนตรวจ label ไว้เป๊ะ ถึงจะ join เจอ
            if is_prototype:
                filename = f"frame_{out_idx:05d}.jpg"
            else:
                filename = f"{clip_name}_{out_idx:06d}_t{t_sec:07.2f}s.jpg"

            label = label_lookup.get(filename)
            if label is not None:  # ข้ามเฟรมที่ไม่มี label เดิม (นอกช่วงที่เคย label ไว้)
                timestamp_ms = int((frame_idx_src / src_fps) * 1000)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = detector.detect_for_video(mp_image, timestamp_ms)

                row = {"filename": filename, "clip": clip_name, "Label": label}
                h, w = frame.shape[:2]
                if len(result.pose_landmarks) > 1 and debug_log is not None:
                    areas = []
                    for p in result.pose_landmarks:
                        xs = [p[i].x * w for i in RIGHT_SIDE_LANDMARKS]
                        ys = [p[i].y * h for i in RIGHT_SIDE_LANDMARKS]
                        areas.append((max(xs) - min(xs)) * (max(ys) - min(ys)))
                    areas.sort(reverse=True)
                    debug_log.append({
                        "clip": clip_name, "filename": filename,
                        "n_candidates": len(result.pose_landmarks),
                        "chosen_area": round(areas[0], 1),
                        "runner_up_area": round(areas[1], 1) if len(areas) > 1 else None,
                    })
                pose = select_largest_person(result.pose_landmarks, w, h)
                if pose is not None:
                    for idx, name in RIGHT_SIDE_LANDMARKS.items():
                        lm = pose[idx]
                        row[f"{name}_x"] = round(lm.x * w, 2)
                        row[f"{name}_y"] = round(lm.y * h, 2)
                        row[f"{name}_v"] = lm.visibility
                else:
                    for name in RIGHT_SIDE_LANDMARKS.values():
                        row[f"{name}_x"] = None
                        row[f"{name}_y"] = None
                        row[f"{name}_v"] = None
                rows.append(row)

            out_idx += 1

        frame_idx_src += 1

    cap.release()
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", default="data/VDO 960x720")
    parser.add_argument("--output", default="landmarks_raw.csv")
    parser.add_argument("--student-csv", default="data/CSV Student.csv")
    parser.add_argument("--prototype-csv", default="data/CSV Prototype.csv")
    args = parser.parse_args()

    label_lookup = load_existing_labels(args.student_csv, args.prototype_csv)
    print(f"โหลด label เดิม {len(label_lookup)} เฟรม")

    videos = sorted(f for f in os.listdir(args.video_dir) if f.lower().endswith(".mp4"))
    print(f"พบวิดีโอ {len(videos)} ไฟล์ใน {args.video_dir}\n")

    all_rows = []
    multi_person_log = []  # เก็บทุกเฟรมที่ MediaPipe เจอคนเกิน 1 คน — ไว้ตรวจว่า select_largest_person() ทำงานถูกจุด
    for filename in videos:
        clip_name = os.path.splitext(filename)[0]
        is_prototype = clip_name == "Tyes"
        video_path = os.path.join(args.video_dir, filename)

        detector = create_detector()  # ต้องสร้างใหม่ทุกคลิป (running_mode=VIDEO ต้องการ timestamp ต่อเนื่อง)
        t0 = time.time()
        df = extract_clip(detector, video_path, clip_name, label_lookup, is_prototype,
                           debug_log=multi_person_log)
        detector.close()
        elapsed = time.time() - t0

        n_detected = df["right_hip_x"].notna().sum() if len(df) else 0
        print(f"{clip_name}: {len(df)} เฟรม (จับคู่ label ได้), ตรวจเจอคน {n_detected}/{len(df)} "
              f"ใช้เวลา {elapsed:.1f}s")
        all_rows.append(df)

    result = pd.concat(all_rows, ignore_index=True)
    result.to_csv(args.output, index=False)
    print(f"\nรวม {len(result)} เฟรม -> {args.output}")

    if multi_person_log:
        debug_df = pd.DataFrame(multi_person_log)
        debug_df.to_csv("multi_person_debug.csv", index=False)
        print(f"\nพบเฟรมที่มีคนมากกว่า 1 คน {len(debug_df)} เฟรม (จาก {debug_df['clip'].nunique()} คลิป) "
              f"-> multi_person_debug.csv (ดูว่าเลือกคนถูกไหมจากพื้นที่ chosen_area vs runner_up_area)")
    else:
        print("\nไม่พบเฟรมไหนที่ MediaPipe เจอคนเกิน 1 คนเลย (num_poses=2 ไม่พบผลกระทบในชุดข้อมูลนี้)")


if __name__ == "__main__":
    main()
