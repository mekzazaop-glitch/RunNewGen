import os
import csv

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import แยกเฟรม  # ใช้ SPLIT_DIRS (ชื่อโฟลเดอร์ที่เก็บเฟรม) ให้ตรงกับตอนแยกเฟรม

# ชุดข้อมูล -> โฟลเดอร์เฟรมของชุดนั้น (อ่านภาพที่แยกไว้แล้ว ไม่ต้องเปิดวิดีโอซ้ำ)
FRAMES_DIRS = แยกเฟรม.SPLIT_DIRS

# ชุดข้อมูล -> ไฟล์ CSV ปลายทาง (main.py import ตัวนี้ไปใช้ จะได้มีชื่อไฟล์อยู่ที่เดียว)
CSV_NAMES = {
    "Train": "landmarks_train.csv",
    "Vaildation": "landmarks_validation.csv",
    "Test": "landmarks_test.csv",
}

# ไฟล์โมเดลของ MediaPipe Tasks API (เวอร์ชันนี้ไม่มี mp.solutions แล้ว ต้องใช้ตัวนี้แทน)
# รุ่น lite = เร็วที่สุด แม่นยำน้อยสุด, ถ้าต้องการแม่นยำขึ้นเปลี่ยนไปดาวน์โหลดรุ่น full/heavy แทน:
# https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_(lite|full|heavy)/float16/1/pose_landmarker_(lite|full|heavy).task
MODEL_PATH = "pose_landmarker.task"

NUM_POSE_LANDMARKS = 33  # จำนวนจุดที่ Pose Landmarker คืนมาต่อคน (คงที่ตาม MediaPipe)

# เก็บเฉพาะจุดที่ใช้วิเคราะห์ท่าวิ่ง: ไหล่ (คำนวณมุมเอียงลำตัว), สะโพก, เข่า, ข้อเท้า
# ตัด ใบหน้า/แขน/มือ/ปลายเท้า ออก เพราะไม่จำเป็นกับการวิเคราะห์ท่าวิ่งช่วงล่าง+ลำตัว
KEEP_LANDMARKS = {
    11: "left_shoulder", 12: "right_shoulder",
    23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle",
}


def create_detector(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"ไม่พบไฟล์โมเดล {model_path} — ดาวน์โหลดจาก "
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task "
            "แล้ววางไว้โฟลเดอร์เดียวกับสคริปต์นี้ (เปลี่ยนชื่อไฟล์เป็น pose_landmarker.task)"
        )

    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        num_poses=1,  # วิเคราะห์คนวิ่ง 1 คนต่อเฟรม
        running_mode=vision.RunningMode.IMAGE,  # ใช้โหมดภาพนิ่ง เพราะแต่ละไฟล์ในโฟลเดอร์เป็นภาพนิ่งแยกจากกัน
    )
    return vision.PoseLandmarker.create_from_options(options)


def parse_label(frame_filename):
    """ดึง label จากชื่อไฟล์เฟรม เช่น 'Tbank_0007.jpg' -> prefix='T', word='bank'
    (ชื่อคลิปต้นทางคือส่วนก่อน '_' สุดท้าย ตามที่ แยกเฟรม.py ตั้งชื่อไว้)"""
    name = os.path.splitext(frame_filename)[0]  # 'Tbank_0007'
    video_name = name.rsplit("_", 1)[0]         # 'Tbank'
    prefix = video_name[0]                      # 'T' หรือ 'F'
    word = video_name[1:]                       # ส่วนที่เหลือ เช่น 'bank'
    return prefix, word


def load_frame(path):
    """อ่านภาพจาก path ที่อาจมีตัวอักษรไทย
    ห้ามใช้ cv2.imread ตรงๆ เพราะบน Windows มันใช้ API แบบ ANSI ภายใน
    ทำให้ path ที่ไม่ใช่ตัวอักษรละติน (เช่นภาษาไทย) อ่านไฟล์ไม่ได้ และคืนค่า None แบบเงียบๆ ไม่ error ให้เห็น
    วิธีแก้คืออ่าน bytes ด้วย Python's open() ปกติ แล้ว decode ภาพด้วย cv2.imdecode แทน"""
    with open(path, "rb") as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def extract_landmarks_from_frame(detector, frame):
    """ส่ง 1 เฟรมเข้า PoseLandmarker แล้วคืน list ของค่า landmark ที่ flatten แล้ว (x,y,z,visibility ต่อจุด)
    คืน None ถ้าตรวจไม่เจอคนในเฟรมนั้น"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # MediaPipe ต้องการ RGB ไม่ใช่ BGR ของ OpenCV
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = detector.detect(mp_image)

    if not result.pose_landmarks:
        return None

    # result.pose_landmarks: List[List[landmark]] -> เอาคนแรก (num_poses=1)
    pose = result.pose_landmarks[0]

    row = []
    for i in sorted(KEEP_LANDMARKS):  # เก็บเฉพาะจุดใน KEEP_LANDMARKS ตามลำดับ index
        lm = pose[i]
        row.extend([lm.x, lm.y, lm.z, lm.visibility])

    return row


def iter_frame_files(frames_dir):
    """วนไฟล์ภาพทุกไฟล์ในโฟลเดอร์เฟรมที่แยกไว้แล้ว"""
    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(
            f"ไม่พบโฟลเดอร์ {frames_dir} — ต้องรัน แยกเฟรม.py ก่อน เพื่อสร้างเฟรมไว้ในโฟลเดอร์นี้"
        )

    for filename in sorted(os.listdir(frames_dir)):
        if filename.lower().endswith(".jpg"):
            yield filename, os.path.join(frames_dir, filename)


def build_dataset(output_csv, frames_dir):
    rows_written = 0
    skipped_no_person = 0
    detector = create_detector()

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # หัวตาราง: 8 จุด (ไหล่/สะโพก/เข่า/ข้อเท้า) x (x,y,z,visibility) = 32 คอลัมน์ + label
        header = [f"{name}_{axis}" for _, name in sorted(KEEP_LANDMARKS.items()) for axis in ("x", "y", "z", "v")]
        writer.writerow(header + ["prefix", "word", "frame_file"])

        for filename, path in iter_frame_files(frames_dir):
            frame = load_frame(path)
            if frame is None:
                print(f"อ่านภาพไม่ได้ (ข้าม): {filename}")
                continue

            landmarks = extract_landmarks_from_frame(detector, frame)
            if landmarks is None:
                skipped_no_person += 1
                continue  # ตรงนี้ข้ามเฟรมที่ตรวจไม่เจอคน

            prefix, word = parse_label(filename)
            writer.writerow(landmarks + [prefix, word, filename])
            rows_written += 1

    print(f"บันทึกผลลัพธ์ {rows_written} แถว ไปที่ {output_csv} (ข้าม {skipped_no_person} เฟรมที่ตรวจไม่เจอคน)")


def build_all(skip_existing=True):
    """สกัด landmark ของทุกชุดข้อมูล แยกเป็น CSV คนละไฟล์
    ชุดที่มี CSV อยู่แล้วจะถูกข้าม เพราะ build_dataset เปิดไฟล์ด้วยโหมด "w" ถ้าไม่ข้ามจะเขียนทับของเดิมทิ้ง
    (ตั้ง skip_existing=False ถ้าอยากสกัดใหม่ทั้งหมด)"""
    for split, frames_dir in FRAMES_DIRS.items():
        output_csv = CSV_NAMES[split]

        if skip_existing and os.path.exists(output_csv):
            print(f"พบ {output_csv} อยู่แล้ว (ข้าม)")
            continue

        print(f"[{split}] กำลังสกัด landmark จาก '{frames_dir}' ...")
        build_dataset(output_csv, frames_dir)


if __name__ == "__main__":
    build_all()
