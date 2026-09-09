"""
inference.py — ท่อประมวลผล 1 คลิปที่อัปโหลดเข้ามา ตั้งแต่วิดีโอดิบ -> คลาส (ถูก/ผิด) + ความมั่นใจ

**สำคัญที่สุด (ข้อควรระวัง #13 — train/serve skew)**: import ฟังก์ชันคำนวณจาก
00_resize_videos.py / 01_extract_landmarks.py / 02_prepare_dataset.py โดยตรง ไม่เขียนสูตร
คำนวณ feature ซ้ำเองแม้แต่บรรทัดเดียว — ให้เว็บคำนวณด้วยสูตรเดียวกันเป๊ะกับตอนเทรนโมเดล
ถ้าแก้สูตรใน 3 ไฟล์นั้นทีหลัง ต้องรัน 02_prepare_dataset.py + 03_train_model.py ใหม่เสมอ
"""

import json
import os
import sys
from importlib import import_module

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

resize_mod = import_module("00_resize_videos")
extract_mod = import_module("01_extract_landmarks")
prep_mod = import_module("02_prepare_dataset")

MODEL_PATH = os.path.join(PARENT_DIR, "model.joblib")
FACING_REFERENCE_PATH = os.path.join(PARENT_DIR, prep_mod.FACING_REFERENCE_PATH)
SCORE_CONFIG_PATH = os.path.join(PARENT_DIR, "score_config.json")
POSE_MODEL_PATH = os.path.join(PARENT_DIR, "pose_landmarker.task")

ANGLE_LABELS_TH = {
    "knee_angle": "มุมเข่า", "hip_angle": "มุมสะโพก", "ankle_angle": "มุมข้อเท้า",
    "thigh_angle": "มุมต้นขา", "shank_angle": "มุมหน้าแข้ง", "foot_angle": "มุมเท้า",
}

# คำแนะนำเชิงโค้ช ผูกกับแต่ละมุม + งานอ้างอิงทางชีวกลศาสตร์การวิ่ง (สำหรับอ้างอิงในรายงาน/สอบ)
# ⚠️ นี่คือคำแนะนำทั่วไปจากวรรณกรรม ไม่ใช่ผลจากการ calibrate กับชุดข้อมูลนี้โดยตรง (ดูคำเตือนใน
# 05_score_calibration.py — สูตรให้คะแนนรายมุมจากชุดข้อมูลนี้เองแยกกลุ่มถูก/ผิดไม่ได้จริง)
# ใช้เป็น "ข้อมูลประกอบ" คู่กับ angle_context เท่านั้น
ADVICE_LIBRARY = {
    "knee_angle": {
        "high": "เข่าเหยียดตรงเกินไปตอนเท้าแตะพื้น (เหมือนก้าวยาว/เดิน) เพิ่มความเสี่ยงแรงกระแทกขึ้นขา "
                "ลองเพิ่มจังหวะก้าว (cadence) ให้ถี่ขึ้นและให้เข่างอรับน้ำหนักมากขึ้นเล็กน้อย",
        "low": "เข่างอมากเกินไปตลอดช่วงยืนขา อาจทำให้ใช้พลังงานเกินความจำเป็นและกล้ามเนื้อต้นขาล้าเร็ว "
               "ลองวิ่งให้ลำตัวตกลงมาบนขาที่รับน้ำหนักมากขึ้นแทนการงอเข่าเยอะ",
        "reference": "Novacheck TF. The biomechanics of running. Gait & Posture. 1998;7(1):77-95.",
    },
    "hip_angle": {
        "high": "สะโพกเหยียดน้อย (ก้าวสั้น/ไม่ส่งขาไปข้างหลังเต็มที่) ลองเพิ่มระยะเหยียดสะโพกตอนดันตัวออก",
        "low": "สะโพกงอมาก อาจมาจากก้าวยาวเกินไปด้านหน้า ลองลดระยะก้าวและวางเท้าใกล้จุดใต้ลำตัวมากขึ้น",
        "reference": "Souza RB. An Evidence-Based Videotaped Running Biomechanics Analysis. "
                     "Phys Med Rehabil Clin N Am. 2016;27(1):217-236.",
    },
    "ankle_angle": {
        "high": "ข้อเท้ากระดกน้อย (มีแนวโน้มลงส้นเท้าแรง) ลองฝึกลงเท้าด้วยกลางเท้าและกระดกข้อเท้าขึ้นมากขึ้น",
        "low": "ข้อเท้างอมาก อาจลงเท้าด้วยปลายเท้ามากเกินไป เพิ่มภาระน่อง ลองลงเท้าด้วยกลางเท้าแทน",
        "reference": "Dicharry J. Kinematics and kinetics of gait: from lab to clinic. "
                     "Clin Sports Med. 2010;29(3):347-364.",
    },
    "thigh_angle": {
        "high": "ต้นขายกไปข้างหน้ามาก (ก้าวยาวเกิน) เชื่อมโยงกับแรงเบรกที่ข้อเท้าตอนลงเท้า "
                "ลองลดความยาวก้าวและเพิ่มจังหวะก้าวแทน",
        "low": "ต้นขายกไปข้างหน้าน้อย ก้าวอาจสั้นเกินไป ลองเพิ่มแรงส่งจากสะโพกขณะดันตัวไปข้างหน้า",
        "reference": "Heiderscheit BC, et al. Effects of step rate manipulation on joint mechanics "
                     "during running. Med Sci Sports Exerc. 2011;43(2):296-302.",
    },
    "shank_angle": {
        "high": "หน้าแข้งเอียงมาก ตอนลงเท้าอาจอยู่ไกลหน้าลำตัวเกินไป (overstriding) "
                "ลองวางเท้าให้ใกล้จุดใต้สะโพกมากขึ้น",
        "low": "หน้าแข้งตั้งชันเกินไป อาจก้าวสั้นเกินไปหรือลงเท้าใต้ลำตัวมากเกินไป ลองผ่อนก้าวให้เป็นธรรมชาติขึ้น",
        "reference": "Heiderscheit BC, et al. Effects of step rate manipulation on joint mechanics "
                     "during running. Med Sci Sports Exerc. 2011;43(2):296-302.",
    },
    "foot_angle": {
        "high": "ปลายเท้าชี้ขึ้นมาก (ส้นเท้ากระแทกก่อน) เพิ่มแรงกระแทกที่ส้นเท้าและหน้าแข้ง "
                "ลองฝึกลงเท้าด้วยกลางเท้าแทนส้นเท้า",
        "low": "ปลายเท้าชี้ลงมาก (ลงปลายเท้าจัด) เพิ่มภาระกล้ามเนื้อน่อง/เอ็นร้อยหวาย ลองลงเท้าให้แบนขึ้นเล็กน้อย",
        "reference": "Lieberman DE, et al. Foot strike patterns and collision forces in habitually "
                     "barefoot versus shod runners. Nature. 2010;463(7280):531-535.",
    },
}

# ⚠️ ค่านี้คุมแค่ตอนวิเคราะห์คลิปผู้ใช้ (ไม่กระทบข้อมูลเทรนซึ่งอยู่คนละไฟล์ คนละ pipeline)
# แต่ละเฟรมคำนวณฟีเจอร์อิสระจากกัน แล้วนำ P(ถูก) มาเฉลี่ยรวมเป็นคะแนนเดียว การลดจำนวนเฟรมที่สุ่ม
# จึงแค่ลดจำนวนตัวอย่างที่เฉลี่ย ไม่เปลี่ยนค่าฟีเจอร์ต่อเฟรมเลย — ทดสอบกับข้อมูลจริงทั้ง 21 คลิป
# (เทียบคำตัดสินที่ 10fps vs 5fps) แล้วพบว่าคำตัดสินไม่เปลี่ยนแม้แต่คลิปเดียว จึงลดเหลือ 5fps ได้
# ลดเวลาวิเคราะห์ลงประมาณครึ่งหนึ่ง (MediaPipe heavy คือคอขวดจริง ไม่ใช่การอ่าน/ย่อวิดีโอ)
SAMPLE_FPS = 5
MAX_DURATION_SEC = 60
MAX_FILE_SIZE_MB = 500  # วิดีโอต้นฉบับ (4K 60fps) อาจใหญ่ถึง ~330MB เผื่อไว้ให้พอ
# มั่นใจต่ำกว่านี้ -> เตือนผู้ใช้ว่าผลอาจไม่แม่น ดีกว่าตอบผิดอย่างมั่นใจ
# ค่านี้อยู่บนสเกลของ confidence_from_threshold() (0.5 = อยู่บนเส้นแบ่งพอดี, 1.0 = ห่างเส้นแบ่งสุด)
# ตั้งไว้ที่ 0.52 เพราะค่า P(ถูก) เฉลี่ยรายคลิปของข้อมูลจริงเกาะกลุ่มใกล้เส้นแบ่งมาก (0.35-0.47)
# ถ้าตั้งสูงกว่านี้จะขึ้นคำเตือนแทบทุกคลิปจนกลายเป็นสัญญาณรบกวนที่ผู้ใช้เลิกสนใจ
REJECT_CONFIDENCE_THRESHOLD = 0.52
MIN_USABLE_FRAMES = 10  # เฟรมที่ใช้ได้น้อยกว่านี้ถือว่าวิดีโอสั้น/คุณภาพต่ำเกินไป

LABEL_MAP = {1: "correct", 0: "incorrect"}

_model_bundle = None
_facing_reference = None
_score_config = None


def get_model_bundle():
    global _model_bundle
    if _model_bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"ไม่พบ {MODEL_PATH} — ต้องรัน 03_train_model.py ให้เสร็จก่อน")
        _model_bundle = joblib.load(MODEL_PATH)
    return _model_bundle


def get_facing_reference():
    global _facing_reference
    if _facing_reference is None:
        if not os.path.exists(FACING_REFERENCE_PATH):
            raise FileNotFoundError(
                f"ไม่พบ {FACING_REFERENCE_PATH} — ต้องรัน 02_prepare_dataset.py ให้เสร็จก่อน "
                "(ไฟล์นี้บอกว่าโมเดลเรียนรู้ทิศทางไหนไว้)"
            )
        with open(FACING_REFERENCE_PATH, encoding="utf-8") as f:
            _facing_reference = json.load(f)["majority_sign"]
    return _facing_reference


def get_score_config():
    """โหลด score_config.json (ช่วง 'ปกติของคนวิ่งถูก' ต่อมุม, calibrate จากข้อมูลจริงใน
    05_score_calibration.py) — ใช้แค่แสดง 'ข้อมูลประกอบ' ไม่ใช่คำนวณคะแนน (สูตรบวกคะแนนแบบ
    รายมุมทดสอบแล้วแยกกลุ่มถูก/ผิดไม่ได้จริง คะแนนหลักใช้ P(ถูก) จากโมเดลแทน)"""
    global _score_config
    if _score_config is None:
        if os.path.exists(SCORE_CONFIG_PATH):
            with open(SCORE_CONFIG_PATH, encoding="utf-8") as f:
                _score_config = json.load(f)
        else:
            _score_config = {}
    return _score_config


def build_angle_context(df, top_n=3):
    """เทียบมุมเฉลี่ยทั้งวิดีโอกับช่วง 'ปกติของคนวิ่งถูก' คืนรายการที่ต่างจากช่วงปกติมากสุด
    top_n รายการ — เป็นข้อมูลประกอบให้ผู้ใช้เข้าใจว่าโมเดลน่าจะสนใจจุดไหน ไม่ใช่คำตัดสินแยกต่างหาก"""
    specs = get_score_config()
    notes = []
    for col, spec in specs.items():
        if col not in df.columns:
            continue
        value = float(df[col].mean())
        lo, hi = spec["target_low"], spec["target_high"]
        mid = (lo + hi) / 2
        half_range = max((hi - lo) / 2, 1e-6)
        deviation = abs(value - mid) / half_range  # 0 = อยู่กลางช่วงปกติ, >1 = ออกนอกช่วงปกติแล้ว

        note = {
            "label": ANGLE_LABELS_TH.get(col, col),
            "value": round(value, 1),
            "typical_range": [round(lo, 1), round(hi, 1)],
            "deviation": round(deviation, 2),
        }

        advice_spec = ADVICE_LIBRARY.get(col)
        if advice_spec and deviation > 0.15:  # ต่างจากช่วงปกติพอสมควรแล้วค่อยให้คำแนะนำ กันสัญญาณรบกวนเล็กน้อย
            direction = "high" if value > mid else "low"
            note["advice"] = advice_spec[direction]
            note["reference"] = advice_spec["reference"]

        notes.append(note)

    notes.sort(key=lambda n: n["deviation"], reverse=True)
    return notes[:top_n]


def score_from_threshold(proba_1, threshold):
    """แปลง P(ถูก) เป็นคะแนน 0-100 โดยให้ 'เส้นแบ่งการตัดสินใจ' อยู่ที่ 50 คะแนนพอดี

    ทำไมต้องแปลง: โมเดลนี้ใช้ threshold = 0.43 ไม่ใช่ 0.5 (ปรับให้ recall สองคลาสสมดุลกัน)
    ถ้าเอา P(ถูก) x 100 มาแสดงตรง ๆ จะเกิดผลลัพธ์ที่ขัดแย้งกันเอง เช่น P=0.469 -> โมเดลตัดสินว่า
    'ท่าถูก' (เพราะ 0.469 >= 0.43) แต่หน้าจอแสดง 46.9/100 ซึ่งผู้ใช้อ่านว่า 'สอบตก'
    การ map ให้ threshold = 50 ทำให้ คะแนน >= 50 เท่ากับ 'ท่าถูก' เสมอ ไม่ขัดกันอีก
    (ลำดับคะแนนยังเรียงตาม P(ถูก) เหมือนเดิมทุกประการ เป็นการยืดสเกลแบบ monotonic เท่านั้น)
    """
    if proba_1 >= threshold:
        span = 1.0 - threshold
        return 50.0 + 50.0 * ((proba_1 - threshold) / span if span > 0 else 0.0)
    return 50.0 * (proba_1 / threshold if threshold > 0 else 0.0)


def confidence_from_threshold(proba_1, threshold, pred_class_int):
    """ความมั่นใจ 0.5-1.0 วัดจาก 'ระยะห่างจากเส้นแบ่ง' ไม่ใช่ระยะห่างจาก 0.5

    เหตุผลเดียวกับ score_from_threshold: สูตรเดิม (confidence = P ของคลาสที่ทาย) ทำให้คลิปที่
    ทายว่า 'ถูก' ได้ confidence สูงสุดแค่เท่ากับ P(ถูก) ซึ่งอยู่แถว 0.43-0.50 เสมอ
    ผลคือคลิปที่ทายว่า 'ถูก' ทุกคลิปจะติดธง low_confidence ตลอด ทั้งที่โมเดลตัดสินใจได้ชัดเจน
    สูตรใหม่ให้ค่า 0.5 พอดีตรงเส้นแบ่ง และเข้าใกล้ 1.0 เมื่อห่างจากเส้นแบ่งมากขึ้น
    """
    if pred_class_int == 1:
        span = 1.0 - threshold
        margin = (proba_1 - threshold) / span if span > 0 else 0.0
    else:
        margin = (threshold - proba_1) / threshold if threshold > 0 else 0.0
    return 0.5 + 0.5 * max(0.0, min(1.0, margin))


def validate_video(path, max_duration_sec=MAX_DURATION_SEC, max_size_mb=MAX_FILE_SIZE_MB):
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise ValueError(f"ไฟล์ใหญ่เกินไป ({size_mb:.0f}MB) จำกัดไว้ที่ {max_size_mb}MB")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("เปิดไฟล์วิดีโอไม่ได้ — ตรวจสอบว่าเป็นไฟล์ .mp4/.mov ที่ไม่เสีย")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = n_frames / fps if fps else 0
    cap.release()

    if duration > max_duration_sec:
        raise ValueError(f"วิดีโอยาวเกินไป ({duration:.0f} วินาที) จำกัดไว้ที่ {max_duration_sec} วินาที")
    if duration < 3:
        raise ValueError(f"วิดีโอสั้นเกินไป ({duration:.1f} วินาที) ต้องมีความยาวอย่างน้อย 3 วินาที")
    return duration


def extract_right_side_landmarks(video_path, progress_cb=None):
    """อ่านวิดีโอ -> letterbox เป็น 960x720 (เหมือน 00_resize_videos.py เป๊ะ) -> สุ่มเฟรมทุก
    1/SAMPLE_FPS วินาที (เหมือน 01_extract_landmarks.py) -> สกัด 6 จุดฝั่งขวาด้วย MediaPipe
    คืน DataFrame คอลัมน์ right_*_x/y (พิกเซลในเฟรม 960x720) ต่อเฟรมที่ตรวจเจอคน"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("เปิดวิดีโอไม่ได้")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n_frames_total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1
    frame_interval = max(1, round(src_fps / SAMPLE_FPS))

    detector = extract_mod.create_detector(POSE_MODEL_PATH)

    rows = []
    frame_idx_src = 0
    out_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx_src % frame_interval == 0:
            letterboxed = resize_mod.letterbox_resize(frame)
            timestamp_ms = int((frame_idx_src / src_fps) * 1000)

            rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                pose = result.pose_landmarks[0]
                h, w = letterboxed.shape[:2]
                row = {}
                for idx, name in extract_mod.RIGHT_SIDE_LANDMARKS.items():
                    lm = pose[idx]
                    row[f"{name}_x"] = lm.x * w
                    row[f"{name}_y"] = lm.y * h
                rows.append(row)

            out_idx += 1
            if progress_cb and n_frames_total:
                pct = 10 + int(50 * frame_idx_src / n_frames_total)  # ช่วง 10-60% ของ progress รวม
                progress_cb(min(pct, 60), f"กำลังตรวจจับท่าทาง ({out_idx} เฟรม)")

        frame_idx_src += 1

    cap.release()
    detector.close()
    return pd.DataFrame(rows)


def analyze_video(video_path, progress_cb=None):
    """ประมวลผลวิดีโอ 1 ไฟล์ครบวงจร: landmark -> normalize -> มุมข้อต่อ -> ทำนายคลาส
    progress_cb(percent:int, stage:str) ถูกเรียกเป็นระยะให้ฝั่งเว็บ poll สถานะได้"""

    def report(pct, stage):
        if progress_cb:
            progress_cb(pct, stage)

    report(2, "ตรวจสอบไฟล์วิดีโอ")
    duration = validate_video(video_path)

    bundle = get_model_bundle()
    facing_reference = get_facing_reference()

    report(5, "กำลังเปิดโมเดลตรวจจับท่าทาง (ขั้นตอนนี้ใช้เวลานานสุด)")
    raw_df = extract_right_side_landmarks(video_path, progress_cb=report)

    if len(raw_df) < MIN_USABLE_FRAMES:
        raise ValueError(
            f"ตรวจจับคนในวิดีโอได้แค่ {len(raw_df)} เฟรม (ต้องการอย่างน้อย {MIN_USABLE_FRAMES}) "
            "— ตรวจสอบว่าเห็นตัวเต็มตัว มุมข้าง แสงสว่างเพียงพอ"
        )

    report(65, "กำลังคำนวณตำแหน่งสัมพัทธ์และมุมข้อต่อ")
    df = prep_mod.normalize_row(raw_df)
    df = prep_mod.compute_angles(df)
    df = df.dropna(subset=prep_mod.FEATURE_COLUMNS)

    if len(df) < MIN_USABLE_FRAMES:
        raise ValueError("คุณภาพภาพต่ำเกินไป (คำนวณตำแหน่ง/มุมไม่ได้หลายเฟรม) ลองถ่ายใหม่ในที่แสงสว่างกว่านี้")

    # normalize ทิศทาง: วิดีโอนี้มีคลิปเดียว หาค่า median ของตัวเองไม่ได้ (ไม่มี 'ส่วนใหญ่' ให้เทียบ)
    # ต้องเทียบกับทิศทางอ้างอิงที่บันทึกไว้ตอนเทรนแทน (facing_reference.json)
    video_facing_sign = np.sign(df["ankle_x_norm"].mean())
    if video_facing_sign != 0 and video_facing_sign != facing_reference:
        for col in prep_mod.FACING_FLIP_COLUMNS:
            df[col] = df[col] * -1

    report(85, "กำลังทำนายคลาส")
    feature_cols = bundle["feature_columns"]
    X = df.reindex(columns=feature_cols).fillna(pd.Series(bundle["feature_medians"]))

    model = bundle["model"]
    threshold = bundle.get("decision_threshold", 0.5)
    proba = model.predict_proba(X)
    mean_proba_1 = float(proba[:, 1].mean())  # y เป็น {0,1} เสมอ -> คอลัมน์ 1 คือคลาส 1 (ถูก)

    pred_class_int = 1 if mean_proba_1 >= threshold else 0
    confidence = confidence_from_threshold(mean_proba_1, threshold, pred_class_int)

    report(95, "กำลังสรุปข้อมูลมุมข้อต่อประกอบ")
    result = {
        "duration_sec": round(duration, 1),
        "n_frames_used": len(df),
        "predicted_class": LABEL_MAP[pred_class_int],
        "score": round(score_from_threshold(mean_proba_1, threshold), 1),
        "confidence": round(confidence, 3),
        "class_probabilities": {
            "correct": round(mean_proba_1, 3),
            "incorrect": round(1 - mean_proba_1, 3),
        },
        "low_confidence": confidence < REJECT_CONFIDENCE_THRESHOLD,
        "angle_context": build_angle_context(df, top_n=6),  # ข้อมูลประกอบ+คำแนะนำ ไม่ใช่คะแนนแยก (ดู 05_score_calibration.py)
    }

    report(100, "เสร็จสิ้น")
    return result
