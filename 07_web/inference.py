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
OOD_REFERENCE_PATH = os.path.join(PARENT_DIR, "ood_reference.json")  # สร้างจาก 08_ood_reference.py

ANGLE_LABELS_TH = {
    "knee_angle": "มุมเข่า", "hip_angle": "มุมสะโพก", "ankle_angle": "มุมข้อเท้า",
    "thigh_angle": "มุมต้นขา", "shank_angle": "มุมหน้าแข้ง", "foot_angle": "มุมเท้า",
}
FEATURE_LABELS_TH = {
    **ANGLE_LABELS_TH,
    "trunk_lean": "การเอนลำตัว",
    "shoulder_x_norm": "ตำแหน่งไหล่ (หน้า-หลัง)", "shoulder_y_norm": "ความสูงไหล่",
    "knee_x_norm": "ตำแหน่งเข่า (หน้า-หลัง)", "knee_y_norm": "ความสูงเข่า",
    "ankle_x_norm": "ตำแหน่งข้อเท้า (หน้า-หลัง)", "ankle_y_norm": "ความสูงข้อเท้า",
    "heel_x_norm": "ตำแหน่งส้นเท้า (หน้า-หลัง)", "heel_y_norm": "ความสูงส้นเท้า",
    "foot_index_x_norm": "ตำแหน่งปลายเท้า (หน้า-หลัง)", "foot_index_y_norm": "ความสูงปลายเท้า",
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
SAMPLE_FPS = 10
MAX_DURATION_SEC = 60
# ⚠️ ตัวเลข 2 ค่านี้ต้องตรงกับที่เซิร์ฟเวอร์รับไหวจริง ห้ามตั้งเผื่อลอยๆ — เดิมตั้ง 500MB "เผื่อไว้ให้พอ"
# แต่ความจริงเซิร์ฟเวอร์ตายก่อนถึงครึ่งหนึ่งของค่านั้น (log จริงบน Railway: POST /api/analyze แล้วตามด้วย
# "Killed" ทันที ตอนอัปโหลดไฟล์ 4K แนวตั้ง 199MB) ผู้ใช้เห็นแค่ "เซิร์ฟเวอร์รีสตาร์ท" ทั้งที่หน้าเว็บ
# โฆษณาว่ารับ 500MB — วัด peak RSS จริงด้วยคลิปเดียวกัน (1,893 เฟรมเท่ากัน) ต่างกันแค่ความละเอียด:
#     960x720  (0.69 ล้านพิกเซล) -> 395MB   58 วิ
#     1080x1920 (2.07 ล้านพิกเซล) -> 444MB   23 วิ (ต่อ 12 วิแรกของคลิป)
#     2160x3840 (8.29 ล้านพิกเซล) -> 663MB  138 วิ  <- ตัวนี้ทำให้ container ตาย
# ตัวการคือ "ขนาดเฟรม" ไม่ใช่ขนาดไฟล์ (เฟรม 4K ดิบ = 25MB/เฟรม) และย่อฝั่งเซิร์ฟเวอร์ช่วยไม่ได้
# เพราะ OOM เกิดตอน cv2 ถอดรหัสเฟรม ซึ่งเกิดก่อนที่เราจะได้ย่อ จึงต้องปฏิเสธตั้งแต่ก่อนเริ่มแทน
# ไม่เสียความแม่นยำเลย เพราะไปป์ไลน์ letterbox ทุกคลิปเหลือ 960x720 อยู่แล้ว (ดู 00_resize_videos.py)
MAX_FRAME_PIXELS = 2_100_000  # ครอบคลุม Full HD ทั้ง 1920x1080 และ 1080x1920 พอดี
MAX_FILE_SIZE_MB = 200  # 1080p60 วัดได้ ~2.8MB/วิ x 60 วิ ≈ 168MB เผื่อเหลือนิดหน่อย

# ── ตรวจว่าเป็น "คลิปวิ่ง ถ่ายจากด้านข้าง" จริงไหม ────────────────────────────────────────────
# ทำไมต้องมี: โมเดลเรียนจากคลิปมุมข้างที่วิ่งบนลู่ล้วนๆ ถ้าป้อนคลิปอื่น (ภาพนิ่ง, คนยืนเฉยๆ, ถ่ายมุมหน้า)
# มันจะยังคายคะแนนออกมาได้ตามปกติ แต่เป็นตัวเลขที่ไม่มีความหมายเลย ผู้ใช้อ่านแล้วเข้าใจผิดว่าวิเคราะห์ได้จริง
#
# เกณฑ์ทุกค่าวัดจากคลิปจริงทั้ง 22 คลิปในชุดข้อมูล (ทุกคลิปเป็นมุมข้างและเป็นการวิ่งจริงทั้งหมด)
# เทียบกับคลิปทดสอบเชิงลบที่สร้างขึ้น (เอา 1 เฟรมมาวนซ้ำ = คนยืนนิ่ง):
#                          คลิปวิ่งจริง 22 คลิป      คนยืนนิ่ง      เกณฑ์ที่ตั้ง
#   ไหล่ซ้าย-ขวา/ลำตัว      0.046 - 0.169 (กลาง 0.112)     -          > 0.40 ปฏิเสธ
#   พิสัยมุมเข่า p95-p5      43 - 97 องศา                 0.00        < 15   ปฏิเสธ
#   สะโพกเด้ง/ขนาดตัว       0.036 - 0.081               0.0003       < 0.012 ปฏิเสธ
# ทุกเกณฑ์ห่างจากคลิปจริงที่ "แย่ที่สุด" ประมาณ 2.4-3 เท่า จึงแทบไม่มีโอกาสปฏิเสธคลิปที่ใช้ได้จริง
#
# ⚠️ ข้อจำกัดที่ต้องรู้: เกณฑ์มุมกล้องยังไม่ได้ทดสอบกับคลิปวิ่งมุมหน้าตรงจริง เพราะชุดข้อมูลนี้เป็นมุมข้างหมด
# อ้างอิงจากเรขาคณิตแทน — ถ่ายจากด้านข้างไหล่ซ้าย/ขวาทับกันเกือบสนิท (ค่าเข้าใกล้ 0) ส่วนถ่ายจากด้านหน้า
# ไหล่กางออกเต็มความกว้างจริงซึ่งใกล้เคียงความยาวลำตัวคน (ค่าราว 0.6-0.9) ถ้าภายหลังมีคลิปมุมหน้าจริง
# ควรวัดซ้ำเพื่อยืนยันค่า 0.40 อีกครั้ง
MAX_SIDE_VIEW_RATIO = 0.40
MIN_KNEE_RANGE_DEG = 15.0
MIN_HIP_OSCILLATION = 0.012
# ดัชนี landmark ของ MediaPipe Pose ฝั่งซ้าย (ฝั่งขวาที่โมเดลใช้อยู่ใน extract_mod.RIGHT_SIDE_LANDMARKS)
# ใช้เป็น "ตัวตรวจ" เท่านั้น ไม่ได้เพิ่มเข้าไปเป็นฟีเจอร์ของโมเดล (โมเดลอ่านเฉพาะ FEATURE_COLUMNS)
LEFT_SHOULDER_IDX, RIGHT_SHOULDER_IDX = 11, 12
LEFT_HIP_IDX, RIGHT_HIP_IDX = 23, 24
# มั่นใจต่ำกว่านี้ -> เตือนผู้ใช้ว่าผลอาจไม่แม่น ดีกว่าตอบผิดอย่างมั่นใจ
# ค่านี้อยู่บนสเกลของ confidence_from_threshold() (0.5 = อยู่บนเส้นแบ่งพอดี, 1.0 = ห่างเส้นแบ่งสุด)
# ตั้งไว้ที่ 0.52 เพราะค่า P(ถูก) เฉลี่ยรายคลิปของข้อมูลจริงเกาะกลุ่มใกล้เส้นแบ่งมาก (0.35-0.47)
# ถ้าตั้งสูงกว่านี้จะขึ้นคำเตือนแทบทุกคลิปจนกลายเป็นสัญญาณรบกวนที่ผู้ใช้เลิกสนใจ
REJECT_CONFIDENCE_THRESHOLD = 0.52
# ความยาวขั้นต่ำของช่วงวิ่งที่ใช้ได้ (10 เฟรม = 1 วินาที ที่ SAMPLE_FPS=10) — วัดจากข้อมูลจริงทั้ง 22 คลิป:
# สุ่มหน้าต่างต่อเนื่องจากแต่ละคลิปแล้วดูว่าคำตัดสินตรงกับทั้งคลิปกี่ %
#   1 วิ 75% (คลิปแย่สุด 44% ≈ โยนเหรียญ) · 3 วิ 85% · 5 วิ 89% · 8 วิ 91%
# เดิมรับแค่ 10 เฟรม (1 วิ ไม่ถึงรอบก้าวครบด้วยซ้ำ) ผลจึงขึ้นกับว่าบังเอิญจับได้จังหวะไหนของก้าว
MIN_USABLE_FRAMES = 30   # < 3 วินาที ปฏิเสธ ให้ถ่ายใหม่
SHORT_CLIP_FRAMES = 50   # < 5 วินาที วิเคราะห์ได้ แต่แจ้งว่าผลแกว่งได้ง่าย (ตรงกับเงื่อนไขการถ่ายบนหน้าเว็บ)

LABEL_MAP = {1: "correct", 0: "incorrect"}

_model_bundle = None
_facing_reference = None
_score_config = None
_ood_reference = None


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


# เทียบ "ค่าเฉลี่ยทั้งคลิป" ได้เฉพาะมุมที่ไม่วนรอบเท่านั้น
# - เข่า/สะโพก/ข้อเท้า (UNSIGNED_ANGLE_COLUMNS) คำนวณด้วย arccos -> ได้ 0-180° เสมอ เฉลี่ยตรงๆ ได้
# - ต้นขา/หน้าแข้ง/เท้า/การเอนลำตัว (SIGNED_ANGLE_COLUMNS) คำนวณด้วย arctan2 -> วนรอบที่ ±180°
#   ค่าเฉลี่ยเลขคณิตของมุมที่วนรอบ "ผิดทางคณิตศาสตร์": -179° กับ +179° แปลว่าเกือบชี้ทางเดียวกัน
#   แต่เฉลี่ยแล้วได้ 0° (ตรงข้ามกันพอดี) ตรวจกับข้อมูลจริง 22 คลิป: ค่าเฉลี่ยเพี้ยนเกิน 5° ถึง 41 จาก 88
#   กรณี เพี้ยนสูงสุด 158° และทำให้คลิป Running Analysis ขึ้นว่า "มุมเท้า 36.8° นอกช่วงปกติ" ทั้งที่
#   ค่าเฉลี่ยเชิงมุมจริงคือ 147° ซึ่งอยู่ในช่วงปกติ -> ให้คำแนะนำผิดมาตลอด
# แก้แบบตรงหลักการที่สุดโดยไม่แตะโมเดล/ไม่แตะ score_config.json คือไม่เอามุมที่วนรอบมาแสดงผลเลย
# (ถ้าจะเอามุมเท้ากลับมา ต้องคำนวณช่วงปกติใหม่ด้วยสถิติเชิงวงกลมใน 05_score_calibration.py ก่อน)
CONTEXT_ANGLE_COLUMNS = ["knee_angle", "hip_angle", "ankle_angle"]


def build_angle_context(df, top_n=3):
    """เทียบมุมเฉลี่ยทั้งวิดีโอกับช่วง 'ปกติของคนวิ่งถูก' คืนรายการที่ต่างจากช่วงปกติมากสุด
    top_n รายการ — เป็นข้อมูลประกอบให้ผู้ใช้เข้าใจว่าโมเดลน่าจะสนใจจุดไหน ไม่ใช่คำตัดสินแยกต่างหาก"""
    specs = get_score_config()
    notes = []
    for col, spec in specs.items():
        if col not in df.columns or col not in CONTEXT_ANGLE_COLUMNS:
            continue
        value = float(df[col].mean())
        lo, hi = spec["target_low"], spec["target_high"]
        mid = (lo + hi) / 2
        half_range = max((hi - lo) / 2, 1e-6)
        deviation = abs(value - mid) / half_range  # 0 = อยู่กลางช่วงปกติ, >1 = ออกนอกช่วงปกติแล้ว

        note = {
            "key": col,
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


# มุมที่ส่งเป็นกราฟตามเวลา — เฉพาะมุมที่ช่วงปกติแคบพอจะอ่านความหมายได้ (ต้นขา/หน้าแข้ง/เท้า มีช่วงปกติ
# กว้างเกินครึ่งวงกลม แรเงาแล้วครอบทั้งกราฟ ไม่ได้ช่วยให้ผู้ใช้อ่านง่ายขึ้น)
TIMELINE_ANGLES = ["knee_angle", "hip_angle", "ankle_angle"]


def build_timeline(df, frame_size):
    """ข้อมูลรายเฟรมสำหรับหน้าเว็บ: เวลา, พิกัด 6 จุดบนภาพต้นฉบับ (สัดส่วน 0-1) และมุมข้อต่อ
    ใช้วาดโครงร่างทับวิดีโอและกราฟตามเวลา — เป็นข้อมูลแสดงผลเท่านั้น ไม่มีผลต่อคำตัดสินของโมเดล"""
    names = list(extract_mod.RIGHT_SIDE_LANDMARKS.values())
    specs = get_score_config()
    points = np.stack(
        [np.stack([df[f"ov_{n}_x"].to_numpy(), df[f"ov_{n}_y"].to_numpy()], axis=1) for n in names], axis=1
    )
    angles = [c for c in TIMELINE_ANGLES if c in df.columns]
    return {
        "t": [round(float(v), 2) for v in df["t_sec"]],
        "points": np.round(points, 4).tolist(),
        "point_names": names,
        "angles": {c: [round(float(v), 1) for v in df[c]] for c in angles},
        "ranges": {c: [round(specs[c]["target_low"], 1), round(specs[c]["target_high"], 1)]
                   for c in angles if c in specs},
        "labels": {c: ANGLE_LABELS_TH.get(c, c) for c in angles},
        "frame_size": list(frame_size) if frame_size else None,
    }


def get_ood_reference():
    global _ood_reference
    if _ood_reference is None:
        if os.path.exists(OOD_REFERENCE_PATH):
            with open(OOD_REFERENCE_PATH, encoding="utf-8") as f:
                _ood_reference = json.load(f)
        else:
            _ood_reference = {}
    return _ood_reference


def build_ood_check(df):
    """คลิปนี้อยู่ห่างจากกลุ่มคลิปที่โมเดลเคยเห็นตอนเทรนแค่ไหน (ดูเหตุผล/วิธีเลือกเส้นใน 08_ood_reference.py)
    ไม่เปลี่ยนคำตัดสินหรือคะแนนของโมเดล — แค่บอกหน้าเว็บว่าผลนี้อยู่นอกขอบเขตข้อมูลที่โมเดลเรียนรู้"""
    ref = get_ood_reference()
    feats = [c for c in ref.get("features", []) if c in df.columns and ref["clip_sd"].get(c)]
    if not feats:
        return None
    z = {c: (float(df[c].mean()) - ref["clip_mean"][c]) / ref["clip_sd"][c] for c in feats}
    rms = float(np.sqrt(np.mean([v * v for v in z.values()])))
    top = sorted(z, key=lambda c: abs(z[c]), reverse=True)[:3]
    return {
        "flag": rms >= ref["rms_z_threshold"],
        "rms_z": round(rms, 2),
        "threshold": ref["rms_z_threshold"],
        "n_reference_clips": ref.get("n_reference_clips"),
        "top": [{"key": c, "label": FEATURE_LABELS_TH.get(c, c), "z": round(z[c], 1)} for c in top],
    }


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


def _side_view_ratio(pose, width, height):
    """ระยะห่างไหล่ซ้าย-ขวาในแนวนอน หารด้วยความยาวลำตัว (ไหล่กลาง -> สะโพกกลาง)
    ถ่ายจากด้านข้าง ไหล่สองข้างทับกันเกือบสนิทเมื่อมองจากกล้อง -> ค่าเข้าใกล้ 0
    ถ่ายจากด้านหน้า/หลัง ไหล่กางออกเต็มความกว้างจริง -> ค่าสูง
    หารด้วยความยาวลำตัวเพื่อให้ไม่ขึ้นกับว่าคนอยู่ใกล้หรือไกลกล้อง"""
    l_sh, r_sh = pose[LEFT_SHOULDER_IDX], pose[RIGHT_SHOULDER_IDX]
    l_hip, r_hip = pose[LEFT_HIP_IDX], pose[RIGHT_HIP_IDX]
    sh_mid_x, sh_mid_y = (l_sh.x + r_sh.x) / 2 * width, (l_sh.y + r_sh.y) / 2 * height
    hip_mid_x, hip_mid_y = (l_hip.x + r_hip.x) / 2 * width, (l_hip.y + r_hip.y) / 2 * height
    torso = float(np.hypot(sh_mid_x - hip_mid_x, sh_mid_y - hip_mid_y))
    if torso < 1:
        return float("nan")  # ลำตัวสั้นผิดปกติ (ตรวจจับเพี้ยน) ปล่อยผ่าน ให้ตัวกรอง visibility จัดการแทน
    return abs(l_sh.x - r_sh.x) * width / torso


def validate_clip_content(df):
    """ตรวจว่าคลิปที่อัปโหลดมาเป็น "การวิ่ง" ถ่าย "จากด้านข้าง" จริงไหม (ที่มาของเกณฑ์อยู่ที่ค่าคงที่ด้านบน)
    ต้องทำหลังสกัด landmark เสร็จ เพราะตัดสินจากการเคลื่อนไหวตลอดคลิป ดูจากตัวไฟล์วิดีโออย่างเดียวไม่ได้"""
    if "sv_ratio" in df.columns:
        sv = df["sv_ratio"].median()
        if pd.notna(sv) and sv > MAX_SIDE_VIEW_RATIO:
            raise ValueError(
                "คลิปนี้ดูเหมือนไม่ได้ถ่ายจากด้านข้าง (เห็นไหล่ทั้งสองข้างกางออก แปลว่ากล้องอยู่ด้านหน้า "
                "หรือด้านหลังผู้วิ่ง) ระบบวิเคราะห์ได้เฉพาะคลิปที่ถ่ายจากด้านข้างและเห็นฝั่งขวาของผู้วิ่งชัดเจน "
                "— ตั้งกล้องไว้ข้างลู่วิ่งให้เห็นผู้วิ่งจากด้านข้างเต็มตัว แล้วถ่ายใหม่"
            )

    knee_range = float(df["knee_angle"].quantile(0.95) - df["knee_angle"].quantile(0.05))
    scale = float(df["body_scale_px"].median())
    hip_osc = float(df["right_hip_y"].std() / scale) if scale > 0 else 0.0
    # ใช้เงื่อนไข "และ" (ต้องนิ่งทั้งสองตัวชี้วัด) เพื่อกันการปฏิเสธคลิปจริงผิดพลาดให้มากที่สุด
    if knee_range < MIN_KNEE_RANGE_DEG and hip_osc < MIN_HIP_OSCILLATION:
        raise ValueError(
            "ไม่พบการเคลื่อนไหวแบบการวิ่งในคลิปนี้ (มุมเข่าและระดับสะโพกแทบไม่เปลี่ยนเลยตลอดคลิป) "
            "— ตรวจสอบว่าเป็นคลิปที่กำลังวิ่งอยู่จริง ไม่ใช่ภาพนิ่ง คนยืนอยู่กับที่ หรือคลิปที่ผู้วิ่ง"
            "อยู่ไกลจนระบบจับการเคลื่อนไหวไม่ได้"
        )


def validate_video(path, max_duration_sec=MAX_DURATION_SEC, max_size_mb=MAX_FILE_SIZE_MB):
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise ValueError(f"ไฟล์ใหญ่เกินไป ({size_mb:.0f}MB) จำกัดไว้ที่ {max_size_mb}MB")

    cap = open_video(path)  # จำกัด thread ตั้งแต่ตอนเปิดครั้งแรก (ดู DECODE_THREADS ด้านล่าง)
    if not cap.isOpened():
        raise ValueError("เปิดไฟล์วิดีโอไม่ได้ — ตรวจสอบว่าเป็นไฟล์ .mp4/.mov ที่ไม่เสีย")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = n_frames / fps if fps else 0
    cap.release()

    # ต้องตรวจความละเอียดด้วย ไม่ใช่แค่ขนาดไฟล์ — เฟรมใหญ่คือสาเหตุจริงที่ทำให้เซิร์ฟเวอร์ถูก OOM kill
    # (ดูตัวเลขที่วัดไว้ตรง MAX_FRAME_PIXELS) ปฏิเสธตรงนี้ใช้เวลาไม่ถึงวินาที เพราะยังไม่ได้โหลด MediaPipe
    if width * height > MAX_FRAME_PIXELS:
        raise ValueError(
            f"วิดีโอความละเอียดสูงเกินไป ({width}x{height}) เซิร์ฟเวอร์รับไหวถึงระดับ Full HD "
            "(1920x1080) เท่านั้น — ตั้งกล้องในมือถือเป็น 1080p แล้วถ่ายใหม่ หรือย่อไฟล์ก่อนอัปโหลด "
            "ผลวิเคราะห์จะไม่ต่างกัน เพราะระบบย่อทุกคลิปเหลือ 960x720 ก่อนวิเคราะห์อยู่แล้ว"
        )

    if duration > max_duration_sec:
        raise ValueError(f"วิดีโอยาวเกินไป ({duration:.0f} วินาที) จำกัดไว้ที่ {max_duration_sec} วินาที")
    if duration < 3:
        raise ValueError(f"วิดีโอสั้นเกินไป ({duration:.1f} วินาที) ต้องมีความยาวอย่างน้อย 3 วินาที")
    return duration


# จำนวน thread ของตัวถอดรหัสวิดีโอ (FFmpeg) — ห้ามปล่อยเป็นค่าเริ่มต้น
# ค่าเริ่มต้น FFmpeg เปิด thread เท่าจำนวนคอร์ที่ "มองเห็น" และแต่ละ thread จองบัฟเฟอร์เฟรมของตัวเอง
# วัดจริงกับคลิป 4K HEVC (baikaw.MOV): 16 thread = decoder ใช้แรม 938MB, 2 thread = 279MB
# บน container มักมองเห็นคอร์ของเครื่อง host ทั้งเครื่อง (มากกว่า vCPU ที่ได้จริง) จึงยิ่งอันตราย
# — เคยทำให้เซิร์ฟเวอร์บน Railway (เพดานแรม 1GB) ถูก OOM killer ฆ่าทิ้งกลางการวิเคราะห์
# จำนวน thread ไม่เปลี่ยนพิกเซลที่ถอดรหัสได้ ผลลัพธ์จึงเหมือนเดิมเป๊ะ (ไม่เกิด train-serve skew)
DECODE_THREADS = int(os.environ.get("DECODE_THREADS", "2"))


def open_video(path):
    """เปิดวิดีโอโดยจำกัด thread ของตัวถอดรหัส (ดูเหตุผลที่ DECODE_THREADS ด้านบน)"""
    return cv2.VideoCapture(path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, DECODE_THREADS])


def _letterbox_geometry(fw, fh, target_w=resize_mod.TARGET_W, target_h=resize_mod.TARGET_H):
    """ตำแหน่งของภาพต้นฉบับภายในกรอบ letterbox — สูตรต้องตรงกับ 00_resize_videos.letterbox_resize เป๊ะ
    คืน (กว้าง, สูง, x_offset, y_offset, new_w, new_h) ใช้แปลงพิกัดโครงร่างกลับไปวางทับวิดีโอต้นฉบับ"""
    scale = min(target_w / fw, target_h / fh)
    new_w, new_h = round(fw * scale), round(fh * scale)
    return fw, fh, (target_w - new_w) // 2, (target_h - new_h) // 2, new_w, new_h


def extract_right_side_landmarks(video_path, progress_cb=None):
    """อ่านวิดีโอ -> letterbox เป็น 960x720 (เหมือน 00_resize_videos.py เป๊ะ) -> สุ่มเฟรมทุก
    1/SAMPLE_FPS วินาที (เหมือน 01_extract_landmarks.py) -> สกัด 6 จุดฝั่งขวาด้วย MediaPipe
    คืน DataFrame คอลัมน์ right_*_x/y (พิกเซลในเฟรม 960x720) ต่อเฟรมที่ตรวจเจอคน"""
    cap = open_video(video_path)
    if not cap.isOpened():
        raise ValueError("เปิดวิดีโอไม่ได้")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n_frames_total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1
    frame_interval = max(1, round(src_fps / SAMPLE_FPS))

    detector = extract_mod.create_detector(POSE_MODEL_PATH)

    rows = []
    frame_idx_src = 0
    out_idx = 0
    geo = None

    while True:
        # เฟรมที่ไม่ได้สุ่มใช้ ให้ grab() เฉยๆ ไม่ต้อง retrieve() — grab() เลื่อนตำแหน่งไปเฟรมถัดไป
        # โดยไม่ถอดรหัสเป็นภาพและไม่จองอาร์เรย์ใหม่ ส่วน cap.read() = grab() + retrieve() จึงได้
        # "เฟรมเดียวกันเป๊ะ" กับโค้ดเดิมทุกประการ (ไม่กระทบผลลัพธ์/ไม่เกิด train-serve skew)
        #
        # สำคัญกับเซิร์ฟเวอร์ที่แรมจำกัด: คลิป 4K 60fps มี ~1,870 เฟรม แต่ใช้จริงแค่ ~310 เฟรม
        # โค้ดเดิม read() ทุกเฟรมจึงจองอาร์เรย์เฟรมละ ~25MB รวมกว่า 1,500 ครั้งโดยเปล่าประโยชน์
        # จนถูก OOM killer ฆ่าทิ้งบน Railway (เพดาน 1GB) — เจอมาแล้วตอน deploy จริง
        if frame_idx_src % frame_interval != 0:
            if not cap.grab():
                break
            frame_idx_src += 1
            continue

        ret, frame = cap.read()
        if not ret:
            break

        letterboxed = resize_mod.letterbox_resize(frame)
        timestamp_ms = int((frame_idx_src / src_fps) * 1000)
        if geo is None:
            geo = _letterbox_geometry(frame.shape[1], frame.shape[0])
        del frame  # คืนอาร์เรย์เฟรมต้นฉบับ (4K = ~25MB) ทันที ไม่ต้องรอถึงรอบถัดไป

        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect_for_video(mp_image, timestamp_ms)

        h, w = letterboxed.shape[:2]
        # ต้องเรียก select_largest_person() เหมือนกับ 01_extract_landmarks.py เป๊ะ (ข้อควรระวัง #13
        # ข้างบน) — เลือกคนที่กรอบครอบ 6 จุดใหญ่สุด แทนเชื่อ pose_landmarks[0] ตรงๆ ซึ่งเคยทำให้
        # โมเดลหลงจับคนพื้นหลังที่เดิน/ยืนอยู่ไกลกว่าแทนนักวิ่งหลักบนลู่วิ่ง
        pose = extract_mod.select_largest_person(result.pose_landmarks, w, h)
        if pose is not None:
            row = {"t_sec": timestamp_ms / 1000.0}
            row["sv_ratio"] = _side_view_ratio(pose, w, h)  # ตัวตรวจมุมกล้อง ไม่ใช่ฟีเจอร์ของโมเดล
            _, _, x_off, y_off, new_w, new_h = geo
            for idx, name in extract_mod.RIGHT_SIDE_LANDMARKS.items():
                lm = pose[idx]
                row[f"{name}_x"] = lm.x * w
                row[f"{name}_y"] = lm.y * h
                row[f"{name}_v"] = lm.visibility  # ใช้กรองเฟรมไม่มั่นใจแบบเดียวกับตอนเทรน (ดู analyze_video)
                # พิกัดสัดส่วน 0-1 บนภาพต้นฉบับ (ถอดแถบดำ letterbox ออก) ให้หน้าเว็บวาดโครงร่างทับวิดีโอได้ตรง
                # ไม่ใช่ feature ของโมเดล — โมเดลอ่านเฉพาะคอลัมน์ใน FEATURE_COLUMNS ผลทำนายจึงไม่เปลี่ยน
                row[f"ov_{name}_x"] = (lm.x * w - x_off) / new_w
                row[f"ov_{name}_y"] = (lm.y * h - y_off) / new_h
            rows.append(row)

        out_idx += 1
        if progress_cb and n_frames_total:
            pct = 10 + int(50 * frame_idx_src / n_frames_total)  # ช่วง 10-60% ของ progress รวม
            progress_cb(min(pct, 60), f"กำลังตรวจจับท่าทาง ({out_idx} เฟรม)")

        frame_idx_src += 1

    cap.release()
    detector.close()
    out = pd.DataFrame(rows)
    out.attrs["frame_size"] = (geo[0], geo[1]) if geo else None
    return out


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
    frame_size = raw_df.attrs.get("frame_size")

    # ทิ้งเฟรมที่ MediaPipe ไม่มั่นใจในจุดใดจุดหนึ่ง — กฎเดียวกับ 02_prepare_dataset.load_and_clean เป๊ะ
    # เดิมเว็บไม่ได้กรองเลย (train/serve skew): โมเดลเทรน/วัดผลบนเฟรมที่ผ่านตัวกรองนี้ แต่เว็บเอาทุกเฟรม
    # มาเฉลี่ย เช่น Tyes เว็บใช้ 320 เฟรม ขณะที่ตอนเทรนเหลือ 202 (อีก 118 เฟรมคือเฟรมที่หลงจับขาผิดข้าง
    # เพราะเส้นวิเคราะห์ที่วาดทับภาพ) ไม่ได้แตะโมเดล แค่ทำให้ข้อมูลที่ป้อนเข้าโมเดลตรงกับตอนเทรน
    if len(raw_df):
        vis_min = raw_df[prep_mod.VISIBILITY_COLUMNS].min(axis=1)
        raw_df = raw_df[vis_min >= prep_mod.MIN_VISIBILITY]

    if len(raw_df) < MIN_USABLE_FRAMES:
        # แยกเป็นตัวแปรแล้วต่อด้วย + เพราะ Python ต่อสตริงอัตโนมัติได้เฉพาะระหว่าง literal เท่านั้น
        # (เอานิพจน์เงื่อนไขในวงเล็บไปวางชนกับ f-string ข้างหลังตรงๆ จะเป็น SyntaxError)
        lead = ("ไม่พบคนในคลิปนี้เลย — ตรวจสอบว่าเป็นคลิปที่มีคนกำลังวิ่งอยู่จริง และเห็นตัวเต็มตัว "
                if len(raw_df) == 0 else
                f"ตรวจจับท่าวิ่งได้ชัดเจนแค่ประมาณ {len(raw_df) / SAMPLE_FPS:.1f} วินาที ")
        raise ValueError(
            lead
            + f"(ต้องการอย่างน้อย {MIN_USABLE_FRAMES / SAMPLE_FPS:.0f} วินาทีเพื่อให้ผลเชื่อถือได้) "
            + "— ถ่ายให้ยาวขึ้น เห็นเต็มตัว มุมข้าง แสงสว่างเพียงพอ"
        )

    report(65, "กำลังคำนวณตำแหน่งสัมพัทธ์และมุมข้อต่อ")
    df = prep_mod.normalize_row(raw_df)
    df = prep_mod.compute_angles(df)
    df = df.dropna(subset=prep_mod.FEATURE_COLUMNS)

    if len(df) < MIN_USABLE_FRAMES:
        raise ValueError("คุณภาพภาพต่ำเกินไป (คำนวณตำแหน่ง/มุมไม่ได้หลายเฟรม) ลองถ่ายใหม่ในที่แสงสว่างกว่านี้")

    # กันคลิปที่ไม่ใช่ "การวิ่งถ่ายจากด้านข้าง" ไม่ให้หลุดไปถึงโมเดล (โมเดลจะคายคะแนนที่ไม่มีความหมายออกมา)
    validate_clip_content(df)

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
        "short_clip": len(df) < SHORT_CLIP_FRAMES,  # สั้นกว่า 5 วิ -> หน้าเว็บแจ้งว่าผลแกว่งได้ง่าย
        "out_of_distribution": build_ood_check(df),  # คลิปต่างจากข้อมูลเทรนมากไหม (ไม่กระทบคำตัดสิน)
        "timeline": build_timeline(df, frame_size),  # โครงร่างทับวิดีโอ + กราฟตามเวลา (แสดงผลเท่านั้น)
    }

    report(100, "เสร็จสิ้น")
    return result
