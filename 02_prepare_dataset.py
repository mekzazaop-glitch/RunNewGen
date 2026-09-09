"""
ขั้นที่ 2: เตรียมข้อมูลจาก landmarks_raw.csv -> normalize พิกัด -> แบ่ง train/val/test ตามคลิป

⚠️ ปรับปรุงจากรอบก่อน (ตามคำแนะนำ): เดิมใช้พิกัดพิกเซลดิบตรงๆ ทำให้โมเดลพึ่งพา 'ตำแหน่งสะโพก
ในเฟรม' เป็นหลัก (เห็นจาก permutation importance) ซึ่งคือตำแหน่ง/ระยะห่างจากกล้อง ไม่ใช่ท่าทาง
จริง — เป็นสาเหตุที่คน 'yes' (กล้องคนละมุม) ถูกทำนายผิดทุกเฟรม

แก้ด้วยการ normalize พิกัดเป็น 'ตำแหน่งสัมพัทธ์กับสะโพก หารด้วยระยะไหล่-สะโพก':
  1. ย้ายจุดกำเนิดไปที่สะโพก (translation invariant — ไม่สนใจว่าสะโพกอยู่ตรงไหนของเฟรม)
  2. หารด้วยระยะไหล่-สะโพก (scale invariant — ไม่สนใจว่าคนอยู่ใกล้/ไกลกล้องแค่ไหน)
ยังคงใช้ 6 จุดเดิมตาม format ที่กำหนด ไม่ได้เพิ่มจุดใหม่ แค่เปลี่ยนวิธีคำนวณค่าที่ป้อนเข้าโมเดล
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

RANDOM_STATE = 42

RAW_COLUMNS = [
    "right_shoulder_x", "right_shoulder_y",
    "right_hip_x", "right_hip_y",
    "right_knee_x", "right_knee_y",
    "right_ankle_x", "right_ankle_y",
    "right_heel_x", "right_heel_y",
    "right_foot_index_x", "right_foot_index_y",
]

VISIBILITY_COLUMNS = [
    "right_shoulder_v", "right_hip_v", "right_knee_v",
    "right_ankle_v", "right_heel_v", "right_foot_index_v",
]
# บางคลิปต้นแบบ (Tyes, Running Analysis) เป็นวิดีโอที่มีเส้น/โครงกระดูกวิเคราะห์ท่าวิ่งจาก
# ซอฟต์แวร์อื่นวาดทับตัวคนไว้อยู่แล้ว (ไม่ใช่ภาพดิบ) ตรวจสอบด้วยตาพบว่า MediaPipe ยังตรวจจับ
# ร่างกายจริงได้ถูกต้องเกือบทั้งหมด (visibility เฉลี่ย 0.94-1.00) แต่พบเฟรมที่มันหลงไปจับขาผิดข้าง
# ในคลิป Tyes ตรงกับที่ visibility ของจุดนั้นตกลงต่ำผิดปกติ (0.73 เทียบเพื่อนบ้าน ~1.0) จึงกรอง
# เฟรมที่ MediaPipe เองไม่มั่นใจออก แทนที่จะพยายามลบเส้นในภาพ (เสี่ยงกว่าและไม่แม่นเท่า)
MIN_VISIBILITY = 0.5

# จุดที่ normalize แล้วใช้เป็น feature จริง (ตัดสะโพกออกเพราะกลายเป็น (0,0) คงที่หลัง normalize
# ไม่มีข้อมูลอะไรเพิ่ม — เหลือจุดที่ยังมีความหมาย 5 จุด x (x,y) = 10 ฟีเจอร์)
NORM_POINTS = ["shoulder", "knee", "ankle", "heel", "foot_index"]
POINT_FEATURE_COLUMNS = [f"{p}_{axis}_norm" for p in NORM_POINTS for axis in ("x", "y")]

# มุมข้อต่อ — คำนวณเพิ่มจาก 6 จุดเดิมที่มีอยู่แล้ว (ไม่ต้องถ่าย/สกัดวิดีโอเพิ่ม) สัมพันธ์กับ
# 'ท่าถูก/ผิด' ตรงกว่าตำแหน่งพิกัดดิบ (ตามคำแนะนำที่แนะนำไว้ — เพิ่มความแม่นยำโดยไม่ต้องมีข้อมูลใหม่)
UNSIGNED_ANGLE_COLUMNS = ["knee_angle", "hip_angle", "ankle_angle"]  # มุมงอข้อต่อ ไม่ขึ้นกับทิศที่หันหน้า
SIGNED_ANGLE_COLUMNS = ["trunk_lean", "thigh_angle", "shank_angle", "foot_angle"]  # ขึ้นกับทิศที่หันหน้า ต้อง normalize ทิศด้วย
ANGLE_FEATURE_COLUMNS = UNSIGNED_ANGLE_COLUMNS + SIGNED_ANGLE_COLUMNS

FEATURE_COLUMNS = POINT_FEATURE_COLUMNS + ANGLE_FEATURE_COLUMNS


def _angle_3pt(ax, ay, bx, by, cx, cy):
    """มุมที่จุด b ระหว่างเวกเตอร์ b->a และ b->c (องศา 0-180) invariant ต่อทิศทาง/สเกล/ตำแหน่ง"""
    v1x, v1y = ax - bx, ay - by
    v2x, v2y = cx - bx, cy - by
    dot = v1x * v2x + v1y * v2y
    norm = np.sqrt(v1x ** 2 + v1y ** 2) * np.sqrt(v2x ** 2 + v2y ** 2)
    cos_angle = (dot / norm.replace(0, np.nan)).clip(-1, 1)
    return np.degrees(np.arccos(cos_angle))


def _angle_from_vertical(dx, dy):
    """มุมของเวกเตอร์ (dx,dy) เทียบแนวดิ่ง มีเครื่องหมาย (y ในภาพชี้ลง จึงใช้ -dy ให้ 0 องศา=ชี้ขึ้น)
    ยังไม่ normalize ทิศทาง ต้องคูณ facing_sign เพิ่มทีหลัง (ทำใน normalize_facing_direction)"""
    return np.degrees(np.arctan2(dx, -dy))


def compute_angles(df):
    """คำนวณมุมข้อต่อจากพิกัดพิกเซลดิบ 6 จุด — ใช้พิกัดดิบ (ไม่ใช่ normalized) เพราะมุมเป็นค่าที่
    ไม่ขึ้นกับสเกล/ตำแหน่งอยู่แล้วโดยธรรมชาติ ไม่ต้อง normalize ซ้ำ"""
    p = {name: (df[f"right_{name}_x"], df[f"right_{name}_y"]) for name in
         ("shoulder", "hip", "knee", "ankle", "heel", "foot_index")}

    out = df.copy()
    out["knee_angle"] = _angle_3pt(*p["hip"], *p["knee"], *p["ankle"])
    out["hip_angle"] = _angle_3pt(*p["shoulder"], *p["hip"], *p["knee"])
    out["ankle_angle"] = _angle_3pt(*p["knee"], *p["ankle"], *p["foot_index"])

    # มุมเอียงเทียบแนวดิ่ง/แนวนอน — มีเครื่องหมาย (ยังไม่ normalize ทิศ)
    out["trunk_lean"] = _angle_from_vertical(p["shoulder"][0] - p["hip"][0], p["shoulder"][1] - p["hip"][1])
    out["thigh_angle"] = _angle_from_vertical(p["knee"][0] - p["hip"][0], p["knee"][1] - p["hip"][1])
    out["shank_angle"] = _angle_from_vertical(p["ankle"][0] - p["knee"][0], p["ankle"][1] - p["knee"][1])
    out["foot_angle"] = _angle_from_vertical(p["foot_index"][0] - p["heel"][0], p["foot_index"][1] - p["heel"][1])
    return out


def normalize_row(df):
    """แปลงพิกัดพิกเซลดิบ -> พิกัดสัมพัทธ์กับสะโพก หารด้วยระยะไหล่-สะโพก (เวกเตอร์ไรซ์ทั้ง DataFrame)"""
    hip_x, hip_y = df["right_hip_x"], df["right_hip_y"]
    shoulder_x, shoulder_y = df["right_shoulder_x"], df["right_shoulder_y"]

    scale = np.sqrt((shoulder_x - hip_x) ** 2 + (shoulder_y - hip_y) ** 2)
    scale = scale.replace(0, np.nan)  # กันหารด้วยศูนย์ (เฟรมที่ไหล่กับสะโพกทับกันพอดี ไม่ควรเกิดขึ้นจริง)

    out = df.copy()
    for point in NORM_POINTS:
        out[f"{point}_x_norm"] = (df[f"right_{point}_x"] - hip_x) / scale
        out[f"{point}_y_norm"] = (df[f"right_{point}_y"] - hip_y) / scale

    out["body_scale_px"] = scale  # เก็บไว้ดูคุณภาพข้อมูล ไม่ใช้เป็น feature (จะทำให้กลับไม่ invariant)
    return out


# คอลัมน์ที่ต้องพลิกเครื่องหมายพร้อมกันเวลาแก้ทิศทาง (x_norm ทุกจุด + มุมมีเครื่องหมาย)
FACING_FLIP_COLUMNS = [c for c in POINT_FEATURE_COLUMNS if c.endswith("_x_norm")] + SIGNED_ANGLE_COLUMNS
FACING_REFERENCE_PATH = "facing_reference.json"


def normalize_facing_direction(df, save_reference=True):
    """แก้ปัญหาคนหันหน้าคนละทิศ (พบว่า 20 คลิปหันทิศหนึ่ง มีแค่ Tyes หันทิศตรงข้าม — ดูจาก
    ankle_x_norm เฉลี่ยทั้งคลิป: 20 คลิปติดลบหมด มี Tyes ตัวเดียวเป็นบวก) ทำให้โมเดลทำนาย Tyes
    ผิดทุกเฟรม (LOSO fold Tyes = 0.000) เพราะพิกัด x ทุกจุดมีเครื่องหมายกลับด้านกับที่เรียนรู้มา

    วิธีแก้: หาทิศทางหลักที่ 'ส่วนใหญ่' หันจากค่าเฉลี่ย ankle_x_norm ทั้งชุด แล้วพลิกเครื่องหมาย
    แกน x (คูณ -1) ให้เฉพาะคลิปที่หันสวนทาง — ทำให้ทุกคลิปมีทิศทางเดียวกันหมดก่อนเข้าโมเดล

    บันทึกทิศทางอ้างอิง (majority_sign) ลงไฟล์ facing_reference.json ด้วย — ตอนเว็บ (07_web)
    ได้รับวิดีโอใหม่ 1 คลิป ต้องรู้ว่า 'ทิศไหนคือทิศที่โมเดลเรียนมา' ถึงจะพลิกให้ตรงกันได้ถูก
    (วิดีโอเดี่ยวหาค่า median ของตัวเองไม่ได้ เพราะมีแค่คลิปเดียว ไม่มี 'ส่วนใหญ่' ให้เทียบ)"""
    clip_facing = df.groupby("clip")["ankle_x_norm"].mean()
    majority_sign = float(np.sign(clip_facing.median()))  # ทิศทางที่คลิปส่วนใหญ่หันไป

    if save_reference:
        with open(FACING_REFERENCE_PATH, "w", encoding="utf-8") as f:
            json.dump({"majority_sign": majority_sign}, f)

    flip_sign = clip_facing.apply(lambda v: -1 if np.sign(v) != majority_sign else 1)
    flipped_clips = flip_sign[flip_sign == -1].index.tolist()
    if flipped_clips:
        print(f"พลิกทิศทาง (mirror แกน x) ให้คลิป: {flipped_clips}")

    out = df.copy()
    row_flip = out["clip"].map(flip_sign)
    for col in FACING_FLIP_COLUMNS:
        out[col] = out[col] * row_flip
    return out


def load_and_clean(csv_path="landmarks_raw.csv"):
    df = pd.read_csv(csv_path)
    n_before = len(df)

    df = df.dropna(subset=RAW_COLUMNS)  # ทิ้งเฟรมที่ตรวจไม่เจอคน
    n_after_detect = len(df)

    # ทิ้งเฟรมที่ MediaPipe เองไม่มั่นใจในจุดใดจุดหนึ่ง (เช่น หลงไปจับขาผิดข้างเพราะเส้นทับภาพ)
    min_v = df[VISIBILITY_COLUMNS].min(axis=1)
    df = df[min_v >= MIN_VISIBILITY]
    n_after_vis = len(df)

    df = normalize_row(df)
    df = compute_angles(df)
    n_after_norm = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS)  # ทิ้งเฟรมที่ scale=0 หรือมุมคำนวณไม่ได้ (ถ้ามี)
    df = normalize_facing_direction(df)

    print(f"โหลด {n_before} เฟรม, ทิ้ง {n_before - n_after_detect} เฟรมที่ตรวจไม่เจอคน, "
          f"ทิ้งเพิ่ม {n_after_detect - n_after_vis} เฟรมที่มั่นใจต่ำกว่า {MIN_VISIBILITY} "
          f"(<{MIN_VISIBILITY} ในจุดใดจุดหนึ่ง), "
          f"ทิ้งเพิ่ม {n_after_vis - len(df)} เฟรมที่ scale=0 เหลือ {len(df)} เฟรม")

    df["subject_id"] = df["clip"]  # T กับ F ของแต่ละชื่อคือคนละคน (ยืนยันจากผู้ใช้)
    df["prefix"] = df["clip"].str[0]
    return df


def split_by_subject(df, test_size=0.2, seed=RANDOM_STATE):
    """แบ่งแค่ train/test ตามคลิป (ตัด val ออกตามที่ตกลง) — การหา decision_threshold
    (03_train_model.py) ใช้ cross-validation บนชุด train เองแทนการมี val แยก จึงไม่ต้องเสีย
    ข้อมูลไปเป็นชุดที่สาม แต่ threshold ก็ยังไม่เห็นชุด test เลยเหมือนเดิม"""
    subjects = df["subject_id"].to_numpy()
    labels = df["Label"].to_numpy()

    n_splits_test = max(2, round(1 / test_size))
    sgkf_test = StratifiedGroupKFold(n_splits=n_splits_test, shuffle=True, random_state=seed)
    train_idx, test_idx = next(sgkf_test.split(df, labels, groups=subjects))

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    return train_df, test_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="landmarks_raw.csv")
    parser.add_argument("--output", default="splits.json")
    parser.add_argument("--train-out", default="dataset_train.parquet")
    parser.add_argument("--test-out", default="dataset_test.parquet")
    args = parser.parse_args()

    df = load_and_clean(args.csv)

    n_subjects = df["subject_id"].nunique()
    print(f"\nคน/คลิป: {n_subjects} {sorted(df['subject_id'].unique())}")
    print(f"Label: {df['Label'].value_counts().to_dict()}")
    print(f"body_scale_px: min={df['body_scale_px'].min():.1f}, max={df['body_scale_px'].max():.1f}, "
          f"median={df['body_scale_px'].median():.1f} (ยิ่งต่างกันมาก ยิ่งแสดงว่า normalize จำเป็น)")

    train_df, test_df = split_by_subject(df)

    train_subj = set(train_df["subject_id"])
    test_subj = set(test_df["subject_id"])
    overlap = train_subj & test_subj
    assert not overlap, f"พบคนซ้ำข้ามชุด: {overlap}"

    for name, split_df in (("Train", train_df), ("Test", test_df)):
        print(
            f"{name}: {len(split_df)} เฟรม, {split_df['subject_id'].nunique()} คน "
            f"{sorted(split_df['subject_id'].unique())}, Label={split_df['Label'].value_counts().to_dict()}"
        )

    train_df.to_parquet(args.train_out, index=False)
    test_df.to_parquet(args.test_out, index=False)

    splits_info = {
        "random_state": RANDOM_STATE,
        "feature_columns": FEATURE_COLUMNS,
        "raw_columns": RAW_COLUMNS,
        "train_subjects": sorted(train_subj),
        "test_subjects": sorted(test_subj),
        "n_train": len(train_df), "n_test": len(test_df),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(splits_info, f, ensure_ascii=False, indent=2)
    print(f"\nบันทึก {args.output}, {args.train_out}, {args.test_out}")


if __name__ == "__main__":
    main()
