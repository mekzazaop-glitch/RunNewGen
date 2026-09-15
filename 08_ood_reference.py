"""
08_ood_reference.py — สร้าง ood_reference.json สำหรับเตือน "ท่าวิ่งต่างจากข้อมูลที่ระบบเรียนรู้"

ไม่ได้แตะโมเดล (model.joblib / threshold / ชุดข้อมูล ไม่เปลี่ยนเลย) — แค่อ่าน dataset_train.parquet
มาหาค่าเฉลี่ยรายคลิปของแต่ละฟีเจอร์ในคลิปที่ใช้เทรน แล้วบันทึกไว้ให้เว็บเทียบว่าคลิปที่อัปโหลดมา
อยู่ห่างจากกลุ่มตัวอย่างที่โมเดลเคยเห็นแค่ไหน

เหตุผล: โมเดลเรียนรู้คำว่า "ท่าถูก" จากนักศึกษา 12 คนเท่านั้น คลิปที่รูปแบบต่างออกไปมาก (เช่น คลิปต้นแบบ
Running Analysis ซึ่งโมเดลไม่เคยเห็นตอนเทรน: มุมเท้าห่างจากค่าเฉลี่ยของคลิปเทรน ~10 SD) โมเดลจะ
ตัดสินจากสิ่งที่มันไม่เคยเรียน ผลจึงไม่ควรนำเสนอเหมือนผลปกติ

วิธีวัด: z ของค่าเฉลี่ยทั้งคลิปแต่ละฟีเจอร์ เทียบกับการกระจายของค่าเฉลี่ยรายคลิปของคลิปเทรน
แล้วรวมเป็น rms_z = sqrt(mean(z^2)) (ตัดมุมต้นขา/หน้าแข้งออก เพราะค่าวนรอบ ±180° ใช้ค่าเฉลี่ยตรงๆ ไม่ได้)

รัน: python 08_ood_reference.py   (ต้องมี splits.json + dataset_train.parquet + dataset_test.parquet)
"""

import json

import numpy as np
import pandas as pd

SPLITS_PATH = "splits.json"
TRAIN_PATH = "dataset_train.parquet"
TEST_PATH = "dataset_test.parquet"
OUTPUT_PATH = "ood_reference.json"

EXCLUDE_FEATURES = {"thigh_angle", "shank_angle"}  # มุมเทียบแนวดิ่งที่ค่าวนรอบ ±180°

# เลือกจากการตรวจแบบ leave-one-clip-out ทั้ง 22 คลิป (ตารางที่สคริปต์นี้พิมพ์ออกมา): คลิปนักศึกษาทุกคลิป
# rms_z <= 2.0, Running Analysis = 3.55 — ตั้งเส้นที่ 2.5 ให้ห่างจากคลิปปกติพอสมควรกันเตือนพร่ำเพรื่อ
RMS_Z_THRESHOLD = 2.5


def rms_z(clip_means, ref):
    z = (clip_means - ref.mean()) / ref.std()
    return float(np.sqrt((z ** 2).mean())), z


def main():
    with open(SPLITS_PATH, encoding="utf-8") as f:
        splits = json.load(f)
    features = [c for c in splits["feature_columns"] if c not in EXCLUDE_FEATURES]

    train = pd.read_parquet(TRAIN_PATH)
    train_means = train.groupby("clip")[features].mean()

    reference = {
        "features": features,
        "clip_mean": {k: round(float(v), 6) for k, v in train_means.mean().items()},
        "clip_sd": {k: round(float(v), 6) for k, v in train_means.std().items()},
        "n_reference_clips": int(len(train_means)),
        "reference_clips": sorted(train_means.index),
        "rms_z_threshold": RMS_Z_THRESHOLD,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(reference, f, ensure_ascii=False, indent=1)
    print(f"บันทึก {OUTPUT_PATH}: {len(features)} ฟีเจอร์, อ้างอิง {len(train_means)} คลิปเทรน, เส้นเตือน rms_z >= {RMS_Z_THRESHOLD}")

    # ตรวจความถูกต้องของเส้นเตือน: คลิปเทรนเทียบกับคลิปเทรนอื่นที่เหลือ (ไม่นับตัวเอง) คลิปเทสต์เทียบกับคลิปเทรนทั้งหมด
    test = pd.read_parquet(TEST_PATH)
    all_means = pd.concat([train, test]).groupby("clip")[features].mean()
    rows = []
    for clip in all_means.index:
        ref = train_means.drop(index=clip, errors="ignore")
        score, z = rms_z(all_means.loc[clip], ref)
        top = z.abs().sort_values(ascending=False).index[:3]
        rows.append((clip, "train" if clip in train_means.index else "test", round(score, 2),
                     "เตือน" if score >= RMS_Z_THRESHOLD else "-", ", ".join(f"{k}={z[k]:+.1f}" for k in top)))
    report = pd.DataFrame(rows, columns=["clip", "split", "rms_z", "ผล", "ฟีเจอร์ที่ต่างมากสุด"])
    print(report.sort_values("rms_z", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
