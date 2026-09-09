"""
ขั้นที่ 5: หาช่วงเป้าหมาย + น้ำหนักของแต่ละมุม 'จากข้อมูลจริง' เพื่อคำนวณคะแนนท่าวิ่ง 0-100

⚠️ บทเรียนจากรอบก่อน: เคยลองใช้ช่วงเป้าหมายจากตำรา biomechanics ทั่วไป (เช่น cadence 170-180,
เอนตัว 5-10°) แล้วพบว่า **แยกกลุ่มถูก/ผิดของข้อมูลชุดนี้ไม่ได้เลย** (คลิปผิดได้คะแนนเฉลี่ยสูงกว่า
คลิปถูกด้วยซ้ำ) เพราะกลุ่มตัวอย่าง/ความเร็ว/มุมกล้องไม่ตรงกับที่ตำราอ้างอิงมา

รอบนี้แก้ด้วยการ **คำนวณช่วงเป้าหมายจากข้อมูล Train จริงที่มี label กำกับแล้ว** แทน:
  - target range = ค่าเฉลี่ย ± 1SD ของกลุ่ม 'ถูก' จริง (Label=1) ในข้อมูล train
  - น้ำหนักแต่ละมุม = Cohen's d (ความแรงของการแยกกลุ่มถูก/ผิด) ยิ่งแยกได้ชัด ยิ่งมีน้ำหนักมาก
  - ตัด trunk_lean ทิ้ง (Cohen's d ≈ 0.04 แทบไม่มีนัยสำคัญในชุดข้อมูลนี้)
  - คำนวณจากชุด Train เท่านั้น แล้ว validate กับชุด Test แยกต่างหาก (ไม่ปนกัน)
"""

import argparse
import json

import numpy as np
import pandas as pd

ANGLE_COLUMNS = ["knee_angle", "hip_angle", "ankle_angle", "thigh_angle", "shank_angle", "foot_angle"]
MIN_COHEN_D = 0.05  # ต่ำกว่านี้ถือว่าไม่มีนัยสำคัญ ตัดออกจากคะแนน (เช่น trunk_lean)


def cohens_d(a, b):
    pooled_std = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    return abs(a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0.0


def calibrate(train_df):
    correct = train_df[train_df["Label"] == 1]
    incorrect = train_df[train_df["Label"] == 0]

    specs = {}
    for col in ANGLE_COLUMNS:
        d = cohens_d(correct[col], incorrect[col])
        if d < MIN_COHEN_D:
            print(f"  ตัด {col} ทิ้ง (Cohen's d={d:.3f} ต่ำเกินไป ไม่มีนัยสำคัญ)")
            continue

        mean, std = correct[col].mean(), correct[col].std()
        specs[col] = {
            "target_low": mean - std,
            "target_high": mean + std,
            "zero_low": mean - 3 * std,
            "zero_high": mean + 3 * std,
            "cohens_d": d,
        }
        print(f"  {col:15s} เป้าหมาย=[{mean - std:7.2f}, {mean + std:7.2f}]  Cohen's d={d:.3f}")

    total_d = sum(s["cohens_d"] for s in specs.values())
    for s in specs.values():
        s["weight"] = round(100 * s["cohens_d"] / total_d, 2)

    return specs


def score_frame(row, specs):
    """คะแนน 1 เฟรม (0-100) จาก spec ที่ calibrate ไว้ — คืน (score, breakdown list)"""
    total_score = 0.0
    breakdown = []
    for col, spec in specs.items():
        value = row[col]
        lo, hi = spec["target_low"], spec["target_high"]
        zlo, zhi = spec["zero_low"], spec["zero_high"]
        weight = spec["weight"]

        if lo <= value <= hi:
            got = weight
        elif value < lo:
            got = weight * max(0.0, min(1.0, (value - zlo) / (lo - zlo))) if lo != zlo else 0
        else:
            got = weight * max(0.0, min(1.0, (zhi - value) / (zhi - hi))) if zhi != hi else 0

        total_score += got
        breakdown.append({"metric": col, "value": value, "target": (round(lo, 1), round(hi, 1)),
                           "score": round(got, 1), "weight": weight})
    return total_score, breakdown


def validate(specs, test_df):
    """เช็คว่าคะแนนแยกกลุ่มถูก/ผิดของชุด Test (ที่ไม่เคยใช้ calibrate) ได้จริงไหม"""
    scores = test_df.apply(lambda row: score_frame(row, specs)[0], axis=1)
    test_df = test_df.assign(score=scores)

    by_clip = test_df.groupby("clip").agg(label=("Label", "mean"), score=("score", "mean"))
    print("\nคะแนนเฉลี่ยต่อคลิป (ชุด Test, ไม่เคยใช้ตอน calibrate):")
    print(by_clip.sort_values("score", ascending=False))

    correct_scores = test_df.loc[test_df["Label"] == 1, "score"]
    incorrect_scores = test_df.loc[test_df["Label"] == 0, "score"]
    print(f"\nคะแนนเฉลี่ยรายเฟรม: ถูก={correct_scores.mean():.1f}  ผิด={incorrect_scores.mean():.1f}  "
          f"(ส่วนต่าง={correct_scores.mean() - incorrect_scores.mean():+.1f})")
    if correct_scores.mean() > incorrect_scores.mean():
        print("✓ แยกกลุ่มได้ถูกทิศทาง (ถูก > ผิด)")
    else:
        print("⚠️  แยกกลุ่มผิดทิศทาง — ห้ามใช้ค่านี้จริง ต้องทบทวนวิธี calibrate ใหม่")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="dataset_train.parquet")
    parser.add_argument("--test", default="dataset_test.parquet")
    parser.add_argument("--output", default="score_config.json")
    args = parser.parse_args()

    train_df = pd.read_parquet(args.train)
    print(f"Calibrate จาก {len(train_df)} เฟรม (ชุด Train เท่านั้น)")
    specs = calibrate(train_df)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(specs, f, ensure_ascii=False, indent=2)
    print(f"\nบันทึก {args.output}")

    test_df = pd.read_parquet(args.test)
    validate(specs, test_df)


if __name__ == "__main__":
    main()
