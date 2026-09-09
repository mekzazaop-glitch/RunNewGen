"""
ขั้นที่ 3: ประเมินผลโมเดล — ต้องดูหลายมุมพร้อมกัน ห้ามเชื่อตัวเลขเดียว

  - macro-F1 คือตัวเลขหลัก ไม่ใช่ accuracy (accuracy หลอกตาเมื่อคลาสไม่สมดุล)
  - รายงาน 2 ระดับ: ระดับเฟรม และระดับคลิป (เฉลี่ย probability ของทุกเฟรมในคลิปนั้น)
  - Leave-One-Subject-Out CV รายงาน mean ± std ข้าม fold — ชุดข้อมูลมีแค่ ~11 คน ตัวเลขเดี่ยวๆ
    จากการแบ่ง train/val/test ครั้งเดียวเชื่อไม่ได้ ต้องดู LOSO ควบคู่ไปด้วยเสมอ
  - Confusion matrix ดูว่าสับสนคู่ไหน
  - Permutation importance ไม่ใช่ feature importance แบบ impurity
  - ถ้า accuracy/macro-F1 > 0.98 ให้สงสัยว่ามี data leakage ไว้ก่อนเสมอ
"""

import argparse
from importlib import import_module

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

prep_mod = import_module("02_prepare_dataset")


def predict_with_threshold(model, X, threshold):
    """ทำนายด้วย decision_threshold ที่ปรับจากชุด val แทนค่า default 0.5 ของ .predict()
    (แก้ปัญหาโมเดลลำเอียงเข้าหาคลาส 0 ที่พบตอน evaluate — ดู 03_train_model.py)"""
    proba_1 = model.predict_proba(X)[:, list(model.classes_).index(1)]
    return (proba_1 >= threshold).astype(int)


def clip_level_report(model, X, y, clip_ids, title, threshold=0.5):
    """รวมทำนายระดับเฟรม -> ระดับคลิป ด้วยการเฉลี่ย probability ของทุกเฟรมในคลิปเดียวกัน"""
    proba = model.predict_proba(X)
    df = pd.DataFrame(proba, columns=model.classes_)
    df["clip"] = clip_ids.to_numpy()
    df["true"] = y.to_numpy()

    clip_proba = df.groupby("clip")[list(model.classes_)].mean()
    clip_true = df.groupby("clip")["true"].first().astype(int)
    clip_pred = (clip_proba[1] >= threshold).astype(int)  # ใช้ threshold เดียวกับระดับเฟรม
    clip_confidence = clip_proba.max(axis=1)

    print(f"\n--- {title} (ระดับคลิป, {len(clip_true)} คลิป) ---")
    for c in clip_true.index:
        mark = "✓" if clip_pred[c] == clip_true[c] else "✗"
        print(f"  {mark} {c}: จริง={clip_true[c]} ทาย={clip_pred[c]} (มั่นใจ {clip_confidence[c]:.1%})")

    acc = (clip_pred == clip_true).mean()
    f1 = f1_score(clip_true, clip_pred, average="macro", zero_division=0)
    print(f"  accuracy={acc:.3f}, macro-F1={f1:.3f}")
    return acc, f1


def window_level_report(model, X, y, title, threshold=0.5):
    y_pred = predict_with_threshold(model, X, threshold)
    acc = (y_pred == y.to_numpy()).mean()
    f1 = f1_score(y, y_pred, average="macro", zero_division=0)

    print(f"\n--- {title} (ระดับเฟรม, {len(y)} เฟรม, threshold={threshold:.2f}) ---")
    print(f"  accuracy={acc:.3f}, macro-F1={f1:.3f}")
    print(classification_report(y, y_pred, zero_division=0))

    if acc > 0.98 or f1 > 0.98:
        print("  ⚠️  accuracy/macro-F1 สูงกว่า 0.98 — สงสัยว่ามี data leakage ไว้ก่อนเสมอ")

    cm = confusion_matrix(y, y_pred, labels=sorted(y.unique()))
    print(f"  confusion matrix (labels={sorted(y.unique())}):\n    {cm}")
    return acc, f1


def loso_cv(all_df, feature_cols):
    """Leave-One-Subject-Out CV บนข้อมูลทั้งหมด (รวม train+val+test) — ชุดข้อมูลมีแค่ ~11 คน
    การแบ่ง train/val/test ครั้งเดียวให้ตัวเลขที่เชื่อถือไม่ได้ LOSO ทดสอบทีละคนแทน"""
    X = all_df[feature_cols]
    y = all_df["Label"]
    groups = all_df["subject_id"]

    n_subjects = groups.nunique()
    cv = GroupKFold(n_splits=n_subjects)

    fold_scores = []
    print(f"\n=== Leave-One-Subject-Out CV ({n_subjects} fold, รวม train+val+test, ระดับเฟรม) ===")
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups)):
        held_out_subject = groups.iloc[test_idx].iloc[0]
        clf = RandomForestClassifier(class_weight="balanced", n_estimators=200, random_state=42)
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = clf.predict(X.iloc[test_idx])
        f1 = f1_score(y.iloc[test_idx], y_pred, average="macro", zero_division=0)
        acc = (y_pred == y.iloc[test_idx].to_numpy()).mean()
        fold_scores.append(f1)
        print(f"  fold {fold_idx + 1} (ทดสอบกับคน '{held_out_subject}'): accuracy={acc:.3f}, macro-F1={f1:.3f}")

    fold_scores = np.array(fold_scores)
    print(f"\n  LOSO macro-F1 เฉลี่ย: {fold_scores.mean():.3f} (+/- {fold_scores.std():.3f})")
    if fold_scores.std() > 0.2:
        print("  ⚠️  ส่วนเบี่ยงเบนสูงมาก — ผลต่างกันมากตามว่าทดสอบกับใคร ข้อมูลยังน้อยเกินกว่าจะสรุปได้")
    return fold_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model.joblib")
    parser.add_argument("--landmarks-csv", default="landmarks_raw.csv")
    parser.add_argument("--splits", default="splits.json")
    parser.add_argument("--train", default="dataset_train.parquet")
    parser.add_argument("--test", default="dataset_test.parquet")
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    model = bundle["model"]
    feature_cols = bundle["feature_columns"]
    threshold = bundle.get("decision_threshold", 0.5)
    print(f"โมเดล: {bundle['model_name']}, {len(feature_cols)} ฟีเจอร์, คลาส {bundle['classes']}, "
          f"decision_threshold={threshold:.2f}")

    scores = {}
    for name, path in (("Train", args.train), ("Test", args.test)):
        df = pd.read_parquet(path)
        X = df[feature_cols]
        y = df["Label"]
        acc, f1 = window_level_report(model, X, y, name, threshold)
        clip_level_report(model, X, y, df["clip"], name, threshold)
        scores[name] = (acc, f1)

    train_test_gap = scores["Train"][0] - scores["Test"][0]
    print(f"\n=== ตรวจ overfit: ช่องว่าง accuracy Train-Test = {train_test_gap:+.3f} ===")
    if train_test_gap > 0.15:
        print("  ⚠️  ยังห่างเกิน 0.15 — โมเดลยังจำ train มากกว่าที่ควร ลอง regularize เพิ่มอีก")
    else:
        print("  อยู่ในเกณฑ์ที่รับได้ (< 0.15)")

    print("\n=== Permutation Importance (วัดจากชุด Test) ===")
    test_df = pd.read_parquet(args.test)
    X_val = test_df[feature_cols]
    y_val = test_df["Label"]
    result = permutation_importance(model, X_val, y_val, n_repeats=20, random_state=42, scoring="f1_macro")
    order = result.importances_mean.argsort()[::-1]
    for i in order:
        print(f"  {feature_cols[i]:22s} {result.importances_mean[i]:+.4f} (+/- {result.importances_std[i]:.4f})")

    all_df = prep_mod.load_and_clean(args.landmarks_csv)
    loso_cv(all_df, feature_cols)


if __name__ == "__main__":
    main()
