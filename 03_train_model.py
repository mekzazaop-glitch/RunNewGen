"""
ขั้นที่ 3: เทรนโมเดลจำแนกท่าวิ่งรายเฟรม จากพิกัดพิกเซล 6 จุดฝั่งขวา + มุมข้อต่อ (Prototype + Student)

  - ใช้ RandomForest เท่านั้น — ล็อกตามที่ตกลงกันไว้
  - จูน hyperparameter ด้วย GroupKFold ภายในชุด train เท่านั้น (nested CV) ห้ามแตะ test

⚠️ ปรับปรุงสะสมจากหลายรอบก่อนหน้า (ไม่ต้องเก็บข้อมูลเพิ่มเลยสักข้อ):
  1. Overfit: min_samples_leaf/max_depth คุมความลึกของต้นไม้
  2. Feature: เพิ่มมุมข้อต่อ 7 ตัว (คำนวณจาก 6 จุดเดิม ไม่ต้องสกัดวิดีโอใหม่) — ดู
     02_prepare_dataset.py: compute_angles()
  3. Bias ระหว่างคลาส: ทดลองมาหลายวิธี (SMOTE, BorderlineSMOTE, SMOTETomek,
     BalancedRandomForest, ตัดฟีเจอร์) พบว่า SMOTE ให้ macro-F1 รวมดีที่สุด (0.708) แต่ recall
     คลาส 1 (ถูก) ยังต่ำ (60%) เทียบกับคลาส 0 (81%) — เปลี่ยนมาใช้ class_weight={0:1, 1:2} +
     เลือก threshold ที่ทำให้ recall สองคลาส 'สมดุล' ที่สุด (แทนที่จะเลือก threshold ที่ทำให้
     macro-F1 สูงสุด) ตามที่ตัดสินใจไว้ — แลก macro-F1 รวมลดลงเหลือ ~0.68 เพื่อให้ recall ทั้งสอง
     คลาสใกล้เคียงกัน (~69% ทั้งคู่) ไม่ลำเอียงเข้าหาคลาสใดคลาสหนึ่งเป็นพิเศษ
"""

import argparse
from importlib import import_module

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import GroupKFold, GridSearchCV, cross_val_predict

prep_mod = import_module("02_prepare_dataset")

RANDOM_STATE = 42
FEATURE_COLUMNS = prep_mod.FEATURE_COLUMNS
MODEL_NAME = "RandomForest (balanced recall)"
CLASS_WEIGHT = {0: 1, 1: 2}  # ถ่วงคลาส 1 (ถูก) หนักขึ้น 2 เท่า — ชดเชยที่มีตัวอย่างน้อยกว่าและยากกว่า


def build_model(**rf_params):
    return RandomForestClassifier(random_state=RANDOM_STATE, class_weight=CLASS_WEIGHT, **rf_params)


def tune_model(X, y, groups, param_grid, n_splits=4):
    n_splits = min(n_splits, groups.nunique())
    if n_splits < 2:
        print("  คนในชุด train น้อยเกินกว่าจะทำ GroupKFold จูนพารามิเตอร์ (ใช้ค่า default แทน)")
        model = build_model()
        model.fit(X, y)
        return model

    cv = GroupKFold(n_splits=n_splits)
    splits = list(cv.split(X, y, groups=groups))  # ต้องแปลงเป็น list ก่อน กัน n_jobs=-1 pickle generator ไม่ได้
    search = GridSearchCV(build_model(), param_grid, scoring="f1_macro", cv=splits, n_jobs=-1)
    search.fit(X, y)
    print(f"  best params: {search.best_params_}, best macro-F1 (nested CV): {search.best_score_:.3f}")
    return search.best_estimator_


def find_balanced_threshold(model, X_train, y_train, groups_train, n_splits=4):
    """หา decision threshold ที่ทำให้ recall คลาส 0 กับคลาส 1 'ใกล้เคียงกันที่สุด' (แทนการเลือก
    threshold ที่ทำ macro-F1 สูงสุดแบบเดิม) — ตามที่ตัดสินใจไว้ว่าเน้นความยุติธรรมระหว่าง 2 คลาส
    มากกว่าตัวเลขรวม คำนวณจาก out-of-fold prediction ของ GroupKFold บนชุด train เอง ไม่แตะ test"""
    n_splits = min(n_splits, groups_train.nunique())
    if n_splits < 2:
        print("  คนน้อยเกินกว่าจะทำ cross_val_predict หา threshold (ใช้ 0.5 default)")
        return 0.5

    cv = GroupKFold(n_splits=n_splits)
    splits = list(cv.split(X_train, y_train, groups=groups_train))
    oof_proba = cross_val_predict(model, X_train, y_train, cv=splits, method="predict_proba")
    proba_1 = oof_proba[:, 1]  # y เป็น {0,1} เสมอ -> sklearn เรียง classes_ ขึ้น -> คอลัมน์ 1 คือคลาส 1

    best_threshold, best_gap = 0.5, 1.0
    for t in np.arange(0.20, 0.81, 0.01):
        pred = (proba_1 >= t).astype(int)
        recall_1 = recall_score(y_train, pred, pos_label=1)
        recall_0 = recall_score(y_train, pred, pos_label=0)
        gap = abs(recall_1 - recall_0)
        if gap < best_gap:
            best_gap, best_threshold = gap, t

    pred = (proba_1 >= best_threshold).astype(int)
    f1 = f1_score(y_train, pred, average="macro")
    r1 = recall_score(y_train, pred, pos_label=1)
    r0 = recall_score(y_train, pred, pos_label=0)
    print(f"  threshold ที่สมดุลที่สุด (out-of-fold บนชุด train): {best_threshold:.2f} "
          f"(macro-F1 {f1:.3f}, recall(0)={r0:.3f}, recall(1)={r1:.3f})")
    return float(best_threshold)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="dataset_train.parquet")
    parser.add_argument("--output", default="model.joblib")
    args = parser.parse_args()

    train_df = pd.read_parquet(args.train)
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["Label"]
    groups_train = train_df["subject_id"]

    print(f"เทรนด้วย {len(X_train)} เฟรม, {groups_train.nunique()} คน, {len(FEATURE_COLUMNS)} ฟีเจอร์")
    print(f"Label: {y_train.value_counts().to_dict()}")

    print(f"\n=== {MODEL_NAME} (class_weight={CLASS_WEIGHT}) ===")
    param_grid = {
        "n_estimators": [300, 500],
        "max_depth": [4, 6, 8],
        "min_samples_leaf": [10, 20, 40, 60],
        "max_features": ["sqrt", 0.5],
    }
    model = tune_model(X_train, y_train, groups_train, param_grid)

    tuned_params = {
        k: v for k, v in model.get_params().items()
        if k in ("n_estimators", "max_depth", "min_samples_leaf", "max_features")
    }
    threshold = find_balanced_threshold(build_model(**tuned_params), X_train, y_train, groups_train)

    model.fit(X_train, y_train)

    bundle = {
        "model": model,
        "model_name": MODEL_NAME,
        "feature_columns": FEATURE_COLUMNS,
        "classes": sorted(y_train.unique().tolist()),
        "feature_medians": X_train.median(numeric_only=True).to_dict(),
        "decision_threshold": threshold,  # ใช้แทน 0.5 ตอนทำนายจริง (predict_proba >= threshold -> 1)
    }
    joblib.dump(bundle, args.output)
    print(f"\nบันทึกโมเดล -> {args.output}")


if __name__ == "__main__":
    main()
