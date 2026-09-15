"""
ขั้นที่ 3: ประเมินผลโมเดล — ต้องดูหลายมุมพร้อมกัน ห้ามเชื่อตัวเลขเดียว

  - macro-F1 คือตัวเลขหลัก ไม่ใช่ accuracy (accuracy หลอกตาเมื่อคลาสไม่สมดุล)
  - รายงาน 2 ระดับ: ระดับเฟรม และระดับคลิป (เฉลี่ย probability ของทุกเฟรมในคลิปนั้น)
  - Leave-One-Subject-Out CV รายงาน mean ± std ข้าม fold — ชุดข้อมูลมีแค่ 12 คนจริง (ไม่ใช่ 22
    เหมือนจำนวนคลิป) ตัวเลขเดี่ยวๆ จากการแบ่ง train/val/test ครั้งเดียวเชื่อไม่ได้ ต้องดู LOSO
    ควบคู่ไปด้วยเสมอ
  - Confusion matrix ดูว่าสับสนคู่ไหน
  - Permutation importance ไม่ใช่ feature importance แบบ impurity
  - ถ้า accuracy/macro-F1 > 0.98 ให้สงสัยว่ามี data leakage ไว้ก่อนเสมอ

⚠️ แก้ไขสำคัญ (รอบตรวจสอบ overfit): 2 บั๊กที่ทำให้ตัวเลขก่อนหน้านี้เชื่อถือไม่ได้ — สคริปต์นี้แก้ไข
ให้ถูกต้องแล้ว แต่**ไม่แตะ model.joblib ที่ deploy อยู่เลย** (ยังคงเป็น RandomForest ตัวเดิมที่เทรน
ไว้แล้ว สคริปต์นี้แค่ประเมินผลมัน ไม่ได้เทรนใหม่ทับ):

  1. LOSO เดิมแบ่ง fold ตาม "clip" (subject_id ในไฟล์ข้อมูล = ชื่อคลิป เช่น Fmek/Tmek) ทำให้
     Fmek กับ Tmek ถูกนับเป็นคนละ fold ทั้งที่**เป็นคนเดียวกันที่ถ่าย 2 คลิป** (ยืนยันจากผู้ใช้แล้ว —
     ดูคอมเมนต์ที่ 02_prepare_dataset.py บรรทัด subject_id ซึ่งเคยเขียนผิดไว้ว่าเป็นคนละคน) ผลคือ
     ตอนทดสอบ Fmek โมเดลเคยเห็นท่าวิ่งของคนคนเดียวกันจาก Tmek ในชุด train มาแล้ว ทำให้ตัวเลข LOSO
     เดิม (0.706) สูงเกินจริง ตัวเลขที่ถูกต้องคือแบ่งตาม "คน" (ตัด F/T ออกจากชื่อคลิป) ได้ต่ำกว่า
     ชัดเจน (~0.64) — ดูฟังก์ชัน person_of() ด้านล่าง
  2. clip_level_report() เดิมเอา label ของ "เฟรมแรก" ของคลิปมาเป็นความจริงระดับคลิป
     (`groupby("clip")["true"].first()`) คลิปที่ label รายเฟรมสลับไปมาตามจังหวะก้าว (ไม่ใช่ทุกเฟรม
     ของคลิป "T" จะ label ว่าถูก) จึงได้ความจริงตามเฟรมแรกที่บังเอิญเจอ ไม่ใช่ F/T ของคลิปจริง —
     แก้เป็นอ่านจากชื่อคลิป (F=ผิด, T=ถูก) แทน ดูฟังก์ชัน clip_truth() ด้านล่าง
"""

import argparse
import re
from importlib import import_module

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

prep_mod = import_module("02_prepare_dataset")
train_mod = import_module("03_train_model")  # reuse build_model()/find_balanced_threshold() เดียวกับตอนเทรนจริง
                                              # กัน LOSO ใช้วิธีเลือก threshold คนละแบบกับโมเดลที่ deploy จริง


def person_of(clip_name):
    """ตัด F/T นำหน้าออกจากชื่อคลิป -> ได้ชื่อคน (Fmek/Tmek -> mek) คลิปที่ไม่ได้ขึ้นต้นด้วย F/T+ตัวเล็ก
    (เช่น 'Running Analysis' ซึ่งเป็นคลิปต้นแบบ ไม่ใช่คลิปนักศึกษา) ถือเป็นคนของตัวเอง 1 คน"""
    return re.sub(r"^[FT](?=[a-z])", "", clip_name)


def clip_truth(clip_name, frame_labels):
    """ความจริงระดับคลิปจากชื่อคลิป (F=ผิด, T=ถูก) แทนการเดาจาก label ของเฟรมแรก — คลิปต้นแบบที่ไม่ได้
    ขึ้นต้นด้วย F/T (เช่น 'Running Analysis') ใช้ label ส่วนใหญ่ของเฟรมในคลิปนั้นแทน (fallback)"""
    if re.match(r"^F[a-z]", clip_name):
        return 0
    if re.match(r"^T[a-z]", clip_name):
        return 1
    return int(round(frame_labels.mean()))


def predict_with_threshold(model, X, threshold):
    """ทำนายด้วย decision_threshold ที่ปรับจากชุด val แทนค่า default 0.5 ของ .predict()
    (แก้ปัญหาโมเดลลำเอียงเข้าหาคลาส 0 ที่พบตอน evaluate — ดู 03_train_model.py)"""
    proba_1 = model.predict_proba(X)[:, list(model.classes_).index(1)]
    return (proba_1 >= threshold).astype(int)


def clip_level_report(model, X, y, clip_ids, title, threshold=0.5):
    """รวมทำนายระดับเฟรม -> ระดับคลิป ด้วยการเฉลี่ย probability ของทุกเฟรมในคลิปเดียวกัน
    ความจริงระดับคลิปอ่านจากชื่อคลิป (clip_truth) ไม่ใช่เดาจาก label ของเฟรมแรกเหมือนเดิม"""
    proba = model.predict_proba(X)
    df = pd.DataFrame(proba, columns=model.classes_)
    df["clip"] = clip_ids.to_numpy()
    df["true"] = y.to_numpy()

    clip_proba = df.groupby("clip")[list(model.classes_)].mean()
    clip_true = pd.Series({name: clip_truth(name, g["true"]) for name, g in df.groupby("clip")})
    clip_pred = (clip_proba[1] >= threshold).astype(int)  # ใช้ threshold เดียวกับระดับเฟรม
    clip_confidence = clip_proba.max(axis=1)

    print(f"\n--- {title} (ระดับคลิป, {len(clip_true)} คลิป) ---")
    for c in clip_true.index:
        mark = "✓" if clip_pred[c] == clip_true[c] else "✗"
        print(f"  {mark} {c}: จริง={clip_true[c]} ทาย={clip_pred[c]} (มั่นใจ {clip_confidence[c]:.1%})")

    acc = (clip_pred == clip_true).mean()
    f1 = f1_score(clip_true, clip_pred, average="macro", zero_division=0)
    auc = roc_auc_score(clip_true, clip_proba[1]) if clip_true.nunique() > 1 else float("nan")
    print(f"  accuracy={acc:.3f}, macro-F1={f1:.3f}, ROC AUC (แยกคลิป F/T)={auc:.3f}")
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


def loso_cv(all_df, feature_cols, rf_params):
    """Leave-One-Subject-Out CV บนข้อมูลทั้งหมด (รวม train+val+test) แบ่ง fold ตาม "คนจริง"
    (person_of ตัด F/T ออกจากชื่อคลิป) — มีคนจริงแค่ 12 คน ไม่ใช่ 22 เหมือนจำนวนคลิป ถ้าแบ่งตาม
    คลิปเหมือนเดิม Fmek กับ Tmek จะแยกกันคนละ fold ทั้งที่เป็นคนเดียวกัน ทำให้ตัวเลขสูงเกินจริง

    ใช้ build_model() และ find_balanced_threshold() ชุดเดียวกับ 03_train_model.py (ไม่ได้คิดค้น
    วิธีใหม่ในสคริปต์นี้) เพื่อให้ LOSO วัด "วิธีการเทรนที่ deploy จริง" ไม่ใช่วิธีอื่นที่ไม่มีใครใช้จริง —
    โมเดลที่ fit ในนี้เป็นโมเดลชั่วคราวสำหรับวัดผลเท่านั้น ไม่ได้เขียนทับ model.joblib"""
    X = all_df[feature_cols]
    y = all_df["Label"]
    clip = all_df["subject_id"]
    person = clip.map(person_of)

    n_subjects = person.nunique()
    cv = GroupKFold(n_splits=n_subjects)

    fold_scores = []
    clip_rows = []
    print(f"\n=== Leave-One-Subject-Out CV ({n_subjects} fold แบ่งตามคนจริง, รวม train+val+test, ระดับเฟรม) ===")
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=person)):
        held_out_person = person.iloc[test_idx].iloc[0]
        X_tr, y_tr, groups_tr = X.iloc[train_idx], y.iloc[train_idx], person.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]

        clf = train_mod.build_model(**rf_params)
        clf.fit(X_tr, y_tr)
        # threshold สมดุลต้องหาจาก out-of-fold บน train ของ fold นี้เท่านั้น ห้ามแตะคนที่ถูกทดสอบ
        threshold = train_mod.find_balanced_threshold(clf, X_tr, y_tr, groups_tr)
        proba_1 = clf.predict_proba(X_te)[:, list(clf.classes_).index(1)]
        y_pred = (proba_1 >= threshold).astype(int)

        f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
        acc = (y_pred == y_te.to_numpy()).mean()
        fold_scores.append(f1)
        print(f"  fold {fold_idx + 1} (ทดสอบกับคน '{held_out_person}', threshold={threshold:.2f}): "
              f"accuracy={acc:.3f}, macro-F1={f1:.3f}")

        for c, g in pd.DataFrame({"clip": clip.iloc[test_idx].to_numpy(), "true": y_te.to_numpy(),
                                   "p": proba_1}).groupby("clip"):
            clip_rows.append({"clip": c, "truth": clip_truth(c, g["true"]), "mean_p": g["p"].mean()})

    fold_scores = np.array(fold_scores)
    print(f"\n  LOSO macro-F1 เฉลี่ย: {fold_scores.mean():.3f} (+/- {fold_scores.std():.3f})")
    if fold_scores.std() > 0.2:
        print("  ⚠️  ส่วนเบี่ยงเบนสูงมาก — ผลต่างกันมากตามว่าทดสอบกับใคร ข้อมูลยังน้อยเกินกว่าจะสรุปได้")

    cr = pd.DataFrame(clip_rows)
    cr["pred"] = (cr["mean_p"] >= 0.5).astype(int)  # เกณฑ์ดูภาพรวม ไม่ใช่ threshold ของแต่ละ fold (ต่างกันไปตามคน)
    clip_acc = (cr["pred"] == cr["truth"]).mean()
    clip_auc = roc_auc_score(cr["truth"], cr["mean_p"]) if cr["truth"].nunique() > 1 else float("nan")
    print(f"  ระดับคลิป (out-of-fold ทั้งหมด, {len(cr)} คลิป): ทายทั้งคลิปถูก {clip_acc:.3f}, "
          f"ROC AUC (แยกคลิป F/T) {clip_auc:.3f}  <- 0.5 = เดาสุ่ม")
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

    # LOSO ต้องเทรนโมเดลชั่วคราวใหม่ทุก fold (จำเป็นต่อการวัดผลแบบ leave-one-out) — ใช้ hyperparameter
    # ชุดเดียวกับ model.joblib ที่ deploy อยู่จริง (อ่านจากตัวโมเดลเอง ไม่ได้เดา/เขียนทับไฟล์นี้เลย)
    rf_params = {k: v for k, v in model.get_params().items()
                 if k in ("n_estimators", "max_depth", "min_samples_leaf", "max_features")}
    print(f"\n(LOSO จะเทรนโมเดลชั่วคราวด้วยค่าเดียวกับที่ deploy อยู่: {rf_params} — ไม่เขียนทับ {args.model})")

    all_df = prep_mod.load_and_clean(args.landmarks_csv)
    loso_cv(all_df, feature_cols, rf_params)


if __name__ == "__main__":
    main()
