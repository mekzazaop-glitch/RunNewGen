"""
ทดลอง (รอบ 2 — เทียบแฟร์ๆ): LOSO CV เต็มรูปแบบสำหรับ ensemble (soft-voting RF+GB+MLP) เทียบกับ
RF เดี่ยว โดยหา decision threshold ที่สมดุล (out-of-fold บน train เท่านั้น) ให้ทั้งสองฝั่ง เหมือนวิธี
ที่โมเดลจริงที่ deploy อยู่ใช้ (03_train_model.py) — ไม่ใช่ threshold 0.5 คงที่แบบรอบแรกที่เทียบแค่
"คุณภาพโมเดลดิบ" แต่ไม่ได้เทียบกับของจริงที่ deploy ตรงๆ

รัน: python 07_ensemble_loso.py > logs/ensemble_loso_log.txt
"""
import time
from importlib import import_module

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import GroupKFold

cmp_mod = import_module("06_compare_models")
prep_mod = import_module("02_prepare_dataset")
FC = prep_mod.FEATURE_COLUMNS

USE_SW = {"RandomForest": True, "GradientBoosting": True, "MLP Neural Network": False}
INNER_SPLITS = 4  # จำนวน fold ภายในสำหรับหา threshold (เหมือน find_balanced_threshold เดิม)


def fit_predict(name, build_fn, X_tr, y_tr, X_va):
    m = build_fn()
    if USE_SW[name]:
        sw = cmp_mod.get_sample_weight(y_tr, cmp_mod.CLASS_WEIGHT)
        m.fit(X_tr, y_tr, sample_weight=sw)
    else:
        m.fit(X_tr, y_tr)
    return m, m.predict_proba(X_va)[:, 1]


def balanced_threshold(oof_proba, y):
    """หา threshold ที่ทำให้ |recall(1) - recall(0)| น้อยที่สุด — สูตรเดียวกับที่ใช้เลือก
    threshold ของโมเดลจริงที่ deploy อยู่ (03_train_model.py) เพื่อเทียบกันแฟร์ๆ"""
    best_t, best_gap = 0.5, 1.0
    for t in np.arange(0.20, 0.81, 0.02):
        pred = (oof_proba >= t).astype(int)
        gap = abs(recall_score(y, pred, pos_label=1, zero_division=0)
                  - recall_score(y, pred, pos_label=0, zero_division=0))
        if gap < best_gap:
            best_gap, best_t = gap, t
    return float(best_t)


def oof_proba_for_ensemble(X_tr, y_tr, groups_tr):
    """เฉลี่ยความน่าจะเป็น out-of-fold ของ 3 โมเดล (fit ใน inner fold ของ train เท่านั้น) —
    ใช้หา threshold ที่สมดุลสำหรับ ensemble โดยไม่แตะข้อมูล test ของ LOSO fold นอกเลย"""
    n_splits = min(INNER_SPLITS, groups_tr.nunique())
    cv = GroupKFold(n_splits=n_splits)
    splits = list(cv.split(X_tr, y_tr, groups=groups_tr))

    oof = {name: np.zeros(len(y_tr)) for name in cmp_mod.MODELS}
    for name, build_fn in cmp_mod.MODELS.items():
        for tr_idx, va_idx in splits:
            _, p = fit_predict(name, build_fn, X_tr.iloc[tr_idx], y_tr.iloc[tr_idx], X_tr.iloc[va_idx])
            oof[name][va_idx] = p
    return np.mean([oof[n] for n in oof], axis=0)


def oof_proba_for_rf(X_tr, y_tr, groups_tr):
    return oof_proba_single("RandomForest", X_tr, y_tr, groups_tr)


def oof_proba_single(name, X_tr, y_tr, groups_tr):
    n_splits = min(INNER_SPLITS, groups_tr.nunique())
    cv = GroupKFold(n_splits=n_splits)
    splits = list(cv.split(X_tr, y_tr, groups=groups_tr))
    build_fn = cmp_mod.MODELS[name]
    oof = np.zeros(len(y_tr))
    for tr_idx, va_idx in splits:
        _, p = fit_predict(name, build_fn, X_tr.iloc[tr_idx], y_tr.iloc[tr_idx], X_tr.iloc[va_idx])
        oof[va_idx] = p
    return oof


def main():
    print("โหลดข้อมูลทั้งหมด (train+val+test รวมกัน) สำหรับ LOSO...")
    all_df = prep_mod.load_and_clean()
    X = all_df[FC]
    y = all_df["Label"]
    groups = all_df["subject_id"]
    n_subjects = groups.nunique()
    cv = GroupKFold(n_splits=n_subjects)

    print(f"\n=== Ensemble LOSO CV ({n_subjects} fold) — RF เทียบ ensemble ด้วย threshold สมดุลทั้งคู่ ===")
    ens_scores, rf_scores = [], []
    t_all = time.time()
    for i, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups=groups), 1):
        subj = groups.iloc[va_idx].iloc[0]
        t0 = time.time()
        X_tr, y_tr, groups_tr = X.iloc[tr_idx], y.iloc[tr_idx], groups.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

        # 1) หา threshold สมดุลของ RF และ ensemble แยกกัน จาก out-of-fold บน train เท่านั้น
        rf_oof = oof_proba_for_rf(X_tr, y_tr, groups_tr)
        ens_oof = oof_proba_for_ensemble(X_tr, y_tr, groups_tr)
        th_rf = balanced_threshold(rf_oof, y_tr)
        th_ens = balanced_threshold(ens_oof, y_tr)

        # 2) fit โมเดลจริงบน train เต็ม แล้วทำนายกับคนที่ถูกตัดออก (fold นอก)
        proba_va = {}
        for name, build_fn in cmp_mod.MODELS.items():
            _, p = fit_predict(name, build_fn, X_tr, y_tr, X_va)
            proba_va[name] = p
        p_ens_va = np.mean([proba_va[n] for n in proba_va], axis=0)

        pred_rf = (proba_va["RandomForest"] >= th_rf).astype(int)
        pred_ens = (p_ens_va >= th_ens).astype(int)

        f1_rf = f1_score(y_va, pred_rf, average="macro", zero_division=0)
        f1_ens = f1_score(y_va, pred_ens, average="macro", zero_division=0)
        rf_scores.append(f1_rf)
        ens_scores.append(f1_ens)

        elapsed = time.time() - t0
        print(f"  fold {i:2d}/{n_subjects} ('{subj}'): RF={f1_rf:.3f} (th={th_rf:.2f})  "
              f"Ensemble={f1_ens:.3f} (th={th_ens:.2f})  "
              f"({'ดีขึ้น' if f1_ens > f1_rf else 'แย่ลง' if f1_ens < f1_rf else 'เท่าเดิม'}) "
              f"[{elapsed:.0f}s]")

    ens_scores = np.array(ens_scores)
    rf_scores = np.array(rf_scores)

    print(f"\n=== สรุป (รวมเวลา {time.time()-t_all:.0f}s) ===")
    print(f"  RF เดี่ยว (threshold สมดุลต่อ fold):      {rf_scores.mean():.3f} ± {rf_scores.std():.3f}")
    print(f"  Ensemble (threshold สมดุลต่อ fold):      {ens_scores.mean():.3f} ± {ens_scores.std():.3f}")
    n_better = int((ens_scores > rf_scores).sum())
    n_worse = int((ens_scores < rf_scores).sum())
    n_same = len(ens_scores) - n_better - n_worse
    print(f"  Ensemble ดีขึ้น {n_better} fold, แย่ลง {n_worse} fold, เท่าเดิม {n_same} fold (จาก {len(ens_scores)})")

    print("\n=== ตัดสินใจ ===")
    diff = ens_scores.mean() - rf_scores.mean()
    if diff > 0.01:
        print(f"  Ensemble ดีขึ้นชัดเจน (+{diff:.3f}) -> แนะนำให้ใช้ ensemble")
    elif diff < -0.01:
        print(f"  Ensemble แย่ลง ({diff:.3f}) -> เก็บ RF เดี่ยวไว้เหมือนเดิม")
    else:
        print(f"  ต่างกันเล็กน้อย ({diff:+.3f}) อยู่ในช่วง noise -> ไม่คุ้มเพิ่มความซับซ้อน เก็บ RF เดี่ยวไว้")


if __name__ == "__main__":
    main()
