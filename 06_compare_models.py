"""
ขั้นที่ 6 (เสริม): เปรียบเทียบ 3 โมเดล — RandomForest / GradientBoosting / MLP Neural Network
ใช้ข้อมูล ฟีเจอร์ และวิธีวัดผลชุดเดียวกันทั้งหมด (Train/Test split เดิม, GroupKFold เดิม) เพื่อให้
เทียบกันได้ตรง ๆ ไม่ใช่คนละเงื่อนไข

⚠️ ข้อจำกัดที่ต้องบอกไว้ตรงๆ: MLPClassifier ของ sklearn **ไม่รองรับ sample_weight/class_weight**
(RF และ GB รองรับทั้งคู่) เลยใช้ SMOTE oversample คลาส 1 ก่อนเทรน MLP แทนเพื่อแก้ปัญหาความไม่สมดุล
ในทางที่ใกล้เคียงกันที่สุดเท่าที่ทำได้ — ไม่ใช่กลไกเดียวกันเป๊ะ ต้องรู้ไว้ตอนอ่านผล
"""

import time
from importlib import import_module

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, recall_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

prep_mod = import_module("02_prepare_dataset")
FEATURE_COLUMNS = prep_mod.FEATURE_COLUMNS
RANDOM_STATE = 42
CLASS_WEIGHT = {0: 1, 1: 2}  # เดียวกับที่ใช้ตัดสินใจไว้ตอนแก้ bias — ให้ RF/GB ยุติธรรมเท่ากัน


def build_rf():
    return RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=10, max_features="sqrt",
        class_weight=CLASS_WEIGHT, random_state=RANDOM_STATE,
    )


def build_gb():
    return GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=RANDOM_STATE,
    )


def build_mlp():
    # MLP ไม่มี class_weight/sample_weight -> ใช้ SMOTE ในตัว pipeline แทน (fit เฉพาะตอน .fit()
    # เหมือน imblearn.Pipeline ปกติ) + StandardScaler เพราะ neural net ต้องการฟีเจอร์ scale ใกล้กัน
    # (RF/GB ไม่ต้อง scale เพราะเป็นโมเดลอิงลำดับ/threshold ของค่า ไม่ใช่ระยะทาง)
    from imblearn.pipeline import Pipeline as ImbPipeline
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(32, 16), activation="relu", alpha=1e-3,
            max_iter=2000, early_stopping=True, n_iter_no_change=20,
            random_state=RANDOM_STATE,
        )),
    ])


MODELS = {"RandomForest": build_rf, "GradientBoosting": build_gb, "MLP Neural Network": build_mlp}


def get_sample_weight(y, class_weight):
    return y.map(class_weight).to_numpy()


def find_balanced_threshold(build_fn, X, y, groups, use_sample_weight, n_splits=4):
    n_splits = min(n_splits, groups.nunique())
    cv = GroupKFold(n_splits=n_splits)
    splits = list(cv.split(X, y, groups=groups))

    fit_params = {}
    if use_sample_weight:
        sw = get_sample_weight(y, CLASS_WEIGHT)
        # cross_val_predict ไม่รองรับ fit_params ตรงๆ ง่ายๆ กับทุก estimator แบบ pipeline
        # จึงทำ manual out-of-fold loop แทนสำหรับกรณีที่ต้องใช้ sample_weight
        oof_proba = np.zeros(len(y))
        for train_idx, val_idx in splits:
            model = build_fn()
            model.fit(X.iloc[train_idx], y.iloc[train_idx], sample_weight=sw[train_idx])
            oof_proba[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    else:
        oof_proba = cross_val_predict(build_fn(), X, y, cv=splits, method="predict_proba")[:, 1]

    best_t, best_gap = 0.5, 1.0
    for t in np.arange(0.20, 0.81, 0.02):
        pred = (oof_proba >= t).astype(int)
        gap = abs(recall_score(y, pred, pos_label=1) - recall_score(y, pred, pos_label=0))
        if gap < best_gap:
            best_gap, best_t = gap, t
    return float(best_t)


def evaluate_model(name, build_fn, use_sample_weight, X_train, y_train, groups_train, X_test, y_test):
    print(f"\n=== {name} ===")
    t0 = time.time()

    model = build_fn()
    if use_sample_weight:
        sw = get_sample_weight(y_train, CLASS_WEIGHT)
        model.fit(X_train, y_train, sample_weight=sw)
    else:
        model.fit(X_train, y_train)

    threshold = find_balanced_threshold(build_fn, X_train, y_train, groups_train, use_sample_weight)
    elapsed = time.time() - t0

    y_pred_test = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
    report = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)
    acc = report["accuracy"]

    print(f"  threshold={threshold:.2f}  เวลาเทรน={elapsed:.1f}s")
    print(classification_report(y_test, y_pred_test, zero_division=0))

    return {
        "name": name, "model": model, "threshold": threshold,
        "test_accuracy": acc,
        "test_precision_0": report["0"]["precision"], "test_recall_0": report["0"]["recall"], "test_f1_0": report["0"]["f1-score"],
        "test_precision_1": report["1"]["precision"], "test_recall_1": report["1"]["recall"], "test_f1_1": report["1"]["f1-score"],
        "test_macro_f1": report["macro avg"]["f1-score"],
        "train_time_sec": elapsed,
    }


def loso_cv(all_df, build_fn, use_sample_weight, n_max_folds=None):
    X = all_df[FEATURE_COLUMNS]
    y = all_df["Label"]
    groups = all_df["subject_id"]
    n_subjects = groups.nunique()
    cv = GroupKFold(n_splits=n_subjects)

    scores = []
    for train_idx, test_idx in cv.split(X, y, groups=groups):
        model = build_fn()
        if use_sample_weight:
            sw = get_sample_weight(y.iloc[train_idx], CLASS_WEIGHT)
            model.fit(X.iloc[train_idx], y.iloc[train_idx], sample_weight=sw)
        else:
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        scores.append(f1_score(y.iloc[test_idx], pred, average="macro", zero_division=0))

    scores = np.array(scores)
    return scores.mean(), scores.std()


def main():
    train_df = pd.read_parquet("dataset_train.parquet")
    test_df = pd.read_parquet("dataset_test.parquet")
    X_train, y_train, groups_train = train_df[FEATURE_COLUMNS], train_df["Label"], train_df["subject_id"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["Label"]

    use_sample_weight = {"RandomForest": True, "GradientBoosting": True, "MLP Neural Network": False}

    results = []
    for name, build_fn in MODELS.items():
        res = evaluate_model(name, build_fn, use_sample_weight[name], X_train, y_train, groups_train, X_test, y_test)
        results.append(res)

    print("\n\n=== กำลังรัน LOSO CV (21 fold) ทั้ง 3 โมเดล — ใช้เวลาสักครู่ ===")
    all_df = prep_mod.load_and_clean()
    for res in results:
        name = res["name"]
        build_fn = MODELS[name]
        t0 = time.time()
        mean_f1, std_f1 = loso_cv(all_df, build_fn, use_sample_weight[name])
        print(f"{name}: LOSO macro-F1 = {mean_f1:.3f} ± {std_f1:.3f} ({time.time()-t0:.0f}s)")
        res["loso_macro_f1_mean"] = mean_f1
        res["loso_macro_f1_std"] = std_f1

    print("\n\n=== ตารางสรุปเปรียบเทียบ ===")
    summary = pd.DataFrame(results)[[
        "name", "test_accuracy", "test_precision_0", "test_recall_0", "test_f1_0",
        "test_precision_1", "test_recall_1", "test_f1_1", "test_macro_f1",
        "loso_macro_f1_mean", "loso_macro_f1_std", "train_time_sec",
    ]]
    pd.set_option("display.width", 160)
    print(summary.to_string(index=False))
    summary.to_csv("model_comparison.csv", index=False)
    print("\nบันทึก model_comparison.csv")


if __name__ == "__main__":
    main()
