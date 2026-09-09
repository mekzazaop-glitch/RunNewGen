import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

import สกัดจุด  # ขั้นตอนสกัด landmark ด้วย MediaPipe

# ===== ค่าที่ปรับได้ =====
MODEL_PATH = "model.joblib"

# "prefix" = จำแนกท่าถูก/ผิด (1/0)
# ระวัง: ถ้าเปลี่ยนเป็น "word" ชุด Test จะพัง เพราะมีคำว่า 'not' ที่ไม่เคยอยู่ในชุด Train
TARGET_COLUMN = "prefix"

FORCE_REBUILD_CSV = False  # ตั้ง True ถ้าอยากสกัด landmark ใหม่ทุกครั้งแม้มี CSV เดิมอยู่แล้ว
RANDOM_STATE = 42
N_SPLITS = 5  # จำนวน fold ของ GroupKFold

# คอลัมน์ label/ข้อมูลกำกับ ไม่ใช่ feature — ต้องตัดออกก่อนเทรนเสมอ
# โดยเฉพาะ frame_file ที่เป็นชื่อไฟล์ (string) ถ้าหลุดเข้าไปเป็น feature โมเดลจะ fit ไม่ได้
# เพราะ RandomForest แปลงค่า string เป็นตัวเลขไม่ได้ -> ValueError: could not convert string to float
NON_FEATURE_COLUMNS = ("prefix", "word", "frame_file")


def ensure_dataset(force=FORCE_REBUILD_CSV):
    """ถ้ายังไม่มี CSV ของชุดไหน (หรือสั่ง force) ให้สกัด landmark ของชุดนั้นใหม่"""
    สกัดจุด.build_all(skip_existing=not force)


def clip_name(frame_file_series):
    """ชื่อคลิปต้นทางของแต่ละเฟรม เช่น 'Tbank_0007.jpg' -> 'Tbank'"""
    return frame_file_series.str.replace(r"\.jpg$", "", regex=True).str.rsplit("_", n=1).str[0]


def load_split(split, target_column=TARGET_COLUMN):
    """อ่าน CSV ของชุดข้อมูลหนึ่ง คืน (features, label, ชื่อคลิปของแต่ละแถว)"""
    df = pd.read_csv(สกัดจุด.CSV_NAMES[split])

    feature_columns = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    return df[feature_columns], df[target_column], clip_name(df["frame_file"])


def evaluate(clf, X, y, title):
    """รายงานผลระดับ 'เฟรม' — ดูว่าโมเดลทายแต่ละภาพถูกกี่ %"""
    y_pred = clf.predict(X)

    print(f"\n=== {title} ===")
    print(f"Accuracy (รายเฟรม): {accuracy_score(y, y_pred):.4f}  จาก {len(y)} เฟรม")
    print(classification_report(y, y_pred, zero_division=0))

    return y_pred


def clip_report(y, y_pred, clips):
    """รายงานผลระดับ 'คลิป' — โหวตจากเฟรมทั้งหมดในคลิปนั้น (เสียงข้างมาก)
    ใกล้เคียงการใช้งานจริงมากกว่าดูทีละเฟรม เพราะเวลาใช้จริงเราตัดสินทั้งคลิป ไม่ใช่ทีละภาพ"""
    df = pd.DataFrame({"true": y.values, "pred": y_pred, "clip": clips.values})

    correct = 0
    print("  สรุปรายคลิป (เสียงข้างมากของเฟรมในคลิป):")

    for clip, group in df.groupby("clip"):
        true = group["true"].iloc[0]
        votes = group["pred"].value_counts()
        pred = votes.idxmax()
        confidence = votes.max() / len(group)

        correct += pred == true
        mark = "ถูก" if pred == true else "ผิด <<<"
        print(f"    {clip:12s} จริง={true}  ทาย={pred} ({confidence:.0%} ของ {len(group)} เฟรม)  -> {mark}")

    print(f"  ทายถูก {correct}/{len(df['clip'].unique())} คลิป")


def cross_validate_all_clips(clf):
    """รวมทุกคลิปจากทั้ง 3 ชุด แล้ววัดผลด้วย GroupKFold โดยแบ่งกลุ่มตามคลิป
    เฟรมจากคลิปเดียวกันจะอยู่ fold เดียวกันเสมอ จึงไม่มี data leakage แบบสุ่มระดับเฟรม
    ทำแบบนี้เพราะชุด Vaildation/Test มีแค่ชุดละ 3 คลิป ตัวเลขเหวี่ยงเกินกว่าจะเชื่อได้อย่างเดียว"""
    splits = [load_split(s) for s in สกัดจุด.CSV_NAMES]

    X = pd.concat([s[0] for s in splits], ignore_index=True)
    y = pd.concat([s[1] for s in splits], ignore_index=True)
    groups = pd.concat([s[2] for s in splits], ignore_index=True)

    n_clips = groups.nunique()
    print(f"\n=== GroupKFold {N_SPLITS} fold (รวม {n_clips} คลิป, {len(X)} เฟรม) ===")

    scores = cross_val_score(clf, X, y, groups=groups, cv=GroupKFold(n_splits=N_SPLITS))

    for i, score in enumerate(scores, start=1):
        print(f"  fold {i}: {score:.4f}")
    print(f"  Accuracy เฉลี่ย: {scores.mean():.4f} (+/- {scores.std():.4f})")


def main():
    ensure_dataset()

    X_train, y_train, clips_train = load_split("Train")
    print(f"\nชุด Train: {len(X_train)} เฟรม จาก {clips_train.nunique()} คลิป, "
          f"{X_train.shape[1]} feature, {y_train.nunique()} คลาส ({sorted(y_train.unique())})")

    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)

    # (ก) วัดผลตามโฟลเดอร์ที่แบ่งไว้ — Vaildation/Test คือคลิปที่โมเดลไม่เคยเห็น
    evaluate(clf, X_train, y_train, "Train (ดูว่าโมเดล fit ข้อมูลได้ไหม)")

    for split in ("Vaildation", "Test"):
        X, y, clips = load_split(split)
        y_pred = evaluate(clf, X, y, f"{split} (คลิปที่โมเดลไม่เคยเห็น)")
        clip_report(y, y_pred, clips)

    # (ข) วัดผลข้ามทั้ง 21 คลิป — ตัวเลขอ้างอิงที่เชื่อถือได้กว่า เพราะไม่ได้ขึ้นกับแค่ 3 คลิป
    cross_validate_all_clips(
        RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    )

    joblib.dump(clf, MODEL_PATH)
    print(f"\nบันทึกโมเดลไปที่ {MODEL_PATH}")


if __name__ == "__main__":
    main()
