import cv2
import os
import sys

# terminal บางตัวบน Windows ใช้ codepage cp1252 ซึ่ง print ข้อความภาษาไทยแล้วจะพัง UnicodeEncodeError
# แก้โดยสั่งให้ stdout/stderr เข้ารหัสแบบ utf-8 และแทนอักขระที่พิมพ์ไม่ได้ด้วย ? แทนที่จะ error
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ชุดข้อมูล (โฟลเดอร์คลิปต้นทาง) -> โฟลเดอร์ปลายทางที่จะเก็บเฟรมของชุดนั้น
# แยกโฟลเดอร์ปลายทางตามชุด เพื่อให้ตอนเทรนแยก train / validation / test ออกจากกันได้จริง
SPLIT_DIRS = {
    "Train": "แยกเฟรม train",
    "Vaildation": "แยกเฟรม validation",
    "Test": "แยกเฟรม test",
}
VIDEO_EXTS = (".mov", ".mp4")

TARGET_FPS = 10  # จำนวนเฟรมที่อยากดึงต่อวินาที

# ถ้าคลิปไหนเคยแยกเฟรมไว้แล้ว ให้ข้าม จะได้ไม่ต้องเสียเวลาทำซ้ำตอนรันรอบถัดๆ ไป
SKIP_EXISTING = True


def extract_frames(video_path, target_fps=TARGET_FPS):
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print(f"เปิดวิดีโอไม่ได้: {video_path}")
        return []

    fps = video.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30  # ค่า fallback ถ้าอ่าน fps จากไฟล์ไม่ได้

    frame_interval = max(1, int(fps / target_fps))  # กัน frame_interval = 0

    frame_count = 0
    selected_frames = []

    while True:
        ret, frame = video.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # ตรงนี้ส่ง frame เข้า MediaPipe
            selected_frames.append(frame)

        frame_count += 1

    video.release()
    return selected_frames


def save_frame(frame, out_path):
    """บันทึกภาพลง path ที่อาจมีตัวอักษรไทย
    ห้ามใช้ cv2.imwrite ตรงๆ เพราะบน Windows มันใช้ API แบบ ANSI ภายใน
    ทำให้ path ที่ไม่ใช่ตัวอักษรละติน (เช่นภาษาไทย) เขียนไฟล์ไม่ได้ และคืนค่า False แบบเงียบๆ ไม่ error ให้เห็น
    วิธีแก้คือ encode ภาพเป็น bytes ด้วย cv2 แล้วเขียนไฟล์ด้วย Python's open() ปกติแทน"""
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        print(f"encode ภาพไม่สำเร็จ: {out_path}")
        return False

    with open(out_path, "wb") as f:
        f.write(buf.tobytes())
    return True


def iter_video_files(folders=SPLIT_DIRS):
    """วนหาไฟล์วิดีโอทุกไฟล์ในทุกโฟลเดอร์ที่ระบุ พร้อมคืนชื่อโฟลเดอร์ (label ชุดข้อมูล) และพาธไฟล์"""
    for folder in folders:
        if not os.path.isdir(folder):
            print(f"ไม่พบโฟลเดอร์: {folder} (ข้าม)")
            continue
        for filename in os.listdir(folder):
            if filename.lower().endswith(VIDEO_EXTS):
                yield folder, filename, os.path.join(folder, filename)


if __name__ == "__main__":
    for folder, output_dir in SPLIT_DIRS.items():
        os.makedirs(output_dir, exist_ok=True)  # สร้างโฟลเดอร์ปลายทางถ้ายังไม่มี

        for _, filename, path in iter_video_files([folder]):
            base_name = os.path.splitext(filename)[0]  # ชื่อไฟล์ไม่รวมนามสกุล เช่น Tbank

            # ถ้าเฟรมแรกของคลิปนี้มีอยู่แล้ว แปลว่าเคยแยกไปแล้ว ข้ามไปเลย
            first_frame = os.path.join(output_dir, f"{base_name}_0000.jpg")
            if SKIP_EXISTING and os.path.exists(first_frame):
                print(f"[{folder}] {filename}: มีเฟรมอยู่แล้ว (ข้าม)")
                continue

            frames = extract_frames(path)
            for i, frame in enumerate(frames):
                out_path = os.path.join(output_dir, f"{base_name}_{i:04d}.jpg")
                save_frame(frame, out_path)

            print(f"[{folder}] {filename}: ได้ {len(frames)} เฟรม -> บันทึกที่ '{output_dir}'")
