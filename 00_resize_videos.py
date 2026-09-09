"""
ขั้นที่ 0: ปรับวิดีโอทุกไฟล์ให้เป็นขนาดเดียวกับ Prototype (960x720) ด้วยวิธี letterbox

ปัญหาที่แก้: วิดีโอ Student เป็นแนวตั้ง (เช่น 1080x1920, 2160x3840) แต่ Prototype (Tyes.mp4)
เป็นแนวนอน 960x720 — พิกัดพิกเซลที่สกัดได้จากสองฝั่งเลยอยู่คนละช่วงกัน (ตรวจพบตอนโมเดล
ทำนายคน 'yes' ผิดหมดทุกเฟรม เพราะไม่เคยเห็นพิกัดช่วงนั้นเลยตอนเทรน)

วิธีแก้ (letterbox): ย่อภาพลงคงสัดส่วนเดิม (ไม่ยืด/บีบ ไม่ทำให้สัดส่วนร่างกายผิดเพี้ยน) ให้พอดี
กับกรอบ 960x720 แล้วเติมแถบดำรอบๆ ส่วนที่เหลือ — ทำกับทุกไฟล์เหมือนกันหมด รวมถึง Tyes.mp4
เองด้วย (แม้ขนาดจะตรงอยู่แล้ว) เพื่อให้ผ่าน pipeline เดียวกันแบบเดียวกันทุกไฟล์

⚠️ ไม่เขียนทับไฟล์ต้นฉบับ — วิดีโอดิบเป็นข้อมูลที่ถ่ายมาแล้วกู้คืนไม่ได้ถ้าไฟล์เสีย บันทึกเป็น
ไฟล์ใหม่ในโฟลเดอร์ 'VDO 960x720/' แทน
"""

import argparse
import os
import time

import cv2
import numpy as np

TARGET_W, TARGET_H = 960, 720


def letterbox_resize(frame, target_w=TARGET_W, target_h=TARGET_H):
    """ย่อ frame คงสัดส่วนเดิมให้พอดีกรอบ target แล้วเติมแถบดำรอบๆ ส่วนที่เหลือ (ไม่ยืด/บีบภาพ)"""
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = round(w * scale), round(h * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


def resize_video(src_path, dst_path):
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise IOError(f"เปิดวิดีโอไม่ได้: {src_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(dst_path, fourcc, fps, (TARGET_W, TARGET_H))
    if not writer.isOpened():
        cap.release()
        raise IOError(f"เปิด VideoWriter ไม่ได้: {dst_path}")

    written = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(letterbox_resize(frame))
        written += 1

    cap.release()
    writer.release()
    return written, n_frames, fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-dir", default="data/VDO Student")
    parser.add_argument("--prototype-dir", default="data/VDO Prototype")
    parser.add_argument("--output-dir", default="data/VDO 960x720")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    sources = []
    for folder in (args.student_dir, args.prototype_dir):
        if not os.path.isdir(folder):
            print(f"ไม่พบโฟลเดอร์: {folder} (ข้าม)")
            continue
        for filename in sorted(os.listdir(folder)):
            if filename.lower().endswith((".mov", ".mp4")):
                sources.append(os.path.join(folder, filename))

    print(f"พบวิดีโอทั้งหมด {len(sources)} ไฟล์ (ทั้ง Student + Prototype) เป้าหมาย {TARGET_W}x{TARGET_H}\n")

    for src_path in sources:
        base_name = os.path.splitext(os.path.basename(src_path))[0]
        dst_path = os.path.join(args.output_dir, f"{base_name}.mp4")

        if args.skip_existing and os.path.exists(dst_path):
            print(f"{base_name}: มีไฟล์ผลลัพธ์อยู่แล้ว (ข้าม)")
            continue

        t0 = time.time()
        written, n_frames, fps = resize_video(src_path, dst_path)
        elapsed = time.time() - t0
        print(f"{base_name}: {written}/{n_frames} เฟรม @ {fps:.1f}fps ใช้เวลา {elapsed:.1f}s -> {dst_path}")


if __name__ == "__main__":
    main()
