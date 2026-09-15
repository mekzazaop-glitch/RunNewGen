# Dockerfile สำหรับรัน backend (FastAPI + MediaPipe) บน Hugging Face Spaces
# Spaces ต้องการให้ app ฟังที่ port 7860 (ตั้งค่าผ่าน env PORT ด้านล่าง)

FROM python:3.12-slim

# libgl1/libglib จำเป็นสำหรับ OpenCV และ MediaPipe บน Linux headless
# libegl1/libgles2 จำเป็นสำหรับ MediaPipe Tasks โดยเฉพาะ (ตอนสร้าง PoseLandmarker จะ dlopen
# libEGL.so.1 แม้จะรันบน CPU ล้วน ถ้าไม่มีจะ error "libEGL.so.1: cannot open shared object file")
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libegl1 \
    libgles2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกเฉพาะไฟล์ที่ backend ใช้งานจริง (ไม่เอาสคริปต์ทดลอง/รูปภาพ/คลิปทดสอบ)
COPY 00_resize_videos.py 01_extract_landmarks.py 02_prepare_dataset.py ./
COPY 07_web ./07_web
# model.joblib = โมเดลที่เทรนแล้ว, pose_landmarker.task = โมเดล MediaPipe
# facing_reference.json = ทิศทางอ้างอิงที่โมเดลเรียนรู้ไว้ (จาก 02_prepare_dataset.py) — ขาดไม่ได้
# score_config.json = ช่วงมุมปกติสำหรับคิดคะแนน
# ood_reference.json = ค่าอ้างอิงสำหรับเตือนคลิปที่ต่างจากข้อมูลเทรนมาก (จาก 08_ood_reference.py)
COPY model.joblib pose_landmarker.task facing_reference.json score_config.json ood_reference.json ./

# Hugging Face Spaces ต้องการให้ container ฟังที่ port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "cd 07_web && uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
