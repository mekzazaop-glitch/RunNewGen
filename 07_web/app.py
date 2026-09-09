"""
app.py — FastAPI backend: อัปโหลดคลิป -> ประมวลผลเบื้องหลัง -> poll สถานะ -> ดูผล

รันด้วย (จากในโฟลเดอร์ 07_web/):
  python -m uvicorn app:app --reload --port 8000

⚠️ ข้อจำกัดตอนนี้ (ต้องบอกผู้ใช้เว็บให้ชัดเจน):
  - โมเดลเทรนจากคนแค่ ~21 คลิป (20 คน) LOSO macro-F1 = 0.714 ± 0.125 — ยังไม่แม่นยำพอสำหรับ
    การใช้งานจริงจัง เป็น demo ต้นแบบเทคนิคที่ทำถูกวิธี (ไม่มี data leakage, ตรวจ overfit/bias แล้ว)
  - ใช้ได้เฉพาะคลิปที่ถ่ายคล้ายชุดข้อมูลเทรน: ลู่วิ่ง กล้องนิ่ง มุมข้าง เห็นเต็มตัว เห็นฝั่งขวาชัด
"""

import os
import threading
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import inference

APP_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(APP_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="วิเคราะห์ท่าวิ่ง (demo)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# เก็บสถานะงานไว้ในหน่วยความจำ (พอสำหรับ demo คนเดียว/เครื่องเดียว)
JOBS = {}
JOBS_LOCK = threading.Lock()

MAX_UPLOAD_MB = inference.MAX_FILE_SIZE_MB


def _process_job(job_id, video_path):
    def progress_cb(pct, stage):
        with JOBS_LOCK:
            JOBS[job_id]["progress"] = pct
            JOBS[job_id]["stage"] = stage

    try:
        result = inference.analyze_video(video_path, progress_cb=progress_cb)
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "done"
            JOBS[job_id]["result"] = result
    except Exception as e:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = str(e)
    finally:
        try:
            os.remove(video_path)
        except OSError:
            pass


@app.on_event("startup")
def check_model_on_startup():
    try:
        inference.get_model_bundle()
        inference.get_facing_reference()
        inference.get_score_config()
        print("โหลดโมเดลสำเร็จ พร้อมใช้งาน")
    except Exception as e:
        print(f"⚠️  โหลดโมเดลไม่สำเร็จ: {e}")
        print("   เว็บจะเปิดได้แต่ /api/analyze จะ error จนกว่าจะแก้ปัญหานี้")


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4", ".mov", ".m4v")):
        raise HTTPException(400, "รองรับเฉพาะไฟล์ .mp4 / .mov")

    job_id = uuid.uuid4().hex
    video_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")

    size = 0
    with open(video_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_MB * 1024 * 1024:
                f.close()
                os.remove(video_path)
                raise HTTPException(400, f"ไฟล์ใหญ่เกิน {MAX_UPLOAD_MB}MB")
            f.write(chunk)

    with JOBS_LOCK:
        JOBS[job_id] = {"status": "processing", "progress": 0, "stage": "เข้าคิวรอประมวลผล", "result": None, "error": None}

    thread = threading.Thread(target=_process_job, args=(job_id, video_path), daemon=True)
    thread.start()

    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
def status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "ไม่พบงานนี้ (job_id ผิด หรือเซิร์ฟเวอร์รีสตาร์ทไปแล้ว)")
    return {"status": job["status"], "progress": job["progress"], "stage": job["stage"]}


@app.get("/api/result/{job_id}")
def result(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "ไม่พบงานนี้")
    if job["status"] == "processing":
        raise HTTPException(409, "ยังประมวลผลไม่เสร็จ")
    if job["status"] == "failed":
        raise HTTPException(422, job["error"])
    return job["result"]


@app.get("/")
def index():
    return FileResponse(os.path.join(APP_DIR, "static", "index.html"))


app.mount("/static", StaticFiles(directory=os.path.join(APP_DIR, "static")), name="static")
