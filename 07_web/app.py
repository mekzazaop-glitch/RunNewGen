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
import time
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

# ถ้าไม่เก็บกวาด JOBS จะโตไม่มีที่สิ้นสุด (แต่ละงานมี result เต็มรวม timeline หลายร้อยจุด) —
# บนเซิร์ฟเวอร์ที่มีแรมจำกัด (Railway 1GB) เปิดทิ้งไว้หลายวันแล้วมีคนทดสอบเยอะอาจกิน RAM จนล่มได้
# ไม่เกี่ยวกับความแม่นยำของโมเดล เป็นแค่การดูแลหน่วยความจำของเซิร์ฟเวอร์
JOB_TTL_SEC = 2 * 60 * 60  # เก็บผลไว้ให้ดึงดู 2 ชั่วโมง พอสำหรับ session ใช้งานจริง


def _prune_old_jobs():
    cutoff = time.time() - JOB_TTL_SEC
    with JOBS_LOCK:
        stale = [jid for jid, j in JOBS.items() if j.get("created", cutoff) < cutoff and j["status"] != "processing"]
        for jid in stale:
            del JOBS[jid]

MAX_UPLOAD_MB = inference.MAX_FILE_SIZE_MB

# จำกัดจำนวนคลิปที่วิเคราะห์พร้อมกัน — ห้ามปล่อยให้ทุก upload เปิด thread วิเคราะห์ของตัวเองทันที
# แต่ละงานสร้าง MediaPipe detector + ตัวถอดรหัสวิดีโอของตัวเอง ใช้แรมสูงสุด ~660MB ต่อคลิป 4K
# เซิร์ฟเวอร์ Railway มีเพดาน 1GB — log จริงเห็น POST /api/analyze 2 ครั้งติดกันแล้วตามด้วย "Killed"
# (สองงานพร้อมกัน ~1.3GB) งานที่ส่งมาทีหลังจึงต้องรอคิวแทน ปรับได้ผ่าน env var ถ้าย้ายไปเครื่องแรมเยอะ
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "1"))
_ANALYZE_SLOTS = threading.Semaphore(MAX_CONCURRENT_JOBS)


def _process_job(job_id, video_path):
    def progress_cb(pct, stage):
        with JOBS_LOCK:
            JOBS[job_id]["progress"] = pct
            JOBS[job_id]["stage"] = stage

    try:
        if not _ANALYZE_SLOTS.acquire(blocking=False):
            progress_cb(0, "รอคิว — มีคลิปอื่นกำลังวิเคราะห์อยู่ จะเริ่มให้อัตโนมัติเมื่อคิวว่าง")
            _ANALYZE_SLOTS.acquire()
        try:
            result = inference.analyze_video(video_path, progress_cb=progress_cb)
        finally:
            _ANALYZE_SLOTS.release()
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
def clean_leftover_uploads():
    """ลบไฟล์วิดีโอค้างใน uploads/ ตอนเริ่มเซิร์ฟเวอร์ — ปกติ _process_job() ลบให้เองทุกงาน แต่ถ้า
    process ถูก kill กลางคัน (เคยเกิดจริงตอนโดน OOM) ไฟล์คลิปหลักร้อย MB จะค้างกินดิสก์ถาวร"""
    removed = 0
    for name in os.listdir(UPLOAD_DIR):
        # ห้ามลบ .gitkeep (ไฟล์ว่างที่ git ใช้รักษาโฟลเดอร์ uploads/ ไว้ใน repo) และ dotfile อื่นๆ —
        # เคยลบจริงตอนรันทดสอบในเครื่อง ทำให้โฟลเดอร์หลุดจาก git ไปเฉยๆ
        if name.startswith("."):
            continue
        try:
            os.remove(os.path.join(UPLOAD_DIR, name))
            removed += 1
        except OSError:
            pass
    if removed:
        print(f"ลบไฟล์อัปโหลดค้างจากรอบก่อน {removed} ไฟล์")


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
    # .webm เพิ่มมาเพื่อรองรับไฟล์ที่หน้าเว็บย่อขนาดให้อัตโนมัติก่อนส่ง (ดู autoDownscale ใน index.html)
    # — เบราว์เซอร์เข้ารหัสวิดีโอเองได้แค่ webm (MediaRecorder) ไม่ใช่ mp4/mov
    if not file.filename.lower().endswith((".mp4", ".mov", ".m4v", ".webm")):
        raise HTTPException(400, "รองรับเฉพาะไฟล์ .mp4 / .mov / .webm")

    _prune_old_jobs()

    job_id = uuid.uuid4().hex
    # ⚠️ ต้องใช้ os.path.basename() ตัด path ออกจากชื่อไฟล์ที่ผู้ใช้ส่งมาก่อนเสมอ — ถ้าเอา
    # file.filename มาต่อ path ตรงๆ ผู้ใช้ที่ตั้งชื่อไฟล์เป็น "../../something" จะเขียนไฟล์หลุดออกจาก
    # UPLOAD_DIR ได้ (path traversal) ก่อนแก้ไม่มีการกรองชื่อไฟล์เลยสักจุด
    safe_name = os.path.basename(file.filename).replace("\\", "_").lstrip(".") or "clip.mp4"
    video_path = os.path.join(UPLOAD_DIR, f"{job_id}_{safe_name}")

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
        JOBS[job_id] = {"status": "processing", "progress": 0, "stage": "เข้าคิวรอประมวลผล",
                         "result": None, "error": None, "created": time.time()}

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
