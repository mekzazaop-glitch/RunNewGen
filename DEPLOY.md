# วิธี Deploy ขึ้นเว็บจริง (Railway)

## 🌐 URL ที่ใช้งานอยู่จริงตอนนี้

**https://runnewgen-web-production.up.railway.app**

URL เดียวได้ครบทั้งหน้าเว็บและตัวประมวลผล เพราะ `07_web/app.py` เสิร์ฟทั้งสองอย่างในตัวเดียวกันอยู่แล้ว
(`GET /` คืนหน้าเว็บ, `POST /api/analyze` รับคลิปไปวิเคราะห์) จึงไม่ต้องตั้งค่า `API_BASE` และไม่มีปัญหา CORS

---

## ทำไมเลือก Railway (ไม่ใช่ Vercel)

| | Vercel | Railway |
|---|---|---|
| รัน MediaPipe + วิเคราะห์ 15-90 วินาที/คลิป | ❌ serverless function จำกัดเวลา ไม่เหมาะกับ native library ใหญ่ | ✅ รัน container ค้างได้ยาว |
| เสิร์ฟหน้าเว็บ + API ใน URL เดียว | ❌ ต้องแยก backend ไปอยู่ที่อื่น | ✅ ได้ในตัวเดียว |
| เข้าถึงได้จากเน็ตในไทย | ⚠️ **ทดสอบแล้วเข้าไม่ได้** — DNS ของ ISP ชี้ `vercel.app` ไปที่ `::1` (localhost) ทำให้หน้าเว็บโหลดไม่ขึ้น | ✅ เข้าได้ปกติ |

> **หมายเหตุสำคัญ:** ตอนทดสอบจากเครื่องที่พัฒนา `nslookup web-frontend-taupe-gamma.vercel.app` คืนค่า
> `::1` และ `125.26.170.3` ซึ่งไม่ใช่ IP จริงของ Vercel (`76.76.21.x`) และ `curl` timeout ทุกครั้ง
> นี่คือสาเหตุที่หน้าเว็บบน Vercel อัปโหลดค้างที่ 0% — ไม่ใช่บั๊กในโค้ด แต่เป็นการบล็อกระดับผู้ให้บริการเน็ต

---

## วิธี deploy ใหม่ (เมื่อแก้โค้ดแล้วอยากอัปขึ้นเว็บ)

รันจาก**รากโปรเจกต์** (ไม่ใช่ในโฟลเดอร์ย่อย):

```bash
npx @railway/cli up --service runnewgen-web --ci
```

ครั้งแรกต้อง login ก่อน (เปิดเบราว์เซอร์ยืนยัน):
```bash
npx @railway/cli login
npx @railway/cli link --project runnewgen --service runnewgen-web --environment production
```

### ⚠️ ห้ามลบ `.railwayignore`

Railway CLI **ไม่อ่าน `.dockerignore`** ตอนอัปโหลด build context — ถ้าไม่มี `.railwayignore` มันจะพยายาม
อัปโหลดคลิปวิดีโอทั้งหมดในโฟลเดอร์ `data/` (รวมกว่า 4GB) แล้ว **CLI จะ crash** ด้วยข้อความ
`assertion failed: buf.len() <= u32::MAX` (เจอมาแล้วตอน deploy จริง)

ต้องมีทั้งสองไฟล์ และเนื้อหาเหมือนกัน:
- `.railwayignore` — Railway ใช้ตอนอัปโหลด
- `.dockerignore` — Docker ใช้ตอน build

---

## ไฟล์ที่เกี่ยวกับการ deploy

| ไฟล์ | หน้าที่ |
|---|---|
| `Dockerfile` | สร้าง image — ติดตั้ง Python 3.12 + ไลบรารีระบบ + คัดลอกเฉพาะไฟล์ที่ backend ใช้จริง |
| `.railwayignore` / `.dockerignore` | กันไม่ให้อัปโหลด/คัดลอกคลิปวิดีโอหลาย GB เข้าไปใน build |
| `requirements.txt` | รายการไลบรารี Python |
| `README_hf_space.md` | metadata เผื่อย้ายไป Hugging Face Spaces (ทางเลือกสำรอง) |

### ไลบรารีระบบที่ขาดไม่ได้ (ใน Dockerfile)

```
libgl1  libglib2.0-0  libegl1  libgles2
```

`libegl1`/`libgles2` จำเป็นแม้จะรันบน CPU ล้วน เพราะ MediaPipe Tasks จะ `dlopen("libEGL.so.1")` ตอน
สร้าง PoseLandmarker ถ้าขาดจะ error `libEGL.so.1: cannot open shared object file` ตอนวิเคราะห์คลิป
(job ขึ้น `failed` ที่ progress 5%) — เจอมาแล้วตอน deploy จริง

### ไฟล์ข้อมูลที่ backend ต้องมีครบ

```
model.joblib            โมเดล Random Forest ที่เทรนแล้ว
pose_landmarker.task    โมเดล MediaPipe (heavy)
facing_reference.json   ทิศทางอ้างอิงที่โมเดลเรียนรู้ไว้ (จาก 02_prepare_dataset.py)
score_config.json       ช่วงมุมปกติสำหรับคิดคะแนน
```

ถ้าขาด `facing_reference.json` เว็บจะเปิดได้แต่ขึ้น `⚠️ โหลดโมเดลไม่สำเร็จ` ใน log และ `/api/analyze`
จะ error ทุกครั้ง — เจอมาแล้วตอน deploy จริงเช่นกัน

---

## ตรวจสอบว่า deploy สำเร็จจริง

```bash
# 1) หน้าเว็บขึ้นไหม
curl -o /dev/null -w "%{http_code}\n" https://runnewgen-web-production.up.railway.app/

# 2) โมเดลโหลดสำเร็จไหม (ต้องเห็น "โหลดโมเดลสำเร็จ พร้อมใช้งาน")
npx @railway/cli logs --service runnewgen-web

# 3) วิเคราะห์ได้จริงไหม (ทดสอบครบวงจร)
curl -X POST https://runnewgen-web-production.up.railway.app/api/analyze \
  -F "file=@data/VDO 960x720/Fatom.mp4;type=video/mp4"
# แล้วเอา job_id ที่ได้ไปเช็ค /api/status/<job_id> จนขึ้น "done" และดูผลที่ /api/result/<job_id>
```

ผลทดสอบล่าสุด: อัปโหลด `Fatom.mp4` (15MB) → วิเคราะห์ 306 เฟรม → `status: done` พร้อมคะแนนและคำแนะนำครบ 6 มุม

---

## ทางเลือกสำรอง: Hugging Face Spaces

ถ้าต้องการย้าย (เช่น Railway หมดเครดิต) — Spaces ให้ RAM มากกว่าและฟรีถาวร:

1. สร้าง Space ใหม่ที่ https://huggingface.co/new-space → เลือก **SDK = Docker**
2. อัปโหลดไฟล์เหล่านี้ขึ้นไป: `Dockerfile`, `requirements.txt`, `00_resize_videos.py`,
   `01_extract_landmarks.py`, `02_prepare_dataset.py`, `07_web/`, `model.joblib`,
   `pose_landmarker.task`, `facing_reference.json`, `score_config.json`
3. เปลี่ยนชื่อ `README_hf_space.md` เป็น `README.md` (Spaces อ่าน metadata จากหัวไฟล์นี้)
4. Space จะ build เอง — Dockerfile ตั้ง port 7860 ไว้ตรงกับที่ Spaces ต้องการอยู่แล้ว
