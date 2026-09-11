# วิธี Deploy ขึ้นเว็บจริง (Vercel + Render)

## ทำไมแยกเป็น 2 ที่ (ไม่ใช้ Vercel ทั้งหมด)

Vercel รองรับ **frontend/static** และ serverless function ที่ทำงานสั้น ๆ (จำกัดเวลา 10-60 วินาที
ต่อ request) แต่ระบบวิเคราะห์ท่าวิ่งของเราใช้เวลา **15-90 วินาทีต่อคลิป** และต้องรัน MediaPipe
(native library ขนาดใหญ่) ซึ่ง**ไม่เข้ากับข้อจำกัดของ Vercel serverless function เลย**

จึงแยกเป็น 2 ส่วน:

| ส่วน | Deploy ที่ไหน | เพราะอะไร |
|---|---|---|
| **หน้าเว็บ** (`web-frontend/index.html`) | **Vercel** | static file ล้วน ไม่มี backend logic |
| **ตัวประมวลผล** (`07_web/`, FastAPI + MediaPipe) | **Render** | รัน process ค้างได้ยาว ไม่จำกัดเวลาต่อ request แบบ serverless |

---

## ขั้นที่ 1 — Deploy Backend ขึ้น Render ก่อน

1. เข้า https://dashboard.render.com (สมัครฟรีด้วย GitHub account)
2. กด **New** → **Blueprint**
3. เลือก repo `RunNewGen` (ต้อง push ขึ้น GitHub ให้เรียบร้อยก่อน)
4. Render จะเจอไฟล์ `render.yaml` ที่เตรียมไว้แล้วอัตโนมัติ กด **Apply** ได้เลย
5. รอ build เสร็จ (ครั้งแรกช้าเพราะต้องโหลด MediaPipe + ไฟล์โมเดลผ่าน Git LFS — อาจใช้เวลา 5-10 นาที)
6. ได้ URL แบบ `https://runnewgen-backend.onrender.com` มา — **จดไว้ ใช้ในขั้นที่ 2**

### ⚠️ ตรวจสอบก่อนใช้จริง (สำคัญมาก)

**เช็คว่าไฟล์โมเดลโหลดมาครบจริง ไม่ใช่แค่ pointer ของ Git LFS:**
เปิด Render Dashboard → service → **Shell** แล้วรัน:
```bash
ls -lh model.joblib pose_landmarker.task
```
ต้องเห็นขนาดไฟล์จริง (`model.joblib` ~6MB, `pose_landmarker.task` ~30MB) **ถ้าเห็นแค่ไม่กี่ร้อยไบต์
แปลว่า Git LFS ดึงไฟล์จริงไม่สำเร็จ** ต้องแก้ก่อนใช้งานได้ (ติดต่อ Render support หรือเปลี่ยนไปใช้
แพลตฟอร์มอื่นที่รองรับ LFS เต็มรูปแบบ เช่น Railway/Fly.io)

**RAM ของแพลนฟรีอาจไม่พอ:**
MediaPipe heavy model + OpenCV + ประมวลผลวิดีโอ 4K ใช้ RAM ค่อนข้างมาก แพลนฟรีของ Render (512MB)
**มีความเสี่ยงสูงที่จะพังกลางทาง** (เซิร์ฟเวอร์ restart เอง / job ค้าง) ระหว่างวิเคราะห์คลิปจริง
ถ้าเจอปัญหานี้ต้องอัปเกรดเป็นแพลนที่ RAM สูงกว่า (เริ่มต้น ~$7/เดือน มี RAM 512MB-2GB ขึ้นกับแพลน)

**แพลนฟรีจะ "หลับ" หลังไม่มีคนใช้ 15 นาที:**
คำขอแรกหลังเซิร์ฟเวอร์หลับจะช้า (cold start ~30-50 วินาที) ก่อนจะตอบสนองปกติ — ปกติของแพลนฟรี
ไม่ใช่บั๊ก ถ้าต้องการให้พร้อมใช้ตลอดเวลาต้องอัปเกรดแพลน

---

## ขั้นที่ 2 — แก้ URL backend ในหน้าเว็บ แล้ว Deploy Frontend ขึ้น Vercel

1. เปิดไฟล์ `web-frontend/index.html`
2. หาบรรทัด:
   ```js
   var API_BASE = 'https://YOUR-BACKEND.onrender.com';
   ```
3. แก้ `YOUR-BACKEND.onrender.com` เป็น URL จริงที่ได้จากขั้นที่ 1 (ไม่ต้องมี `/` ปิดท้าย)
4. เข้า https://vercel.com (สมัครฟรีด้วย GitHub account)
5. กด **Add New → Project** → เลือก repo `RunNewGen`
6. ตั้งค่า **Root Directory** เป็น `web-frontend`
7. Framework Preset เลือก **Other** (ไม่ต้อง build อะไร เป็น static HTML ล้วน)
8. กด **Deploy**
9. ได้โดเมนแบบ `https://your-project.vercel.app` มา — เปิดใช้งานได้เลย

---

## ทดสอบหลัง deploy เสร็จ

1. เปิดโดเมน Vercel ในเบราว์เซอร์
2. ลองอัปโหลดคลิปวิ่งจริง 1 คลิป
3. ถ้าค้างที่ "กำลังอัปโหลด..." ตลอด — เช็ค:
   - Browser DevTools (F12) → tab Console มี error `CORS` หรือ `Failed to fetch` ไหม
   - URL ใน `API_BASE` พิมพ์ถูกไหม (ไม่มี `/` ท้าย, เป็น `https://` ไม่ใช่ `http://`)
   - Backend บน Render ยังรันอยู่ไหม (เข้า Render Dashboard เช็คสถานะ service)

---

## สรุปไฟล์ที่เตรียมไว้ให้แล้ว

| ไฟล์ | หน้าที่ |
|---|---|
| `render.yaml` | ตั้งค่า deploy backend อัตโนมัติบน Render |
| `web-frontend/index.html` | หน้าเว็บที่แก้ให้เรียก backend ข้ามโดเมนได้แล้ว |
| `web-frontend/vercel.json` | ตั้งค่า cache header เล็กน้อยสำหรับ Vercel |
| `requirements.txt` | รายการไลบรารีที่ Render ใช้ตอน build (มีอยู่แล้วจากก่อนหน้านี้) |

**ไฟล์เดิม `07_web/static/index.html` ยังใช้รันทดสอบในเครื่อง (localhost) ได้ตามปกติ ไม่กระทบกัน**
เพราะ `web-frontend/index.html` เป็นไฟล์คัดลอกแยกต่างหากสำหรับ deploy จริงเท่านั้น
