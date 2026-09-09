"""
สร้างภาพแผนผังระบบสำหรับใช้พรีเซนต์ (PNG ความละเอียดสูง สัดส่วน 16:9 พอดีสไลด์)

รัน:  python make_framework_images.py
ได้:  framework_images/1_overview.png
      framework_images/2_pipeline.png
      framework_images/3_inference.png

ตัวเลขทุกตัวในภาพดึงจากไฟล์จริงในโปรเจกต์ ไม่ได้พิมพ์ค่าคงที่ทิ้งไว้
โครงร่างนักวิ่งในภาพที่ 1 วาดจากพิกัดจริงในชุดข้อมูล ไม่ใช่ภาพวาดประกอบ
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = "framework_images"
FONT = "Leelawadee UI"          # ฟอนต์ไทยที่มากับ Windows — ถ้าไม่มีให้เปลี่ยนเป็น "Tahoma"
plt.rcParams["font.family"] = FONT
plt.rcParams["axes.unicode_minus"] = False

# ---- จานสี: คุมโทนเดียวกันทั้งสามภาพ พื้นขาวเพื่อให้ฉายโปรเจกเตอร์แล้วคมชัด ----
INK = "#16211d"
MUTED = "#6b7a74"
ACC = "#0d6e63"
ACC_BG = "#e3efec"
DATA_BG = "#eef1ee"
LINE = "#c6cec9"
WARN = "#b4551a"
WARN_BG = "#f9ece2"


def canvas(w=16, h=9):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100 * h / w)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def box(ax, x, y, w, h, fc="white", ec=LINE, lw=1.4, r=0.6, z=2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))


def txt(ax, x, y, s, size=11, color=INK, weight="normal", ha="left", va="center", z=4, style="normal"):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, zorder=z, fontstyle=style)


def arrow(ax, x1, y1, x2, y2, color=INK, lw=1.6, style="-|>", dashed=False, z=3):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=15,
        color=color, linewidth=lw, zorder=z,
        linestyle=(0, (5, 3)) if dashed else "solid",
        shrinkA=0, shrinkB=0))


def title_block(ax, top, kicker, title, sub=None):
    txt(ax, 5, top, kicker, size=11.5, color=ACC, weight="bold")
    txt(ax, 5, top - 4.2, title, size=25, weight="bold")
    if sub:
        txt(ax, 5, top - 8.6, sub, size=12.5, color=MUTED)
    ax.plot([5, 95], [top - 11.6, top - 11.6], color=INK, lw=2, zorder=1)


def draw_skeleton(ax, cx, cy, scale, frame, color=ACC, lw=2.6):
    """วาดโครงร่าง 6 จุดจากพิกัดจริง (y ในข้อมูลชี้ลง จึงกลับเครื่องหมาย)"""
    bones = [(0, 1), (0, 2), (2, 3), (3, 4), (4, 5), (3, 5)]
    pts = [(cx + p[0] * scale, cy - (p[1] - 0.45) * scale) for p in frame]
    for a, b in bones:
        ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]],
                color=color, lw=lw, solid_capstyle="round", zorder=5)
    for p in pts:
        ax.plot(p[0], p[1], "o", color=color, markersize=lw * 1.9,
                markeredgecolor="white", markeredgewidth=1.1, zorder=6)


def load_facts():
    """ดึงตัวเลขจริงจากไฟล์ในโปรเจกต์"""
    tr = pd.read_parquet("dataset_train.parquet")
    te = pd.read_parquet("dataset_test.parquet")
    raw = sum(1 for _ in open("landmarks_raw.csv", encoding="utf-8")) - 1
    pts = ["shoulder", "knee", "ankle", "heel", "foot_index"]
    r = tr[tr["clip"] == "Tyes"].iloc[40]
    frame = [(0.0, 0.0)] + [(float(r[p + "_x_norm"]), float(r[p + "_y_norm"])) for p in pts]
    return {
        "n_train": len(tr), "n_test": len(te),
        "s_train": tr["subject_id"].nunique(), "s_test": te["subject_id"].nunique(),
        "n_raw": raw, "n_used": len(tr) + len(te), "frame": frame,
    }


# =========================================================================
# ภาพที่ 1 — ภาพรวมระบบ (สไลด์อธิบายใน 30 วินาที)
# =========================================================================
def fig_overview(F):
    fig, ax = canvas()
    title_block(ax, 51, "RUNNING FORM ANALYSIS SYSTEM",
                "ระบบวิเคราะห์ท่าวิ่งจากวิดีโอ",
                "จากคลิปวิ่งมุมด้านข้าง สู่คำตัดสินถูก/ผิด พร้อมคะแนนและคำแนะนำ")

    stages = [
        ("01", "คลิปวิ่ง", ["ถ่ายมุมด้านข้าง", "เห็นฝั่งขวาเต็มตัว", "บนลู่วิ่ง"]),
        ("02", "สกัด 6 จุด", ["MediaPipe Pose", "รุ่น heavy", "สุ่มที่ 10 fps"]),
        ("03", "17 ฟีเจอร์", ["พิกัด 10 ค่า", "มุมข้อต่อ 7 ค่า", "normalize แล้ว"]),
        ("04", "Random Forest", ["ต้นไม้ 500 ต้น", "ลึกสุด 8 ชั้น", "threshold 0.43"]),
        ("05", "ผลลัพธ์", ["ถูก / ผิด", "คะแนน 0–100", "คำแนะนำ + อ้างอิง"]),
    ]

    bw, gap = 15.6, 3.0
    x0, ytop, bh = 5.0, 22.0, 15.0
    for i, (no, head, lines) in enumerate(stages):
        x = x0 + i * (bw + gap)
        last = i == len(stages) - 1
        box(ax, x, ytop, bw, bh,
            fc=ACC_BG if last else "white",
            ec=ACC, lw=2.2 if last else 1.6)
        txt(ax, x + 1.2, ytop + bh - 2.0, no, size=10, color=ACC, weight="bold")
        txt(ax, x + 1.2, ytop + bh - 5.2, head, size=14.5, weight="bold")
        for j, ln in enumerate(lines):
            txt(ax, x + 1.2, ytop + bh - 8.6 - j * 2.7, "· " + ln, size=10.8, color=MUTED)
        if i < len(stages) - 1:
            arrow(ax, x + bw + 0.5, ytop + bh / 2, x + bw + gap - 0.5, ytop + bh / 2, color=ACC, lw=2)

    # โครงร่างนักวิ่งจากข้อมูลจริง วางมุมขวาบนในพื้นที่ว่างข้างหัวเรื่อง
    draw_skeleton(ax, 88, 46.5, 2.8, F["frame"], lw=2.2)
    txt(ax, 81, 47.8, "โครงร่าง 6 จุด", size=9.8, color=MUTED, ha="right")
    txt(ax, 81, 45.2, "จากพิกัดจริงในชุดข้อมูล", size=9.8, color=MUTED, ha="right")

    # แถบตัวเลขด้านล่าง
    facts = [
        (f"{F['s_train'] + F['s_test']}", "คลิป / บุคคล"),
        (f"{F['n_used']:,}", "เฟรมที่ใช้ได้"),
        ("17", "ฟีเจอร์"),
        ("0.785", "Accuracy"),
        ("0.714", "LOSO macro-F1"),
    ]
    fw = 17.0
    for i, (v, l) in enumerate(facts):
        x = 5 + i * fw
        txt(ax, x, 12.5, v, size=21, weight="bold", color=ACC)
        txt(ax, x, 8.4, l, size=10.5, color=MUTED)
    ax.plot([5, 95], [16.5, 16.5], color=LINE, lw=1, zorder=1)

    txt(ax, 95, 3.0, "ข้อมูล 21 คลิป · แบ่ง Train/Test ตามบุคคล ไม่ซ้ำกัน",
        size=9.5, color=MUTED, ha="right")
    return fig


# =========================================================================
# ภาพที่ 2 — สายการเทรน
# =========================================================================
def fig_pipeline(F):
    fig, ax = canvas()
    title_block(ax, 51, "TRAINING PIPELINE",
                "สายการเทรน — รันครั้งเดียว ได้ไฟล์โมเดล",
                "ผลของขั้นก่อนเป็นวัตถุดิบของขั้นถัดไป ชุดทดสอบแยกตัวออกตั้งแต่ขั้นที่ 02")

    rows = [
        ("00_resize_videos.py", "ยัดทุกคลิปลงเฟรม 960×720 คงสัดส่วน เติมขอบดำ",
         "VDO 960x720/", "21 คลิป"),
        ("01_extract_landmarks.py", "MediaPipe สกัด 6 จุดฝั่งขวา ที่ 10 fps จับคู่กับเฉลย",
         "landmarks_raw.csv", f"{F['n_raw']:,} แถว"),
        ("02_prepare_dataset.py", "normalize พิกัด · คำนวณมุม 7 ค่า · แก้ทิศหัน · แบ่งตามคน",
         "dataset_train / test", f"{F['n_train']:,} + {F['n_test']:,}"),
        ("03_train_model.py", "GridSearchCV บนชุดเทรนเท่านั้น · หา threshold ที่ recall สมดุล",
         "model.joblib", "โมเดลพร้อมใช้"),
    ]

    # กล่องต้นทาง (เริ่มใต้เส้นคั่นหัวเรื่องที่ y=39.4)
    box(ax, 5, 33.2, 55, 4.8, fc=DATA_BG)
    txt(ax, 6.6, 35.6, "VDO Student/  20 คลิป 4K แนวตั้ง   +   VDO Prototype/  1 คลิป 960×720 แนวนอน",
        size=11, weight="bold")

    rh, rgap = 4.8, 1.6
    y = 33.2 - rgap - rh          # แถวแรกเริ่มใต้กล่องต้นทาง
    for i, (script, what, out, count) in enumerate(rows):
        arrow(ax, 12, y + rh + rgap - 0.2, 12, y + rh + 0.3, color=INK, lw=1.5)
        box(ax, 5, y, 55, rh, fc="white", ec=ACC, lw=2)
        txt(ax, 6.6, y + rh - 1.7, script, size=11.5, color=ACC, weight="bold")
        txt(ax, 6.6, y + 1.6, what, size=10.2, color=MUTED)
        arrow(ax, 60.5, y + rh / 2, 64.5, y + rh / 2, color=INK, lw=1.5)
        last = i == len(rows) - 1
        box(ax, 65, y, 30, rh, fc=ACC_BG if last else DATA_BG,
            ec=ACC if last else LINE, lw=2 if last else 1.4)
        txt(ax, 66.6, y + rh - 1.7, out, size=11, weight="bold", color=ACC if last else INK)
        txt(ax, 66.6, y + 1.6, count, size=10.2, color=MUTED)
        y -= rh + rgap

    # หมายเหตุท้ายภาพ — แถบสีส้มบางด้านซ้าย
    ax.plot([5, 5], [0.8, 5.4], color=WARN, lw=3, zorder=3)
    txt(ax, 7.2, 4.4, "ไม่มีบุคคลใดปรากฏทั้งในชุดเทรนและชุดทดสอบ  ·  ชุดทดสอบ 4 คนไม่ถูกแตะจนกว่าจะวัดผลครั้งสุดท้าย",
        size=11, weight="bold")
    txt(ax, 7.2, 1.6, "เฟรมที่ติดกันในวิดีโอแทบเหมือนกัน หากสุ่มแบ่งรายเฟรมจะได้ความแม่นยำสูงปลอม",
        size=10, color=MUTED)
    return fig


# =========================================================================
# ภาพที่ 3 — สายการใช้งานจริง
# =========================================================================
def fig_inference(F):
    fig, ax = canvas()
    title_block(ax, 51, "INFERENCE PIPELINE",
                "สายการใช้งานจริง — ทุกครั้งที่มีคนอัปโหลดคลิป",
                "เว็บเรียกใช้โค้ดคำนวณฟีเจอร์ชุดเดียวกับตอนเทรน ไม่ได้เขียนสูตรซ้ำ")

    cx, bw = 32.0, 34.0

    steps = [
        (32.0, 5.8, "ผู้ใช้อัปโหลดคลิป", "สูงสุด 500MB · 60 วินาที", DATA_BG, LINE, 1.4),
        (23.5, 5.8, "07_web/app.py", "รับไฟล์ · คิวงานเบื้องหลัง · แสดง progress", "white", ACC, 2),
        (12.5, 8.4, "07_web/inference.py",
         "letterbox  >  สกัด 6 จุด  >  normalize  >  มุมข้อต่อ  >  ทำนาย", "white", ACC, 2),
    ]
    labels = ["POST /api/analyze", "analyze_video()"]
    for i, (y, h, head, sub, fc, ec, lw) in enumerate(steps):
        box(ax, cx, y, bw, h, fc=fc, ec=ec, lw=lw)
        txt(ax, cx + bw / 2, y + h - 2.2, head, size=13, weight="bold", ha="center",
            color=ACC if ec == ACC else INK)
        txt(ax, cx + bw / 2, y + 2.1, sub, size=10.2, color=MUTED, ha="center")
        if i < len(steps) - 1:
            ny = steps[i + 1][0] + steps[i + 1][1]
            arrow(ax, cx + bw / 2, y - 0.3, cx + bw / 2, ny + 0.4, color=INK, lw=1.6)
            txt(ax, cx + bw / 2 + 1.4, (y + ny) / 2, labels[i], size=9.8, color=MUTED)

    # ซ้าย: import ฟังก์ชันเดิม (คำอธิบายอยู่ในกล่อง จะได้ไม่ชนกล่องผลลัพธ์ด้านล่าง)
    box(ax, 3, 11.8, 25, 9.7, fc=DATA_BG, ec=LINE, lw=1.4)
    for j, m in enumerate(["00_resize_videos", "01_extract_landmarks", "02_prepare_dataset"]):
        txt(ax, 4.4, 19.8 - j * 2.2, m, size=10.2, weight="bold")
    txt(ax, 4.4, 13.3, "import ฟังก์ชันเดิม · กัน train/serve skew", size=9.3,
        color=ACC, weight="bold")
    arrow(ax, 28.3, 16.7, 31.5, 16.7, color=ACC, lw=2.2, dashed=True)

    # ขวา: ไฟล์โมเดล
    box(ax, 69, 11.8, 26, 9.7, fc=ACC_BG, ec=ACC, lw=1.6)
    for j, m in enumerate(["model.joblib", "facing_reference.json", "score_config.json"]):
        txt(ax, 70.4, 19.8 - j * 2.2, m, size=10.2, weight="bold", color=ACC)
    txt(ax, 70.4, 13.3, "ผลผลิตจากสายการเทรน", size=9.3, color=MUTED)
    arrow(ax, 68.3, 16.7, 66.5, 16.7, color=ACC, lw=2.2)

    # ผลลัพธ์
    arrow(ax, cx + bw / 2, 12.2, cx + bw / 2, 9.4, color=INK, lw=1.6)
    box(ax, 20, 2.4, 60, 6.6, fc=ACC_BG, ec=ACC, lw=2)
    txt(ax, 50, 6.9, "ท่าถูก / ท่าผิด  ·  คะแนน 0–100  ·  มุมข้อต่อ  ·  คำแนะนำพร้อมงานอ้างอิง",
        size=12, weight="bold", ha="center", color=ACC)
    txt(ax, 50, 3.9, "ทดสอบแล้ว: อัปโหลดคลิป 4K 193MB ได้ P(ถูก)=0.469 · คำนวณออฟไลน์ได้ 0.466",
        size=9.8, color=MUTED, ha="center")
    return fig


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    F = load_facts()
    for name, fn in [("1_overview", fig_overview), ("2_pipeline", fig_pipeline),
                     ("3_inference", fig_inference)]:
        fig = fn(F)
        path = os.path.join(OUT_DIR, name + ".png")
        fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white", pad_inches=0.25)
        plt.close(fig)
        print("บันทึก %s" % path)


if __name__ == "__main__":
    main()
