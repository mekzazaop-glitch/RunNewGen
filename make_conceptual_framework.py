"""
สร้างภาพ Conceptual Framework (แก้ไขจากภาพต้นฉบับที่ผู้ใช้ส่งมา) ให้ตรงกับระบบจริง

แก้ไข 3 จุดหลักตามที่ตรวจสอบแล้ว:
  1. เพิ่มขั้น Normalization (hip-relative + scale-invariant + facing-direction correction)
     ซึ่งเป็น contribution ทางเทคนิคที่สำคัญที่สุดของงาน — หายไปในภาพต้นฉบับ
  2. ระบุจำนวนฟีเจอร์ให้ถูกต้อง (17 ฟีเจอร์: พิกัด normalize 10 + มุมข้อต่อ 7)
  3. เพิ่มคะแนน 0-100 ใน Output และทำเครื่องหมาย feedback loop ว่าเป็นงานเสนอ (ยังไม่ implement)

รัน: python make_conceptual_framework.py
ได้: framework_images/conceptual_framework.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

OUT_DIR = "framework_images"
plt.rcParams["font.family"] = "Leelawadee UI"
plt.rcParams["axes.unicode_minus"] = False

# ---- จานสี: อิงโทนเดิมของภาพต้นฉบับ (แต่ละคอลัมน์มีสีต่างกัน) ----
NAVY = "#0d2c54"
INK = "#1c2530"
MUTED = "#5b6672"
WHITE = "#ffffff"

COL = {
    "input":   {"head": "#2f6fb0", "bg": "#eaf2fb", "border": "#2f6fb0"},
    "proc":    {"head": "#3a8a52", "bg": "#eaf7ee", "border": "#3a8a52"},
    "feat":    {"head": "#c98a1a", "bg": "#fdf3e0", "border": "#c98a1a"},
    "ml":      {"head": "#6b3fa0", "bg": "#f2eaf9", "border": "#6b3fa0"},
    "out":     {"head": "#c0392b", "bg": "#fbe9e7", "border": "#c0392b"},
}
NEW_BADGE = "#c0392b"   # สีเน้นจุดที่แก้ไข/เพิ่มใหม่


def canvas(w=16, h=10.4):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100 * h / w)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def rbox(ax, x, y, w, h, fc, ec, lw=1.6, r=0.8, z=2, alpha=1.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z, alpha=alpha))


def txt(ax, x, y, s, size=10, color=INK, weight="normal", ha="center", va="center", z=4, style="normal"):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, zorder=z, fontstyle=style)


def arrow(ax, x1, y1, x2, y2, color=INK, lw=1.8, dashed=False, z=3):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
        color=color, linewidth=lw, zorder=z,
        linestyle=(0, (5, 3)) if dashed else "solid", shrinkA=0, shrinkB=0))


def new_badge(ax, x, y, label="แก้ไข/เพิ่มใหม่"):
    ax.add_patch(Circle((x, y), 1.15, facecolor=NEW_BADGE, edgecolor="white", linewidth=1.2, zorder=6))
    txt(ax, x, y, "!", size=9, color="white", weight="bold", z=7)


def col_header(ax, x, w, ytop, title, colkey):
    c = COL[colkey]
    rbox(ax, x, ytop, w, 5.2, c["head"], c["head"], r=0.6)
    txt(ax, x + w / 2, ytop + 2.6, title, size=12.5, color="white", weight="bold")


def bullet_list(ax, x, y, items, size=8.3, gap=2.55, color=MUTED, w=None):
    for i, it in enumerate(items):
        txt(ax, x, y - i * gap, "•  " + it, size=size, color=color, ha="left", va="top")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = canvas()
    W, H = 100, 65

    # ================= title bar =================
    rbox(ax, 3, H - 8.2, 94, 7.4, NAVY, NAVY, r=0.5)
    txt(ax, 50, H - 4.0, "CONCEPTUAL FRAMEWORK", size=17, color="white", weight="bold")
    txt(ax, 50, H - 6.7, "AI-Based Running Gait Analysis for Injury Prevention in Recreational Runners",
        size=9.5, color="#cfe0f5")

    col_w = 17.6
    gap = 1.2
    x0 = 3
    ytop = H - 11.5
    col_h = 41.5

    cols = ["input", "proc", "feat", "ml", "out"]
    titles = ["Input", "Processing\n(Computer Vision)", "Feature\nRepresentation",
              "Machine Learning", "Output"]
    xs = [x0 + i * (col_w + gap) for i in range(5)]

    # ================= column bodies =================
    for i, key in enumerate(cols):
        c = COL[key]
        rbox(ax, xs[i], ytop - col_h, col_w, col_h, c["bg"], c["border"], lw=1.3, r=0.6, alpha=1)
        rbox(ax, xs[i], ytop, col_w, 4.6, c["head"], c["head"], r=0.5)
        t = titles[i].replace("\n", "\n")
        lines = titles[i].split("\n")
        if len(lines) == 1:
            txt(ax, xs[i] + col_w / 2, ytop + 2.3, lines[0], size=10.5, color="white", weight="bold")
        else:
            txt(ax, xs[i] + col_w / 2, ytop + 3.0, lines[0], size=10.5, color="white", weight="bold")
            txt(ax, xs[i] + col_w / 2, ytop + 1.3, lines[1], size=8.5, color="white", weight="bold")

    # connecting arrows between columns
    for i in range(4):
        ay = ytop - col_h / 2
        arrow(ax, xs[i] + col_w + 0.15, ay, xs[i + 1] - 0.15, ay, color=INK, lw=1.6)

    # ---------------- Input column ----------------
    cx = xs[0] + col_w / 2
    txt(ax, cx, ytop - 3.2, "Running Video", size=9.5, color=INK, weight="bold")
    txt(ax, cx, ytop - 5.4, "(Recreational Runners)", size=7.6, color=MUTED, style="italic")
    rbox(ax, xs[0] + 2.2, ytop - 20, col_w - 4.4, 12.5, WHITE, COL["input"]["border"], lw=1, r=0.5)
    txt(ax, cx, ytop - 13.5, "VIDEO", size=13, color=COL["input"]["border"], weight="bold")
    bullet_list(ax, xs[0] + 1.6, ytop - 23.5,
                ["ถ่ายด้วยกล้อง มุมด้านข้าง (side view)",
                 "วิ่งบนลู่วิ่ง (treadmill) — ควบคุมสภาพ",
                 "เห็นเต็มตัว ตั้งแต่ศีรษะถึงเท้า"], size=8)

    # ---------------- Processing column ----------------
    cx = xs[1] + col_w / 2
    steps_y = ytop - 3.4
    rbox(ax, xs[1] + 1.4, steps_y - 4.4, col_w - 2.8, 4.4, WHITE, COL["proc"]["border"], lw=1, r=0.4)
    txt(ax, cx, steps_y - 1.4, "Frame Sampling", size=8.6, color=INK, weight="bold")
    txt(ax, cx, steps_y - 3.4, "อ่านจากวิดีโอโดยตรง (10 fps)", size=7.2, color=MUTED)

    arrow(ax, cx, steps_y - 4.6, cx, steps_y - 6.2, color=INK, lw=1.3)
    rbox(ax, xs[1] + 1.4, steps_y - 11.0, col_w - 2.8, 4.8, WHITE, COL["proc"]["border"], lw=1, r=0.4)
    txt(ax, cx, steps_y - 7.9, "MediaPipe Pose", size=8.6, color=INK, weight="bold")
    txt(ax, cx, steps_y - 9.9, "รุ่น Heavy · VIDEO mode", size=7.2, color=MUTED)

    arrow(ax, cx, steps_y - 11.2, cx, steps_y - 12.8, color=INK, lw=1.3)
    rbox(ax, xs[1] + 1.4, steps_y - 20.5, col_w - 2.8, 7.7, WHITE, COL["proc"]["border"], lw=1, r=0.4)
    txt(ax, cx, steps_y - 14.7, "Landmark Detection", size=8.6, color=INK, weight="bold")
    txt(ax, cx, steps_y - 16.5, "ฝั่งขวาเท่านั้น (6 จุด):", size=7, color=MUTED)
    txt(ax, cx, steps_y - 18.1, "Shoulder · Hip · Knee", size=7.2, color=INK)
    txt(ax, cx, steps_y - 19.7, "Ankle · Heel · Toe", size=7.2, color=INK)

    # ---------------- Feature Representation column ----------------
    cx = xs[2] + col_w / 2
    fy = ytop - 3.4
    rbox(ax, xs[2] + 1.4, fy - 4.6, col_w - 2.8, 4.6, WHITE, COL["feat"]["border"], lw=1, r=0.4)
    txt(ax, cx, fy - 1.5, "Raw (x, y) Coordinates", size=8.4, color=INK, weight="bold")
    txt(ax, cx, fy - 3.5, "6 จุด × 2 แกน = 12 ค่า", size=7.2, color=MUTED)

    arrow(ax, cx, fy - 4.8, cx, fy - 6.3, color=INK, lw=1.3)

    # normalization — highlighted as the fix
    ny0 = fy - 12.6
    rbox(ax, xs[2] + 1.1, ny0, col_w - 2.2, 6.3, "#fff4d6", NEW_BADGE, lw=1.8, r=0.5)
    txt(ax, cx, ny0 + 4.9, "Normalization", size=8.6, color=INK, weight="bold")
    txt(ax, cx, ny0 + 3.2, "ย้ายจุดกำเนิดไปสะโพก (translation-", size=6.7, color=INK)
    txt(ax, cx, ny0 + 2.0, "invariant) + หารด้วยระยะไหล่-สะโพก", size=6.7, color=INK)
    txt(ax, cx, ny0 + 0.8, "(scale-invariant) + แก้ทิศทางที่หัน", size=6.7, color=INK)
    new_badge(ax, xs[2] + col_w - 1.3, ny0 + 6.0)

    arrow(ax, cx, ny0 - 0.2, cx, ny0 - 1.7, color=INK, lw=1.3)

    ay0 = ny0 - 8.4
    rbox(ax, xs[2] + 1.4, ay0, col_w - 2.8, 6.5, WHITE, COL["feat"]["border"], lw=1, r=0.4)
    txt(ax, cx, ay0 + 5.2, "Joint Angles (7 ค่า)", size=8.4, color=INK, weight="bold")
    txt(ax, cx, ay0 + 3.5, "Knee · Hip · Ankle (unsigned)", size=6.8, color=MUTED)
    txt(ax, cx, ay0 + 2.1, "Trunk lean · Thigh · Shank ·", size=6.8, color=MUTED)
    txt(ax, cx, ay0 + 0.7, "Foot (signed, ต้อง normalize ทิศ)", size=6.8, color=MUTED)

    rbox(ax, xs[2] + 1.4, ay0 - 4.2, col_w - 2.8, 3.2, COL["feat"]["head"], COL["feat"]["head"], r=0.4)
    txt(ax, cx, ay0 - 2.6, "รวม 17 ฟีเจอร์ (10 + 7)", size=8, color="white", weight="bold")

    # ---------------- Machine Learning column ----------------
    cx = xs[3] + col_w / 2
    my = ytop - 3.3
    txt(ax, cx, my, "Classification Models", size=8.8, color=INK, weight="bold")
    for j, (name) in enumerate(["Random Forest", "Gradient Boosting", "MLP Neural Network"]):
        yy = my - 3.0 - j * 3.6
        rbox(ax, xs[3] + 1.6, yy - 2.3, col_w - 3.2, 2.9, WHITE, COL["ml"]["border"], lw=1, r=1.2)
        txt(ax, cx, yy - 0.85, name, size=7.6, color=INK, weight="bold")

    hy = my - 15.0
    txt(ax, cx, hy, "Handling & Calibration", size=8, color=INK, weight="bold")
    rbox(ax, xs[3] + 1.4, hy - 6.4, col_w - 2.8, 5.6, WHITE, COL["ml"]["border"], lw=1, r=0.4)
    txt(ax, cx, hy - 2.0, "class_weight ถ่วงคลาสน้อยกว่า", size=6.8, color=MUTED)
    txt(ax, cx, hy - 3.5, "ปรับ decision threshold ให้ recall", size=6.8, color=MUTED)
    txt(ax, cx, hy - 5.0, "สองคลาสสมดุล (ไม่ใช่ 0.5)", size=6.8, color=MUTED)

    ey = hy - 8.5
    txt(ax, cx, ey, "Model Evaluation", size=8, color=INK, weight="bold")
    rbox(ax, xs[3] + 1.4, ey - 8.0, col_w - 2.8, 7.2, WHITE, COL["ml"]["border"], lw=1, r=0.4)
    bullet_list(ax, xs[3] + 1.9, ey - 1.4,
                ["Accuracy · Precision", "Recall · F1-score",
                 "Leave-One-Subject-", "Out (LOSO) CV"], size=7, gap=1.65)

    # ---------------- Output column ----------------
    cx = xs[4] + col_w / 2
    oy = ytop - 3.3
    rbox(ax, xs[4] + 1.4, oy - 8.0, col_w - 2.8, 7.6, WHITE, COL["out"]["border"], lw=1, r=0.4)
    txt(ax, cx, oy - 1.2, "Gait Classification", size=8.6, color=INK, weight="bold")
    txt(ax, cx - col_w / 4.6, oy - 3.6, "Class 0", size=8, color="#c0392b", weight="bold")
    txt(ax, cx - col_w / 4.6, oy - 5.0, "(ท่าผิด)", size=6.6, color=MUTED)
    txt(ax, cx + col_w / 4.6, oy - 3.6, "Class 1", size=8, color="#2e8b57", weight="bold")
    txt(ax, cx + col_w / 4.6, oy - 5.0, "(ท่าถูก)", size=6.6, color=MUTED)
    txt(ax, cx, oy - 6.9, "+ คะแนน 0–100", size=7.2, color=INK, weight="bold")
    new_badge(ax, xs[4] + col_w - 1.3, oy - 0.4)

    ry0 = oy - 12.4
    rbox(ax, xs[4] + 1.1, ry0 - 5.5, col_w - 2.2, 4.9, "#fdecec", COL["out"]["border"], lw=1, r=0.4)
    txt(ax, cx, ry0 - 1.4, "Recommendation", size=8, color=INK, weight="bold")
    txt(ax, cx, ry0 - 3.0, "คำแนะนำปรับท่า + อ้างอิงงานวิจัย", size=6.6, color=MUTED)
    txt(ax, cx, ry0 - 4.4, "(injury risk reduction — indirect)", size=6.4, color=MUTED, style="italic")

    arrow(ax, cx, oy - 8.2, cx, oy - 9.8, color=INK, lw=1.3)

    # ================= feedback loop (marked as proposed / not implemented) =================
    fby = ytop - col_h - 5.0
    rbox(ax, 8, fby - 4.2, 84, 4.4, "#f0f0f0", "#9aa1ac", lw=1.4, r=0.6)
    txt(ax, 50, fby - 1.2, "Feedback for Continuous Improvement (Runner / Coach / Research)",
        size=9, color=INK, weight="bold")
    txt(ax, 50, fby - 3.2, "[งานที่เสนอไว้สำหรับพัฒนาต่อ — ยังไม่ได้ implement ในระบบปัจจุบัน]",
        size=7.6, color=NEW_BADGE, weight="bold")

    arrow(ax, xs[4] + col_w / 2, ry0 - 5.7, xs[4] + col_w / 2, fby - 4.4 + 4.4, color=MUTED, lw=1.3, dashed=True)
    arrow(ax, xs[4] + col_w / 2, fby - 2.1, 8, fby - 2.1, color=MUTED, lw=1.3, dashed=True)
    arrow(ax, xs[0] + col_w / 2, fby - 2.1, xs[0] + col_w / 2, ytop - col_h - 0.2, color=MUTED, lw=1.3, dashed=True)

    # ================= legend for the fix badge =================
    new_badge(ax, 5.4, 3.3)
    txt(ax, 7.2, 3.3, "จุดที่แก้ไข/เพิ่มจากภาพต้นฉบับ", size=8, color=INK, ha="left", weight="bold")
    txt(ax, 95, 3.3, "RunNewGen · Random Forest (deployed) · LOSO macro-F1 0.709 ± 0.125",
        size=7.5, color=MUTED, ha="right")

    path = os.path.join(OUT_DIR, "conceptual_framework.png")
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white", pad_inches=0.3)
    plt.close(fig)
    print("บันทึก", path)


if __name__ == "__main__":
    main()
