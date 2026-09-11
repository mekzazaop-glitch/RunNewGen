"""
สร้างภาพ Research Framework (แก้ไขจากภาพ 8-ขั้น + RQ ที่ผู้ใช้ส่งมา) ให้ตรงกับระบบจริง

แก้ไข 4 จุดตามที่ตรวจสอบแล้ว:
  1. สลับ Class 0/1 ให้ตรงกับ LABEL_MAP จริง (1=correct/ถูก, 0=incorrect/ผิด) — ภาพเดิมสลับกัน
  2. เพิ่มขั้น Normalization (hip-relative + scale-invariant + facing-direction correction)
  3. เพิ่มมุมข้อต่อ 7 ค่าในตาราง Feature Extraction (เดิมมีแค่พิกัด x,y ดิบ)
  4. เพิ่มคะแนน 0-100 ในผลลัพธ์ (Posture Classification)

รัน: python make_research_framework.py
ได้: framework_images/research_framework.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

OUT_DIR = "framework_images"
plt.rcParams["font.family"] = "Leelawadee UI"
plt.rcParams["axes.unicode_minus"] = False

NAVY = "#123a6b"
INK = "#1c2530"
MUTED = "#5b6672"
WHITE = "#ffffff"
GOOD = "#2e8b57"
BAD = "#c0392b"
NEW_BADGE = "#c0392b"

STEP_HEAD = {
    1: "#2f6fb0", 2: "#3a8a52", 3: "#3a8a52", 4: "#c98a1a",
    5: "#3a8a52", 6: "#2f6fb0", 7: "#c0392b",
}
STEP_BG = {
    1: "#eaf2fb", 2: "#eaf7ee", 3: "#eaf7ee", 4: "#fdf3e0",
    5: "#eaf7ee", 6: "#eaf2fb", 7: "#fbe9e7",
}


def canvas(w=17.5, h=14.7):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100 * h / w)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def rbox(ax, x, y, w, h, fc, ec, lw=1.5, r=0.7, z=2, alpha=1.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z, alpha=alpha))


def txt(ax, x, y, s, size=8.5, color=INK, weight="normal", ha="center", va="center", z=4, style="normal"):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, zorder=z, fontstyle=style)


def arrow(ax, x1, y1, x2, y2, color=INK, lw=1.6, dashed=False, z=3):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
        color=color, linewidth=lw, zorder=z,
        linestyle=(0, (5, 3)) if dashed else "solid", shrinkA=0, shrinkB=0))


def new_badge(ax, x, y):
    ax.add_patch(Circle((x, y), 1.05, facecolor=NEW_BADGE, edgecolor="white", linewidth=1.1, zorder=6))
    txt(ax, x, y, "!", size=8.5, color="white", weight="bold", z=7)


def bullets(ax, x, y, items, size=6.6, gap=2.05, color=MUTED, ha="left"):
    for i, it in enumerate(items):
        txt(ax, x, y - i * gap, "•  " + it, size=size, color=color, ha=ha, va="top")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = canvas()
    H = 84.0

    # ================= title =================
    rbox(ax, 2, H - 7.2, 96, 6.6, NAVY, NAVY, r=0.5)
    txt(ax, 50, H - 3.4, "RESEARCH FRAMEWORK", size=16, color="white", weight="bold")
    txt(ax, 50, H - 5.9, "AI-Based Running Gait Analysis for Injury Prevention in Recreational Runners",
        size=9, color="#cfe0f5")

    # ================= 7 step boxes in a row =================
    n = 7
    gap = 0.9
    x0 = 2
    total_w = 96
    bw = (total_w - gap * (n - 1)) / n
    ytop = H - 10.5
    bh = 40.0
    xs = [x0 + i * (bw + gap) for i in range(n)]

    titles = ["1. Running Video\nInput", "2. Frame\nExtraction", "3. Pose Estimation\n(MediaPipe Pose)",
              "4. Feature\nExtraction", "5. Machine\nLearning Model", "6. Posture\nClassification",
              "7. Recommendation"]

    for i in range(n):
        k = i + 1
        rbox(ax, xs[i], ytop - bh, bw, bh, STEP_BG[k], STEP_HEAD[k], lw=1.2, r=0.5)
        rbox(ax, xs[i], ytop, bw, 4.6, STEP_HEAD[k], STEP_HEAD[k], r=0.45)
        lines = titles[i].split("\n")
        if len(lines) == 1:
            txt(ax, xs[i] + bw / 2, ytop + 2.3, lines[0], size=7.6, color="white", weight="bold")
        else:
            txt(ax, xs[i] + bw / 2, ytop + 3.1, lines[0], size=7.6, color="white", weight="bold")
            txt(ax, xs[i] + bw / 2, ytop + 1.3, lines[1], size=7.6, color="white", weight="bold")

    for i in range(n - 1):
        ay = ytop - bh / 2
        arrow(ax, xs[i] + bw + 0.1, ay, xs[i + 1] - 0.1, ay, color=INK, lw=1.4)

    # ---------------- box 1: Running Video Input ----------------
    cx = xs[0] + bw / 2
    txt(ax, cx, ytop - 3.4, "VIDEO", size=11, color=STEP_HEAD[1], weight="bold")
    rbox(ax, xs[0] + 1.1, ytop - 15.5, bw - 2.2, 10.2, WHITE, STEP_HEAD[1], lw=0.9, r=0.4)
    txt(ax, cx, ytop - 6.9, "treadmill · side view", size=6.4, color=MUTED)
    bullets(ax, xs[0] + 0.9, ytop - 18.5,
            ["Video recording of", "recreational runners",
             "Controlled conditions", "(camera position,", "distance, height,",
             "lighting, speed)"], size=6.2, gap=1.85)

    # ---------------- box 2: Frame Extraction ----------------
    cx = xs[1] + bw / 2
    txt(ax, cx, ytop - 4.5, "[ ][ ][ ][ ]", size=9, color=STEP_HEAD[2])
    bullets(ax, xs[1] + 0.9, ytop - 9.0,
            ["Convert video to", "frames",
             "Extract frames at", "predefined rate", "(10 fps)"], size=6.4, gap=1.95)

    # ---------------- box 3: Pose Estimation ----------------
    cx = xs[2] + bw / 2
    txt(ax, cx, ytop - 4.3, "MediaPipe Pose", size=7.2, color=STEP_HEAD[3], weight="bold")
    txt(ax, cx, ytop - 6.1, "(heavy · VIDEO mode)", size=6, color=MUTED, style="italic")
    bullets(ax, xs[2] + 0.9, ytop - 9.6,
            ["Detect body landmarks", "with MediaPipe Pose",
             "Select right-side key", "landmarks: shoulder, hip,", "knee, ankle, heel, toe"],
            size=6.4, gap=1.95)

    # ---------------- box 4: Feature Extraction (real pixel coords + two independent branches) ----------------
    cx = xs[3] + bw / 2
    fy = ytop - 2.6
    # ตารางพิกัด — ค่าพิกเซลจริงจาก 1 เฟรมในชุดข้อมูล (clip 'Tyes') ไม่ใช่ตัวเลขสมมติ
    rows = [("Shoulder", "371.4", "169.6"), ("Hip", "375.0", "350.4"), ("Knee", "318.4", "482.9"),
            ("Ankle", "282.2", "614.4"), ("Heel", "286.1", "621.8"), ("Toe", "214.6", "612.6")]
    tx0, ty0, rowh, colw = xs[3] + 1.0, fy - 1.2, 1.55, [bw * 0.42, bw * 0.29, bw * 0.29]
    txt(ax, tx0, ty0, "Joint", size=5.8, color=MUTED, weight="bold", ha="left")
    txt(ax, tx0 + colw[0], ty0, "x (px)", size=5.8, color=MUTED, weight="bold")
    txt(ax, tx0 + colw[0] + colw[1], ty0, "y (px)", size=5.8, color=MUTED, weight="bold")
    for r, (name, xv, yv) in enumerate(rows):
        ry = ty0 - (r + 1) * rowh
        txt(ax, tx0, ry, name, size=5.8, color=INK, ha="left")
        txt(ax, tx0 + colw[0], ry, xv, size=5.8, color=INK)
        txt(ax, tx0 + colw[0] + colw[1], ry, yv, size=5.8, color=INK)
    txt(ax, cx, ty0 - 7 * rowh - 0.3, "(ตัวอย่างจริง 1 เฟรม จากคลิป Tyes)", size=5.0, color=MUTED, style="italic")

    branch_y = ty0 - 7 * rowh - 2.0   # จุดแยกสองสาย — เน้นว่าพิกัดดิบถูกใช้ 2 ทางที่เป็นอิสระต่อกัน
    left_x, right_x = xs[3] + bw * 0.28, xs[3] + bw * 0.72
    arrow(ax, cx, ty0 - 7 * rowh - 0.7, cx, branch_y, color=INK, lw=1.2)
    ax.plot([left_x, right_x], [branch_y, branch_y], color=INK, lw=1.2, zorder=3)
    arrow(ax, left_x, branch_y, left_x, branch_y - 1.4, color=INK, lw=1.2)
    arrow(ax, right_x, branch_y, right_x, branch_y - 1.4, color=INK, lw=1.2)
    txt(ax, left_x, branch_y - 1.85, "ใช้ตรง ๆ", size=5.2, color=STEP_HEAD[4], weight="bold")
    txt(ax, right_x, branch_y - 1.85, "arccos/atan2", size=4.9, color=STEP_HEAD[4], weight="bold")

    ny0 = branch_y - 2.9
    rbox(ax, xs[3] + 0.7, ny0 - 5.6, bw - 1.4, 5.6, "#fff4d6", NEW_BADGE, lw=1.5, r=0.4)
    txt(ax, cx, ny0 - 1.1, "Normalization", size=6.8, color=INK, weight="bold")
    txt(ax, cx, ny0 - 2.5, "hip-relative origin +", size=5.8, color=INK)
    txt(ax, cx, ny0 - 3.7, "scale-invariant +", size=5.8, color=INK)
    txt(ax, cx, ny0 - 4.9, "facing-direction fix", size=5.8, color=INK)
    new_badge(ax, xs[3] + bw - 1.1, ny0 - 0.2)
    txt(ax, cx, ny0 - 6.3, "= 10 point features", size=5.4, color=MUTED, style="italic")

    ay0 = ny0 - 6.3 - 3.8
    rbox(ax, xs[3] + 0.7, ay0 - 5.4, bw - 1.4, 5.4, WHITE, STEP_HEAD[4], lw=0.9, r=0.4)
    txt(ax, cx, ay0 - 1.0, "Joint Angles (×7)", size=6.6, color=INK, weight="bold")
    txt(ax, cx, ay0 - 2.4, "Knee · Hip · Ankle", size=5.7, color=MUTED)
    txt(ax, cx, ay0 - 3.6, "Trunk · Thigh · Shank", size=5.7, color=MUTED)
    txt(ax, cx, ay0 - 4.8, "Foot  (จาก 2-3 จุด/มุม)", size=5.5, color=MUTED)

    # เส้นประจากฝั่งขวาของจุดแยก วนลงมาเข้ากล่อง Joint Angles โดยตรง — ไม่ผ่านกล่อง Normalization
    # (ตามโค้ดจริง compute_angles() ใช้พิกัดดิบ ไม่ใช่พิกัดที่ normalize แล้ว)
    lane_x = xs[3] + bw - 1.15
    ax.plot([right_x, lane_x], [branch_y - 1.4, branch_y - 1.4], color=MUTED, lw=1.0,
            linestyle=(0, (3, 2)), zorder=3)
    ax.plot([lane_x, lane_x], [branch_y - 1.4, ay0 + 0.4], color=MUTED, lw=1.0,
            linestyle=(0, (3, 2)), zorder=3)
    arrow(ax, lane_x, ay0 + 0.4, xs[3] + bw - 1.6, ay0 - 0.3, color=MUTED, lw=1.0, dashed=True)

    rbox(ax, xs[3] + 0.7, ay0 - 8.0, bw - 1.4, 2.4, STEP_HEAD[4], STEP_HEAD[4], r=0.4)
    txt(ax, cx, ay0 - 6.8, "17 features (10+7)", size=6.4, color="white", weight="bold")

    # ---------------- box 5: Machine Learning Model ----------------
    cx = xs[4] + bw / 2
    my = ytop - 3.2
    for j, name in enumerate(["Random Forest\n(RF)", "Gradient Boosting\n(GB)", "MLP Neural\nNetwork (MLP)"]):
        yy = my - 2.2 - j * 6.1
        rbox(ax, xs[4] + 0.9, yy - 4.6, bw - 1.8, 4.9, WHITE, STEP_HEAD[5], lw=1, r=1.0)
        lines = name.split("\n")
        txt(ax, cx, yy - 1.7, lines[0], size=6.6, color=INK, weight="bold")
        txt(ax, cx, yy - 3.4, lines[1], size=6.6, color=INK, weight="bold")
    bullets(ax, xs[4] + 0.9, my - 20.5, ["Train models on", "labeled data",
            "Compare model", "performance"], size=6.2, gap=1.85)

    # ---------------- box 6: Posture Classification (fixed labels + score) ----------------
    cx = xs[5] + bw / 2
    oy = ytop - 3.0
    rbox(ax, xs[5] + 1.0, oy - 8.6, bw - 2.0, 8.2, WHITE, BAD, lw=1.2, r=0.5)
    txt(ax, cx, oy - 1.3, "Class 0", size=7.6, color=BAD, weight="bold")
    txt(ax, cx, oy - 2.7, "(Improper / ท่าผิด)", size=5.8, color=MUTED)
    txt(ax, cx, oy - 5.0, "Class 1", size=7.6, color=GOOD, weight="bold")
    txt(ax, cx, oy - 6.4, "(Proper / ท่าถูก)", size=5.8, color=MUTED)
    new_badge(ax, xs[5] + bw - 1.0, oy - 0.3)

    rbox(ax, xs[5] + 1.0, oy - 13.3, bw - 2.0, 4.1, STEP_HEAD[6], STEP_HEAD[6], r=0.4)
    txt(ax, cx, oy - 11.0, "Score 0–100", size=7.2, color="white", weight="bold")
    txt(ax, cx, oy - 12.5, "= P(correct) × 100", size=5.8, color="#dbeafe")
    new_badge(ax, xs[5] + bw - 1.0, oy - 9.4)

    # ---------------- box 7: Recommendation ----------------
    cx = xs[6] + bw / 2
    ry = ytop - 4.0
    bullets(ax, xs[6] + 0.9, ry, ["Provide feedback on", "running posture",
            "Suggest improvements", "to gait pattern",
            "Support injury", "prevention for", "recreational runners"],
            size=6.3, gap=1.9)

    # ================= step 8: Performance Evaluation =================
    e_top = ytop - bh - 3.4
    e_h = 15.0
    rbox(ax, 2, e_top - e_h, 96, e_h, "#e9f2e8", "#3a8a52", lw=1.4, r=0.6)
    txt(ax, 6.5, e_top - 2.1, "8. Performance Evaluation", size=10, color=INK, weight="bold", ha="left")

    rq_w = (96 - 3 * 2) / 3 - 1.0
    rq_x = [2 + 2.0 + i * (rq_w + 1.6) for i in range(3)]
    rq_titles = ["Conventional Classification\nPerformance (RQ1)",
                 "Model Comparison\n(RQ2)",
                 "Generalization to Unseen\nRunners (RQ3)"]
    rq_body = [["Accuracy, Precision,", "Recall, F1-score,", "Confusion Matrix"],
               ["Compare RF, GB, and", "MLP using the same", "evaluation metrics"],
               ["Leave-One-Subject-Out", "(LOSO) Cross-Validation,", "Macro F1 and variability", "across runners"]]
    box_bottom = e_top - e_h + 1.2
    box_top_y = e_top - 3.9
    for i in range(3):
        rbox(ax, rq_x[i], box_bottom, rq_w, box_top_y - box_bottom + 1.0, WHITE, "#3a8a52", lw=1, r=0.5)
        lines = rq_titles[i].split("\n")
        txt(ax, rq_x[i] + rq_w / 2, box_top_y + 0.55, lines[0], size=7, color=INK, weight="bold")
        txt(ax, rq_x[i] + rq_w / 2, box_top_y - 0.9, lines[1], size=7, color=INK, weight="bold")
        for j, b in enumerate(rq_body[i]):
            txt(ax, rq_x[i] + rq_w / 2, box_top_y - 3.0 - j * 1.65, b, size=6.2, color=MUTED)

    # ================= bottom tagline =================
    tby = e_top - e_h - 3.4
    rbox(ax, 2, tby - 4.2, 96, 4.4, "#eaf2fb", "#2f6fb0", lw=1.2, r=0.6)
    txt(ax, 50, tby - 2.1, "Towards Accessible Running Gait Analysis for Injury Prevention",
        size=9.5, color=NAVY, weight="bold")

    # ================= legend =================
    ly = tby - 4.2 - 2.6
    new_badge(ax, 4.2, ly)
    txt(ax, 5.8, ly, "จุดที่แก้ไข/เพิ่มจากภาพต้นฉบับ (Class สลับ · เพิ่ม Normalization · มุมข้อต่อ · คะแนน 0-100)",
        size=7.2, color=INK, ha="left", weight="bold")
    txt(ax, 96, ly, "RunNewGen · Random Forest (deployed) · LOSO macro-F1 0.709 ± 0.125",
        size=6.8, color=MUTED, ha="right")

    path = os.path.join(OUT_DIR, "research_framework.png")
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white", pad_inches=0.3)
    plt.close(fig)
    print("บันทึก", path)


if __name__ == "__main__":
    main()
