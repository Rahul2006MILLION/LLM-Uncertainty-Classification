CLASSES = ["KNOWN", "AMBIGUOUS", "UNKNOWN"]
from sklearn.metrics import confusion_matrix
#!/usr/bin/env python3
"""Stage 6: Visualization Generator for Generalization and Lexical Shortcut Audit.

Generates:
1. figures/model_comparison.png: Standard Grouped CV vs Category-Held-Out across 5 models.
2. figures/category_performance.png: Accuracy breakdown across 15 held-out categories.
3. figures/lexical_audit.png: Accuracy across 7 lexical transformations under both protocols.
4. figures/confusion_matrices.png: Confusion matrices (Standard CV vs Category-Held-Out).

Rendered using PIL with publication-grade layout and styling.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EXP_DIR, "results")
FIGS_DIR = os.path.join(EXP_DIR, "figures")
os.makedirs(FIGS_DIR, exist_ok=True)

# Try loading standard system font or default bitmap font
def get_font(size: int = 14, bold: bool = False):
    font_paths = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for p in font_paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()

# 1. Model Comparison Figure (Standard CV vs Category-Held-Out)
def generate_figure_model_comparison():
    w, h = 1000, 560
    img = Image.new("RGB", (w, h), color="#FFFFFF")
    draw = ImageDraw.Draw(img)

    f_title = get_font(20, bold=True)
    f_sub = get_font(12, bold=False)
    f_axis = get_font(12, bold=True)
    f_label = get_font(11, bold=False)
    f_val = get_font(11, bold=True)

    # Title block
    draw.text((40, 25), "Model Generalization Comparison: Standard CV vs Category-Held-Out", fill="#1A202C", font=f_title)
    draw.text((40, 55), "Evaluated on Batch 2 Combined (N=60 questions). Performance drop highlights domain memorization.", fill="#718096", font=f_sub)

    # Legend
    draw.rectangle([680, 30, 698, 44], fill="#2B6CB0")
    draw.text((706, 30), "Standard Grouped CV (5 Folds)", fill="#2D3748", font=f_label)
    draw.rectangle([680, 52, 698, 66], fill="#E53E3E")
    draw.text((706, 52), "Category-Held-Out (15 Folds)", fill="#2D3748", font=f_label)

    # Data
    models = [
        ("TF-IDF Linear SVM", 0.8333, 0.6500),
        ("TF-IDF Logistic Reg", 0.8167, 0.5333),
        ("Hybrid A Linear SVM", 0.7500, 0.4833),
        ("Embedding Linear SVM", 0.7333, 0.4833),
        ("Baseline Random Forest", 0.6333, 0.5167),
    ]

    # Plot dimensions
    x0, y0 = 100, 110
    plot_w, plot_h = 840, 380
    x1, y1 = x0 + plot_w, y0 + plot_h

    # Gridlines
    for pct in range(0, 101, 20):
        y_pos = y1 - int(plot_h * (pct / 100.0))
        draw.line([x0, y_pos, x1, y_pos], fill="#EDF2F7", width=1)
        draw.text((x0 - 45, y_pos - 7), f"{pct}%", fill="#A0AEC0", font=f_label)

    # Axis lines
    draw.line([x0, y0, x0, y1], fill="#CBD5E0", width=2)
    draw.line([x0, y1, x1, y1], fill="#CBD5E0", width=2)

    # Draw grouped bars
    n_groups = len(models)
    group_width = plot_w / n_groups
    bar_w = 48

    for i, (m_label, cv_val, ho_val) in enumerate(models):
        center_x = x0 + i * group_width + group_width / 2

        # Standard CV bar (blue)
        b1_x0 = center_x - bar_w - 4
        b1_x1 = center_x - 4
        b1_y0 = y1 - int(plot_h * cv_val)
        draw.rectangle([b1_x0, b1_y0, b1_x1, y1], fill="#2B6CB0")
        draw.text((b1_x0 + 4, b1_y0 - 18), f"{cv_val*100:.1f}%", fill="#2B6CB0", font=f_val)

        # Held-out bar (red)
        b2_x0 = center_x + 4
        b2_x1 = center_x + bar_w + 4
        b2_y0 = y1 - int(plot_h * ho_val)
        draw.rectangle([b2_x0, b2_y0, b2_x1, y1], fill="#E53E3E")
        draw.text((b2_x0 + 4, b2_y0 - 18), f"{ho_val*100:.1f}%", fill="#E53E3E", font=f_val)

        # Drop delta tag
        drop = (ho_val - cv_val) * 100
        draw.text((center_x - 18, y1 + 32), f"Δ {drop:+.1f}%", fill="#C53030", font=f_val)

        # Model label (multi-line)
        parts = m_label.split(" ")
        draw.text((center_x - len(parts[0])*3.5, y1 + 8), parts[0], fill="#2D3748", font=f_axis)
        if len(parts) > 1:
            rest = " ".join(parts[1:])
            draw.text((center_x - len(rest)*3.2, y1 + 20), rest, fill="#4A5568", font=f_label)

    out_path = os.path.join(FIGS_DIR, "model_comparison.png")
    img.save(out_path)
    print(f"Saved {out_path}")

# 2. Category Performance Figure
def generate_figure_category_performance():
    df_cat = pd.read_csv(os.path.join(RESULTS_DIR, "per_category_results.csv"))
    w, h = 1100, 560
    img = Image.new("RGB", (w, h), color="#FFFFFF")
    draw = ImageDraw.Draw(img)

    f_title = get_font(18, bold=True)
    f_sub = get_font(12, bold=False)
    f_axis = get_font(11, bold=True)
    f_label = get_font(10, bold=False)
    f_val = get_font(10, bold=True)

    draw.text((40, 25), "Category-Held-Out Accuracy Breakdown Across 15 Domains (N=60)", fill="#1A202C", font=f_title)
    draw.text((40, 52), "Testing on 4 questions per held-out category with 56 training questions. Zero domain overlap.", fill="#718096", font=f_sub)

    # Legend
    draw.rectangle([750, 30, 768, 44], fill="#2B6CB0")
    draw.text((776, 30), "TF-IDF Linear SVM (Mean: 65.0%)", fill="#2D3748", font=f_label)
    draw.rectangle([750, 50, 768, 64], fill="#48BB78")
    draw.text((776, 50), "Baseline Random Forest (Mean: 51.7%)", fill="#2D3748", font=f_label)

    x0, y0 = 80, 100
    plot_w, plot_h = 960, 380
    x1, y1 = x0 + plot_w, y0 + plot_h

    # Gridlines
    for pct in [0, 25, 50, 75, 100]:
        y_pos = y1 - int(plot_h * (pct / 100.0))
        draw.line([x0, y_pos, x1, y_pos], fill="#EDF2F7", width=1)
        draw.text((x0 - 45, y_pos - 7), f"{pct}%", fill="#A0AEC0", font=f_label)

    draw.line([x0, y0, x0, y1], fill="#CBD5E0", width=2)
    draw.line([x0, y1, x1, y1], fill="#CBD5E0", width=2)

    n_cats = len(df_cat)
    cat_width = plot_w / n_cats
    bar_w = 20

    for i, row in df_cat.iterrows():
        c_x = x0 + i * cat_width + cat_width / 2
        cat_name = row["category"]
        cls = row["target_class"]

        tfidf_acc = row["A_TFIDF_LinearSVC_acc"]
        base_acc = row["B_Baseline_RandomForest_acc"]

        # Bar 1 (TFIDF)
        b1_x0 = c_x - bar_w - 2
        b1_x1 = c_x - 2
        b1_y0 = y1 - int(plot_h * tfidf_acc)
        draw.rectangle([b1_x0, b1_y0, b1_x1, y1], fill="#2B6CB0")

        # Bar 2 (Baseline)
        b2_x0 = c_x + 2
        b2_x1 = c_x + bar_w + 2
        b2_y0 = y1 - int(plot_h * base_acc)
        draw.rectangle([b2_x0, b2_y0, b2_x1, y1], fill="#48BB78")

        # Class color indicator
        cls_color = "#3182CE" if cls == "KNOWN" else ("#DD6B20" if cls == "AMBIGUOUS" else "#805AD5")
        draw.rectangle([c_x - 14, y1 + 5, c_x + 14, y1 + 18], fill=cls_color)
        draw.text((c_x - 12, y1 + 6), cls[:3], fill="#FFFFFF", font=f_label)

        draw.text((c_x - 14, y1 + 22), cat_name, fill="#2D3748", font=f_axis)

    out_path = os.path.join(FIGS_DIR, "category_performance.png")
    img.save(out_path)
    print(f"Saved {out_path}")

# 3. Lexical Audit Transformation Figure
def generate_figure_lexical_audit():
    df_lex = pd.read_csv(os.path.join(RESULTS_DIR, "lexical_audit_results.csv"))
    w, h = 1100, 600
    img = Image.new("RGB", (w, h), color="#FFFFFF")
    draw = ImageDraw.Draw(img)

    f_title = get_font(18, bold=True)
    f_sub = get_font(12, bold=False)
    f_axis = get_font(11, bold=True)
    f_label = get_font(10, bold=False)
    f_val = get_font(10, bold=True)

    draw.text((40, 25), "Lexical Shortcut Audit: Performance Under Systematic Text Masking", fill="#1A202C", font=f_title)
    draw.text((40, 52), "Comparing accuracy under Standard CV (blue) vs Category-Held-Out (red). Dropping domain text collapses CV advantage.", fill="#718096", font=f_sub)

    # Legend
    draw.rectangle([750, 30, 768, 44], fill="#2B6CB0")
    draw.text((776, 30), "Standard Grouped CV (5 Folds)", fill="#2D3748", font=f_label)
    draw.rectangle([750, 52, 768, 66], fill="#E53E3E")
    draw.text((776, 52), "Category-Held-Out (15 Folds)", fill="#2D3748", font=f_label)

    x0, y0 = 80, 100
    plot_w, plot_h = 960, 400
    x1, y1 = x0 + plot_w, y0 + plot_h

    for pct in range(0, 101, 20):
        y_pos = y1 - int(plot_h * (pct / 100.0))
        draw.line([x0, y_pos, x1, y_pos], fill="#EDF2F7", width=1)
        draw.text((x0 - 45, y_pos - 7), f"{pct}%", fill="#A0AEC0", font=f_label)

    draw.line([x0, y0, x0, y1], fill="#CBD5E0", width=2)
    draw.line([x0, y1, x1, y1], fill="#CBD5E0", width=2)

    # Filter to main 7 conditions for visual clarity
    conds = [
        ("Original TF-IDF", "ExpA_Original_TFIDF"),
        ("Stopwords Removed", "ExpB_Stopwords_Removed_TFIDF"),
        ("Entity Masked", "ExpC_Entity_Masked_TFIDF"),
        ("Domain-Term Masked", "ExpD_Domain_Term_Masked_TFIDF"),
        ("Domain-Neutral TF-IDF", "ExpG_Domain_Neutral_TFIDF"),
        ("Epistemic-Only TF-IDF", "ExpE_Epistemic_Only_TFIDF"),
        ("Marker Counts (RF)", "ExpF_Marker_Counts_RandomForest"),
    ]

    n_conds = len(conds)
    col_w = plot_w / n_conds
    bar_w = 34

    for i, (label, exp_id) in enumerate(conds):
        row = df_lex[df_lex["experiment"] == exp_id].iloc[0]
        cv_acc = row["cv_accuracy"]
        ho_acc = row["heldout_accuracy"]

        c_x = x0 + i * col_w + col_w / 2

        # CV bar
        b1_x0 = c_x - bar_w - 3
        b1_x1 = c_x - 3
        b1_y0 = y1 - int(plot_h * cv_acc)
        draw.rectangle([b1_x0, b1_y0, b1_x1, y1], fill="#2B6CB0")
        draw.text((b1_x0 + 2, b1_y0 - 15), f"{cv_acc*100:.1f}%", fill="#2B6CB0", font=f_val)

        # Held-out bar
        b2_x0 = c_x + 3
        b2_x1 = c_x + bar_w + 3
        b2_y0 = y1 - int(plot_h * ho_acc)
        draw.rectangle([b2_x0, b2_y0, b2_x1, y1], fill="#E53E3E")
        draw.text((b2_x0 + 2, b2_y0 - 15), f"{ho_acc*100:.1f}%", fill="#E53E3E", font=f_val)

        # Multi-line label
        words = label.split(" ")
        draw.text((c_x - len(words[0])*3.2, y1 + 8), words[0], fill="#2D3748", font=f_axis)
        if len(words) > 1:
            rest = " ".join(words[1:])
            draw.text((c_x - len(rest)*3.0, y1 + 22), rest, fill="#4A5568", font=f_label)

        # Gap indicator
        gap = (cv_acc - ho_acc) * 100
        draw.text((c_x - 14, y1 + 38), f"-{gap:.1f}%", fill="#C53030", font=f_val)

    out_path = os.path.join(FIGS_DIR, "lexical_audit.png")
    img.save(out_path)
    print(f"Saved {out_path}")

# 4. Confusion Matrices Figure
def generate_figure_confusion_matrices():
    df_oof = pd.read_csv(os.path.join(RESULTS_DIR, "oof_predictions.csv"))
    y_true = df_oof["target_label"].to_numpy()

    cm_cv = confusion_matrix(y_true, df_oof["ExpA_Original_TFIDF_StandardCV"], labels=CLASSES)
    cm_ho = confusion_matrix(y_true, df_oof["ExpA_Original_TFIDF_CategoryHeldOut"], labels=CLASSES)

    w, h = 900, 480
    img = Image.new("RGB", (w, h), color="#FFFFFF")
    draw = ImageDraw.Draw(img)

    f_title = get_font(18, bold=True)
    f_sub = get_font(12, bold=False)
    f_panel = get_font(14, bold=True)
    f_axis = get_font(11, bold=True)
    f_cell = get_font(16, bold=True)
    f_note = get_font(10, bold=False)

    draw.text((40, 25), "Confusion Matrix Shift: TF-IDF Linear SVM", fill="#1A202C", font=f_title)
    draw.text((40, 52), "Standard Grouped CV (83.3% Acc) vs Leave-One-Category-Out Generalization (65.0% Acc)", fill="#718096", font=f_sub)

    def draw_cm(cm_mat, offset_x, title_text, overall_acc):
        draw.text((offset_x + 60, 95), title_text, fill="#2B6CB0" if "Standard" in title_text else "#E53E3E", font=f_panel)
        draw.text((offset_x + 90, 118), f"Overall Accuracy: {overall_acc:.1f}% (N=60)", fill="#4A5568", font=f_note)

        # Headers
        cell_size = 80
        cx0, cy0 = offset_x + 70, 160

        draw.text((cx0 + 80, cy0 - 25), "Predicted Class", fill="#2D3748", font=f_axis)
        for ci, c_name in enumerate(["KNOWN", "AMBIG", "UNKN"]):
            draw.text((cx0 + ci * cell_size + 20, cy0 - 8), c_name, fill="#4A5568", font=f_axis)
            draw.text((cx0 - 55, cy0 + ci * cell_size + 30), c_name, fill="#4A5568", font=f_axis)

        # Cells
        for r_i in range(3):
            for c_i in range(3):
                val = cm_mat[r_i, c_i]
                x_box = cx0 + c_i * cell_size
                y_box = cy0 + r_i * cell_size

                # Cell color: diagonal is green/blue, off-diagonal is red/gray
                if r_i == c_i:
                    intensity = int(180 - val * 6)
                    bg_color = (intensity, 230, intensity)
                else:
                    intensity = int(255 - val * 18)
                    bg_color = (255, intensity, intensity) if val > 0 else (247, 250, 252)

                draw.rectangle([x_box, y_box, x_box + cell_size, y_box + cell_size], fill=bg_color, outline="#CBD5E0", width=1)
                draw.text((x_box + 30, y_box + 28), str(val), fill="#1A202C", font=f_cell)

    draw_cm(cm_cv, 40, "A. Standard Grouped CV", 83.3)
    draw_cm(cm_ho, 480, "B. Category-Held-Out Generalization", 65.0)

    out_path = os.path.join(FIGS_DIR, "confusion_matrices.png")
    img.save(out_path)
    print(f"Saved {out_path}")

def main():
    generate_figure_model_comparison()
    generate_figure_category_performance()
    generate_figure_lexical_audit()
    generate_figure_confusion_matrices()
    print("All 4 figures generated successfully!")

if __name__ == "__main__":
    main()
