#!/usr/bin/env python3
"""Generate publication-quality IEEE figures for the research paper.

Produces:
1. paper/figures/architecture.png: End-to-end system architecture
2. paper/figures/performance_comparison.png: Experimental configuration comparison
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FIG_DIR = os.path.dirname(os.path.abspath(__file__)) + "/figures"
os.makedirs(FIG_DIR, exist_ok=True)

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

def generate_architecture_figure():
    w, h = 1200, 680
    img = Image.new("RGB", (w, h), color="#FFFFFF")
    draw = ImageDraw.Draw(img)

    f_title = get_font(18, bold=True)
    f_sub = get_font(12, bold=False)
    f_box_h = get_font(13, bold=True)
    f_box_b = get_font(11, bold=False)
    f_tag = get_font(10, bold=True)

    # Title Banner
    draw.text((40, 20), "Fig. 1. End-to-End Multi-Response Consistency & Linguistic Feature Classification Architecture", fill="#1A202C", font=f_title)
    draw.text((40, 48), "Dual-track feature extraction pipeline mapping stochastic LLM generations to tri-class uncertainty categories.", fill="#4A5568", font=f_sub)

    # 1. Input Box (Left)
    draw.rectangle([40, 200, 200, 380], fill="#EDF2F7", outline="#4A5568", width=2)
    draw.text((55, 215), "Input Query", fill="#2D3748", font=f_box_h)
    draw.text((55, 245), "Natural Language\nPrompt (q)", fill="#1A202C", font=f_box_b)
    draw.text((55, 290), "• Determinate Fact\n• Polysemic Query\n• Epistemic Barrier", fill="#718096", font=f_box_b)

    # Arrow 1
    draw.line([200, 290, 250, 290], fill="#2B6CB0", width=3)
    draw.polygon([(250, 290), (240, 284), (240, 296)], fill="#2B6CB0")

    # 2. Generator Box
    draw.rectangle([250, 180, 420, 400], fill="#EBF8FF", outline="#2B6CB0", width=2)
    draw.text((265, 195), "Response Generation", fill="#2B6CB0", font=f_box_h)
    draw.text((265, 225), "LLM Inference Engine\n(Qwen3:4B Local)", fill="#1A202C", font=f_box_b)
    draw.text((265, 275), "Stochastic Sampling:\n• r0: Greedy (T=0.0)\n• r1..r9: T=0.7, p=0.9\n\nTotal: M = 10 responses", fill="#4A5568", font=f_box_b)

    # Branching arrows
    draw.line([420, 250, 470, 170], fill="#2B6CB0", width=3)
    draw.polygon([(470, 170), (460, 170), (465, 180)], fill="#2B6CB0")

    draw.line([420, 330, 470, 410], fill="#2B6CB0", width=3)
    draw.polygon([(470, 410), (465, 400), (460, 410)], fill="#2B6CB0")

    # 3. Track A: Consistency Track (Top)
    draw.rectangle([470, 90, 710, 250], fill="#F0FFF4", outline="#2F855A", width=2)
    draw.text((485, 102), "Track A: Semantic Consistency", fill="#2F855A", font=f_box_h)
    draw.text((485, 128), "1. Sentence Embeddings:\n   all-MiniLM-L6-v2 (384-dim, L2-norm)\n2. Agglomerative Clustering:\n   Cosine distance, threshold tau=0.20\n3. Information-Theoretic Metrics:\n   • Semantic & Normalized Entropy\n   • Agreement Margins (p1, p2, Delta p)\n   • Cluster Counts & Pairwise Sim", fill="#2D3748", font=f_box_b)
    draw.rectangle([630, 95, 700, 115], fill="#C6F6D5")
    draw.text((635, 98), "9 Features", fill="#22543D", font=f_tag)

    # 4. Track B: Linguistic Track (Bottom)
    draw.rectangle([470, 330, 710, 490], fill="#FEFCBF", outline="#B7791F", width=2)
    draw.text((485, 342), "Track B: Ensemble Linguistic", fill="#B7791F", font=f_box_h)
    draw.text((485, 368), "1. Multi-Response Aggregation:\n   Statistical dispersion across 10 outputs\n2. Lexical & Structural Metrics:\n   • Length Mean & Std (len_mean, len_std)\n   • Distinct 1-gram & 2-gram Ratios\n   • Question-Response Overlap\n   • Hedging Score & Negation Count", fill="#2D3748", font=f_box_b)
    draw.rectangle([630, 335, 700, 355], fill="#FEFCBF")
    draw.text((635, 338), "7 Features", fill="#744210", font=f_tag)

    # Merge arrows
    draw.line([710, 170, 760, 270], fill="#2B6CB0", width=3)
    draw.line([710, 410, 760, 310], fill="#2B6CB0", width=3)

    # 5. Fusion Box
    draw.rectangle([760, 230, 890, 350], fill="#EDF2F7", outline="#4A5568", width=2)
    draw.text((770, 245), "Feature Fusion", fill="#2D3748", font=f_box_h)
    draw.text((770, 275), "Concatenation:\n16-Feature Vector\n\nx in R^16\n(StandardScaler)", fill="#1A202C", font=f_box_b)

    # Arrow to Classifier
    draw.line([890, 290, 930, 290], fill="#2B6CB0", width=3)
    draw.polygon([(930, 290), (920, 284), (920, 296)], fill="#2B6CB0")

    # 6. Classifier & Safeguards Box
    draw.rectangle([930, 180, 1070, 400], fill="#FAF5FF", outline="#6B46C1", width=2)
    draw.text((940, 195), "Classification", fill="#6B46C1", font=f_box_h)
    draw.text((940, 225), "Supervised ML:\n• Random Forest\n• Logistic Reg\n• Linear SVM\n\nRule Safeguards:\n• Superlatives\n• Epistemic Guard", fill="#1A202C", font=f_box_b)

    # Arrow to Outputs
    draw.line([1070, 290, 1100, 290], fill="#2B6CB0", width=3)
    draw.polygon([(1100, 290), (1090, 284), (1090, 296)], fill="#2B6CB0")

    # 7. Output Labels (Right)
    draw.rectangle([1100, 190, 1180, 230], fill="#C6F6D5", outline="#2F855A", width=2)
    draw.text((1115, 202), "KNOWN", fill="#22543D", font=f_box_h)

    draw.rectangle([1100, 270, 1180, 310], fill="#FEFCBF", outline="#B7791F", width=2)
    draw.text((1105, 282), "AMBIGUOUS", fill="#744210", font=f_box_h)

    draw.rectangle([1100, 350, 1180, 390], fill="#FED7D7", outline="#C53030", width=2)
    draw.text((1110, 362), "UNKNOWN", fill="#742A2A", font=f_box_h)

    # Bottom notes
    draw.rectangle([40, 530, 1160, 650], fill="#F7FAFC", outline="#E2E8F0", width=1)
    draw.text((55, 540), "METHODOLOGICAL SAFEGUARDS & AUDIT INVARIANTS:", fill="#2D3748", font=f_box_h)
    draw.text((55, 565), "1. Grouped Partitioning: 5-Fold StratifiedGroupKFold on question_group_id guarantees ZERO train/test question leakage.", fill="#4A5568", font=f_box_b)
    draw.text((55, 585), "2. In-Sample Preprocessing: StandardScaler parameters and feature normalizations fitted strictly on training partitions.", fill="#4A5568", font=f_box_b)
    draw.text((55, 605), "3. Statistical Validation: Nadeau-Bengio corrected resampled t-tests, exact McNemar tests, and 2,000-sample bootstrap CIs.", fill="#4A5568", font=f_box_b)
    draw.text((55, 625), "4. Lexical Audit: Tested under Leave-One-Category-Out (15 folds) and 7 entity/domain masking transformations.", fill="#4A5568", font=f_box_b)

    out_path = os.path.join(FIG_DIR, "architecture.png")
    img.save(out_path, dpi=(300, 300))
    print(f"Saved: {out_path}")

def generate_performance_figure():
    w, h = 1000, 580
    img = Image.new("RGB", (w, h), color="#FFFFFF")
    draw = ImageDraw.Draw(img)

    f_title = get_font(18, bold=True)
    f_sub = get_font(12, bold=False)
    f_axis = get_font(12, bold=True)
    f_label = get_font(11, bold=False)
    f_val = get_font(11, bold=True)
    f_tag = get_font(10, bold=False)

    # Title
    draw.text((40, 25), "Fig. 2. Performance Comparison Across Feature Configurations on Batch 2 Combined (N=60)", fill="#1A202C", font=f_title)
    draw.text((40, 52), "Evaluated using 5-Fold StratifiedGroupKFold (Random Forest Classifier). Whiskers denote fold-level std dev.", fill="#4A5568", font=f_sub)

    # Legend
    draw.rectangle([660, 30, 680, 45], fill="#2B6CB0")
    draw.text((688, 30), "Macro-F1 (Mean)", fill="#2D3748", font=f_label)
    draw.rectangle([820, 30, 840, 45], fill="#4FD1C5")
    draw.text((848, 30), "Accuracy (Mean)", fill="#2D3748", font=f_label)

    # Data
    configs = [
        ("True Single\nResponse (r0)", 0.609, 0.109, 0.633),
        ("Single Random\nResponse (r_rand)", 0.543, 0.116, 0.583),
        ("Consistency\nOnly (9 feats)", 0.460, 0.135, 0.500),
        ("Ensemble\nLinguistic (7 feats)", 0.791, 0.209, 0.800),
        ("Full Hybrid\n(All 16 feats)", 0.789, 0.137, 0.800),
    ]

    x0, y0 = 100, 110
    plot_w, plot_h = 840, 390
    x1, y1 = x0 + plot_w, y0 + plot_h

    # Gridlines
    for pct in range(0, 101, 20):
        y_pos = y1 - int(plot_h * (pct / 100.0))
        draw.line([x0, y_pos, x1, y_pos], fill="#EDF2F7", width=1)
        draw.text((x0 - 45, y_pos - 7), f"{pct}%", fill="#A0AEC0", font=f_label)

    # Axis lines
    draw.line([x0, y0, x0, y1], fill="#CBD5E0", width=2)
    draw.line([x0, y1, x1, y1], fill="#CBD5E0", width=2)

    n_groups = len(configs)
    group_width = plot_w / n_groups
    bar_w = 46

    for i, (name, f1_val, f1_std, acc_val) in enumerate(configs):
        center_x = x0 + i * group_width + group_width / 2

        # Macro-F1 bar (Dark Blue)
        b1_x0 = center_x - bar_w - 4
        b1_x1 = center_x - 4
        b1_y0 = y1 - int(plot_h * f1_val)
        draw.rectangle([b1_x0, b1_y0, b1_x1, y1], fill="#2B6CB0")
        draw.text((b1_x0 + 4, b1_y0 - 20), f"{f1_val:.3f}", fill="#2B6CB0", font=f_val)

        # Whiskers for F1
        w_top = max(y0, y1 - int(plot_h * min(1.0, f1_val + f1_std)))
        w_bot = min(y1, y1 - int(plot_h * max(0.0, f1_val - f1_std)))
        mid_b1 = (b1_x0 + b1_x1) / 2
        draw.line([mid_b1, w_top, mid_b1, w_bot], fill="#1A365D", width=2)
        draw.line([mid_b1 - 6, w_top, mid_b1 + 6, w_top], fill="#1A365D", width=2)
        draw.line([mid_b1 - 6, w_bot, mid_b1 + 6, w_bot], fill="#1A365D", width=2)

        # Accuracy bar (Teal)
        b2_x0 = center_x + 4
        b2_x1 = center_x + bar_w + 4
        b2_y0 = y1 - int(plot_h * acc_val)
        draw.rectangle([b2_x0, b2_y0, b2_x1, y1], fill="#4FD1C5")
        draw.text((b2_x0 + 4, b2_y0 - 20), f"{acc_val:.1%}", fill="#234E52", font=f_val)

        # Config label
        lines = name.split("\n")
        draw.text((center_x - 45, y1 + 12), lines[0], fill="#2D3748", font=f_label)
        if len(lines) > 1:
            draw.text((center_x - 45, y1 + 28), lines[1], fill="#718096", font=f_tag)

    # Significance annotation bracket between Single and Full Hybrid
    mid_single = x0 + 0 * group_width + group_width / 2
    mid_hybrid = x0 + 4 * group_width + group_width / 2
    draw.line([mid_single, y0 + 15, mid_hybrid, y0 + 15], fill="#E53E3E", width=2)
    draw.line([mid_single, y0 + 15, mid_single, y0 + 25], fill="#E53E3E", width=2)
    draw.line([mid_hybrid, y0 + 15, mid_hybrid, y0 + 25], fill="#E53E3E", width=2)
    draw.text((mid_single + 180, y0 - 3), "Delta Macro-F1 = +0.180 (McNemar p = 0.0129, Hedges' g = 1.425)", fill="#C53030", font=f_val)

    out_path = os.path.join(FIG_DIR, "performance_comparison.png")
    img.save(out_path, dpi=(300, 300))
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    generate_architecture_figure()
    generate_performance_figure()
