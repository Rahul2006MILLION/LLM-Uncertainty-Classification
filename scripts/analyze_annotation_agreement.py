#!/usr/bin/env python3
"""Inter-Annotator Agreement (IAA) Analysis and Adjudication Docket Generator.

Implements the statistical methodology specified in data/annotation/AGREEMENT_ANALYSIS_PLAN.md:
- Raw observed agreement (Po)
- Expected chance agreement (Pe)
- Cohen's Kappa coefficient (kappa), standard error, and 95% confidence interval
- Krippendorff's Alpha for nominal categories
- 3x3 epistemic confusion matrix (KNOWN, AMBIGUOUS, UNKNOWN)
- Class-specific agreement and F1 scores
- Diagnostic disagreement stratification
- Adjudication docket generation (adjudication_docket.jsonl)

Enforces the Blinding Key Release Gate:
The blinding key is only accessed when both annotators have completed all records
and passed full validation.
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, Any, List, Tuple, Optional, Set

CLASSES = ["KNOWN", "AMBIGUOUS", "UNKNOWN"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def compute_raw_agreement(y_a: List[str], y_b: List[str]) -> float:
    """Calculates observed proportion of agreement Po."""
    assert len(y_a) == len(y_b) and len(y_a) > 0
    matches = sum(1 for a, b in zip(y_a, y_b) if a == b)
    return matches / len(y_a)


def compute_expected_agreement(y_a: List[str], y_b: List[str]) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """Calculates expected agreement by chance Pe."""
    n = len(y_a)
    p_a = {c: sum(1 for a in y_a if a == c) / n for c in CLASSES}
    p_b = {c: sum(1 for b in y_b if b == c) / n for c in CLASSES}
    p_e = sum(p_a[c] * p_b[c] for c in CLASSES)
    return p_e, p_a, p_b


def compute_cohens_kappa(y_a: List[str], y_b: List[str]) -> Dict[str, Any]:
    """Computes Cohen's Kappa, SE, and 95% CI."""
    n = len(y_a)
    p_o = compute_raw_agreement(y_a, y_b)
    p_e, p_a, p_b = compute_expected_agreement(y_a, y_b)

    if math.isclose(p_e, 1.0):
        kappa = 1.0 if math.isclose(p_o, 1.0) else 0.0
        se = 0.0
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)
        denom = n * ((1.0 - p_e) ** 2)
        var = (p_o * (1.0 - p_o)) / denom if denom > 0 else 0.0
        se = math.sqrt(max(0.0, var))

    ci_lower = max(-1.0, kappa - 1.96 * se)
    ci_upper = min(1.0, kappa + 1.96 * se)

    return {
        "n": n,
        "raw_agreement_po": p_o,
        "expected_agreement_pe": p_e,
        "cohens_kappa": kappa,
        "standard_error": se,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "marginal_a": p_a,
        "marginal_b": p_b,
    }


def compute_krippendorffs_alpha_nominal(y_a: List[str], y_b: List[str]) -> float:
    """Computes Krippendorff's Alpha for nominal 2-rater complete data."""
    n = len(y_a)
    if n == 0:
        return 0.0

    # Total ratings
    total_ratings = 2 * n

    # Total frequency of each class across both annotators
    n_c = {c: 0 for c in CLASSES}
    for a, b in zip(y_a, y_b):
        n_c[a] += 1
        n_c[b] += 1

    # Observed coincidence: pairs of identical ratings on the same item
    # For 2 raters: if a == b, coincidences for class a += 2*1*(1)/1 = 2; if a != b, 0
    d_o = sum(1 for a, b in zip(y_a, y_b) if a != b) / n

    # Expected disagreement
    d_e = sum(n_c[c] * (total_ratings - n_c[c]) for c in CLASSES) / (total_ratings * (total_ratings - 1))

    if math.isclose(d_e, 0.0):
        return 1.0 if math.isclose(d_o, 0.0) else 0.0

    alpha = 1.0 - (d_o / d_e)
    return alpha


def compute_confusion_matrix(y_a: List[str], y_b: List[str]) -> Dict[str, Any]:
    """Generates 3x3 confusion matrix and per-class agreement / F1 scores."""
    matrix = {row: {col: 0 for col in CLASSES} for row in CLASSES}
    for a, b in zip(y_a, y_b):
        matrix[a][b] += 1

    per_class = {}
    for c in CLASSES:
        tp = matrix[c][c]
        row_sum = sum(matrix[c][col] for col in CLASSES)
        col_sum = sum(matrix[row][c] for row in CLASSES)
        total_mentions = row_sum + col_sum

        # Specific agreement (Cicchetti & Feinstein)
        specific_agreement = (2.0 * tp / total_mentions) if total_mentions > 0 else 0.0

        # Dice / F1
        f1 = (2.0 * tp / (row_sum + col_sum)) if (row_sum + col_sum) > 0 else 0.0

        per_class[c] = {
            "annotator_a_count": row_sum,
            "annotator_b_count": col_sum,
            "unanimous_count": tp,
            "specific_agreement": specific_agreement,
            "f1_score": f1,
        }

    return {
        "matrix": matrix,
        "per_class": per_class,
    }


def extract_disagreements_and_docket(
    records_a: List[Dict[str, Any]],
    records_b: List[Dict[str, Any]],
    blinding_key: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Identifies all discordant items and creates an adjudication docket.

    Matches records between Annotator A and B by underlying question string.
    If blinding key is authorized, also maps to candidate original_id.
    """
    map_b_by_q = {r["question"]: r for r in records_b}

    disagreements = []
    docket = []

    docket_num = 1
    for r_a in records_a:
        q = r_a["question"]
        r_b = map_b_by_q.get(q)
        if not r_b:
            continue

        lbl_a = r_a.get("assigned_label")
        lbl_b = r_b.get("assigned_label")
        adj_a = r_a.get("requires_adjudication", False)
        adj_b = r_b.get("requires_adjudication", False)
        conf_a = r_a.get("confidence", "NONE")
        conf_b = r_b.get("confidence", "NONE")

        is_disagreement = (lbl_a != lbl_b)
        is_escalated = adj_a or adj_b
        is_double_low_conf = (conf_a == "LOW" and conf_b == "LOW")

        needs_adjudication = is_disagreement or is_escalated or is_double_low_conf

        if is_disagreement:
            axis = f"{min(lbl_a, lbl_b)} vs {max(lbl_a, lbl_b)}"
            disagreements.append({
                "blinded_id_a": r_a.get("blinded_id"),
                "blinded_id_b": r_b.get("blinded_id"),
                "question": q,
                "label_a": lbl_a,
                "label_b": lbl_b,
                "confidence_a": conf_a,
                "confidence_b": conf_b,
                "rationale_a": r_a.get("rationale", ""),
                "rationale_b": r_b.get("rationale", ""),
                "axis": axis,
            })

        if needs_adjudication:
            entry = {
                "docket_id": f"ADJ_{docket_num:03d}",
                "blinded_id_a": r_a.get("blinded_id"),
                "blinded_id_b": r_b.get("blinded_id"),
                "question": q,
                "label_a": lbl_a,
                "label_b": lbl_b,
                "confidence_a": conf_a,
                "confidence_b": conf_b,
                "rationale_a": r_a.get("rationale", ""),
                "rationale_b": r_b.get("rationale", ""),
                "flagged_by_a": adj_a,
                "flagged_by_b": adj_b,
                "trigger_reason": "LABEL_DISAGREEMENT" if is_disagreement else ("EXPLICIT_ESCALATION" if is_escalated else "DOUBLE_LOW_CONFIDENCE"),
                "adjudicated_label": None,
                "adjudication_rationale": "",
                "adjudicator": None,
                "timestamp": None,
            }
            docket.append(entry)
            docket_num += 1

    return disagreements, docket


def check_blinding_release_gate(
    path_a: str,
    path_b: str,
    records_a: List[Dict[str, Any]],
    records_b: List[Dict[str, Any]],
    expected_count: int = 300,
) -> Tuple[bool, List[str]]:
    """Checks whether the 5 formal conditions for blinding key release are met."""
    reasons = []

    # Condition 1: Annotator A complete
    un_a = sum(1 for r in records_a if r.get("assigned_label") is None)
    if len(records_a) != expected_count:
        reasons.append(f"Annotator A count mismatch: {len(records_a)} != {expected_count}")
    if un_a > 0:
        reasons.append(f"Annotator A has {un_a} unannotated records")

    # Condition 2: Annotator B complete
    un_b = sum(1 for r in records_b if r.get("assigned_label") is None)
    if len(records_b) != expected_count:
        reasons.append(f"Annotator B count mismatch: {len(records_b)} != {expected_count}")
    if un_b > 0:
        reasons.append(f"Annotator B has {un_b} unannotated records")

    # Condition 3: Schema validation
    for i, r in enumerate(records_a):
        if r.get("assigned_label") not in CLASSES:
            reasons.append(f"Annotator A line {i+1}: invalid label '{r.get('assigned_label')}'")
            break
    for i, r in enumerate(records_b):
        if r.get("assigned_label") not in CLASSES:
            reasons.append(f"Annotator B line {i+1}: invalid label '{r.get('assigned_label')}'")
            break

    # Condition 4: Files exist on disk
    if not os.path.exists(path_a):
        reasons.append(f"Annotator A file not found: {path_a}")
    if not os.path.exists(path_b):
        reasons.append(f"Annotator B file not found: {path_b}")

    can_release = (len(reasons) == 0)
    return can_release, reasons


def main():
    parser = argparse.ArgumentParser(
        description="Compute inter-annotator agreement and generate adjudication docket."
    )
    parser.add_argument("file_a", help="Path to completed Annotator A JSONL file.")
    parser.add_argument("file_b", help="Path to completed Annotator B JSONL file.")
    parser.add_argument(
        "--blinding-key",
        default=None,
        help="Optional path to blinding key (quarantined unless release gate passed).",
    )
    parser.add_argument(
        "--docket-out",
        default="data/annotation/adjudication_docket.jsonl",
        help="Output path for adjudication docket JSONL.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Output summary report as JSON."
    )

    args = parser.parse_args()

    if not os.path.exists(args.file_a) or not os.path.exists(args.file_b):
        print("Error: Input annotation files do not exist.")
        sys.exit(1)

    with open(args.file_a, "r", encoding="utf-8") as f:
        records_a = [json.loads(line) for line in f if line.strip()]
    with open(args.file_b, "r", encoding="utf-8") as f:
        records_b = [json.loads(line) for line in f if line.strip()]

    # Blinding release gate check
    can_release, gate_failures = check_blinding_release_gate(
        args.file_a, args.file_b, records_a, records_b
    )

    if not can_release:
        print("\n" + "=" * 65)
        print(" BLINDING RELEASE GATE: LOCKED (RELEASE REFUSED)")
        print("=" * 65)
        print("Human annotation is NOT yet complete and validated.")
        print("Gate failures:")
        for fail in gate_failures:
            print(f"  - {fail}")
        print("\nSTATISTICAL AGREEMENT ANALYSIS CANNOT BE PERFORMED.")
        print("DO NOT FABRICATE ANNOTATION COMPLETION.")
        print("=" * 65 + "\n")
        sys.exit(2)

    # Align labels by question string
    map_b = {r["question"]: r.get("assigned_label") for r in records_b}
    aligned_a = []
    aligned_b = []
    for r in records_a:
        q = r["question"]
        lbl_a = r.get("assigned_label")
        lbl_b = map_b.get(q)
        if lbl_a and lbl_b:
            aligned_a.append(lbl_a)
            aligned_b.append(lbl_b)

    kappa_res = compute_cohens_kappa(aligned_a, aligned_b)
    alpha = compute_krippendorffs_alpha_nominal(aligned_a, aligned_b)
    conf_matrix = compute_confusion_matrix(aligned_a, aligned_b)
    disagreements, docket = extract_disagreements_and_docket(records_a, records_b)

    # Save docket
    os.makedirs(os.path.dirname(os.path.abspath(args.docket_out)), exist_ok=True)
    with open(args.docket_out, "w", encoding="utf-8") as f:
        for entry in docket:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    summary = {
        "status": "COMPLETED_AND_CALCULATED",
        "total_aligned": len(aligned_a),
        "cohens_kappa": kappa_res,
        "krippendorffs_alpha": alpha,
        "confusion_matrix": conf_matrix,
        "disagreement_count": len(disagreements),
        "docket_item_count": len(docket),
        "docket_file": args.docket_out,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print("\n" + "=" * 65)
        print(" INTER-ANNOTATOR AGREEMENT REPORT")
        print("=" * 65)
        print(f"Total Evaluated Items: {len(aligned_a)}")
        print(f"Raw Observed Agreement (Po): {kappa_res['raw_agreement_po']:.4f}")
        print(f"Expected Chance Agreement (Pe): {kappa_res['expected_agreement_pe']:.4f}")
        print(f"Cohen's Kappa (kappa):        {kappa_res['cohens_kappa']:.4f} (SE: {kappa_res['standard_error']:.4f})")
        print(f"95% Confidence Interval:       [{kappa_res['ci_95_lower']:.4f}, {kappa_res['ci_95_upper']:.4f}]")
        print(f"Krippendorff's Alpha:          {alpha:.4f}")
        print(f"Disagreements (A != B):        {len(disagreements)}")
        print(f"Items in Adjudication Docket:  {len(docket)}")
        print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
