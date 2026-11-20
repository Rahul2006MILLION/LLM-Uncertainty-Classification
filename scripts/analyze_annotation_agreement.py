#!/usr/bin/env python3
"""Inter-Annotator Agreement (IAA) Analysis Pipeline and Adjudication Docket Generator.

Implements the formal statistical methodology specified in data/annotation/AGREEMENT_ANALYSIS_PLAN.md
and Milestone 7.5.2 protocol:
- Independent loading and validation of Annotator A and B records
- Robust underlying question alignment via blinding key mapping (never by file order or assumed sequence)
- Annotation completeness gate (halts if annotations are incomplete or null)
- Observed Raw Agreement (Po) with exact numerator, denominator, and percentage
- Cohen's Kappa coefficient (kappa) with observed, chance expected (Pe), SE, and 95% CI
- Krippendorff's Alpha (nominal formulation)
- Canonical 3x3 Epistemic Confusion Matrix (Rows = Annotator A, Columns = Annotator B)
- Mathematically sound class-wise specific agreement
- Independent class distribution reporting
- Non-leaking disagreement extraction and Adjudication Docket generation
- Reproducible machine-readable JSON and human-readable Markdown reporting
"""

import argparse
import datetime
import json
import math
import os
import sys
from typing import Dict, Any, List, Tuple, Optional, Set

CLASSES = ["KNOWN", "AMBIGUOUS", "UNKNOWN"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

DEFAULT_A_PATH = "data/annotation/blinded_annotation_annotator_a.jsonl"
DEFAULT_B_PATH = "data/annotation/blinded_annotation_annotator_b.jsonl"
DEFAULT_KEY_PATH = "data/annotation/annotation_blinding_key.json"
DEFAULT_OUT_JSON = "data/annotation/agreement_analysis_results.json"
DEFAULT_OUT_MD = "data/annotation/AGREEMENT_ANALYSIS_REPORT.md"
DEFAULT_DOCKET = "data/annotation/adjudication_docket.jsonl"


def load_annotation_file(filepath: str, expected_annotator: Optional[str] = None) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Loads and validates JSONL records for an individual annotator."""
    errors = []
    records = []

    if not os.path.exists(filepath):
        errors.append(f"File not found: {filepath}")
        return records, errors

    seen_ids = set()
    with open(filepath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                errors.append(f"Line {idx}: Empty line in JSONL file")
                continue
            try:
                rec = json.loads(line_str)
            except json.JSONDecodeError as jde:
                errors.append(f"Line {idx}: Malformed JSON - {jde}")
                continue

            if not isinstance(rec, dict):
                errors.append(f"Line {idx}: Record is not a JSON object")
                continue

            bid = rec.get("blinded_id")
            if not bid or not isinstance(bid, str):
                errors.append(f"Line {idx}: Missing or invalid 'blinded_id'")
            else:
                if expected_annotator:
                    prefix = f"ANN_{expected_annotator.upper()}_"
                    if not bid.startswith(prefix):
                        errors.append(f"Line {idx}: Blinded ID '{bid}' does not match expected prefix '{prefix}'")
                if bid in seen_ids:
                    errors.append(f"Line {idx}: Duplicate blinded_id '{bid}'")
                seen_ids.add(bid)

            if not rec.get("question") or not isinstance(rec.get("question"), str):
                errors.append(f"Line {idx}: Missing or invalid 'question' field")

            records.append(rec)

    return records, errors


def check_completeness_gate(
    records_a: List[Dict[str, Any]],
    records_b: List[Dict[str, Any]],
    expected_count: int = 300,
) -> Tuple[bool, Dict[str, Any]]:
    """Evaluates whether both annotation files meet the completeness requirements."""
    completed_a = sum(1 for r in records_a if r.get("assigned_label") in CLASSES)
    unannotated_a = sum(1 for r in records_a if r.get("assigned_label") is None)
    invalid_a = [r.get("assigned_label") for r in records_a if r.get("assigned_label") is not None and r.get("assigned_label") not in CLASSES]

    completed_b = sum(1 for r in records_b if r.get("assigned_label") in CLASSES)
    unannotated_b = sum(1 for r in records_b if r.get("assigned_label") is None)
    invalid_b = [r.get("assigned_label") for r in records_b if r.get("assigned_label") is not None and r.get("assigned_label") not in CLASSES]

    failures = []
    if len(records_a) != expected_count:
        failures.append(f"Annotator A total records ({len(records_a)}) != expected ({expected_count})")
    if unannotated_a > 0:
        failures.append(f"Annotator A has {unannotated_a} unannotated records (assigned_label is null)")
    if invalid_a:
        failures.append(f"Annotator A has {len(invalid_a)} invalid labels: {set(invalid_a)}")

    if len(records_b) != expected_count:
        failures.append(f"Annotator B total records ({len(records_b)}) != expected ({expected_count})")
    if unannotated_b > 0:
        failures.append(f"Annotator B has {unannotated_b} unannotated records (assigned_label is null)")
    if invalid_b:
        failures.append(f"Annotator B has {len(invalid_b)} invalid labels: {set(invalid_b)}")

    is_complete = (len(failures) == 0 and completed_a == expected_count and completed_b == expected_count)

    stats = {
        "is_complete": is_complete,
        "annotator_a": {
            "total": len(records_a),
            "completed": completed_a,
            "remaining": unannotated_a,
            "invalid_count": len(invalid_a),
        },
        "annotator_b": {
            "total": len(records_b),
            "completed": completed_b,
            "remaining": unannotated_b,
            "invalid_count": len(invalid_b),
        },
        "failures": failures,
    }
    return is_complete, stats


def align_annotations_via_mapping(
    records_a: List[Dict[str, Any]],
    records_b: List[Dict[str, Any]],
    mapping_data: Dict[str, Any],
) -> Tuple[bool, List[Dict[str, Any]], List[str]]:
    """Aligns Annotator A and B records to the same underlying question using the blinding mapping.

    Guarantees that records are aligned strictly by canonical underlying question ID
    and NOT by file position, line number, or randomized presentation order.
    """
    errors = []
    map_a = mapping_data.get("mapping_annotator_a", {})
    map_b = mapping_data.get("mapping_annotator_b", {})

    if not map_a or not map_b:
        errors.append("Invalid mapping structure: missing 'mapping_annotator_a' or 'mapping_annotator_b'")
        return False, [], errors

    # Map blinded_id -> underlying question ID
    bid_to_qid_a = {bid: v["original_id"] for bid, v in map_a.items()}
    bid_to_qid_b = {bid: v["original_id"] for bid, v in map_b.items()}

    # Index annotator records by underlying question ID
    records_by_qid_a: Dict[str, Dict[str, Any]] = {}
    for r in records_a:
        bid = r.get("blinded_id")
        qid = bid_to_qid_a.get(bid)
        if not qid:
            errors.append(f"Annotator A blinded_id '{bid}' not found in mapping key")
            continue
        if qid in records_by_qid_a:
            errors.append(f"Annotator A multiple records map to same question ID '{qid}'")
        records_by_qid_a[qid] = r

    records_by_qid_b: Dict[str, Dict[str, Any]] = {}
    for r in records_b:
        bid = r.get("blinded_id")
        qid = bid_to_qid_b.get(bid)
        if not qid:
            errors.append(f"Annotator B blinded_id '{bid}' not found in mapping key")
            continue
        if qid in records_by_qid_b:
            errors.append(f"Annotator B multiple records map to same question ID '{qid}'")
        records_by_qid_b[qid] = r

    # Determine common underlying question IDs
    qids_a = set(records_by_qid_a.keys())
    qids_b = set(records_by_qid_b.keys())

    missing_in_b = qids_a - qids_b
    missing_in_a = qids_b - qids_a
    if missing_in_b:
        errors.append(f"Question IDs present in Annotator A but missing in B: {len(missing_in_b)}")
    if missing_in_a:
        errors.append(f"Question IDs present in Annotator B but missing in A: {len(missing_in_a)}")

    aligned_records = []
    common_qids = sorted(list(qids_a.intersection(qids_b)))

    for qid in common_qids:
        rec_a = records_by_qid_a[qid]
        rec_b = records_by_qid_b[qid]

        # Verify underlying question text identity
        if rec_a.get("question") != rec_b.get("question"):
            errors.append(f"Question text mismatch for aligned question ID '{qid}' between annotators")

        aligned_records.append({
            "underlying_question_id": qid,
            "blinded_id_a": rec_a.get("blinded_id"),
            "blinded_id_b": rec_b.get("blinded_id"),
            "question": rec_a.get("question"),
            "label_a": rec_a.get("assigned_label"),
            "label_b": rec_b.get("assigned_label"),
            "confidence_a": rec_a.get("confidence"),
            "confidence_b": rec_b.get("confidence"),
            "rationale_a": rec_a.get("rationale", ""),
            "rationale_b": rec_b.get("rationale", ""),
            "flag_a": rec_a.get("requires_adjudication", False),
            "flag_b": rec_b.get("requires_adjudication", False),
        })

    is_aligned = (len(errors) == 0 and len(aligned_records) > 0 and len(aligned_records) == len(records_a))
    return is_aligned, aligned_records, errors


def compute_raw_agreement(y_a: List[str], y_b: List[str]) -> Dict[str, Any]:
    """Calculates observed raw agreement proportion Po with numerator and denominator."""
    assert len(y_a) == len(y_b) and len(y_a) > 0
    total = len(y_a)
    agreements = sum(1 for a, b in zip(y_a, y_b) if a == b)
    disagreements = total - agreements
    proportion = agreements / total
    return {
        "numerator": agreements,
        "denominator": total,
        "disagreements": disagreements,
        "raw_agreement_po": proportion,
        "percentage": round(proportion * 100.0, 2),
    }


def compute_expected_agreement(y_a: List[str], y_b: List[str]) -> float:
    """Calculates chance-expected agreement Pe across canonical classes."""
    n = len(y_a)
    assert n > 0 and len(y_b) == n
    p_a = {c: sum(1 for a in y_a if a == c) / n for c in CLASSES}
    p_b = {c: sum(1 for b in y_b if b == c) / n for c in CLASSES}
    return sum(p_a[c] * p_b[c] for c in CLASSES)


def compute_cohens_kappa(y_a: List[str], y_b: List[str]) -> Dict[str, Any]:
    """Calculates Cohen's Kappa, chance expected agreement, SE, and 95% CI."""
    n = len(y_a)
    raw = compute_raw_agreement(y_a, y_b)
    p_o = raw["raw_agreement_po"]

    p_a = {c: sum(1 for a in y_a if a == c) / n for c in CLASSES}
    p_b = {c: sum(1 for b in y_b if b == c) / n for c in CLASSES}
    p_e = compute_expected_agreement(y_a, y_b)

    if math.isclose(p_e, 1.0):
        # Degenerate case: both raters assign 100% of items to one single class
        kappa = 1.0 if math.isclose(p_o, 1.0) else 0.0
        se = 0.0
        ci_lower, ci_upper = kappa, kappa
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)
        denom = n * ((1.0 - p_e) ** 2)
        var = (p_o * (1.0 - p_o)) / denom if denom > 0 else 0.0
        se = math.sqrt(max(0.0, var))
        ci_lower = max(-1.0, kappa - 1.96 * se)
        ci_upper = min(1.0, kappa + 1.96 * se)

    return {
        "n": n,
        "observed_agreement_po": p_o,
        "expected_agreement_pe": p_e,
        "cohens_kappa": kappa,
        "standard_error": se,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "marginal_annotator_a": p_a,
        "marginal_annotator_b": p_b,
    }


def compute_krippendorffs_alpha_nominal(y_a: List[str], y_b: List[str]) -> float:
    """Calculates Krippendorff's Alpha for nominal 2-rater complete data."""
    n = len(y_a)
    if n == 0:
        return 0.0

    total_ratings = 2 * n
    n_c = {c: 0 for c in CLASSES}
    for a, b in zip(y_a, y_b):
        n_c[a] += 1
        n_c[b] += 1

    d_o = sum(1 for a, b in zip(y_a, y_b) if a != b) / n
    d_e = sum(n_c[c] * (total_ratings - n_c[c]) for c in CLASSES) / (total_ratings * (total_ratings - 1))

    if math.isclose(d_e, 0.0):
        return 1.0 if math.isclose(d_o, 0.0) else 0.0

    return 1.0 - (d_o / d_e)


def compute_confusion_matrix(y_a: List[str], y_b: List[str]) -> Dict[str, Any]:
    """Generates canonical 3x3 confusion matrix and class-wise specific agreement."""
    # Rows = Annotator A, Columns = Annotator B
    matrix = {row: {col: 0 for col in CLASSES} for row in CLASSES}
    for a, b in zip(y_a, y_b):
        matrix[a][b] += 1

    class_wise = {}
    for c in CLASSES:
        tp = matrix[c][c]
        row_sum = sum(matrix[c][col] for col in CLASSES)  # Annotator A assigned c
        col_sum = sum(matrix[row][c] for row in CLASSES)  # Annotator B assigned c
        denom = row_sum + col_sum

        # Fleiss / Cicchetti specific agreement: 2*A_c / (R_A + R_B)
        specific_agreement = (2.0 * tp / denom) if denom > 0 else 0.0

        class_wise[c] = {
            "annotator_a_count": row_sum,
            "annotator_b_count": col_sum,
            "unanimous_count": tp,
            "specific_agreement": specific_agreement,
            "percentage": round(specific_agreement * 100.0, 2),
        }

    return {
        "rows": "Annotator A",
        "columns": "Annotator B",
        "order": CLASSES,
        "matrix": matrix,
        "class_wise_agreement": class_wise,
    }


def compute_annotation_distributions(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes class frequency distribution and percentages for an annotator."""
    total = len(records)
    counts = {c: 0 for c in CLASSES}
    for r in records:
        lbl = r.get("assigned_label")
        if lbl in counts:
            counts[lbl] += 1

    pcts = {c: round((counts[c] / total * 100.0), 2) if total > 0 else 0.0 for c in CLASSES}
    return {
        "total": total,
        "counts": counts,
        "percentages": pcts,
    }


def extract_disagreements_and_docket(aligned_records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extracts all discordant queries and builds the adjudication docket."""
    disagreements = []
    docket = []

    docket_num = 1
    for rec in aligned_records:
        lbl_a = rec["label_a"]
        lbl_b = rec["label_b"]
        flag_a = rec["flag_a"]
        flag_b = rec["flag_b"]
        conf_a = rec["confidence_a"]
        conf_b = rec["confidence_b"]

        is_disagreement = (lbl_a != lbl_b)
        is_escalated = flag_a or flag_b
        is_double_low = (conf_a == "LOW" and conf_b == "LOW")

        needs_adjudication = is_disagreement or is_escalated or is_double_low

        if is_disagreement:
            axis = f"{min(lbl_a, lbl_b)} vs {max(lbl_a, lbl_b)}"
            disagreements.append({
                "underlying_question_id": rec["underlying_question_id"],
                "blinded_id_a": rec["blinded_id_a"],
                "blinded_id_b": rec["blinded_id_b"],
                "question": rec["question"],
                "label_a": lbl_a,
                "label_b": lbl_b,
                "confidence_a": conf_a,
                "confidence_b": conf_b,
                "rationale_a": rec["rationale_a"],
                "rationale_b": rec["rationale_b"],
                "axis": axis,
            })

        if needs_adjudication:
            trigger = "LABEL_DISAGREEMENT" if is_disagreement else ("EXPLICIT_ESCALATION" if is_escalated else "DOUBLE_LOW_CONFIDENCE")
            docket.append({
                "docket_id": f"ADJ_{docket_num:03d}",
                "underlying_question_id": rec["underlying_question_id"],
                "blinded_id_a": rec["blinded_id_a"],
                "blinded_id_b": rec["blinded_id_b"],
                "question": rec["question"],
                "label_a": lbl_a,
                "label_b": lbl_b,
                "confidence_a": conf_a,
                "confidence_b": conf_b,
                "rationale_a": rec["rationale_a"],
                "rationale_b": rec["rationale_b"],
                "flagged_by_a": flag_a,
                "flagged_by_b": flag_b,
                "trigger_reason": trigger,
                "adjudicated_label": None,
                "adjudication_rationale": "",
                "adjudicator": None,
                "timestamp": None,
            })
            docket_num += 1

    return disagreements, docket


def generate_markdown_report(
    results: Dict[str, Any],
    filepath: str,
):
    """Generates human-readable Markdown report conforming to Phase 5."""
    raw = results["raw_agreement"]
    kappa = results["cohens_kappa"]
    dist_a = results["distribution_annotator_a"]
    dist_b = results["distribution_annotator_b"]
    cm = results["confusion_matrix"]

    lines = [
        "# Human Annotation Agreement Analysis Report",
        "",
        f"**Project:** LLM Uncertainty Classification Using Multi-Response Consistency and Feature-Based Machine Learning  ",
        f"**Generated:** {results['timestamp']}  ",
        f"**Benchmark Questions:** {results['total_aligned']}  ",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        f"- **Raw Observed Agreement ($P_o$):** {raw['percentage']}% ({raw['numerator']} / {raw['denominator']})",
        f"- **Cohen's Kappa ($\\kappa$):** {kappa['cohens_kappa']:.4f} (SE: {kappa['standard_error']:.4f}, 95% CI: [{kappa['ci_95_lower']:.4f}, {kappa['ci_95_upper']:.4f}])",
        f"- **Krippendorff's Alpha ($\\alpha$):** {results['krippendorffs_alpha']:.4f}",
        f"- **Total Disagreements:** {len(results['disagreements'])} ({round(len(results['disagreements'])/results['total_aligned']*100, 2)}%)",
        f"- **Items Requiring Adjudication:** {len(results['adjudication_docket'])}",
        "",
        "---",
        "",
        "## 2. Independent Annotation Distributions",
        "",
        "| Class | Annotator A Count | Annotator A % | Annotator B Count | Annotator B % |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]

    for c in CLASSES:
        lines.append(f"| **{c}** | {dist_a['counts'][c]} | {dist_a['percentages'][c]}% | {dist_b['counts'][c]} | {dist_b['percentages'][c]}% |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Confusion Matrix ($3 \\times 3$)",
        "",
        "*Rows = Annotator A, Columns = Annotator B*",
        "",
        "| Annotator A \\ Annotator B | KNOWN | AMBIGUOUS | UNKNOWN | Total A |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])

    for row_c in CLASSES:
        row_counts = [cm["matrix"][row_c][col_c] for col_c in CLASSES]
        lines.append(f"| **{row_c}** | {row_counts[0]} | {row_counts[1]} | {row_counts[2]} | {sum(row_counts)} |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Class-Wise Specific Agreement",
        "",
        "| Class | Unanimous Matches | Specific Agreement ($P_{\\text{specific}}$) |",
        "| :--- | :---: | :---: |",
    ])
    for c in CLASSES:
        lines.append(f"| **{c}** | {cm['class_wise_agreement'][c]['unanimous_count']} | {cm['class_wise_agreement'][c]['percentage']}% |")

    lines.extend([
        "",
        "---",
        "",
        "## 5. Scientific Limitations",
        "",
        "> **Methodological Notice:**  ",
        "> Inter-annotator agreement measures consistency between annotators. High agreement does not establish that the assigned labels are objectively correct.  ",
        "",
        "- Agreement magnitude depends on rubric clarity and annotator calibration.",
        "- Cohen's Kappa can be affected by class distribution prevalence.",
        "- Agreement statistics do not replace formal adjudication.",
        "- All discordant items require independent human adjudication under established rubric criteria.",
        "",
    ])

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Compute inter-annotator agreement and generate adjudication docket."
    )
    parser.add_argument(
        "--annotator-a",
        default=DEFAULT_A_PATH,
        help="Path to Annotator A JSONL file.",
    )
    parser.add_argument(
        "--annotator-b",
        default=DEFAULT_B_PATH,
        help="Path to Annotator B JSONL file.",
    )
    parser.add_argument(
        "--mapping",
        default=DEFAULT_KEY_PATH,
        help="Path to blinding key mapping file (coordinator access only).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUT_JSON,
        help="Output path for machine-readable JSON results.",
    )
    parser.add_argument(
        "--report",
        default=DEFAULT_OUT_MD,
        help="Output path for human-readable Markdown report.",
    )
    parser.add_argument(
        "--docket",
        default=DEFAULT_DOCKET,
        help="Output path for adjudication docket JSONL.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=300,
        help="Expected total number of benchmark questions (default: 300).",
    )

    args = parser.parse_args()

    recs_a, errs_a = load_annotation_file(args.annotator_a, expected_annotator="A")
    recs_b, errs_b = load_annotation_file(args.annotator_b, expected_annotator="B")

    is_complete, comp_stats = check_completeness_gate(
        recs_a, recs_b, expected_count=args.expected_count
    )

    if not is_complete:
        print("\n" + "=" * 50)
        print("ANNOTATION AGREEMENT ANALYSIS")
        print("=" * 50)
        print("\nAnnotator A:")
        print(f"Completed: {comp_stats['annotator_a']['completed']}/{args.expected_count}")
        print(f"Remaining: {comp_stats['annotator_a']['remaining']}")
        print("\nAnnotator B:")
        print(f"Completed: {comp_stats['annotator_b']['completed']}/{args.expected_count}")
        print(f"Remaining: {comp_stats['annotator_b']['remaining']}")
        print("\nAlignment:\nNOT VERIFIED")
        print("\nValidation:\nINCOMPLETE")
        print("\nAgreement Analysis:\nBLOCKED")
        print("\nReason:")
        print("Human annotation is incomplete.")
        for f in comp_stats["failures"]:
            print(f"  - {f}")
        print("\n" + "=" * 50 + "\n")
        sys.exit(2)

    # Check mapping file
    if not os.path.exists(args.mapping):
        print(f"Error: Mapping file not found: {args.mapping}")
        sys.exit(1)

    with open(args.mapping, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)

    is_aligned, aligned_records, align_errors = align_annotations_via_mapping(
        recs_a, recs_b, mapping_data
    )

    if not is_aligned:
        print("Error: Alignment validation failed:")
        for err in align_errors:
            print(f"  - {err}")
        sys.exit(1)

    y_a = [r["label_a"] for r in aligned_records]
    y_b = [r["label_b"] for r in aligned_records]

    raw = compute_raw_agreement(y_a, y_b)
    kappa = compute_cohens_kappa(y_a, y_b)
    alpha = compute_krippendorffs_alpha_nominal(y_a, y_b)
    cm = compute_confusion_matrix(y_a, y_b)
    dist_a = compute_annotation_distributions(recs_a)
    dist_b = compute_annotation_distributions(recs_b)
    disagreements, docket = extract_disagreements_and_docket(aligned_records)

    results = {
        "status": "COMPLETED_AND_CALCULATED",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_aligned": len(aligned_records),
        "raw_agreement": raw,
        "cohens_kappa": kappa,
        "krippendorffs_alpha": alpha,
        "confusion_matrix": cm,
        "distribution_annotator_a": dist_a,
        "distribution_annotator_b": dist_b,
        "disagreements": disagreements,
        "adjudication_docket": docket,
    }

    # Save outputs
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    generate_markdown_report(results, args.report)

    os.makedirs(os.path.dirname(os.path.abspath(args.docket)), exist_ok=True)
    with open(args.docket, "w", encoding="utf-8") as f:
        for entry in docket:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("\n" + "=" * 50)
    print("ANNOTATION AGREEMENT ANALYSIS: SUCCESS")
    print("=" * 50)
    print(f"Total Aligned Questions:      {len(aligned_records)}")
    print(f"Raw Observed Agreement (Po):  {raw['percentage']}% ({raw['numerator']}/{raw['denominator']})")
    print(f"Cohen's Kappa:                {kappa['cohens_kappa']:.4f} (SE: {kappa['standard_error']:.4f})")
    print(f"Krippendorff's Alpha:         {alpha:.4f}")
    print(f"Disagreements:                {len(disagreements)}")
    print(f"Adjudication Docket Items:    {len(docket)}")
    print(f"Reports saved to:             {args.output} and {args.report}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
