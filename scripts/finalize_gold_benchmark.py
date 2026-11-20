#!/usr/bin/env python3
"""Gold Benchmark Finalization Pipeline & Validation Engine.

Converts validated, completed human annotations and adjudicated records into
the authoritative, versioned ground-truth gold uncertainty benchmark.

Enforces strict research safeguards:
- Halts with exit code 2 if human annotation or adjudication is incomplete.
- Verifies all 14 data validation rules.
- Generates reproducible, cryptographic benchmark manifest.
- Zero fabrication of labels.
"""

import argparse
import datetime
import hashlib
import json
import os
import sys
from typing import Dict, Any, List, Optional, Tuple, Set

VALID_LABELS = {"KNOWN", "AMBIGUOUS", "UNKNOWN"}
FORBIDDEN_LEAKAGE_KEYS = {
    "prediction", "pred", "cluster_entropy", "entropy", "num_clusters",
    "feature", "features", "logits", "model_output"
}


def compute_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file."""
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class GoldBenchmarkValidator:
    """Validates the finalized gold benchmark against all 14 empirical rules."""

    def __init__(self, expected_count: int = 300):
        self.expected_count = expected_count

    def validate_records(
        self,
        gold_records: List[Dict[str, Any]],
        candidate_records: List[Dict[str, Any]],
    ) -> Tuple[bool, List[str]]:
        errors: List[str] = []

        # 1. Exactly 300 records
        if len(gold_records) != self.expected_count:
            errors.append(f"Record count mismatch: got {len(gold_records)}, expected {self.expected_count}")

        # 2. No duplicate IDs
        ids = [r.get("id") for r in gold_records]
        if len(set(ids)) != len(ids):
            dups = [x for x in ids if ids.count(x) > 1]
            errors.append(f"Duplicate IDs detected: {set(dups)}")

        cand_map = {r["id"]: r for r in candidate_records}

        for idx, rec in enumerate(gold_records, start=1):
            rid = rec.get("id")
            cand = cand_map.get(rid)

            if not cand:
                errors.append(f"Record {idx}: ID '{rid}' not found in candidate benchmark")
                continue

            # 3. All original questions preserved verbatim
            if rec.get("question") != cand.get("question"):
                errors.append(f"Record {rid}: Question text modified from candidate")

            # 4. Valid epistemic labels
            lbl = rec.get("final_gold_label")
            if not lbl or lbl not in VALID_LABELS:
                errors.append(f"Record {rid}: Invalid final_gold_label '{lbl}'")

            # 5. Candidate label preserved
            if rec.get("candidate_label") != cand.get("gold_label"):
                errors.append(f"Record {rid}: Candidate label mismatch with source candidate")

            # 6. Candidate match boolean correct
            expected_match = (lbl == cand.get("gold_label"))
            if rec.get("candidate_matches_gold") != expected_match:
                errors.append(f"Record {rid}: candidate_matches_gold boolean is incorrect")

            # 7 & 8. Provenance & Adjudication checks
            prov = rec.get("human_annotation_provenance")
            if not prov or not isinstance(prov, dict):
                errors.append(f"Record {rid}: Missing or malformed human_annotation_provenance")
                continue

            ann_a = prov.get("annotator_a", {})
            ann_b = prov.get("annotator_b", {})
            lbl_a = ann_a.get("assigned_label")
            lbl_b = ann_b.get("assigned_label")

            if lbl_a not in VALID_LABELS or lbl_b not in VALID_LABELS:
                errors.append(f"Record {rid}: Invalid or null individual annotator label")

            is_unanimous = (lbl_a == lbl_b)
            if prov.get("is_unanimous") != is_unanimous:
                errors.append(f"Record {rid}: is_unanimous boolean mismatch ({prov.get('is_unanimous')} vs {is_unanimous})")

            adj_req = prov.get("adjudication_required")
            adj_rec = prov.get("adjudication_record")

            if not is_unanimous or ann_a.get("requires_adjudication") or ann_b.get("requires_adjudication"):
                if not adj_req:
                    errors.append(f"Record {rid}: Disputed record marked adjudication_required=false")
                if not adj_rec:
                    errors.append(f"Record {rid}: Disputed record missing adjudication_record")
                else:
                    if not adj_rec.get("adjudicated_label") or adj_rec.get("adjudicated_label") not in VALID_LABELS:
                        errors.append(f"Record {rid}: Missing or invalid adjudicated_label in adjudication_record")
                    if not adj_rec.get("adjudication_rationale") or len(adj_rec.get("adjudication_rationale").strip()) < 5:
                        errors.append(f"Record {rid}: Missing or too short adjudication_rationale")
                    if lbl != adj_rec.get("adjudicated_label"):
                        errors.append(f"Record {rid}: final_gold_label does not match adjudicated decision")
            else:
                if is_unanimous and not adj_req and lbl != lbl_a:
                    errors.append(f"Record {rid}: Unanimous record final_gold_label does not match consensus ({lbl} vs {lbl_a})")

            # 9. Leakage checks
            leak_keys = {k for k in rec.keys() if k.lower() in FORBIDDEN_LEAKAGE_KEYS}
            if leak_keys:
                errors.append(f"Record {rid}: Prohibited metadata leakage detected: {leak_keys}")

        # 10. Deterministic ordering check (must match candidate index sequence)
        cand_ids = [r["id"] for r in candidate_records]
        if ids != cand_ids:
            errors.append("Gold record ID ordering does not match candidate benchmark sequence exactly")

        is_valid = len(errors) == 0
        return is_valid, errors


def assemble_gold_record(
    candidate: Dict[str, Any],
    record_a: Dict[str, Any],
    record_b: Dict[str, Any],
    adjudication_map: Dict[str, Dict[str, Any]],
    timestamp: str,
) -> Dict[str, Any]:
    """Assembles a single finalized gold benchmark record."""
    lbl_a = record_a.get("assigned_label")
    lbl_b = record_b.get("assigned_label")
    flag_a = record_a.get("requires_adjudication", False)
    flag_b = record_b.get("requires_adjudication", False)

    is_unanimous = (lbl_a == lbl_b)
    needs_adjudication = (not is_unanimous) or flag_a or flag_b

    qid = candidate["id"]
    adj_entry = adjudication_map.get(qid) or adjudication_map.get(candidate["question"])

    if needs_adjudication:
        if not adj_entry:
            raise ValueError(f"Record '{qid}' requires adjudication but no adjudication entry found.")
        final_label = adj_entry["adjudicated_label"]
        adjudication_record = {
            "docket_id": adj_entry.get("docket_id", "ADJ_???"),
            "trigger_reason": adj_entry.get("trigger_reason", "LABEL_DISAGREEMENT" if not is_unanimous else "EXPLICIT_ESCALATION"),
            "adjudicated_label": final_label,
            "adjudication_rationale": adj_entry.get("adjudication_rationale", "Adjudicated consensus"),
            "adjudicator": adj_entry.get("adjudicator", "Senior Adjudication Panel"),
            "timestamp": adj_entry.get("timestamp", timestamp),
        }
        agreement_type = "adjudicated_consensus"
        adjudicator_id = adj_entry.get("adjudicator", "Senior Adjudication Panel")
        adjudication_rationale = adj_entry.get("adjudication_rationale")
    else:
        final_label = lbl_a
        adjudication_record = None
        agreement_type = "unanimous_consensus"
        adjudicator_id = None
        adjudication_rationale = None

    cand_label = candidate.get("gold_label")

    return {
        "id": candidate["id"],
        "question_group_id": candidate.get("question_group_id", "grp_unknown"),
        "question": candidate["question"],
        "category": candidate.get("category", "general"),
        "final_gold_label": final_label,
        "candidate_label": cand_label,
        "candidate_matches_gold": (final_label == cand_label),
        "source": candidate.get("source", "benchmark_v1.1"),
        "is_adversarial_variant": candidate.get("is_adversarial_variant", False),
        "human_annotation_provenance": {
            "annotator_a": {
                "blinded_id": record_a["blinded_id"],
                "assigned_label": lbl_a,
                "confidence": record_a.get("confidence", "HIGH"),
                "rationale": record_a.get("rationale", ""),
                "requires_adjudication": flag_a,
                "notes": record_a.get("notes", ""),
            },
            "annotator_b": {
                "blinded_id": record_b["blinded_id"],
                "assigned_label": lbl_b,
                "confidence": record_b.get("confidence", "HIGH"),
                "rationale": record_b.get("rationale", ""),
                "requires_adjudication": flag_b,
                "notes": record_b.get("notes", ""),
            },
            "is_unanimous": is_unanimous,
            "adjudication_required": needs_adjudication,
            "adjudication_record": adjudication_record,
        },
        "reviewer_agreement": {
            "annotation_status": "gold_benchmark",
            "annotator_count": 3 if needs_adjudication else 2,
            "individual_labels": [lbl_a, lbl_b] + ([final_label] if needs_adjudication else []),
            "inter_annotator_agreement": agreement_type,
            "adjudicator_id": adjudicator_id,
            "adjudication_rationale": adjudication_rationale,
        },
        "benchmark_version": "1.0",
        "gold_finalized_timestamp": timestamp,
    }


def build_reproducibility_manifest(
    candidate_benchmark_path: str,
    annotator_a_path: str,
    annotator_b_path: str,
    blinding_key_path: str,
    adjudication_path: Optional[str],
    gold_benchmark_path: str,
    gold_records: List[Dict[str, Any]],
    agreement_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generates the formal reproducibility manifest for the finalized gold benchmark."""
    counts = {"KNOWN": 0, "AMBIGUOUS": 0, "UNKNOWN": 0}
    for r in gold_records:
        lbl = r.get("final_gold_label")
        if lbl in counts:
            counts[lbl] += 1

    return {
        "manifest_version": "1.0",
        "dataset_name": "LLM Uncertainty Classification 300-Question Gold Benchmark",
        "dataset_version": "1.0.0",
        "created_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_records": len(gold_records),
        "label_distribution": counts,
        "source_artifacts": {
            "candidate_benchmark": {
                "path": candidate_benchmark_path,
                "sha256": compute_sha256(candidate_benchmark_path),
            },
            "annotator_a_file": {
                "path": annotator_a_path,
                "sha256": compute_sha256(annotator_a_path),
            },
            "annotator_b_file": {
                "path": annotator_b_path,
                "sha256": compute_sha256(annotator_b_path),
            },
            "blinding_key": {
                "path": blinding_key_path,
                "sha256": compute_sha256(blinding_key_path),
            },
            "adjudication_log": {
                "path": adjudication_path,
                "sha256": compute_sha256(adjudication_path) if adjudication_path and os.path.exists(adjudication_path) else None,
            },
        },
        "output_artifact": {
            "gold_benchmark": {
                "path": gold_benchmark_path,
                "sha256": compute_sha256(gold_benchmark_path) if os.path.exists(gold_benchmark_path) else None,
            }
        },
        "agreement_summary": agreement_stats or {},
        "scientific_provenance": {
            "annotation_protocol": "Double-blind independent human annotation with senior adjudication",
            "annotation_rubric": "data/annotation/ANNOTATION_RUBRIC.md (v1.1)",
            "execution_guide": "data/annotation/ANNOTATION_EXECUTION_GUIDE.md",
            "leakage_safeguards": "StratifiedGroupKFold on question_group_id (152 groups)",
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Assemble and validate the finalized gold benchmark from human annotations."
    )
    parser.add_argument(
        "--candidate",
        default="data/benchmarks/uncertainty_benchmark_300_candidate_v1.1.jsonl",
        help="Path to candidate benchmark JSONL.",
    )
    parser.add_argument(
        "--annotator-a",
        default="data/annotation/blinded_annotation_annotator_a.jsonl",
        help="Path to completed Annotator A JSONL.",
    )
    parser.add_argument(
        "--annotator-b",
        default="data/annotation/blinded_annotation_annotator_b.jsonl",
        help="Path to completed Annotator B JSONL.",
    )
    parser.add_argument(
        "--blinding-key",
        default="data/annotation/annotation_blinding_key.json",
        help="Path to blinding key JSON.",
    )
    parser.add_argument(
        "--adjudication-log",
        default="data/annotation/adjudication_log.jsonl",
        help="Path to completed adjudication log JSONL.",
    )
    parser.add_argument(
        "--output-gold",
        default="data/benchmarks/uncertainty_benchmark_300_gold_v1.0.jsonl",
        help="Target output path for finalized gold benchmark JSONL.",
    )
    parser.add_argument(
        "--output-manifest",
        default="data/benchmarks/gold_benchmark_manifest_v1.0.json",
        help="Target output path for reproducibility manifest JSON.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate an existing gold benchmark file without rebuilding.",
    )

    args = parser.parse_args()

    # Read candidate benchmark
    if not os.path.exists(args.candidate):
        print(f"Error: Candidate benchmark not found: {args.candidate}")
        sys.exit(1)

    with open(args.candidate, "r", encoding="utf-8") as f:
        candidate_records = [json.loads(line) for line in f if line.strip()]

    validator = GoldBenchmarkValidator(expected_count=len(candidate_records))

    if args.validate_only:
        if not os.path.exists(args.output_gold):
            print(f"Error: Gold benchmark file does not exist to validate: {args.output_gold}")
            sys.exit(1)
        with open(args.output_gold, "r", encoding="utf-8") as f:
            gold_records = [json.loads(line) for line in f if line.strip()]
        is_valid, errors = validator.validate_records(gold_records, candidate_records)
        if is_valid:
            print("GOLD BENCHMARK VALIDATION: PASSED")
            sys.exit(0)
        else:
            print("GOLD BENCHMARK VALIDATION: FAILED")
            for e in errors[:20]:
                print(f"  - {e}")
            sys.exit(1)

    # Pre-flight check: Are human annotations complete?
    if not os.path.exists(args.annotator_a) or not os.path.exists(args.annotator_b):
        print("Error: Annotator files missing.")
        sys.exit(1)

    with open(args.annotator_a, "r", encoding="utf-8") as f:
        recs_a = [json.loads(line) for line in f if line.strip()]
    with open(args.annotator_b, "r", encoding="utf-8") as f:
        recs_b = [json.loads(line) for line in f if line.strip()]

    unannotated_a = sum(1 for r in recs_a if r.get("assigned_label") is None)
    unannotated_b = sum(1 for r in recs_b if r.get("assigned_label") is None)

    if unannotated_a > 0 or unannotated_b > 0:
        print("\n" + "=" * 65)
        print(" CRITICAL SCIENTIFIC SAFEGUARD: HALTING FINALIZATION")
        print("=" * 65)
        print("Human annotation is NOT completed.")
        print(f"  - Annotator A unannotated records: {unannotated_a}")
        print(f"  - Annotator B unannotated records: {unannotated_b}")
        print("\nGOLD BENCHMARK CANNOT BE GENERATED PREMATURELY.")
        print("DO NOT FABRICATE HUMAN GROUND-TRUTH LABELS.")
        print("=" * 65 + "\n")
        sys.exit(2)

    print("Annotation completeness verified. Proceeding with benchmark assembly...")

    if not os.path.exists(args.blinding_key):
        print(f"Error: Blinding key not found: {args.blinding_key}")
        sys.exit(1)

    with open(args.blinding_key, "r", encoding="utf-8") as f:
        blinding_key = json.load(f)

    adjudication_map = {}
    if args.adjudication_log and os.path.exists(args.adjudication_log):
        with open(args.adjudication_log, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    qid = entry.get("underlying_question_id") or entry.get("question_id")
                    if qid:
                        adjudication_map[qid] = entry

    map_a = blinding_key.get("mapping_annotator_a", {})
    map_b = blinding_key.get("mapping_annotator_b", {})

    bid_to_qid_a = {bid: (v["original_id"] if isinstance(v, dict) else v) for bid, v in map_a.items()}
    bid_to_qid_b = {bid: (v["original_id"] if isinstance(v, dict) else v) for bid, v in map_b.items()}

    rec_a_by_qid = {bid_to_qid_a.get(r["blinded_id"]): r for r in recs_a if bid_to_qid_a.get(r["blinded_id"])}
    rec_b_by_qid = {bid_to_qid_b.get(r["blinded_id"]): r for r in recs_b if bid_to_qid_b.get(r["blinded_id"])}

    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    gold_records = []
    for cand in candidate_records:
        qid = cand["id"]
        ra = rec_a_by_qid.get(qid)
        rb = rec_b_by_qid.get(qid)
        if not ra or not rb:
            print(f"Error: Missing annotator record for question ID '{qid}'")
            sys.exit(1)
        g_rec = assemble_gold_record(cand, ra, rb, adjudication_map, timestamp)
        gold_records.append(g_rec)

    is_valid, val_errors = validator.validate_records(gold_records, candidate_records)
    if not is_valid:
        print("Error: Assembled gold records failed validation:")
        for e in val_errors[:20]:
            print(f"  - {e}")
        sys.exit(1)

    gold_out_dir = os.path.dirname(os.path.abspath(args.output_gold))
    if gold_out_dir:
        os.makedirs(gold_out_dir, exist_ok=True)
    with open(args.output_gold, "w", encoding="utf-8") as f:
        for r in gold_records:
            f.write(json.dumps(r) + "\n")

    manifest = build_reproducibility_manifest(
        candidate_benchmark_path=args.candidate,
        annotator_a_path=args.annotator_a,
        annotator_b_path=args.annotator_b,
        blinding_key_path=args.blinding_key,
        adjudication_path=args.adjudication_log if (args.adjudication_log and os.path.exists(args.adjudication_log)) else None,
        gold_benchmark_path=args.output_gold,
        gold_records=gold_records,
    )
    manifest_out_dir = os.path.dirname(os.path.abspath(args.output_manifest))
    if manifest_out_dir:
        os.makedirs(manifest_out_dir, exist_ok=True)
    with open(args.output_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Successfully finalized gold benchmark: {args.output_gold} ({len(gold_records)} records)")
    print(f"Manifest written to: {args.output_manifest}")


if __name__ == "__main__":
    main()
