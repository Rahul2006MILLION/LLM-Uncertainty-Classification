"""End-to-end execution script for the 30-Question Pilot Dataset.

Covers:
1. Candidate validation & benchmark 300 isolation enforcement
2. 10-response multi-generation per question via local qwen3:4b
3. Embedding (all-minilm) and agglomerative semantic clustering
4. 16-feature schema extraction and validation
5. Unified dataset serialization
6. Statistical analysis and scientific audit reporting
"""

import os
import json
import time
import logging
from typing import List, Dict, Any
import numpy as np

from src.generation.ollama_client import OllamaClient
from src.generation.generator import ResponseGenerator
from src.preprocessing.embeddings import EmbeddingEngine
from src.features.clustering import SemanticClusterer
from src.features.feature_extractor import FeatureExtractor, FEATURE_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("pilot_30_pipeline")

PILOT_CANDIDATES: List[Dict[str, Any]] = [
    # ==========================
    # 10 KNOWN QUESTIONS
    # ==========================
    {
        "id": "pilot_known_01",
        "question": "What year was the first commercial iPhone released by Apple?",
        "target_label": "KNOWN",
        "question_group_id": "grp_pilot_k_iphone_year",
        "source_dataset": "AmbigNQ",
        "source_id": "ambignq_single_2001",
        "original_source_label": "singleAnswer",
        "mapping_rationale": "Sufficiently clear factual query with single verifiable historical ground truth (2007).",
        "category": "factual_trivia",
    },
    {
        "id": "pilot_known_02",
        "question": "Who was the author of the dystopian novel 1984?",
        "target_label": "KNOWN",
        "question_group_id": "grp_pilot_k_1984_author",
        "source_dataset": "AmbigNQ",
        "source_id": "ambignq_single_2002",
        "original_source_label": "singleAnswer",
        "mapping_rationale": "Singular documented authorship (George Orwell) in public literary record.",
        "category": "factual_trivia",
    },
    {
        "id": "pilot_known_03",
        "question": "What is the capital city of Norway?",
        "target_label": "KNOWN",
        "question_group_id": "grp_pilot_k_norway_capital",
        "source_dataset": "AmbigNQ",
        "source_id": "ambignq_single_2003",
        "original_source_label": "singleAnswer",
        "mapping_rationale": "Official statutory capital of sovereign nation (Oslo) under ISO standards.",
        "category": "factual_geography",
    },
    {
        "id": "pilot_known_04",
        "question": "What element has the chemical symbol Au on the periodic table?",
        "target_label": "KNOWN",
        "question_group_id": "grp_pilot_k_gold_symbol",
        "source_dataset": "AmbigNQ",
        "source_id": "ambignq_single_2004",
        "original_source_label": "singleAnswer",
        "mapping_rationale": "IUPAC international standard chemical nomenclature (Gold).",
        "category": "scientific_standard",
    },
    {
        "id": "pilot_known_05",
        "question": "Who is credited with inventing the telephone in 1876?",
        "target_label": "KNOWN",
        "question_group_id": "grp_pilot_k_telephone_patent",
        "source_dataset": "SelfAware",
        "source_id": "selfaware_ans_101",
        "original_source_label": "answerable",
        "mapping_rationale": "Settled historical patent record (Alexander Graham Bell).",
        "category": "historical_fact",
    },
    {
        "id": "pilot_known_06",
        "question": "What is the largest planet in our solar system by volume and mass?",
        "target_label": "KNOWN",
        "question_group_id": "grp_pilot_k_jupiter_planet",
        "source_dataset": "SelfAware",
        "source_id": "selfaware_ans_102",
        "original_source_label": "answerable",
        "mapping_rationale": "Singular astronomical consensus fact (Jupiter).",
        "category": "scientific_fact",
    },
    {
        "id": "pilot_known_07",
        "question": "On which continent is the country of Egypt primarily located?",
        "target_label": "KNOWN",
        "question_group_id": "grp_pilot_k_egypt_continent",
        "source_dataset": "SelfAware",
        "source_id": "selfaware_ans_103",
        "original_source_label": "answerable",
        "mapping_rationale": "Determinate geographic consensus (Africa).",
        "category": "factual_geography",
    },
    {
        "id": "pilot_known_08",
        "question": "Who directed the 1993 science fiction film Jurassic Park?",
        "target_label": "KNOWN",
        "question_group_id": "grp_pilot_k_jurassic_director",
        "source_dataset": "SelfAware",
        "source_id": "selfaware_ans_104",
        "original_source_label": "answerable",
        "mapping_rationale": "Public cinematographic record (Steven Spielberg).",
        "category": "pop_culture_fact",
    },
    {
        "id": "pilot_known_09",
        "question": "What is the primary internal organ responsible for pumping blood throughout the human body?",
        "target_label": "KNOWN",
        "question_group_id": "grp_pilot_k_heart_pumping",
        "source_dataset": "SelfAware",
        "source_id": "selfaware_ans_105",
        "original_source_label": "answerable",
        "mapping_rationale": "Standard biological anatomy definition (Heart).",
        "category": "concept_definition",
    },
    {
        "id": "pilot_known_10",
        "question": "What is the official currency of the United Kingdom?",
        "target_label": "KNOWN",
        "question_group_id": "grp_pilot_k_uk_currency",
        "source_dataset": "AmbigNQ",
        "source_id": "ambignq_single_2005",
        "original_source_label": "singleAnswer",
        "mapping_rationale": "Sovereign legal currency (Pound Sterling / GBP).",
        "category": "factual_standard",
    },

    # ==========================
    # 10 AMBIGUOUS QUESTIONS
    # ==========================
    {
        "id": "pilot_ambig_01",
        "question": "Who is the current manager of the Manchester United soccer club?",
        "target_label": "AMBIGUOUS",
        "question_group_id": "grp_pilot_a_mufc_manager",
        "source_dataset": "AmbigNQ",
        "source_id": "ambignq_multi_3001",
        "original_source_label": "multipleQAs",
        "mapping_rationale": "Temporal ambiguity: answer changes dynamically depending on the unstated season, year, or month.",
        "category": "temporal_ambiguity",
    },
    {
        "id": "pilot_ambig_02",
        "question": "What is the capital city of Washington?",
        "target_label": "AMBIGUOUS",
        "question_group_id": "grp_pilot_a_washington_capital",
        "source_dataset": "AmbigNQ",
        "source_id": "ambignq_multi_3002",
        "original_source_label": "multipleQAs",
        "mapping_rationale": "Geographic polysemy: Olympia (Washington State) vs Washington D.C. (national capital).",
        "category": "entity_polysemy",
    },
    {
        "id": "pilot_ambig_03",
        "question": "What are the official national languages of Switzerland?",
        "target_label": "AMBIGUOUS",
        "question_group_id": "grp_pilot_a_swiss_languages",
        "source_dataset": "AmbigNQ",
        "source_id": "ambignq_multi_3003",
        "original_source_label": "multipleQAs",
        "mapping_rationale": "Plural constitutionally recognized languages (German, French, Italian, Romansh) lacking singular specification.",
        "category": "plural_referents",
    },
    {
        "id": "pilot_ambig_04",
        "question": "What is the total resident population of Arlington?",
        "target_label": "AMBIGUOUS",
        "question_group_id": "grp_pilot_a_arlington_pop",
        "source_dataset": "AmbigNQ",
        "source_id": "ambignq_multi_3004",
        "original_source_label": "multipleQAs",
        "mapping_rationale": "Entity polysemy: refers to prominent municipal cities Arlington, Texas (~400k) or Arlington, Virginia (~240k).",
        "category": "entity_polysemy",
    },
    {
        "id": "pilot_ambig_05",
        "question": "Who is the lead singer of the rock band Queen?",
        "target_label": "AMBIGUOUS",
        "question_group_id": "grp_pilot_a_queen_singer",
        "source_dataset": "AmbigNQ",
        "source_id": "ambignq_multi_3005",
        "original_source_label": "multipleQAs",
        "mapping_rationale": "Era / role ambiguity: Freddie Mercury (classic era) vs Paul Rodgers vs Adam Lambert (Queen + Adam Lambert).",
        "category": "temporal_role_ambiguity",
    },
    {
        "id": "pilot_ambig_06",
        "question": "Who was the drummer for the rock band Nirvana?",
        "target_label": "AMBIGUOUS",
        "question_group_id": "grp_pilot_a_nirvana_drummer",
        "source_dataset": "AmbigNQ",
        "source_id": "ambignq_multi_3006",
        "original_source_label": "multipleQAs",
        "mapping_rationale": "Plural historical members across album releases: Dave Grohl (Nevermind), Chad Channing (Bleach), Aaron Burckhard.",
        "category": "temporal_role_ambiguity",
    },
    {
        "id": "pilot_ambig_07",
        "question": "When does the NFL football season officially start each year?",
        "target_label": "AMBIGUOUS",
        "question_group_id": "grp_pilot_a_nfl_start",
        "source_dataset": "AmbigNQ",
        "source_id": "ambignq_multi_3007",
        "original_source_label": "multipleQAs",
        "mapping_rationale": "Criterion ambiguity: depends on whether preseason kickoff (Hall of Fame Game in August) or regular season opening week (September) is intended.",
        "category": "criterion_ambiguity",
    },
    {
        "id": "pilot_ambig_08",
        "question": "Is it easier or harder to become financially successful today compared to 50 years ago?",
        "target_label": "AMBIGUOUS",
        "question_group_id": "grp_pilot_a_success_comparison",
        "source_dataset": "SelfAware",
        "source_id": "selfaware_unans_2638",
        "original_source_label": "SelfAware_subjective",
        "mapping_rationale": "Subjective value judgment with unconstrained criteria; changes based on metric (purchasing power vs technology access).",
        "category": "subjective_criterion",
    },
    {
        "id": "pilot_ambig_09",
        "question": "Who won the baseball game between the Dodgers and the Giants?",
        "target_label": "AMBIGUOUS",
        "question_group_id": "grp_pilot_a_dodgers_giants",
        "source_dataset": "SelfAware",
        "source_id": "selfaware_unans_2800",
        "original_source_label": "SelfAware_underspecified",
        "mapping_rationale": "Severe temporal under-specification: rivalry includes over 2,500 historic games with no specified date or year.",
        "category": "temporal_under_specification",
    },
    {
        "id": "pilot_ambig_10",
        "question": "What is the tallest building in the world including architectural spires and antennas?",
        "target_label": "AMBIGUOUS",
        "question_group_id": "grp_pilot_a_tallest_building",
        "source_dataset": "AmbigNQ",
        "source_id": "ambignq_multi_3008",
        "original_source_label": "multipleQAs",
        "mapping_rationale": "Criterion ambiguity: depends on whether Council on Tall Buildings and Urban Habitat (CTBUH) architectural height, tip height, or highest occupied floor is measured.",
        "category": "criterion_ambiguity",
    },

    # ==========================
    # 10 UNKNOWN QUESTIONS
    # ==========================
    {
        "id": "pilot_unk_01",
        "question": "Who decided to put and use the letter s in the word lisp?",
        "target_label": "UNKNOWN",
        "question_group_id": "grp_pilot_u_lisp_etymology",
        "source_dataset": "SelfAware",
        "source_id": "selfaware_unans_2338",
        "original_source_label": "SelfAware_unanswerable",
        "mapping_rationale": "Epistemic barrier: historical phonological linguistic evolution with no single individual decision-maker documented.",
        "category": "unrecorded_history",
    },
    {
        "id": "pilot_unk_02",
        "question": "How long will it take for Chicago to average 50,000 drones flying overhead on a daily basis?",
        "target_label": "UNKNOWN",
        "question_group_id": "grp_pilot_u_chicago_drones",
        "source_dataset": "SelfAware",
        "source_id": "selfaware_unans_2371",
        "original_source_label": "SelfAware_unanswerable",
        "mapping_rationale": "Stochastic future contingency: unpredictable future regulatory, technological, and municipal outcome.",
        "category": "future_contingency",
    },
    {
        "id": "pilot_unk_03",
        "question": "What will be the exact worldwide gross box office receipts in US dollars for Avatar 5?",
        "target_label": "UNKNOWN",
        "question_group_id": "grp_pilot_u_avatar5_box_office",
        "source_dataset": "SelfAware",
        "source_id": "selfaware_unans_future_03",
        "original_source_label": "SelfAware_future",
        "mapping_rationale": "Future commercial contingency: film has not been produced or released; economic outcome is completely unestablished.",
        "category": "future_contingency",
    },
    {
        "id": "pilot_unk_04",
        "question": "What was the brand of bicycle owned by Albert Einstein while living in Bern in 1903?",
        "target_label": "UNKNOWN",
        "question_group_id": "grp_pilot_u_einstein_bike",
        "source_dataset": "SelfAware",
        "source_id": "selfaware_unans_inaccessible_01",
        "original_source_label": "SelfAware_unrecorded",
        "mapping_rationale": "Private historical possession: unrecorded minutiae absent from any archival biography or patent office record.",
        "category": "inaccessible_history",
    },
    {
        "id": "pilot_unk_05",
        "question": "What was the exact serial number on the personal typewriter used by George Orwell to write Animal Farm?",
        "target_label": "UNKNOWN",
        "question_group_id": "grp_pilot_u_orwell_serial",
        "source_dataset": "SelfAware",
        "source_id": "selfaware_unans_inaccessible_02",
        "original_source_label": "SelfAware_unrecorded",
        "mapping_rationale": "Lost historical artifact: machine serial number was never recorded in correspondence or inventory records.",
        "category": "inaccessible_history",
    },
    {
        "id": "pilot_unk_06",
        "question": "What is the official currency of the sovereign nation of Atlantis?",
        "target_label": "UNKNOWN",
        "question_group_id": "grp_pilot_u_atlantis_currency",
        "source_dataset": "Curated",
        "source_id": "curated_false_premise_01",
        "original_source_label": "false_premise",
        "mapping_rationale": "Presupposition failure: Atlantis is a mythological concept with no sovereign state or currency.",
        "category": "false_presupposition",
    },
    {
        "id": "pilot_unk_07",
        "question": "What was the exact heart rate in beats per minute of Abraham Lincoln during the Gettysburg Address in 1863?",
        "target_label": "UNKNOWN",
        "question_group_id": "grp_pilot_u_lincoln_heart_rate",
        "source_dataset": "Curated",
        "source_id": "curated_immeasurable_01",
        "original_source_label": "immeasurable_physical",
        "mapping_rationale": "Immeasurable biological quantity: biometric data at that specific 19th-century historical timestamp never existed.",
        "category": "immeasurable_state",
    },
    {
        "id": "pilot_unk_08",
        "question": "What will be the exact winning ticket number in the Spanish El Gordo Christmas lottery in December 2050?",
        "target_label": "UNKNOWN",
        "question_group_id": "grp_pilot_u_el_gordo_2050",
        "source_dataset": "SelfAware",
        "source_id": "selfaware_unans_future_02",
        "original_source_label": "SelfAware_future",
        "mapping_rationale": "Pure stochastic future event: outcome determined by physical random ball drawing decades in the future.",
        "category": "future_contingency",
    },
    {
        "id": "pilot_unk_09",
        "question": "What was the secret personal safe combination of Al Capone when he died in Florida in 1947?",
        "target_label": "UNKNOWN",
        "question_group_id": "grp_pilot_u_capone_safe",
        "source_dataset": "Curated",
        "source_id": "curated_private_info_01",
        "original_source_label": "private_state",
        "mapping_rationale": "Private credential: deceased subject took private passcode to grave; no public record exists.",
        "category": "private_state",
    },
    {
        "id": "pilot_unk_10",
        "question": "How many total handwritten pages were in the personal journal burned by Leonardo da Vinci in 1505?",
        "target_label": "UNKNOWN",
        "question_group_id": "grp_pilot_u_davinci_burned_journal",
        "source_dataset": "Curated",
        "source_id": "curated_destroyed_record_01",
        "original_source_label": "destroyed_record",
        "mapping_rationale": "Destroyed historical object: physical item was destroyed without enumeration or transcription.",
        "category": "inaccessible_history",
    },
]


def verify_benchmark_isolation(candidates: List[Dict[str, Any]], benchmark_path: str, max_allowed_sim: float = 0.75):
    """Enforce strict isolation from the 300 benchmark using all-minilm cosine similarity."""
    logger.info("Verifying benchmark isolation against %s...", benchmark_path)
    engine = EmbeddingEngine(embedding_model="all-minilm")
    
    with open(benchmark_path, "r", encoding="utf-8") as f:
        bench_records = [json.loads(line) for line in f]
    bench_qs = [r["question"].strip() for r in bench_records]
    bench_embs = engine.embed_batch(bench_qs)
    
    cand_qs = [c["question"].strip() for c in candidates]
    cand_embs = engine.embed_batch(cand_qs)
    
    sim_matrix = np.dot(cand_embs, bench_embs.T)
    
    violations = []
    for i, c in enumerate(candidates):
        max_sim = float(np.max(sim_matrix[i]))
        best_match_idx = int(np.argmax(sim_matrix[i]))
        c["max_benchmark_similarity"] = round(max_sim, 4)
        c["closest_benchmark_question"] = bench_qs[best_match_idx]
        
        if max_sim >= max_allowed_sim:
            violations.append((c["id"], c["question"], max_sim, bench_qs[best_match_idx]))
            
    if violations:
        for vid, vq, vsim, vb in violations:
            logger.error("LEAKAGE VIOLATION: [%s] '%s' has similarity %.3f with benchmark '%s'", vid, vq, vsim, vb)
        raise ValueError(f"Found {len(violations)} benchmark leakage violations (similarity >= {max_allowed_sim})")
        
    logger.info("PASS: All %d candidates have max_sim < %.2f with the 300 benchmark.", len(candidates), max_allowed_sim)


def run_pipeline():
    total_start = time.time()
    
    # 1. Isolation check
    benchmark_path = "data/benchmarks/uncertainty_benchmark_300_draft.jsonl"
    verify_benchmark_isolation(PILOT_CANDIDATES, benchmark_path, max_allowed_sim=0.75)
    
    # Save candidate questions
    cand_out_path = "data/pilot_30_candidates.jsonl"
    with open(cand_out_path, "w", encoding="utf-8") as f:
        for c in PILOT_CANDIDATES:
            f.write(json.dumps(c) + "\n")
    logger.info("Saved 30 pilot candidates to %s", cand_out_path)
    
    # 2. Multi-Response Generation
    client = OllamaClient()
    model_name = "qwen3:4b"
    generator = ResponseGenerator(client=client, model=model_name, temperature=0.7, top_p=0.9)
    
    raw_responses_out = "data/raw/pilot_30_responses.jsonl"
    os.makedirs(os.path.dirname(raw_responses_out), exist_ok=True)
    
    all_raw_data = []
    gen_times = []
    total_gens_attempted = 0
    successful_gens = 0
    failed_gens = 0
    
    logger.info("Starting multi-response generation for 30 questions (10 responses each, model: %s)...", model_name)
    with open(raw_responses_out, "w", encoding="utf-8") as raw_f:
        for idx, item in enumerate(PILOT_CANDIDATES):
            qid = item["id"]
            qtext = item["question"]
            label = item["target_label"]
            
            logger.info("[%02d/30] Generating for '%s' (%s: %s)...", idx + 1, qid, label, qtext[:40])
            q_start = time.time()
            
            # Canonical multi-response generation: T=0.0 initial, T=0.7 stochastic (10 total)
            resps = generator.generate_multi(qtext, target_total_count=10)
            q_elapsed = time.time() - q_start
            gen_times.append(q_elapsed)
            
            total_gens_attempted += len(resps)
            for r in resps:
                if r.get("text", "").strip():
                    successful_gens += 1
                else:
                    failed_gens += 1
                    
            raw_record = {
                "id": qid,
                "question": qtext,
                "target_label": label,
                "question_group_id": item["question_group_id"],
                "source_dataset": item["source_dataset"],
                "source_id": item["source_id"],
                "original_source_label": item["original_source_label"],
                "model": model_name,
                "generation_parameters": {
                    "model": model_name,
                    "target_total_count": 10,
                    "greedy_temperature": 0.0,
                    "stochastic_temperature": 0.7,
                    "top_p": 0.9,
                },
                "responses": resps,
                "q_elapsed_seconds": round(q_elapsed, 2),
            }
            all_raw_data.append(raw_record)
            raw_f.write(json.dumps(raw_record) + "\n")
            raw_f.flush()
            logger.info("    -> Done in %.2fs (%.2fs/resp)", q_elapsed, q_elapsed / len(resps))

    logger.info(
        "Generation complete! Total attempted: %d, Success: %d, Failed: %d, Total Gen Time: %.1fs",
        total_gens_attempted, successful_gens, failed_gens, sum(gen_times)
    )

    # 3. Embeddings & Clustering & Feature Extraction
    logger.info("Computing embeddings and extracting features...")
    embedding_engine = EmbeddingEngine(embedding_model="all-minilm")
    clusterer = SemanticClusterer(similarity_threshold=0.80)
    extractor = FeatureExtractor(clusterer=clusterer)
    
    features_out = "data/features/pilot_30_features.jsonl"
    unified_out = "data/pilot_30_unified.jsonl"
    os.makedirs(os.path.dirname(features_out), exist_ok=True)
    
    unified_records = []
    features_records = []
    
    with open(features_out, "w", encoding="utf-8") as feat_f, open(unified_out, "w", encoding="utf-8") as uni_f:
        for raw_item in all_raw_data:
            qid = raw_item["id"]
            qtext = raw_item["question"]
            label = raw_item["target_label"]
            resps = raw_item["responses"]
            resp_texts = [r["text"] for r in resps]
            
            # Embed responses
            resp_embs = embedding_engine.embed_batch(resp_texts)
            
            # Extract features
            feats = extractor.extract_features(
                question=qtext,
                responses=resp_texts,
                embeddings=resp_embs,
            )
            
            # Validate feature vector
            for fn in FEATURE_NAMES:
                if fn not in feats:
                    raise ValueError(f"Missing feature '{fn}' for question '{qid}'")
                val = feats[fn]
                if np.isnan(val) or np.isinf(val):
                    raise ValueError(f"Invalid feature value '{fn}={val}' for question '{qid}'")
            
            feat_record = {
                "id": qid,
                "question": qtext,
                "label": label,
                "features": feats,
            }
            features_records.append(feat_record)
            feat_f.write(json.dumps(feat_record) + "\n")
            feat_f.flush()
            
            # Unified record
            unified_record = {
                "id": qid,
                "question": qtext,
                "target_label": label,
                "question_group_id": raw_item["question_group_id"],
                "source_dataset": raw_item["source_dataset"],
                "source_id": raw_item["source_id"],
                "original_source_label": raw_item["original_source_label"],
                "mapping_rationale": next(c["mapping_rationale"] for c in PILOT_CANDIDATES if c["id"] == qid),
                "model_config": raw_item["generation_parameters"],
                "responses": resps,
                "features": feats,
            }
            unified_records.append(unified_record)
            uni_f.write(json.dumps(unified_record) + "\n")
            uni_f.flush()

    total_pipeline_time = time.time() - total_start
    logger.info("Pipeline complete! All 30 unified records written to %s in %.1fs", unified_out, total_pipeline_time)
    
    # 4. Statistical & Distribution Analysis
    compute_and_log_statistics(unified_records, gen_times, total_gens_attempted, successful_gens, failed_gens, total_pipeline_time)


def compute_and_log_statistics(
    unified_records: List[Dict[str, Any]],
    gen_times: List[float],
    total_gens: int,
    success_gens: int,
    fail_gens: int,
    total_time: float,
):
    stats = {}
    classes = ["KNOWN", "AMBIGUOUS", "UNKNOWN"]
    
    print("\n" + "="*80)
    print("PILOT 30-QUESTION EXECUTION & FEATURE EVALUATION SUMMARY")
    print("="*80)
    print(f"Total Questions: {len(unified_records)} (10 per class)")
    print(f"Total Generations: {total_gens} (Success: {success_gens}, Failed: {fail_gens})")
    print(f"Total Pipeline Runtime: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"Mean Generation Time per Question (10 resps): {np.mean(gen_times):.2f}s (std: {np.std(gen_times):.2f}s)")
    print(f"Mean Generation Time per Single Response: {np.mean(gen_times)/10:.2f}s")
    
    # Response lengths
    all_lens = []
    class_lens = {c: [] for c in classes}
    for r in unified_records:
        lens = [len(resp["text"]) for resp in r["responses"]]
        all_lens.extend(lens)
        class_lens[r["target_label"]].extend(lens)
        
    print(f"\nResponse Character Lengths: Overall Mean={np.mean(all_lens):.1f} chars (std={np.std(all_lens):.1f})")
    for c in classes:
        print(f"  - {c:10s}: Mean={np.mean(class_lens[c]):.1f} chars (std={np.std(class_lens[c]):.1f})")

    # Feature distributions per class
    print("\n" + "-"*80)
    print("CLASS-WISE FEATURE DISTRIBUTIONS (Mean ± Std | Median)")
    print("-"*80)
    header = f"{'Feature Name':26s} | {'KNOWN':17s} | {'AMBIGUOUS':17s} | {'UNKNOWN':17s}"
    print(header)
    print("-" * len(header))
    
    feature_table = {}
    for feat in FEATURE_NAMES:
        feature_table[feat] = {}
        row_str = f"{feat:26s} | "
        for c in classes:
            vals = [r["features"][feat] for r in unified_records if r["target_label"] == c]
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals))
            med_v = float(np.median(vals))
            feature_table[feat][c] = {"mean": mean_v, "std": std_v, "median": med_v}
            row_str += f"{mean_v:6.3f}±{std_v:5.3f} ({med_v:5.3f}) | "
        print(row_str)

    # Save summary json
    summary_out = "data/pilot_30_evaluation_metrics.json"
    summary_data = {
        "execution_metrics": {
            "total_questions": len(unified_records),
            "total_generations_attempted": total_gens,
            "successful_generations": success_gens,
            "failed_generations": fail_gens,
            "total_pipeline_seconds": round(total_time, 2),
            "mean_gen_seconds_per_question": round(float(np.mean(gen_times)), 2),
            "mean_gen_seconds_per_response": round(float(np.mean(gen_times)/10), 2),
        },
        "response_lengths": {
            "overall_mean_chars": round(float(np.mean(all_lens)), 2),
            "class_mean_chars": {c: round(float(np.mean(class_lens[c])), 2) for c in classes},
        },
        "feature_distributions": feature_table,
    }
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nEvaluation metrics written to {summary_out}")


if __name__ == "__main__":
    run_pipeline()
