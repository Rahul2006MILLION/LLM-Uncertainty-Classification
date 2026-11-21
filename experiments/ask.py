#!/usr/bin/env python3
"""Interactive and CLI Adaptive QA Pipeline for LLM Uncertainty Classification."""

import sys
import os
import time
import json
import argparse
import datetime
import logging
from typing import Dict, Any, List, Optional
import yaml

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generation.ollama_client import OllamaClient, OllamaClientError
from src.generation.generator import ResponseGenerator
from src.preprocessing.embeddings import EmbeddingEngine
from src.features.clustering import SemanticClusterer
from src.features.entropy import compute_semantic_entropy_metrics
from src.features.feature_extractor import FeatureExtractor
from src.classification.router import AdaptiveRouter
from src.classification.classifier import UncertaintyClassifier
from src.classification.ambiguity_assessor import CriterionAmbiguityAssessor
from src.classification.inaccessibility_assessor import InaccessibilityAssessor
from src.classification.fast_path_assessor import FastPathAnswerabilityAssessor
from src.classification.clarity_assessor import QuestionClarityAssessor
from src.classification.hitl_engine import HITLEngine, ClarificationDecision, ClarificationState, ConversationState, AmbiguityType
from src.classification.hitl_prompts import format_clarification_prompt
from data.schema import ExperimentRecord

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ask")


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """Load configuration YAML with safe fallbacks."""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def format_terminal_box(title: str, lines: List[str], width: int = 70) -> str:
    """Format clean ASCII box for readable terminal display."""
    border = "=" * width
    sub_border = "-" * width
    out = [border, f"  {title.center(width - 4)}", sub_border]
    for line in lines:
        out.append(f"  {line}")
    out.append(border)
    return "\n".join(out)


def run_pipeline(
    question: str,
    config: Optional[Dict[str, Any]] = None,
    force_multi_response: bool = False,
    generator: Optional[Any] = None,
    classifier: Optional[Any] = None,
    router: Optional[Any] = None,
    embedder: Optional[Any] = None,
    interactive_hitl: bool = False,
    hitl_engine: Optional[Any] = None,
    conversation_state: Optional[ConversationState] = None,
    clarification_callback: Optional[Any] = None,
) -> ExperimentRecord:
    """Execute the end-to-end adaptive QA pipeline on a given question."""
    start_total_t = time.time()
    cfg = config or load_config()

    if conversation_state is None:
        conversation_state = ConversationState(original_question=question, current_question=question)
    if interactive_hitl and hitl_engine is None:
        hitl_engine = HITLEngine()

    ollama_cfg = cfg.get("ollama", {})
    host = ollama_cfg.get("host", "http://localhost:11434")
    gen_model = ollama_cfg.get("generation_model", "qwen2.5:3b")
    emb_model = ollama_cfg.get("embedding_model", "all-minilm")
    temperature = float(ollama_cfg.get("temperature", 0.7))
    top_p = float(ollama_cfg.get("top_p", 0.9))

    pipe_cfg = cfg.get("pipeline", {})
    routing_cfg = cfg.get("routing", {})
    fast_path_enabled = (
        pipe_cfg.get("fast_path_enabled", routing_cfg.get("fast_path_enabled", True))
    ) and not force_multi_response
    multi_count = int(pipe_cfg.get("multi_response_count", routing_cfg.get("multi_response_count", 10)))

    clust_cfg = cfg.get("clustering", {})
    sim_threshold = float(clust_cfg.get("similarity_threshold", 0.80))
    linkage = clust_cfg.get("linkage", "average")

    out_cfg = cfg.get("output_selection", {})
    expl_threshold = float(out_cfg.get("explanation_threshold", 0.25))

    clf_cfg = cfg.get("classifier", {})
    model_save_path = clf_cfg.get("model_save_path", "models/uncertainty_classifier.joblib")

    # 1. Initialize Clients and Modules
    client = None
    if generator is None or embedder is None:
        client = OllamaClient(host=host, request_timeout=float(ollama_cfg.get("request_timeout", 45.0)))

    if generator is None:
        if not client.is_available():
            raise OllamaClientError(
                f"Cannot connect to Ollama at {host}. Please verify that the Ollama service is running."
            )
        generator = ResponseGenerator(client=client, model=gen_model, temperature=temperature, top_p=top_p)

    if router is None:
        router = AdaptiveRouter(fast_path_enabled=fast_path_enabled)

    clusterer = SemanticClusterer(similarity_threshold=sim_threshold, linkage=linkage)
    feature_extractor = FeatureExtractor(clusterer=clusterer)
    if embedder is None:
        embedder = EmbeddingEngine(client=client, embedding_model=emb_model)

    # Load classifier if present and not provided
    if classifier is None:
        for candidate_path in [model_save_path, "models/uncertainty_classifier.joblib", "models/uncertainty_classifier.json"]:
            if os.path.exists(candidate_path):
                try:
                    classifier = UncertaintyClassifier.load(candidate_path)
                    break
                except Exception as e:
                    logger.warning("Could not load classifier from %s: %s", candidate_path, e)

    # 2. Step 0: Question Clarity Assessment
    clarity_assessment = QuestionClarityAssessor.assess(question)

    # 3. Step 1: Initial Response Generation
    t0 = time.time()
    initial_res = generator.generate_single(question, temperature=0.0)  # Greedy for initial response
    initial_text = initial_res["text"]

    # 4. Step 2: Route question
    decision = router.route(question, initial_text)

    # Gate 1: Early Clarification Gate (Polysemy & Missing Criteria)
    if interactive_hitl and hitl_engine is not None and conversation_state.can_clarify():
        early_dec = hitl_engine.evaluate_early(
            question=question,
            initial_response=initial_text,
            router_triggers=decision.triggers,
            conversation_state=conversation_state,
        )
        if early_dec.decision == ClarificationState.CLARIFICATION_REQUIRED:
            user_input = None
            if clarification_callback:
                user_input = clarification_callback(early_dec, conversation_state)
            if user_input:
                conversation_state, clarified_q, success = hitl_engine.apply_clarification(
                    conversation_state, user_input, early_dec
                )
                if success:
                    logger.info("Gate 1 clarification applied: '%s' -> '%s'", question, clarified_q)
                    return run_pipeline(
                        question=clarified_q,
                        config=config,
                        force_multi_response=force_multi_response,
                        generator=generator,
                        classifier=classifier,
                        router=router,
                        embedder=embedder,
                        interactive_hitl=interactive_hitl,
                        hitl_engine=hitl_engine,
                        conversation_state=conversation_state,
                        clarification_callback=clarification_callback,
                    )

    # 4. Path Branching
    if decision.path == "FAST_PATH":
        # Explicit fast-path answerability assessment
        # Avoids feeding single-response features into 10-response multi-class classifier
        fast_assessment = FastPathAnswerabilityAssessor.assess(
            question=question,
            response=initial_text,
        )

        if not fast_assessment.is_assessed or fast_assessment.predicted_label is None:
            # Uncertain fast-path response: escalate to multi-response consistency analysis
            logger.info("Fast path assessment uncertain for '%s'; escalating to multi-response analysis.", question)
            decision.path = "MULTI_RESPONSE"
        else:
            elapsed_total = time.time() - start_total_t
            # Extract genuine single-response features for logging/record schema
            single_emb = embedder.embed_batch([initial_text])
            features = feature_extractor.extract_features(
                question=question,
                responses=[initial_text],
                embeddings=single_emb,
            )

            pred_label = fast_assessment.predicted_label
            pred_conf = fast_assessment.confidence
            clf_status = fast_assessment.status

            # Fast path epistemic and interaction outcome determination
            if conversation_state.round_count >= conversation_state.max_rounds:
                fast_decision = "UNRESOLVED"
                fast_interaction = "MAX_ROUNDS_REACHED"
                fast_ambig = conversation_state.detected_ambiguity_type or "UNDERSPECIFIED_CONTEXT"
                fast_reason = "Maximum clarification rounds reached without ambiguity resolution."
            elif conversation_state.user_declined:
                fast_decision = "UNRESOLVED"
                fast_interaction = "USER_DECLINED"
                fast_ambig = conversation_state.detected_ambiguity_type or "UNDERSPECIFIED_CONTEXT"
                fast_reason = "Clarification declined or skipped by user; ambiguity remains unresolved."
            elif conversation_state.is_clarified:
                if pred_label == "UNKNOWN":
                    fast_decision = "ABSTAIN"
                    fast_interaction = "RESOLVED"
                    fast_ambig = conversation_state.detected_ambiguity_type
                    fast_reason = fast_assessment.reason or "Epistemic refusal or unanswerable query on clarified fast path."
                else:
                    fast_decision = "ANSWERABLE"
                    fast_interaction = "RESOLVED"
                    fast_ambig = conversation_state.detected_ambiguity_type
                    fast_reason = "Ambiguity successfully resolved through human clarification."
            elif pred_label == "AMBIGUOUS" or clarity_assessment.status == "AMBIGUOUS":
                fast_decision = "CLARIFICATION_REQUIRED"
                fast_interaction = "PENDING"
                fast_ambig = clarity_assessment.ambiguity_type or conversation_state.detected_ambiguity_type or "MISSING_REFERENT"
                fast_reason = clarity_assessment.reason or fast_assessment.reason
            else:
                fast_decision = "ABSTAIN" if pred_label == "UNKNOWN" else "ANSWERABLE"
                fast_interaction = "NOT_REQUIRED"
                fast_ambig = None
                fast_reason = fast_assessment.reason if pred_label == "UNKNOWN" else None

            rec = ExperimentRecord(
                timestamp=datetime.datetime.now().isoformat(),
                question=question,
                path="FAST_PATH",
                final_answer=initial_text,
                response_count=1,
                cluster_count=1,
                dominant_cluster_agreement=1.0,
                second_cluster_agreement=0.0,
                agreement_margin=1.0,
                semantic_entropy=0.0,
                normalized_entropy=0.0,
                mean_pairwise_similarity=1.0,
                elapsed_seconds=elapsed_total,
                classifier_status=clf_status,
                predicted_label=pred_label,
                confidence=pred_conf,
                features=features,
                responses=[initial_res],
                clusters={"labels": [0], "cluster_counts": [1], "dominant_rep_idx": 0, "second_rep_idx": None},
                model=gen_model,
                embedding_model=emb_model,
                original_question=conversation_state.original_question,
                clarified_question=conversation_state.current_question if conversation_state.is_clarified else None,
                is_clarified=conversation_state.is_clarified,
                clarification_rounds=conversation_state.round_count,
                clarification_history=conversation_state.history,
                clarification_decision=fast_decision,
                interaction_status=fast_interaction,
                ambiguity_type=fast_ambig,
                clarification_reason=fast_reason,
                clarity_status=clarity_assessment.status,
                clarity_confidence=clarity_assessment.confidence,
                clarification_required=clarity_assessment.clarification_required,
            )
            return rec

    # 5. Multi-Response Path: exactly 10 responses total
    all_responses = generator.generate_multi(
        question=question,
        initial_response=initial_res,
        target_total_count=multi_count,
    )
    response_texts = [r["text"] for r in all_responses]

    # 6. Preprocessing and Semantic Embeddings
    embeddings = embedder.embed_batch(response_texts)

    # 7. Semantic Clustering
    cluster_res = clusterer.cluster_responses(response_texts, embeddings)
    cluster_counts = cluster_res["cluster_counts"]
    dom_rep_idx = cluster_res["dominant_representative_idx"]
    sec_rep_idx = cluster_res["second_representative_idx"]

    # 8. Entropy, Agreement & Features
    features = feature_extractor.extract_features(
        question=question,
        responses=response_texts,
        embeddings=embeddings,
        cluster_result=cluster_res,
    )

    # 9. Output Selection & Synthesis
    dom_text = response_texts[dom_rep_idx] if dom_rep_idx >= 0 and dom_rep_idx < len(response_texts) else initial_text
    p1 = features["dominant_agreement"]
    p2 = features["second_agreement"]

    if p2 >= expl_threshold and sec_rep_idx is not None and sec_rep_idx < len(response_texts):
        sec_text = response_texts[sec_rep_idx]
        final_answer = (
            f"Multiple plausible interpretations detected:\n"
            f"  • [Interpretation 1 ({p1:.0%} consensus)]: {dom_text}\n"
            f"  • [Interpretation 2 ({p2:.0%} consensus)]: {sec_text}"
        )
    else:
        final_answer = dom_text

    # 10. Classifier Inference & Criterion Ambiguity Assessment
    clf_status = "CLASSIFIER NOT TRAINED"
    pred_label = None
    pred_conf = None
    if classifier is not None and classifier.is_trained:
        pred_res = classifier.predict(features)
        clf_status = pred_res.get("status", "PREDICTED")
        pred_label = pred_res.get("predicted_label")
        pred_conf = pred_res.get("confidence")

    # Evaluate Criterion Ambiguity: decouple response consensus from objective truth
    ambig_check = CriterionAmbiguityAssessor.assess(
        question=question,
        responses=response_texts,
        features=features,
    )
    if ambig_check.is_ambiguous and pred_label != "AMBIGUOUS":
        pred_label = "AMBIGUOUS"
        pred_conf = ambig_check.confidence
        clf_status = f"ASSESSED_AMBIGUOUS_RULE_HEURISTIC ({ambig_check.evidence[0]})"

    # Evaluate Inaccessibility Safeguard: prevent multi-cluster refusal masquerading as ambiguity
    inacc_check = InaccessibilityAssessor.assess(
        question=question,
        responses=response_texts,
        features=features,
    )
    if inacc_check.is_inaccessible:
        pred_label = "UNKNOWN"
        pred_conf = inacc_check.confidence
        clf_status = f"ASSESSED_INACCESSIBLE_RULE_HEURISTIC ({inacc_check.evidence[0]})"

    # Gate 2: Post-Consistency Clarification Gate (Cluster Polysemy & Unstated Criteria)
    post_dec = None
    if interactive_hitl and hitl_engine is not None and conversation_state.can_clarify():
        post_dec = hitl_engine.evaluate_post_consistency(
            question=question,
            responses=response_texts,
            features=features,
            cluster_result=cluster_res,
            classifier_label=pred_label,
            inaccessibility_assessment=inacc_check,
            ambiguity_assessment=ambig_check,
            conversation_state=conversation_state,
        )
        if post_dec.decision == ClarificationState.CLARIFICATION_REQUIRED:
            user_input = None
            if clarification_callback:
                user_input = clarification_callback(post_dec, conversation_state)
            if user_input:
                conversation_state, clarified_q, success = hitl_engine.apply_clarification(
                    conversation_state, user_input, post_dec
                )
                if success:
                    logger.info("Gate 2 clarification applied: '%s' -> '%s'", question, clarified_q)
                    return run_pipeline(
                        question=clarified_q,
                        config=config,
                        force_multi_response=force_multi_response,
                        generator=generator,
                        classifier=classifier,
                        router=router,
                        embedder=embedder,
                        interactive_hitl=interactive_hitl,
                        hitl_engine=hitl_engine,
                        conversation_state=conversation_state,
                        clarification_callback=clarification_callback,
                    )

    # 11. Final Decision & Synthesis Formatting
    if post_dec and post_dec.decision == ClarificationState.MODEL_DISAGREEMENT:
        sec_text = response_texts[sec_rep_idx] if (sec_rep_idx is not None and sec_rep_idx < len(response_texts)) else "Alternative claim"
        final_answer = (
            f"Conflicting factual claims detected across model generations (model disagreement / unreliability):\n"
            f"  • Claim 1 ({p1:.0%} consensus): {dom_text}\n"
            f"  • Claim 2 ({p2:.0%} consensus): {sec_text}"
        )

    # Determine epistemic decision, interaction status, and ambiguity provenance
    if conversation_state.round_count >= conversation_state.max_rounds:
        rec_interaction_status = "MAX_ROUNDS_REACHED"
        rec_decision = "UNRESOLVED"
        rec_ambig_type = conversation_state.detected_ambiguity_type or (post_dec.ambiguity_type.value if (post_dec and post_dec.ambiguity_type != AmbiguityType.NONE) else "UNDERSPECIFIED_CONTEXT")
        rec_reason = "Maximum clarification rounds reached without ambiguity resolution."
    elif conversation_state.user_declined:
        rec_interaction_status = "USER_DECLINED"
        rec_decision = "UNRESOLVED"
        rec_ambig_type = conversation_state.detected_ambiguity_type or (post_dec.ambiguity_type.value if (post_dec and post_dec.ambiguity_type != AmbiguityType.NONE) else "UNDERSPECIFIED_CONTEXT")
        rec_reason = "Clarification declined or skipped by user; ambiguity remains unresolved."
    elif conversation_state.is_clarified:
        # Clarified query evaluated through post-consistency analysis
        if post_dec and post_dec.decision == ClarificationState.ANSWERABLE:
            rec_decision = "ANSWERABLE"
            rec_interaction_status = "RESOLVED"
            rec_ambig_type = conversation_state.detected_ambiguity_type
            rec_reason = "Ambiguity successfully resolved through human clarification."
        elif post_dec and post_dec.decision == ClarificationState.MODEL_DISAGREEMENT:
            rec_decision = "MODEL_DISAGREEMENT"
            rec_interaction_status = "RESOLVED"
            rec_ambig_type = conversation_state.detected_ambiguity_type or post_dec.ambiguity_type.value
            rec_reason = post_dec.reason or "Response divergence reflects factual inconsistency on clarified query."
        elif post_dec and post_dec.decision == ClarificationState.ABSTAIN:
            rec_decision = "ABSTAIN"
            rec_interaction_status = "RESOLVED"
            rec_ambig_type = conversation_state.detected_ambiguity_type or post_dec.ambiguity_type.value
            rec_reason = post_dec.reason or "Clarified query identified as epistemically unanswerable or inaccessible."
        elif post_dec and post_dec.decision == ClarificationState.CLARIFICATION_REQUIRED:
            if conversation_state.can_clarify():
                rec_decision = "CLARIFICATION_REQUIRED"
                rec_interaction_status = "PENDING"
            else:
                rec_decision = "UNRESOLVED"
                rec_interaction_status = "MAX_ROUNDS_REACHED"
            rec_ambig_type = conversation_state.detected_ambiguity_type or post_dec.ambiguity_type.value
            rec_reason = post_dec.reason or "Ambiguity persists following clarification attempt."
        elif post_dec and post_dec.decision == ClarificationState.UNRESOLVED:
            rec_decision = "UNRESOLVED"
            rec_interaction_status = "MAX_ROUNDS_REACHED" if conversation_state.round_count >= conversation_state.max_rounds else "USER_DECLINED"
            rec_ambig_type = conversation_state.detected_ambiguity_type or post_dec.ambiguity_type.value
            rec_reason = post_dec.reason or "Clarification ended without ambiguity resolution."
        else:
            rec_decision = "ANSWERABLE"
            rec_interaction_status = "RESOLVED"
            rec_ambig_type = conversation_state.detected_ambiguity_type
            rec_reason = "Ambiguity successfully resolved through human clarification."
    elif post_dec and post_dec.decision == ClarificationState.MODEL_DISAGREEMENT:
        rec_interaction_status = "NOT_REQUIRED"
        rec_decision = "MODEL_DISAGREEMENT"
        rec_ambig_type = post_dec.ambiguity_type.value if post_dec.ambiguity_type != AmbiguityType.NONE else "MODEL_DISAGREEMENT"
        rec_reason = post_dec.reason
    elif post_dec and post_dec.decision == ClarificationState.ABSTAIN:
        rec_interaction_status = "NOT_REQUIRED"
        rec_decision = "ABSTAIN"
        rec_ambig_type = post_dec.ambiguity_type.value if post_dec.ambiguity_type != AmbiguityType.NONE else None
        rec_reason = post_dec.reason
    elif post_dec and post_dec.decision == ClarificationState.CLARIFICATION_REQUIRED:
        if conversation_state.can_clarify():
            rec_decision = "CLARIFICATION_REQUIRED"
            rec_interaction_status = "PENDING"
        else:
            rec_decision = "UNRESOLVED"
            rec_interaction_status = "MAX_ROUNDS_REACHED"
        rec_ambig_type = post_dec.ambiguity_type.value if post_dec.ambiguity_type != AmbiguityType.NONE else None
        rec_reason = post_dec.reason
    elif post_dec and post_dec.decision == ClarificationState.UNRESOLVED:
        rec_interaction_status = "NOT_REQUIRED"
        rec_decision = "UNRESOLVED"
        rec_ambig_type = post_dec.ambiguity_type.value if post_dec.ambiguity_type != AmbiguityType.NONE else None
        rec_reason = post_dec.reason
    elif pred_label == "AMBIGUOUS" and not conversation_state.is_clarified:
        rec_interaction_status = "PENDING" if conversation_state.can_clarify() else "NOT_REQUIRED"
        rec_decision = "CLARIFICATION_REQUIRED" if conversation_state.can_clarify() else "UNRESOLVED"
        rec_ambig_type = clarity_assessment.ambiguity_type or (post_dec.ambiguity_type.value if (post_dec and post_dec.ambiguity_type != AmbiguityType.NONE) else "UNDERSPECIFIED_CONTEXT")
        rec_reason = clarity_assessment.reason or (post_dec.reason if post_dec else "Ambiguity requires context resolution.")
    else:
        rec_interaction_status = "NOT_REQUIRED"
        rec_decision = "ANSWERABLE"
        rec_ambig_type = post_dec.ambiguity_type.value if (post_dec and post_dec.ambiguity_type != AmbiguityType.NONE) else None
        rec_reason = post_dec.reason if post_dec else None

    elapsed_total = time.time() - start_total_t

    # Serialize experiment record
    rec = ExperimentRecord(
        timestamp=datetime.datetime.now().isoformat(),
        question=question,
        path="MULTI_RESPONSE",
        final_answer=final_answer,
        response_count=len(all_responses),
        cluster_count=int(features["cluster_count"]),
        dominant_cluster_agreement=p1,
        second_cluster_agreement=p2,
        agreement_margin=features["agreement_margin"],
        semantic_entropy=features["semantic_entropy"],
        normalized_entropy=features["normalized_entropy"],
        mean_pairwise_similarity=features["mean_pairwise_similarity"],
        elapsed_seconds=elapsed_total,
        classifier_status=clf_status,
        predicted_label=pred_label,
        confidence=pred_conf,
        features=features,
        responses=all_responses,
        clusters={
            "labels": cluster_res["labels"],
            "cluster_counts": cluster_counts,
            "dominant_rep_idx": dom_rep_idx,
            "second_rep_idx": sec_rep_idx,
        },
        model=gen_model,
        embedding_model=emb_model,
        original_question=conversation_state.original_question,
        clarified_question=conversation_state.current_question if conversation_state.is_clarified else None,
        is_clarified=conversation_state.is_clarified,
        clarification_rounds=conversation_state.round_count,
        clarification_history=conversation_state.history,
        clarification_decision=rec_decision,
        interaction_status=rec_interaction_status,
        ambiguity_type=rec_ambig_type,
        clarification_reason=rec_reason,
        clarity_status=clarity_assessment.status,
        clarity_confidence=clarity_assessment.confidence,
        clarification_required=clarity_assessment.clarification_required,
    )
    return rec


def main():
    parser = argparse.ArgumentParser(description="Ask questions using adaptive LLM uncertainty pipeline.")
    parser.add_argument("question", nargs="*", help="Question to ask (if omitted, prompts interactively)")
    parser.add_argument("-q", "--query", type=str, default=None, help="Question text")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--force-multi", action="store_true", help="Force multi-response analysis (bypass fast path)")
    parser.add_argument("--interactive-hitl", dest="interactive_hitl", action="store_true", help="Enable Human-in-the-Loop clarification for ambiguous queries")
    parser.add_argument("--no-hitl", dest="interactive_hitl", action="store_false", help="Disable Human-in-the-Loop clarification")
    parser.set_defaults(interactive_hitl=True)
    parser.add_argument("--json", action="store_true", help="Output raw JSON record to stdout")
    args = parser.parse_args()

    # Determine question
    q_tokens = args.question
    question = args.query or (" ".join(q_tokens) if q_tokens else None)

    if not question:
        try:
            print("\nEnter question (or Ctrl+C to exit):")
            question = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)

    if not question:
        print("Error: Empty question provided.")
        sys.exit(1)

    config = load_config(args.config)

    def terminal_clarification_callback(decision: ClarificationDecision, state: ConversationState) -> Optional[str]:
        prompt_text = format_clarification_prompt(
            question=decision.clarification_prompt or "Please provide additional context for this question:",
            why_it_matters=decision.why_it_matters,
            options=decision.options,
            round_num=state.round_count + 1,
            max_rounds=state.max_rounds,
        )
        print(prompt_text)
        try:
            val = input("> ").strip()
            return val if val else "skip"
        except (KeyboardInterrupt, EOFError):
            print("\nClarification skipped by user.")
            return "skip"

    try:
        record = run_pipeline(
            question=question,
            config=config,
            force_multi_response=args.force_multi,
            interactive_hitl=args.interactive_hitl,
            clarification_callback=terminal_clarification_callback if args.interactive_hitl else None,
        )
    except Exception as e:
        print(f"\n[PIPELINE ERROR]: {e}")
        sys.exit(1)

    # Save timestamped experiment record
    results_dir = config.get("storage", {}).get("results_dir", "experiments/results")
    os.makedirs(results_dir, exist_ok=True)
    ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_filename = os.path.join(results_dir, f"ask_{ts_str}.json")
    with open(out_filename, "w", encoding="utf-8") as f:
        json.dump(record.model_dump(), f, indent=2)

    if args.json:
        print(json.dumps(record.model_dump(), indent=2))
        return

    # Print clean terminal report
    lines = [
        f"QUESTION:          {record.question}",
        f"EXECUTION PATH:    {record.path}",
        f"TOTAL RESPONSES:   {record.response_count}",
        f"CLUSTERS FOUND:    {record.cluster_count}",
        f"DOMINANT AGREEMENT:{record.dominant_cluster_agreement:.1%}",
        f"SECOND AGREEMENT:  {record.second_cluster_agreement:.1%}",
        f"AGREEMENT MARGIN:  {record.agreement_margin:.1%}",
        f"SEMANTIC ENTROPY:  {record.semantic_entropy:.3f} nats (Norm: {record.normalized_entropy:.3f})",
        f"MEAN PAIRWISE SIM: {record.mean_pairwise_similarity:.3f}",
        f"ELAPSED TIME:      {record.elapsed_seconds:.2f}s",
        f"CLASSIFIER STATUS: {record.classifier_status}",
    ]
    if record.predicted_label:
        lines.append(f"PREDICTED CLASS:   {record.predicted_label} (conf: {record.confidence:.2f})")

    # Diagnostic Question Clarity Section
    if record.clarity_status == "UNAMBIGUOUS":
        lines.append(f"QUESTION CLARITY:  UNAMBIGUOUS")
        lines.append(f"CLARITY CONFIDENCE:[{record.clarity_confidence:.2f}]")
    elif record.clarity_status == "AMBIGUOUS":
        lines.append(f"QUESTION CLARITY:  AMBIGUOUS")
        if record.ambiguity_type:
            lines.append(f"AMBIGUITY TYPE:    {record.ambiguity_type}")
        req_str = "YES" if record.clarification_required else "NO"
        lines.append(f"CLARIFICATION REQUIRED: {req_str}")

    if record.is_clarified:
        lines.append(f"ORIGINAL QUERY:    {record.original_question}")
        lines.append(f"CLARIFIED QUERY:   {record.clarified_question}")
        lines.append(f"CLARIFICATION RDS: {record.clarification_rounds}")
    elif record.clarification_decision:
        lines.append(f"HITL DECISION:     {record.clarification_decision}")
    if record.ambiguity_type:
        lines.append(f"AMBIGUITY TYPE:    {record.ambiguity_type}")
    lines.append(f"SAVED RECORD:      {out_filename}")
    lines.append("")
    lines.append("FINAL ANSWER:")
    # Indent answer lines
    for al in record.final_answer.split("\n"):
        lines.append(f"  {al}")

    print("\n" + format_terminal_box("ADAPTIVE QA UNCERTAINTY PIPELINE", lines) + "\n")


if __name__ == "__main__":
    main()
