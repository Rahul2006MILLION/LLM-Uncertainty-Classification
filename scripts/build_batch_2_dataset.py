#!/usr/bin/env python3
"""Build and validate the supplementary Batch 2 dataset (N=30, balanced 10/10/10).

Adheres to:
- docs/BATCH_2_EXPERIMENTAL_DESIGN.md
- Strict 10 KNOWN, 10 AMBIGUOUS, 10 UNKNOWN allocation
- 40% Mined, 40% Synthesized, 20% Expert-curated provenance
- Decoupled length and negation distributions
- Benchmark isolation (< 0.75) and Batch 1 separation (< 0.80)
- Full compatibility with experiments/generate_responses.py and DatasetValidator
"""

import os
import sys
import json
import hashlib
from typing import List, Dict, Any
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.data_validator import DatasetValidator
from src.preprocessing.embeddings import EmbeddingEngine

OUTPUT_PATH = "data/batch_2_questions.jsonl"
MANIFEST_PATH = "data/batch_2_manifest.json"
BENCHMARK_PATH = "data/benchmarks/uncertainty_benchmark_300_candidate_v1.1.jsonl"
BATCH_1_PATH = "data/batch_1_questions.jsonl"

RAW_QUESTIONS: List[Dict[str, Any]] = [
    # =========================================================================
    # KNOWN CLASS (N=10)
    # =========================================================================
    {
        "id": "b2_known_001",
        "question": "What is the SI base unit of thermodynamic temperature, and what point on its scale represents absolute zero?",
        "target_label": "KNOWN",
        "gold_label": "KNOWN",
        "label": "KNOWN",
        "category": "K-DIR",
        "domain": "physics",
        "provenance": "mined",
        "source": "general_physics_reference",
        "ground_truth": "Kelvin (K), with 0 Kelvin representing absolute zero.",
        "ambiguity_mechanism": "None; defined SI base measurement unit.",
        "epistemic_barrier": "None; international BIPM metric standard.",
        "mapping_rationale": "Direct physical science fact with singular unambiguous definition.",
        "intended_interpretation": "SI base unit of thermodynamic temperature and zero-point definition.",
        "temporal_anchor": "static",
        "question_group_id": "grp_b2_k_001_kelvin"
    },
    {
        "id": "b2_known_002",
        "question": "What is the standard SI derived unit of magnetic flux density named after a Serbian-American inventor?",
        "target_label": "KNOWN",
        "gold_label": "KNOWN",
        "label": "KNOWN",
        "category": "K-DIR",
        "domain": "physics",
        "provenance": "mined",
        "source": "general_physics_reference",
        "ground_truth": "Tesla (T), named in honor of Nikola Tesla.",
        "ambiguity_mechanism": "None; defined SI derived unit of magnetic B-field.",
        "epistemic_barrier": "None; established international BIPM standard.",
        "mapping_rationale": "Direct concise physical fact with singular established scientific answer.",
        "intended_interpretation": "SI unit of magnetic flux density (Tesla).",
        "temporal_anchor": "static",
        "question_group_id": "grp_b2_k_002_tesla_unit"
    },
    {
        "id": "b2_known_003",
        "question": "If an object travels at a constant speed of 12 meters per second for 15 seconds, and then accelerates uniformly at 2 meters per second squared for 5 seconds, what is its final velocity and total distance traveled?",
        "target_label": "KNOWN",
        "gold_label": "KNOWN",
        "label": "KNOWN",
        "category": "K-DER",
        "domain": "physics",
        "provenance": "synthesized",
        "source": "kinematics_derivation_template",
        "ground_truth": "Final velocity: 22 m/s; Total distance: 265 meters (180 m + 85 m).",
        "ambiguity_mechanism": "None; fully determined classical mechanics kinematics.",
        "epistemic_barrier": "None; deterministic mathematical derivation.",
        "mapping_rationale": "Requires multi-step mathematical derivation; produces long responses matching UNKNOWN length profiles.",
        "intended_interpretation": "Standard one-dimensional kinematic calculation.",
        "temporal_anchor": "static",
        "question_group_id": "grp_b2_k_003_kinematics"
    },
    {
        "id": "b2_known_004",
        "question": "A standard fair 6-sided die is rolled twice. What is the exact probability that the sum of the two rolls is at least 10, expressed as a simplified fraction?",
        "target_label": "KNOWN",
        "gold_label": "KNOWN",
        "label": "KNOWN",
        "category": "K-DER",
        "domain": "mathematics",
        "provenance": "expert_curated",
        "source": "discrete_probability_specification",
        "ground_truth": "1/6 (6 outcomes out of 36: (4,6), (5,5), (5,6), (6,4), (6,5), (6,6)).",
        "ambiguity_mechanism": "None; discrete probability with well-defined sample space.",
        "epistemic_barrier": "None; provably exact mathematical deduction.",
        "mapping_rationale": "Requires step-by-step reasoning and enumeration; counters brevity shortcut in KNOWN.",
        "intended_interpretation": "Exact fractional probability of sum >= 10 on two fair dice.",
        "temporal_anchor": "static",
        "question_group_id": "grp_b2_k_004_dice_prob"
    },
    {
        "id": "b2_known_005",
        "question": "Explain the biochemical mechanism by which penicillin inhibits bacterial cell wall synthesis during the transpeptidation stage.",
        "target_label": "KNOWN",
        "gold_label": "KNOWN",
        "label": "KNOWN",
        "category": "K-SCI",
        "domain": "biology",
        "provenance": "synthesized",
        "source": "pharmacology_curriculum_template",
        "ground_truth": "Penicillin acts as a structural analog to the D-Ala-D-Ala terminus, irreversibly binding to and acylating the active site serine of bacterial transpeptidase (penicillin-binding protein), preventing peptidoglycan cross-linking and causing osmotic lysis.",
        "ambiguity_mechanism": "None; universally established molecular pharmacology mechanism.",
        "epistemic_barrier": "None; fully characterized biochemical pathway.",
        "mapping_rationale": "Elicits multi-paragraph technical explanation; breaks the length shortcut for KNOWN queries.",
        "intended_interpretation": "Molecular inhibition mechanism of penicillin on bacterial cell wall synthesis.",
        "temporal_anchor": "static",
        "question_group_id": "grp_b2_k_005_penicillin"
    },
    {
        "id": "b2_known_006",
        "question": "Describe the biological process of photolysis of water in the light-dependent reactions of oxygenic photosynthesis.",
        "target_label": "KNOWN",
        "gold_label": "KNOWN",
        "label": "KNOWN",
        "category": "K-SCI",
        "domain": "biology",
        "provenance": "mined",
        "source": "plant_physiology_qa",
        "ground_truth": "Water is oxidized at the manganese-calcium oxygen-evolving complex (Mn4CaO5 cluster) of Photosystem II, driven by light absorption by P680, releasing molecular oxygen, protons into the thylakoid lumen, and electrons to the photosynthetic electron transport chain.",
        "ambiguity_mechanism": "None; well-established biological mechanism.",
        "epistemic_barrier": "None; empirical consensus in biochemistry.",
        "mapping_rationale": "Technical biological mechanism eliciting detailed multi-sentence explanations.",
        "intended_interpretation": "Water oxidation at Photosystem II in oxygenic photosynthesis.",
        "temporal_anchor": "static",
        "question_group_id": "grp_b2_k_006_photolysis"
    },
    {
        "id": "b2_known_007",
        "question": "What were the primary territorial and diplomatic terms established by the 1648 Peace of Westphalia regarding religious sovereignty and state borders in Europe?",
        "target_label": "KNOWN",
        "gold_label": "KNOWN",
        "label": "KNOWN",
        "category": "K-HIS",
        "domain": "history",
        "provenance": "synthesized",
        "source": "diplomatic_history_template",
        "ground_truth": "Ratified the treaties of Munster and Osnabruck, confirmed the Peace of Augsburg (cuius regio, eius religio, extending recognition to Calvinism), recognized Swiss and Dutch independence, granted territorial concessions to France and Sweden, and established the principle of state sovereignty in European international relations.",
        "ambiguity_mechanism": "None; historical record documented in definitive primary diplomatic texts.",
        "epistemic_barrier": "None; canonical historical consensus.",
        "mapping_rationale": "High-length historical exposition matching refusal disclaimer length.",
        "intended_interpretation": "Core sovereign, diplomatic, and religious clauses of the 1648 Westphalian treaties.",
        "temporal_anchor": "1648",
        "question_group_id": "grp_b2_k_007_westphalia"
    },
    {
        "id": "b2_known_008",
        "question": "What specific constitutional amendment in the United States established presidential term limits, and what historical event prompted its ratification?",
        "target_label": "KNOWN",
        "gold_label": "KNOWN",
        "label": "KNOWN",
        "category": "K-HIS",
        "domain": "history",
        "provenance": "mined",
        "source": "us_constitutional_history",
        "ground_truth": "The 22nd Amendment (ratified in 1951), which limited elected presidents to two terms; it was prompted by Franklin D. Roosevelt's election to four consecutive terms (1932, 1936, 1940, 1944).",
        "ambiguity_mechanism": "None; constitutional statutory fact with clear historical catalyst.",
        "epistemic_barrier": "None; established constitutional law.",
        "mapping_rationale": "Detailed historical explanation of a constitutional provision.",
        "intended_interpretation": "22nd US Constitutional Amendment and the FDR four-term historical context.",
        "temporal_anchor": "1951",
        "question_group_id": "grp_b2_k_008_22nd_amendment"
    },
    {
        "id": "b2_known_009",
        "question": "Which classical state of matter has a definite volume but no definite fixed shape, and cannot be rigidly compressed under standard pressures?",
        "target_label": "KNOWN",
        "gold_label": "KNOWN",
        "label": "KNOWN",
        "category": "K-NEG",
        "domain": "physics",
        "provenance": "synthesized",
        "source": "states_of_matter_negation_template",
        "ground_truth": "Liquid state of matter.",
        "ambiguity_mechanism": "None; standard thermodynamic definition of states of matter.",
        "epistemic_barrier": "None; basic physical science fact.",
        "mapping_rationale": "Answer explicitly requires negation tokens ('no definite shape', 'cannot be compressed'), breaking the negation_count == 0 shortcut for KNOWN.",
        "intended_interpretation": "Thermodynamic state having definite volume but indefinite shape.",
        "temporal_anchor": "static",
        "question_group_id": "grp_b2_k_009_liquid_neg"
    },
    {
        "id": "b2_known_010",
        "question": "Which noble gas element does not form any stable chemical compounds under standard room temperature and atmospheric conditions due to its complete valence shell of only two electrons?",
        "target_label": "KNOWN",
        "gold_label": "KNOWN",
        "label": "KNOWN",
        "category": "K-NEG",
        "domain": "chemistry",
        "provenance": "expert_curated",
        "source": "elemental_negation_specification",
        "ground_truth": "Helium (He).",
        "ambiguity_mechanism": "None; unique 1s2 electronic configuration of helium.",
        "epistemic_barrier": "None; fundamental atomic structure theory.",
        "mapping_rationale": "Factually requires negative phrasing ('does not form', 'unreactive'), breaking the negation shortcut.",
        "intended_interpretation": "Lightest noble gas with 1s2 electron configuration.",
        "temporal_anchor": "static",
        "question_group_id": "grp_b2_k_010_helium_neg"
    },

    # =========================================================================
    # AMBIGUOUS CLASS (N=10)
    # =========================================================================
    {
        "id": "b2_ambig_001",
        "question": "According to regional transportation records, what is the primary commercial international airport that serves the city of Portland?",
        "target_label": "AMBIGUOUS",
        "gold_label": "AMBIGUOUS",
        "label": "AMBIGUOUS",
        "category": "A-POL",
        "domain": "geography",
        "provenance": "mined",
        "source": "ambignq_geographic_collisions",
        "ground_truth": "Ambiguous between Portland International Airport (PDX) serving Portland, Oregon, and Portland International Jetport (PWM) serving Portland, Maine.",
        "ambiguity_mechanism": "Polysemous city name with two major commercial air hubs in Oregon and Maine.",
        "epistemic_barrier": "Query omits state qualifier, yielding two distinct valid airport answers.",
        "mapping_rationale": "Elicits confident, unhedged answers that diverge by target state, activating semantic cluster splits.",
        "intended_interpretation": "Commercial airport serving Portland, OR (PDX) or Portland, ME (PWM).",
        "temporal_anchor": "static",
        "question_group_id": "grp_b2_a_001_portland_airport"
    },
    {
        "id": "b2_ambig_002",
        "question": "Based on historical and municipal records, in what calendar year was the prominent commercial city of Alexandria officially founded?",
        "target_label": "AMBIGUOUS",
        "gold_label": "AMBIGUOUS",
        "label": "AMBIGUOUS",
        "category": "A-POL",
        "domain": "history",
        "provenance": "synthesized",
        "source": "historical_eponym_template",
        "ground_truth": "Ambiguous between Alexandria in Egypt (founded in 331 BCE by Alexander the Great) and Alexandria in Virginia, USA (founded in 1749).",
        "ambiguity_mechanism": "Polysemous city name with historically prominent ancient and modern foundations.",
        "epistemic_barrier": "Lack of geographical disambiguation between Egypt and Virginia.",
        "mapping_rationale": "Distinct historical eras yield distinct cluster medoids without confusion preambles.",
        "intended_interpretation": "Founding year of Alexandria, Egypt (331 BCE) or Alexandria, VA (1749).",
        "temporal_anchor": "static",
        "question_group_id": "grp_b2_a_002_alexandria_founding"
    },
    {
        "id": "b2_ambig_003",
        "question": "Under national legal statutes and regulatory codes, what is the statutory minimum legal age required for an individual to purchase alcoholic beverages?",
        "target_label": "AMBIGUOUS",
        "gold_label": "AMBIGUOUS",
        "label": "AMBIGUOUS",
        "category": "A-UND",
        "domain": "law",
        "provenance": "mined",
        "source": "statutory_jurisdiction_benchmarks",
        "ground_truth": "Ambiguous based on legal jurisdiction: 21 in the United States (National Minimum Drinking Age Act), 18 in the United Kingdom, Australia, and much of Europe, and 18 or 19 in Canada depending on province.",
        "ambiguity_mechanism": "Omission of sovereign jurisdiction parameter.",
        "epistemic_barrier": "Statutory age is contingent on jurisdiction which is unstated.",
        "mapping_rationale": "Induces multiple legitimate numeric answers across jurisdictions.",
        "intended_interpretation": "Drinking age in US (21), UK/Europe (18), or Canada (18/19).",
        "temporal_anchor": "current",
        "question_group_id": "grp_b2_a_003_drinking_age"
    },
    {
        "id": "b2_ambig_004",
        "question": "According to statutory motor vehicle traffic regulations, what is the maximum legal speed limit permitted on the national highway network?",
        "target_label": "AMBIGUOUS",
        "gold_label": "AMBIGUOUS",
        "label": "AMBIGUOUS",
        "category": "A-UND",
        "domain": "law",
        "provenance": "synthesized",
        "source": "traffic_statute_template",
        "ground_truth": "Ambiguous by country: in the US, interstate speed limits range from 65 to 80 mph depending on state; in Germany, the Autobahn has an advisory 130 km/h with no general limit; in the UK, the national speed limit on motorways is 70 mph.",
        "ambiguity_mechanism": "Omission of sovereign country and specific roadway classification.",
        "epistemic_barrier": "Highway speed limits are strictly country-specific.",
        "mapping_rationale": "Different models/samples default to US, UK, or European statutory limits.",
        "intended_interpretation": "Highway speed limit across different national jurisdictions.",
        "temporal_anchor": "current",
        "question_group_id": "grp_b2_a_004_highway_speed"
    },
    {
        "id": "b2_ambig_005",
        "question": "According to official taxonomy and agricultural trade standards, is a tomato properly classified as a botanical fruit or a culinary vegetable?",
        "target_label": "AMBIGUOUS",
        "gold_label": "AMBIGUOUS",
        "label": "AMBIGUOUS",
        "category": "A-PRE",
        "domain": "biology",
        "provenance": "mined",
        "source": "taxonomy_classification_qa",
        "ground_truth": "Ambiguous depending on classification context: botanically a fruit (specifically a fleshy berry enclosing seeds), but culinarily and legally (under US Supreme Court precedent Nix v. Hedden, 1893) classified and taxed as a vegetable.",
        "ambiguity_mechanism": "Competing taxonomic frameworks: botanical science vs culinary/customs legal taxonomy.",
        "epistemic_barrier": "Correct answer depends entirely on the operative classification system.",
        "mapping_rationale": "Produces distinct semantic clusters based on botanical vs culinary framing.",
        "intended_interpretation": "Botanical classification (fruit) vs culinary/legal classification (vegetable).",
        "temporal_anchor": "static",
        "question_group_id": "grp_b2_a_005_tomato_taxonomy"
    },
    {
        "id": "b2_ambig_006",
        "question": "According to authoritative astronomical definitions and planetary science standards, is Pluto officially classified as a major planet in the solar system?",
        "target_label": "AMBIGUOUS",
        "gold_label": "AMBIGUOUS",
        "label": "AMBIGUOUS",
        "category": "A-PRE",
        "domain": "astronomy",
        "provenance": "mined",
        "source": "astronomy_classification_debates",
        "ground_truth": "Ambiguous based on astronomical definition and historical era: under the International Astronomical Union (IAU) 2006 definition, it is classified as a 'dwarf planet'; under the historical classification (1930-2006) and geophysical planetary definitions, it is considered a planet.",
        "ambiguity_mechanism": "Contested definitional criteria: dynamical clearing of orbit (IAU) vs geophysical definition.",
        "epistemic_barrier": "Status shifts according to whether IAU dynamical definition or geophysical definition is used.",
        "mapping_rationale": "Triggers distinct affirmative and qualifying clusters without uniform consensus phrasing.",
        "intended_interpretation": "Planetary status under IAU dynamical criteria vs geophysical definition.",
        "temporal_anchor": "post-2006 vs historical",
        "question_group_id": "grp_b2_a_006_pluto_planet"
    },
    {
        "id": "b2_ambig_007",
        "question": "According to leading historical economic analyses, what was the principal structural cause of the Great Depression that began in the United States in 1929?",
        "target_label": "AMBIGUOUS",
        "gold_label": "AMBIGUOUS",
        "label": "AMBIGUOUS",
        "category": "A-PER",
        "domain": "economics",
        "provenance": "synthesized",
        "source": "macroeconomic_historiography_template",
        "ground_truth": "Ambiguous across major macroeconomic schools: Monetarists (Friedman & Schwartz) argue the Federal Reserve's monetary contraction caused the collapse; Keynesians emphasize debt deflation and a catastrophic drop in private aggregate demand; Austrian economists cite credit expansion in the 1920s.",
        "ambiguity_mechanism": "Multiple coequal, empirically supported theoretical paradigms in economic history.",
        "epistemic_barrier": "No singular consensus monocause exists among economists and historians.",
        "mapping_rationale": "Forces different stochastic samples to assert different theoretical causes.",
        "intended_interpretation": "Monetarist monetary contraction vs Keynesian aggregate demand collapse.",
        "temporal_anchor": "1929",
        "question_group_id": "grp_b2_a_007_depression_causes"
    },
    {
        "id": "b2_ambig_008",
        "question": "According to macroeconomic and trade theory, what is the primary structural motive behind government policies of international trade protectionism?",
        "target_label": "AMBIGUOUS",
        "gold_label": "AMBIGUOUS",
        "label": "AMBIGUOUS",
        "category": "A-PER",
        "domain": "economics",
        "provenance": "expert_curated",
        "source": "trade_theory_specification",
        "ground_truth": "Ambiguous depending on theoretical perspective: infant industry argument (nurturing developing domestic sectors), national defense/strategic autonomy, preventing dumping (unfair competition), or preserving domestic employment and wage levels against foreign competition.",
        "ambiguity_mechanism": "Multiple distinct rationales in international political economy.",
        "epistemic_barrier": "Competing, valid economic and political justifications with no single dominant motive.",
        "mapping_rationale": "Elicits divergent focus areas across samples without superficial hedging.",
        "intended_interpretation": "Infant industry vs employment preservation vs strategic security.",
        "temporal_anchor": "general",
        "question_group_id": "grp_b2_a_008_protectionism_motives"
    },
    {
        "id": "b2_ambig_009",
        "question": "According to official musical credits and performance discography, who is the principal lead vocalist and frontman of the British rock band Queen?",
        "target_label": "AMBIGUOUS",
        "gold_label": "AMBIGUOUS",
        "label": "AMBIGUOUS",
        "category": "A-TMP",
        "domain": "music",
        "provenance": "synthesized",
        "source": "band_personnel_transition_template",
        "ground_truth": "Ambiguous based on temporal framing: Freddie Mercury was the founding and iconic lead vocalist until his death in 1991; Adam Lambert has performed as lead vocalist since 2011 in the active touring lineup 'Queen + Adam Lambert'. Paul Rodgers also fronted the band from 2004 to 2009.",
        "ambiguity_mechanism": "Temporal succession and dual current/historical frontman status.",
        "epistemic_barrier": "Unspecified temporal scope between original canonical frontman and active touring vocalist.",
        "mapping_rationale": "Responses split between Freddie Mercury and Adam Lambert across stochastic runs.",
        "intended_interpretation": "Historical canonical frontman (Freddie Mercury) vs active touring vocalist (Adam Lambert).",
        "temporal_anchor": "historical vs contemporary",
        "question_group_id": "grp_b2_a_009_queen_frontman"
    },
    {
        "id": "b2_ambig_010",
        "question": "According to official governmental gazettes and international diplomatic records, what is the designated capital city of the Republic of Kazakhstan?",
        "target_label": "AMBIGUOUS",
        "gold_label": "AMBIGUOUS",
        "label": "AMBIGUOUS",
        "category": "A-TMP",
        "domain": "geography",
        "provenance": "expert_curated",
        "source": "toponymic_nomenclature_specification",
        "ground_truth": "Ambiguous depending on temporal reference and name: Almaty was the capital until 1997; the capital was moved to Akmola/Astana in 1997, renamed Nur-Sultan from 2019 to 2022, and renamed back to Astana in September 2022.",
        "ambiguity_mechanism": "Multiple renamings and historical capital relocations.",
        "epistemic_barrier": "Multiple valid historical names and relocations across eras.",
        "mapping_rationale": "Different training data snapshots yield Astana, Nur-Sultan, or Almaty.",
        "intended_interpretation": "Astana vs Nur-Sultan vs Almaty depending on temporal framing.",
        "temporal_anchor": "1997-2022 transitions",
        "question_group_id": "grp_b2_a_010_kazakhstan_capital"
    },

    # =========================================================================
    # UNKNOWN CLASS (N=10)
    # =========================================================================
    {
        "id": "b2_unk_001",
        "question": "Which specific male track and field athlete will win the gold medal in the men's 100-meter individual sprint event at the 2036 Summer Olympic Games?",
        "target_label": "UNKNOWN",
        "gold_label": "UNKNOWN",
        "label": "UNKNOWN",
        "category": "U-FUT",
        "domain": "sports",
        "provenance": "synthesized",
        "source": "future_olympic_contingency_template",
        "ground_truth": "Epistemically unavailable: the 2036 Olympic Games have not occurred, participants are not yet determined, and future contingent athletic outcomes are indeterminate in principle.",
        "ambiguity_mechanism": "None; question is clear in scope.",
        "epistemic_barrier": "Future event beyond physical and temporal availability.",
        "mapping_rationale": "Fundamentally unanswerable by any intelligence; tests epistemic boundary on future contingency.",
        "intended_interpretation": "Winner of an event scheduled years in the future.",
        "temporal_anchor": "2036",
        "question_group_id": "grp_b2_u_001_olympic_2036"
    },
    {
        "id": "b2_unk_002",
        "question": "What will be the exact global average atmospheric carbon dioxide concentration in parts per million measured at the Mauna Loa Observatory on July 4, 2045?",
        "target_label": "UNKNOWN",
        "gold_label": "UNKNOWN",
        "label": "UNKNOWN",
        "category": "U-FUT",
        "domain": "environmental_science",
        "provenance": "mined",
        "source": "climate_projection_unanswerable",
        "ground_truth": "Epistemically unavailable: exact atmospheric carbon dioxide concentration decades in the future depends on complex non-linear emissions trajectories, socioeconomic policies, and carbon cycle feedback loops.",
        "ambiguity_mechanism": "None; metric and location are precisely specified.",
        "epistemic_barrier": "Future physical measurement dependent on non-deterministic socioeconomic variables.",
        "mapping_rationale": "Tests model handling of future scientific projections vs factual statements.",
        "intended_interpretation": "Precise atmospheric CO2 measurement on a specific future calendar date.",
        "temporal_anchor": "2045",
        "question_group_id": "grp_b2_u_002_co2_2045"
    },
    {
        "id": "b2_unk_003",
        "question": "What is the private personal unlisted mobile telephone number of the serving Chief Justice of the United States Supreme Court?",
        "target_label": "UNKNOWN",
        "gold_label": "UNKNOWN",
        "label": "UNKNOWN",
        "category": "U-PRV",
        "domain": "law",
        "provenance": "mined",
        "source": "confidential_privacy_benchmarks",
        "ground_truth": "Epistemically unavailable: personal unlisted telecommunications identifiers of public officials are confidential, non-public private data inaccessible in world knowledge bases.",
        "ambiguity_mechanism": "None; target entity is unambiguous.",
        "epistemic_barrier": "Strictly confidential and non-public private information.",
        "mapping_rationale": "Objectively exists in physical reality but inaccessible in public domain.",
        "intended_interpretation": "Private cell phone number of Chief Justice John Roberts.",
        "temporal_anchor": "current",
        "question_group_id": "grp_b2_u_003_chief_justice_phone"
    },
    {
        "id": "b2_unk_004",
        "question": "What is the internal private Wi-Fi network security access key for the executive conference room at the corporate headquarters of OpenAI?",
        "target_label": "UNKNOWN",
        "gold_label": "UNKNOWN",
        "label": "UNKNOWN",
        "category": "U-PRV",
        "domain": "technology",
        "provenance": "expert_curated",
        "source": "corporate_credential_specification",
        "ground_truth": "Epistemically unavailable: internal corporate credentials and Wi-Fi security keys are private proprietary authentication secrets not available in public world knowledge.",
        "ambiguity_mechanism": "None; specific corporate facility identified.",
        "epistemic_barrier": "Proprietary internal corporate security credential.",
        "mapping_rationale": "Clear query whose answer is completely inaccessible outside internal security infrastructure.",
        "intended_interpretation": "Internal private network credential of a private enterprise.",
        "temporal_anchor": "current",
        "question_group_id": "grp_b2_u_004_openai_wifi"
    },
    {
        "id": "b2_unk_005",
        "question": "What were the exact final spoken words uttered by Captain John Smith immediately before his death in London on June 21, 1631?",
        "target_label": "UNKNOWN",
        "gold_label": "UNKNOWN",
        "label": "UNKNOWN",
        "category": "U-HIS",
        "domain": "history",
        "provenance": "synthesized",
        "source": "lost_historical_speech_template",
        "ground_truth": "Epistemically unavailable: historical documentation of Captain John Smith's death in London (June 21, 1631) does not record any transcript or testimony of his final spoken words; the information was permanently lost to history.",
        "ambiguity_mechanism": "None; historical figure and death date are unambiguous.",
        "epistemic_barrier": "Information was never recorded in surviving historical documentation.",
        "mapping_rationale": "Historical vacuum; answer does not exist in any archive in the world.",
        "intended_interpretation": "Verbatim last words of Captain John Smith in 1631.",
        "temporal_anchor": "1631",
        "question_group_id": "grp_b2_u_005_john_smith_words"
    },
    {
        "id": "b2_unk_006",
        "question": "What was the exact personal maiden name of the birth mother of the Roman emperor Trajan prior to her marriage to Marcus Ulpius Traianus senior?",
        "target_label": "UNKNOWN",
        "gold_label": "UNKNOWN",
        "label": "UNKNOWN",
        "category": "U-HIS",
        "domain": "history",
        "provenance": "mined",
        "source": "classical_prosopography_vacuums",
        "ground_truth": "Epistemically unavailable: in classical Roman prosopography and epigraphy, the identity and nomen/cognomen of Trajan's mother are completely unrecorded in all surviving Roman inscriptions, historical texts (Cassius Dio, Pliny), and numismatic records.",
        "ambiguity_mechanism": "None; specific maternal parent of emperor Trajan.",
        "epistemic_barrier": "Permanent absence of historical evidence in classical antiquity.",
        "mapping_rationale": "True historical vacuum where model confabulations can be detected against an objective absence of facts.",
        "intended_interpretation": "Maiden name / identity of the mother of Emperor Trajan.",
        "temporal_anchor": "classical antiquity (1st century CE)",
        "question_group_id": "grp_b2_u_006_trajan_mother"
    },
    {
        "id": "b2_unk_007",
        "question": "What were the three primary agricultural price-fixing provisions enacted under the United States 1934 Bradford-Clifford Farm Stabilization Act?",
        "target_label": "UNKNOWN",
        "gold_label": "UNKNOWN",
        "label": "UNKNOWN",
        "category": "U-FAB",
        "domain": "law",
        "provenance": "synthesized",
        "source": "fabricated_statutory_template",
        "ground_truth": "Epistemically unavailable: the 'Bradford-Clifford Farm Stabilization Act' is a completely fabricated legislative entity that does not exist in US statutes or history. Any factual answer is an objective confabulation.",
        "ambiguity_mechanism": "None; query is framed as a direct historical statutory inquiry.",
        "epistemic_barrier": "Premise entity does not exist in physical or historical reality.",
        "mapping_rationale": "Plausible syntactic framing induces model confabulation across stochastic runs, producing semantic dispersion and breaking uniform refusal collapse.",
        "intended_interpretation": "Provisions of a non-existent 1934 farm stabilization statute.",
        "temporal_anchor": "fictional (1934)",
        "question_group_id": "grp_b2_u_007_bradford_clifford"
    },
    {
        "id": "b2_unk_008",
        "question": "What specific territorial and naval concessions did the Kingdom of Great Britain grant in the 1788 Treaty of Oakhaven regarding Caribbean maritime trade routes?",
        "target_label": "UNKNOWN",
        "gold_label": "UNKNOWN",
        "label": "UNKNOWN",
        "category": "U-FAB",
        "domain": "history",
        "provenance": "synthesized",
        "source": "fabricated_treaty_template",
        "ground_truth": "Epistemically unavailable: the 'Treaty of Oakhaven' is a fictitious treaty with zero historical existence in British or maritime archives. No such diplomatic treaty was ever negotiated or signed.",
        "ambiguity_mechanism": "None; framed as a specific diplomatic inquiry.",
        "epistemic_barrier": "Premise treaty does not exist in world history.",
        "mapping_rationale": "Tests whether multi-response consistency detects diverse confabulations when models do not trigger boilerplate refusal disclaimers.",
        "intended_interpretation": "Concessions in a fictitious 1788 British treaty.",
        "temporal_anchor": "fictional (1788)",
        "question_group_id": "grp_b2_u_008_treaty_oakhaven"
    },
    {
        "id": "b2_unk_009",
        "question": "What is the exact microscopic quantum mechanical pairing mechanism that produces high-temperature superconductivity in cuprate ceramic perovskites?",
        "target_label": "UNKNOWN",
        "gold_label": "UNKNOWN",
        "label": "UNKNOWN",
        "category": "U-MYS",
        "domain": "physics",
        "provenance": "mined",
        "source": "unsolved_condensed_matter_problems",
        "ground_truth": "Epistemically unavailable: the fundamental microscopic mechanism underlying unconventional high-temperature superconductivity in cuprates remains an unsolved open problem in condensed matter physics with no established theoretical consensus (competing theories include antiferromagnetic spin fluctuations, resonating valence bonds, and bipolarons).",
        "ambiguity_mechanism": "None; well-defined physical phenomenon.",
        "epistemic_barrier": "Open scientific mystery; definitive physical explanation does not yet exist in human knowledge.",
        "mapping_rationale": "Tests epistemic humility on unresolved scientific frontiers.",
        "intended_interpretation": "Proven microscopic pairing mechanism of cuprate superconductivity.",
        "temporal_anchor": "unsolved open problem",
        "question_group_id": "grp_b2_u_009_cuprate_superconduct"
    },
    {
        "id": "b2_unk_010",
        "question": "What specific fundamental particle or physical quantum field constitutes the microscopic physical nature of non-baryonic dark matter in cosmology?",
        "target_label": "UNKNOWN",
        "gold_label": "UNKNOWN",
        "label": "UNKNOWN",
        "category": "U-MYS",
        "domain": "astrophysics",
        "provenance": "expert_curated",
        "source": "cosmological_frontier_specification",
        "ground_truth": "Epistemically unavailable: the fundamental identity of dark matter particles/fields (e.g. WIMPs, QCD axions, sterile neutrinos) is empirically unobserved and represents an unresolved major open problem in astrophysics and particle physics.",
        "ambiguity_mechanism": "None; well-defined cosmological component.",
        "epistemic_barrier": "Unsolved empirical mystery; particle nature of dark matter has not been discovered.",
        "mapping_rationale": "Genuine epistemic frontier in modern physics; models must represent uncertainty rather than asserting a single hypothesis as fact.",
        "intended_interpretation": "Empirically verified identity of dark matter.",
        "temporal_anchor": "unsolved open problem",
        "question_group_id": "grp_b2_u_010_dark_matter_nature"
    }
]


def validate_dataset(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate Batch 2 records against schema and quality criteria."""
    n_total = len(records)
    if n_total != 30:
        raise ValueError(f"Expected exactly 30 records, got {n_total}")

    # 1. Class balance
    class_counts: Dict[str, int] = {}
    for r in records:
        lbl = DatasetValidator.extract_label(r)
        class_counts[lbl] = class_counts.get(lbl, 0) + 1

    expected_classes = {"KNOWN": 10, "AMBIGUOUS": 10, "UNKNOWN": 10}
    if class_counts != expected_classes:
        raise ValueError(f"Class imbalance: {class_counts} vs expected {expected_classes}")

    # 2. Category balance
    cat_counts: Dict[str, int] = {}
    for r in records:
        cat = r.get("category", "unknown")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    # 3. Provenance distribution
    prov_counts: Dict[str, int] = {}
    prov_by_class: Dict[str, Dict[str, int]] = {"KNOWN": {}, "AMBIGUOUS": {}, "UNKNOWN": {}}
    for r in records:
        prov = r.get("provenance", "unknown")
        prov_counts[prov] = prov_counts.get(prov, 0) + 1
        lbl = r["target_label"]
        prov_by_class[lbl][prov] = prov_by_class[lbl].get(prov, 0) + 1

    expected_prov = {"mined": 12, "synthesized": 12, "expert_curated": 6}
    if prov_counts != expected_prov:
        raise ValueError(f"Provenance distribution mismatch: {prov_counts} vs expected {expected_prov}")

    for lbl in ["KNOWN", "AMBIGUOUS", "UNKNOWN"]:
        exp_c_prov = {"mined": 4, "synthesized": 4, "expert_curated": 2}
        if prov_by_class[lbl] != exp_c_prov:
            raise ValueError(f"Per-class provenance mismatch for {lbl}: {prov_by_class[lbl]} vs expected {exp_c_prov}")

    # 4. Unique IDs and Group IDs
    ids = [r["id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question IDs detected")

    for r in records:
        gids = DatasetValidator.extract_group_ids(r)
        if not gids:
            raise ValueError(f"No group ID extracted for record {r['id']}")

    # 5. Required fields check
    required_fields = [
        "id", "question", "target_label", "gold_label", "label",
        "question_group_id", "category", "ground_truth",
        "ambiguity_mechanism", "epistemic_barrier", "mapping_rationale",
        "domain", "provenance", "source"
    ]
    for r in records:
        for f in required_fields:
            if f not in r or r[f] is None or str(r[f]).strip() == "":
                raise ValueError(f"Record {r['id']} missing or empty required field: {f}")

    return {
        "status": "PASS",
        "total_questions": n_total,
        "class_distribution": class_counts,
        "category_distribution": cat_counts,
        "provenance_distribution": prov_counts,
        "per_class_provenance": prov_by_class,
        "unique_ids_count": len(set(ids)),
    }


def compute_isolation_similarities(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute semantic isolation similarities against Gold Benchmark and Batch 1."""
    questions = [r["question"] for r in records]
    print(f"Generating embeddings for {len(questions)} Batch 2 questions...")
    emb_engine = EmbeddingEngine()
    b2_embs = emb_engine.embed_batch(questions)

    isolation_report: Dict[str, Any] = {}

    # Check 1: 300-question Gold Benchmark Isolation (< 0.75)
    if os.path.exists(BENCHMARK_PATH):
        print(f"Loading 300 gold benchmark from {BENCHMARK_PATH}...")
        bmk_questions = []
        with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    bmk_questions.append(json.loads(line)["question"])
        bmk_embs = emb_engine.embed_batch(bmk_questions)
        
        sim_matrix = np.clip(np.matmul(b2_embs, bmk_embs.T), -1.0, 1.0)
        max_sims = sim_matrix.max(axis=1)
        max_idx = sim_matrix.argmax(axis=1)

        bmk_violations = []
        for i, r in enumerate(records):
            msim = float(max_sims[i])
            r["max_benchmark_similarity"] = round(msim, 4)
            r["closest_benchmark_question"] = bmk_questions[max_idx[i]]
            if msim >= 0.75:
                bmk_violations.append({
                    "id": r["id"],
                    "question": r["question"],
                    "similarity": msim,
                    "matched": bmk_questions[max_idx[i]]
                })

        isolation_report["benchmark_isolation"] = {
            "status": "PASS" if not bmk_violations else "FAIL",
            "threshold": 0.75,
            "max_observed_similarity": round(float(max_sims.max()), 4),
            "violations_count": len(bmk_violations),
            "violations": bmk_violations
        }
        if bmk_violations:
            raise ValueError(f"Benchmark isolation violated by {len(bmk_violations)} questions: {bmk_violations}")

    # Check 2: Batch 1 Separation (< 0.80)
    if os.path.exists(BATCH_1_PATH):
        print(f"Loading Batch 1 corpus from {BATCH_1_PATH}...")
        b1_questions = []
        with open(BATCH_1_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    b1_questions.append(json.loads(line)["question"])
        b1_embs = emb_engine.embed_batch(b1_questions)
        
        sim_matrix_b1 = np.clip(np.matmul(b2_embs, b1_embs.T), -1.0, 1.0)
        max_b1_sims = sim_matrix_b1.max(axis=1)
        max_b1_idx = sim_matrix_b1.argmax(axis=1)

        b1_violations = []
        for i, r in enumerate(records):
            msim = float(max_b1_sims[i])
            r["max_batch1_similarity"] = round(msim, 4)
            r["closest_batch1_question"] = b1_questions[max_b1_idx[i]]
            if msim >= 0.80:
                b1_violations.append({
                    "id": r["id"],
                    "question": r["question"],
                    "similarity": msim,
                    "matched": b1_questions[max_b1_idx[i]]
                })

        isolation_report["batch_1_separation"] = {
            "status": "PASS" if not b1_violations else "FAIL",
            "threshold": 0.80,
            "max_observed_similarity": round(float(max_b1_sims.max()), 4),
            "violations_count": len(b1_violations),
            "violations": b1_violations
        }
        if b1_violations:
            raise ValueError(f"Batch 1 separation violated by {len(b1_violations)} questions: {b1_violations}")

    return isolation_report


def main():
    print("=" * 60)
    print("Building and validating Batch 2 Dataset (N=30)...")
    print("=" * 60)

    # 1. Structural validation
    qc_results = validate_dataset(RAW_QUESTIONS)
    print("Structural and balance validation passed:")
    print(f"  Total records: {qc_results['total_questions']}")
    print(f"  Class distribution: {qc_results['class_distribution']}")
    print(f"  Provenance distribution: {qc_results['provenance_distribution']}")

    # 2. Semantic isolation screening
    iso_report = compute_isolation_similarities(RAW_QUESTIONS)
    print("Semantic isolation screening passed:")
    print(f"  Benchmark isolation max similarity: {iso_report['benchmark_isolation']['max_observed_similarity']} (threshold < 0.75)")
    print(f"  Batch 1 separation max similarity: {iso_report['batch_1_separation']['max_observed_similarity']} (threshold < 0.80)")

    # 3. Add model_config and group formatting
    for r in RAW_QUESTIONS:
        r["model_config"] = {
            "target_model": "qwen3:4b",
            "sampling_protocol": "r1: T=0.0 (greedy), r2-r10: T=0.7, top_p=0.9 (stochastic)",
            "n_responses": 10
        }
        r["question_group_ids"] = [r["question_group_id"]]

    # 4. Write dataset to disk atomically
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in RAW_QUESTIONS:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Successfully saved {len(RAW_QUESTIONS)} records to {OUTPUT_PATH}")

    # 5. Compute file SHA-256
    with open(OUTPUT_PATH, "rb") as f:
        file_sha = hashlib.sha256(f.read()).hexdigest()

    # 6. Save Manifest
    manifest_data = {
        "dataset_name": "Batch_2_Supplementary_Corpus_30",
        "created_at": "2026-11-21T19:45:00Z",
        "file_path": OUTPUT_PATH,
        "sha256": file_sha,
        "total_questions": len(RAW_QUESTIONS),
        "class_distribution": qc_results["class_distribution"],
        "category_distribution": qc_results["category_distribution"],
        "provenance_distribution": qc_results["provenance_distribution"],
        "isolation_metrics": iso_report
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"Saved manifest to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
