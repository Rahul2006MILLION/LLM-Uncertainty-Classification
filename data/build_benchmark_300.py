"""Builder for constructing the 300-question draft uncertainty benchmark.

Generates candidate questions across the 7 planned categories with exact 100/100/100 label balance,
question-group IDs, meaningful paraphrases, and adversarial boundary perturbations.
All records are explicitly marked as 'needs_human_review'.
"""

import json
import os
from typing import List, Dict, Any, Optional
from src.evaluation.benchmark_validator import BenchmarkDraftValidator, TARGET_QUOTAS


def build_draft_benchmark() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    def _add_rec(
        rec_id: str,
        group_id: str,
        question: str,
        category: str,
        intended_interpretation: str,
        gold_label: str,
        rationale: str,
        evidence_type: str,
        citations_or_proof: List[str],
        source: str,
        is_adversarial: bool = False,
        verification_status: str = None,
        temporal_anchor: Optional[str] = None,
        audit_trail: Optional[Dict[str, Any]] = None,
    ):
        if verification_status is None:
            if category in ["private_inaccessible_info", "unknowable_future_events", "subjective_questions"]:
                verification_status = "flagged_for_human_verification"
            else:
                verification_status = "verified_published_standard"

        rec = {
            "id": rec_id,
            "question_group_id": group_id,
            "question": question,
            "category": category,
            "intended_interpretation": intended_interpretation,
            "gold_label": gold_label,
            "rationale": rationale,
            "answerability_evidence": {
                "evidence_type": evidence_type,
                "citations_or_proof": citations_or_proof,
                "verification_status": verification_status,
            },
            "source": source,
            "is_adversarial_variant": is_adversarial,
            "reviewer_agreement": {
                "annotation_status": "needs_human_review",
                "annotator_count": 0,
                "individual_labels": [],
                "inter_annotator_agreement": "pending_human_review",
                "adjudicator_id": None,
                "adjudication_rationale": None,
            },
        }
        if temporal_anchor is not None:
            rec["temporal_anchor"] = temporal_anchor
        if audit_trail is not None:
            rec["audit_trail"] = audit_trail
        records.append(rec)

    # =========================================================================
    # CATEGORY 1: Determinate Factual Questions (Quota: 50, Target: KNOWN)
    # =========================================================================
    # Arithmetic & Number Theory Groups (10 items)
    # Group: grp_arith_2plus2
    _add_rec("draft_known_fact_001", "grp_arith_2plus2", "What is 2 + 2?", "determinate_factual",
             "Elementary Peano arithmetic sum", "KNOWN", "Mathematical identity", "consensus_reference",
             ["Peano axioms", "Standard arithmetic"], "elementary_arithmetic")
    _add_rec("draft_known_fact_002", "grp_arith_2plus2", "What is the sum of two and two?", "determinate_factual",
             "Lexical paraphrase of elementary arithmetic sum", "KNOWN", "Mathematical identity", "consensus_reference",
             ["Peano axioms"], "elementary_arithmetic")
    _add_rec("draft_known_fact_003", "grp_arith_2plus2", "What is the result when 2 is added to 2?", "determinate_factual",
             "Passive syntactic paraphrase of 2 + 2", "KNOWN", "Mathematical identity", "consensus_reference",
             ["Peano axioms"], "elementary_arithmetic")

    # Group: grp_arith_square12
    _add_rec("draft_known_fact_004", "grp_arith_square12", "What is 12 squared?", "determinate_factual",
             "Multiplication of integer 12 by itself", "KNOWN", "Elementary exponentiation 12*12=144", "consensus_reference",
             ["Elementary arithmetic"], "elementary_arithmetic")
    _add_rec("draft_known_fact_005", "grp_arith_square12", "What is the value of 12 raised to the second power?", "determinate_factual",
             "Syntactic paraphrase of 12 squared", "KNOWN", "Elementary exponentiation", "consensus_reference",
             ["Elementary arithmetic"], "elementary_arithmetic")

    # Group: grp_prime_check
    _add_rec("draft_known_fact_006", "grp_prime_check", "What is the smallest prime number?", "determinate_factual",
             "First prime integer in natural numbers", "KNOWN", "The integer 2 is the uniquely smallest prime number by definition", "consensus_reference",
             ["Euclid Elements VII", "Fundamental Theorem of Arithmetic"], "number_theory")
    _add_rec("draft_known_fact_007", "grp_prime_check", "Which natural number is the smallest prime?", "determinate_factual",
             "Paraphrase of smallest prime query", "KNOWN", "Integer 2 is the uniquely smallest prime", "consensus_reference",
             ["Fundamental Theorem of Arithmetic"], "number_theory")

    # Group: grp_cube_root_27
    _add_rec("draft_known_fact_008", "grp_cube_root_27", "What is the real cube root of 27?", "determinate_factual",
             "Principal real root of 27^(1/3)", "KNOWN", "The real number 3 satisfies 3^3 = 27", "consensus_reference",
             ["Elementary algebra"], "elementary_algebra")
    _add_rec("draft_known_fact_009", "grp_cube_root_27", "Which real number multiplied by itself three times equals 27?", "determinate_factual",
             "Verbose definitional paraphrase of cube root of 27", "KNOWN", "3^3 = 27", "consensus_reference",
             ["Elementary algebra"], "elementary_algebra")

    # Group: grp_pythagoras
    _add_rec("draft_known_fact_010", "grp_pythagoras", "In a right triangle with legs of length 3 and 4, what is the length of the hypotenuse?", "determinate_factual",
             "Hypotenuse calculation using Pythagorean theorem", "KNOWN", "sqrt(3^2 + 4^2) = 5", "consensus_reference",
             ["Pythagorean theorem (Euclid I.47)"], "elementary_geometry")

    # Physical Constants & Chemical Standards (15 items)
    # Group: grp_adv_vacuum_wave_speed (adversarial cross-label group with sound in vacuum)
    _add_rec("draft_known_fact_011", "grp_adv_vacuum_wave_speed", "What is the exact speed of light in a vacuum in meters per second?", "determinate_factual",
             "Exact SI physical constant c", "KNOWN", "Defined as exactly 299,792,458 m/s since 1983", "consensus_reference",
             ["17th CGPM Resolution 1 (1983)", "NIST Special Publication 330"], "si_standards", is_adversarial=True)
    _add_rec("draft_known_fact_012", "grp_adv_vacuum_wave_speed", "In meters per second, what is the defined speed of light in vacuum?", "determinate_factual",
             "Paraphrase of speed of light in vacuum", "KNOWN", "Defined constant 299,792,458 m/s", "consensus_reference",
             ["NIST SP 330"], "si_standards", is_adversarial=True)

    # Group: grp_water_freezing
    _add_rec("draft_known_fact_013", "grp_water_freezing", "At standard atmospheric pressure (1 atm), at what temperature does pure water freeze in degrees Celsius?", "determinate_factual",
             "Normal freezing point of pure H2O at 101.325 kPa", "KNOWN", "Freezes at 0 degrees Celsius under standard 1 atm pressure", "consensus_reference",
             ["CRC Handbook of Chemistry and Physics"], "chemistry_standards")
    _add_rec("draft_known_fact_014", "grp_water_freezing", "What is the normal freezing point of pure water in Celsius at 1 atm?", "determinate_factual",
             "Paraphrase of freezing point of water", "KNOWN", "0 degrees Celsius", "consensus_reference",
             ["CRC Handbook of Chemistry and Physics"], "chemistry_standards")

    # Group: grp_water_boiling
    _add_rec("draft_known_fact_015", "grp_water_boiling", "What is the boiling point of pure water in degrees Celsius at standard atmospheric pressure?", "determinate_factual",
             "Normal boiling point of pure H2O at 1 atm", "KNOWN", "Boils at 100 degrees Celsius at 1 atm", "consensus_reference",
             ["IUPAC Compendium of Chemical Terminology"], "chemistry_standards")
    _add_rec("draft_known_fact_016", "grp_water_boiling", "At 1 atm pressure, at what Celsius temperature does water boil?", "determinate_factual",
             "Inverted clause paraphrase of water boiling point", "KNOWN", "100 degrees Celsius", "consensus_reference",
             ["IUPAC Compendium"], "chemistry_standards")

    # Group: grp_atomic_gold
    _add_rec("draft_known_fact_017", "grp_atomic_gold", "What is the atomic number of gold?", "determinate_factual",
             "Proton number of element Au in IUPAC periodic table", "KNOWN", "Gold has atomic number 79", "consensus_reference",
             ["IUPAC Periodic Table of Elements"], "iupac_chemistry")
    _add_rec("draft_known_fact_018", "grp_atomic_gold", "How many protons are in the nucleus of an atom of gold?", "determinate_factual",
             "Physical definitional paraphrase of atomic number of gold", "KNOWN", "79 protons", "consensus_reference",
             ["IUPAC Periodic Table of Elements"], "iupac_chemistry")

    # Group: grp_atomic_helium
    _add_rec("draft_known_fact_019", "grp_atomic_helium", "What is the atomic number of helium?", "determinate_factual",
             "Proton number of element He", "KNOWN", "Helium has atomic number 2", "consensus_reference",
             ["IUPAC Periodic Table"], "iupac_chemistry")

    # Group: grp_avogadro
    _add_rec("draft_known_fact_020", "grp_avogadro", "What is the exact value of the Avogadro constant defined by the 2019 SI redefinition?", "determinate_factual",
             "Exact SI value of Avogadro constant N_A", "KNOWN", "Exactly 6.02214076 x 10^23 mol^-1", "consensus_reference",
             ["BIPM 2019 SI Brochure 9th Edition"], "si_standards")

    # Group: grp_gravity_earth
    _add_rec("draft_known_fact_021", "grp_gravity_earth", "What is the standard acceleration due to gravity on Earth in meters per second squared defined by ISO 80000-3?", "determinate_factual",
             "Standard Earth gravity g_n", "KNOWN", "Defined standard is 9.80665 m/s^2", "consensus_reference",
             ["ISO 80000-3:2019 Quantities and units"], "iso_standards")

    # Group: grp_absolute_zero
    _add_rec("draft_known_fact_022", "grp_absolute_zero", "What is absolute zero in degrees Celsius?", "determinate_factual",
             "Zero kelvin temperature expressed in Celsius", "KNOWN", "-273.15 degrees Celsius", "consensus_reference",
             ["BIPM SI Brochure"], "si_standards")
    _add_rec("draft_known_fact_023", "grp_absolute_zero", "How many degrees Celsius corresponds to zero Kelvin?", "determinate_factual",
             "Syntactic inversion of absolute zero in Celsius", "KNOWN", "-273.15 C", "consensus_reference",
             ["BIPM SI Brochure"], "si_standards")

    # Group: grp_planck_constant
    _add_rec("draft_known_fact_024", "grp_planck_constant", "What is the defined value of the Planck constant in Joule-seconds under the 2019 SI revision?", "determinate_factual",
             "Exact SI value of h", "KNOWN", "Exactly 6.62607015 x 10^-34 J*s", "consensus_reference",
             ["26th CGPM Resolution 1 (2018)"], "si_standards")
    _add_rec("draft_known_fact_025", "grp_planck_constant", "Under the current International System of Units, what is the exact value of Planck's constant?", "determinate_factual",
             "Paraphrase of Planck constant SI value", "KNOWN", "6.62607015 x 10^-34 J*s", "consensus_reference",
             ["26th CGPM Resolution 1"], "si_standards")

    # Geography & Sovereign States (15 items)
    # Group: grp_capital_france
    _add_rec("draft_known_fact_026", "grp_capital_france", "What is the official capital city of France?", "determinate_factual",
             "Current official capital of the French Republic", "KNOWN", "Paris is the constitutional capital of France", "consensus_reference",
             ["Constitution of the French Republic, Art. 2", "ISO 3166-1"], "world_geography")
    _add_rec("draft_known_fact_027", "grp_capital_france", "Which city serves as the official capital of France?", "determinate_factual",
             "Paraphrase of capital of France", "KNOWN", "Paris", "consensus_reference",
             ["ISO 3166-1"], "world_geography")
    _add_rec("draft_known_fact_028", "grp_capital_france", "What is the capital of the French Republic?", "determinate_factual",
             "Formal nomenclature paraphrase for France capital", "KNOWN", "Paris", "consensus_reference",
             ["ISO 3166-1"], "world_geography")

    # Group: grp_capital_japan
    _add_rec("draft_known_fact_029", "grp_capital_japan", "What is the capital city of Japan?", "determinate_factual",
             "Current seat of government and capital of Japan", "KNOWN", "Tokyo is the de facto and legal seat of government of Japan", "consensus_reference",
             ["Capital Area Consolidation Act of Japan", "ISO 3166-1"], "world_geography",
             audit_trail={"borderline_review": "Tokyo recognized as de facto seat of government and capital under ISO 3166-1 standard despite absence of single codified statutory clause."})
    _add_rec("draft_known_fact_030", "grp_capital_japan", "Which city is the capital of Japan?", "determinate_factual",
             "Syntactic paraphrase of capital of Japan", "KNOWN", "Tokyo", "consensus_reference",
             ["ISO 3166-1"], "world_geography",
             audit_trail={"borderline_review": "Tokyo recognized as de facto capital under ISO 3166-1 standard."})

    # Group: grp_capital_germany
    _add_rec("draft_known_fact_031", "grp_capital_germany", "What is the official capital city of Germany?", "determinate_factual",
             "Current constitutional capital of Federal Republic of Germany", "KNOWN", "Berlin is the official capital pursuant to Basic Law Art. 22", "consensus_reference",
             ["Basic Law for the Federal Republic of Germany, Art. 22"], "world_geography")
    _add_rec("draft_known_fact_032", "grp_capital_germany", "Which city is the federal capital of Germany?", "determinate_factual",
             "Paraphrase of German capital city", "KNOWN", "Berlin", "consensus_reference",
             ["German Basic Law Art. 22"], "world_geography")

    # Group: grp_continents_count
    _add_rec("draft_known_fact_033", "grp_continents_count", "Under the standard English-language seven-continent model, how many continents are recognized?", "determinate_factual",
             "Count in the 7-continent geographic convention", "KNOWN", "Exactly 7 continents (Africa, Antarctica, Asia, Australia, Europe, North America, South America)", "consensus_reference",
             ["National Geographic Standard Geography Curriculum"], "world_geography")
    _add_rec("draft_known_fact_034", "grp_continents_count", "How many continents are there in the traditional 7-continent geographic model?", "determinate_factual",
             "Paraphrase of continent count under 7-continent model", "KNOWN", "7 continents", "consensus_reference",
             ["National Geographic Curriculum"], "world_geography")

    # Group: grp_highest_mountain
    _add_rec("draft_known_fact_035", "grp_highest_mountain", "What is the highest mountain peak above sea level on Earth?", "determinate_factual",
             "Highest elevation above mean sea level", "KNOWN", "Mount Everest (8,848.86 m)", "consensus_reference",
             ["Survey Department of Nepal / Ministry of Natural Resources China 2020 Joint Survey"], "geodesy")
    _add_rec("draft_known_fact_036", "grp_highest_mountain", "Which mountain peak has the greatest elevation above sea level on Earth?", "determinate_factual",
             "Paraphrase of highest mountain above sea level", "KNOWN", "Mount Everest", "consensus_reference",
             ["2020 Nepal-China Joint Survey"], "geodesy")

    # Group: grp_largest_ocean
    _add_rec("draft_known_fact_037", "grp_largest_ocean", "What is the largest ocean on Earth by surface area?", "determinate_factual",
             "Largest oceanic division by surface square kilometers", "KNOWN", "The Pacific Ocean covers approximately 165 million square kilometers", "consensus_reference",
             ["NOAA National Ocean Service"], "oceanography")

    # Group: grp_longest_river
    _add_rec("draft_known_fact_038", "grp_longest_river", "According to Guinness World Records and Britannica, what is traditionally considered the longest river in the world?", "determinate_factual",
             "Traditional consensus on world's longest river", "KNOWN", "The Nile River (approximately 6,650 km)", "consensus_reference",
             ["Encyclopaedia Britannica", "Guinness World Records"], "physical_geography")

    # Group: grp_deepest_trench
    _add_rec("draft_known_fact_039", "grp_deepest_trench", "What is the deepest known oceanic trench on Earth?", "determinate_factual",
             "Deepest bathymetric feature", "KNOWN", "The Mariana Trench (Challenger Deep at ~10,994 m)", "consensus_reference",
             ["GEBCO World Ocean Bathymetry", "NOAA"], "oceanography")

    # Group: grp_canada_capital
    _add_rec("draft_known_fact_040", "grp_canada_capital", "What is the national capital of Canada?", "determinate_factual",
             "Official federal capital of Canada", "KNOWN", "Ottawa", "consensus_reference",
             ["Constitution Act 1867, Section 16"], "world_geography")

    # Historical Dates & Documents (10 items)
    # Group: grp_apollo11_date
    _add_rec("draft_known_fact_041", "grp_apollo11_date", "In what calendar year did the Apollo 11 lunar module land on the Moon?", "determinate_factual",
             "Calendar year of first crewed lunar landing", "KNOWN", "July 20, 1969", "consensus_reference",
             ["NASA Apollo 11 Mission Report"], "history_of_spaceflight")
    _add_rec("draft_known_fact_042", "grp_apollo11_date", "What year saw humanity first walk on the lunar surface during Apollo 11?", "determinate_factual",
             "Paraphrase of Apollo 11 landing year", "KNOWN", "1969", "consensus_reference",
             ["NASA Historical Archives"], "history_of_spaceflight")

    # Group: grp_declaration_independence
    _add_rec("draft_known_fact_043", "grp_declaration_independence", "In what year did the Second Continental Congress adopt the United States Declaration of Independence?", "determinate_factual",
             "Year of adoption of US Declaration of Independence", "KNOWN", "Adopted on July 4, 1776", "consensus_reference",
             ["US National Archives Records of the Continental Congress"], "american_history")
    _add_rec("draft_known_fact_044", "grp_declaration_independence", "What year was the United States Declaration of Independence officially adopted?", "determinate_factual",
             "Paraphrase of Declaration of Independence year", "KNOWN", "1776", "consensus_reference",
             ["US National Archives"], "american_history")

    # Group: grp_wwii_end
    _add_rec("draft_known_fact_045", "grp_wwii_end", "In what calendar year did World War II officially conclude with the formal Japanese surrender?", "determinate_factual",
             "Year of conclusion of WWII", "KNOWN", "Concluded in 1945 (Instrument of Surrender signed September 2, 1945)", "consensus_reference",
             ["Japanese Instrument of Surrender, National Archives"], "world_history")
    _add_rec("draft_known_fact_046", "grp_wwii_end", "Which year marked the formal end of the Second World War?", "determinate_factual",
             "Paraphrase of WWII formal conclusion year", "KNOWN", "1945", "consensus_reference",
             ["World War II Historical Records"], "world_history")

    # Group: grp_first_us_president
    _add_rec("draft_known_fact_047", "grp_first_us_president", "Who was the first President of the United States under the US Constitution?", "determinate_factual",
             "First constitutional US president", "KNOWN", "George Washington (served 1789-1797)", "consensus_reference",
             ["US National Archives Presidential Records"], "american_history")
    _add_rec("draft_known_fact_048", "grp_first_us_president", "Under the United States Constitution, who was the initial person to serve as President of the United States?", "determinate_factual",
             "Paraphrase of first constitutional US president", "KNOWN", "George Washington", "consensus_reference",
             ["Library of Congress Presidential Papers"], "american_history",
             audit_trail={"original_question": "Who was the initial person to serve as President of the United States?", "modification_reason": "Added 'Under the United States Constitution' to resolve historical ambiguity with John Hanson under Articles of Confederation."})

    # Group: grp_un_founding
    _add_rec("draft_known_fact_049", "grp_un_founding", "In what year was the United Nations Charter signed in San Francisco?", "determinate_factual",
             "Year of signing of UN Charter", "KNOWN", "Signed on June 26, 1945", "consensus_reference",
             ["United Nations Treaty Collection, Charter of the United Nations"], "international_relations")
    _add_rec("draft_known_fact_050", "grp_un_founding", "What year did the founding conference sign the Charter of the United Nations?", "determinate_factual",
             "Paraphrase of UN charter signing year", "KNOWN", "1945", "consensus_reference",
             ["United Nations Archives"], "international_relations")


    # =========================================================================
    # CATEGORY 2: Established Concept Definitions (Quota: 50, Target: KNOWN)
    # =========================================================================
    # Group: grp_def_data_science
    _add_rec("draft_known_def_001", "grp_def_data_science", "What is data science?", "concept_definition",
             "Academic definition of data science", "KNOWN", "Standard established academic discipline with clear consensus definition", "definitional_standard",
             ["NIST Special Publication 1500-1 Big Data Interoperability", "ACM/IEEE Guidelines"], "computer_science")
    _add_rec("draft_known_def_002", "grp_def_data_science", "Define data science.", "concept_definition",
             "Imperative definition request for data science", "KNOWN", "Standard interdisciplinary field combining computation, statistics, and domain analysis", "definitional_standard",
             ["NIST SP 1500-1"], "computer_science")
    _add_rec("draft_known_def_003", "grp_def_data_science", "What is meant by the academic field of data science?", "concept_definition",
             "Paraphrase of data science definition", "KNOWN", "Standard established academic discipline", "definitional_standard",
             ["ACM Curriculum Guidelines"], "computer_science")

    # Group: grp_def_machine_learning
    _add_rec("draft_known_def_004", "grp_def_machine_learning", "What is machine learning?", "concept_definition",
             "Core computer science subfield definition", "KNOWN", "Subfield of artificial intelligence studying algorithms that improve performance via data", "definitional_standard",
             ["Mitchell (1997) Machine Learning", "Russell & Norvig AIMA 4th Ed"], "computer_science")
    _add_rec("draft_known_def_005", "grp_def_machine_learning", "Define machine learning.", "concept_definition",
             "Imperative definition of machine learning", "KNOWN", "Standard algorithmic subfield of AI", "definitional_standard",
             ["Russell & Norvig AIMA"], "computer_science")

    # Group: grp_def_photosynthesis
    _add_rec("draft_known_def_006", "grp_def_photosynthesis", "What is photosynthesis?", "concept_definition",
             "Biological process definition", "KNOWN", "Biochemical process by which phototrophic organisms convert light energy into chemical energy", "definitional_standard",
             ["Campbell Biology 12th Ed"], "biology")
    _add_rec("draft_known_def_007", "grp_def_photosynthesis", "Define photosynthesis.", "concept_definition",
             "Imperative definition of photosynthesis", "KNOWN", "Biological process of converting solar energy to glucose", "definitional_standard",
             ["Campbell Biology"], "biology")

    # Group: grp_def_entropy
    _add_rec("draft_known_def_008", "grp_def_entropy", "In thermodynamics, what is entropy?", "concept_definition",
             "Thermodynamic state function definition", "KNOWN", "Thermodynamic measure of unavailable thermal energy per unit temperature (dS = dQ_rev/T)", "definitional_standard",
             ["IUPAC Compendium of Chemical Terminology", "Clausius (1865)"], "thermodynamics")
    _add_rec("draft_known_def_009", "grp_def_entropy", "Define thermodynamic entropy.", "concept_definition",
             "Imperative definition of thermodynamic entropy", "KNOWN", "Physical measure of thermal dispersal and microstate probability", "definitional_standard",
             ["IUPAC Compendium"], "thermodynamics")

    # Group: grp_def_osmosis
    _add_rec("draft_known_def_010", "grp_def_osmosis", "What is osmosis?", "concept_definition",
             "Physical chemistry definition of osmosis", "KNOWN", "Net movement of solvent molecules through a selectively permeable membrane into higher solute concentration", "definitional_standard",
             ["IUPAC Gold Book", "Alberts Molecular Biology of the Cell"], "chemistry")
    _add_rec("draft_known_def_011", "grp_def_osmosis", "Define the biological process of osmosis.", "concept_definition",
             "Imperative definition of biological osmosis", "KNOWN", "Solvent transport across semipermeable barrier", "definitional_standard",
             ["IUPAC Gold Book"], "chemistry")

    # Group: grp_def_mitosis
    _add_rec("draft_known_def_012", "grp_def_mitosis", "What is mitosis?", "concept_definition",
             "Eukaryotic cell division definition", "KNOWN", "Part of eukaryotic cell cycle wherein replicated chromosomes are separated into two new nuclei", "definitional_standard",
             ["Campbell Biology 12th Ed"], "biology")
    _add_rec("draft_known_def_013", "grp_def_mitosis", "Define mitosis in cell biology.", "concept_definition",
             "Imperative definition of mitosis", "KNOWN", "Nuclear division in somatic eukaryotic cells", "definitional_standard",
             ["Campbell Biology"], "biology")

    # Group: grp_def_dna
    _add_rec("draft_known_def_014", "grp_def_dna", "What is deoxyribonucleic acid (DNA)?", "concept_definition",
             "Biochemical polymer definition", "KNOWN", "Polymer composed of two polynucleotide chains that coil around each other to form a double helix carrying genetic instructions", "definitional_standard",
             ["IUPAC Gold Book", "Watson & Crick 1953"], "molecular_biology")
    _add_rec("draft_known_def_015", "grp_def_dna", "Define DNA.", "concept_definition",
             "Short imperative definition of DNA", "KNOWN", "Biomolecule encoding genetic information", "definitional_standard",
             ["IUPAC Gold Book"], "molecular_biology")

    # Group: grp_def_inflation
    _add_rec("draft_known_def_016", "grp_def_inflation", "In economics, what is inflation?", "concept_definition",
             "Macroeconomic definition of inflation", "KNOWN", "General progressive increase in price levels of goods and services over time", "definitional_standard",
             ["IMF Glossary of Terms", "Mankiw Macroeconomics 10th Ed"], "economics")
    _add_rec("draft_known_def_017", "grp_def_inflation", "Define economic inflation.", "concept_definition",
             "Imperative definition of economic inflation", "KNOWN", "Sustained increase in general price level", "definitional_standard",
             ["IMF Glossary"], "economics")

    # Group: grp_def_algorithm
    _add_rec("draft_known_def_018", "grp_def_algorithm", "What is an algorithm in computer science?", "concept_definition",
             "Formal definition of algorithm", "KNOWN", "Finite, unambiguous sequence of well-defined computer-implementable instructions to solve a problem", "definitional_standard",
             ["Cormen et al. Introduction to Algorithms 4th Ed", "Knuth TAOCP"], "computer_science")
    _add_rec("draft_known_def_019", "grp_def_algorithm", "Define algorithm.", "concept_definition",
             "Imperative definition of algorithm", "KNOWN", "Finite sequence of computational instructions", "definitional_standard",
             ["CLRS Introduction to Algorithms"], "computer_science")

    # Group: grp_def_superconductivity
    _add_rec("draft_known_def_020", "grp_def_superconductivity", "What is superconductivity?", "concept_definition",
             "Condensed matter physics definition", "KNOWN", "State of matter characterized by exactly zero electrical resistance and expulsion of magnetic fields below critical temperature", "definitional_standard",
             ["Kittel Introduction to Solid State Physics 8th Ed"], "physics")
    _add_rec("draft_known_def_021", "grp_def_superconductivity", "Define superconductivity.", "concept_definition",
             "Imperative definition of superconductivity", "KNOWN", "Zero electrical resistance and Meissner effect", "definitional_standard",
             ["Kittel Solid State Physics"], "physics")

    # Additional standard definitions to complete quota (22-50)
    # Group: grp_def_compiler
    _add_rec("draft_known_def_022", "grp_def_compiler", "What is a compiler in computer science?", "concept_definition",
             "Computer systems definition of compiler", "KNOWN", "Computer program that translates computer code written in one programming language into another language", "definitional_standard",
             ["Aho et al. Compilers: Principles, Techniques, and Tools (Dragon Book)"], "computer_science")
    _add_rec("draft_known_def_023", "grp_def_compiler", "Define a software compiler.", "concept_definition",
             "Imperative definition of compiler", "KNOWN", "Program translating high-level code to target assembly or machine code", "definitional_standard",
             ["Dragon Book"], "computer_science")

    # Group: grp_def_plate_tectonics
    _add_rec("draft_known_def_024", "grp_def_plate_tectonics", "What is plate tectonics in geology?", "concept_definition",
             "Geological scientific theory definition", "KNOWN", "Scientific theory describing large-scale motion of seven large plates and smaller plates of Earth's lithosphere", "definitional_standard",
             ["USGS Plate Tectonics Guide"], "geology")
    _add_rec("draft_known_def_025", "grp_def_plate_tectonics", "Define plate tectonics.", "concept_definition",
             "Imperative definition of plate tectonics", "KNOWN", "Scientific model of lithospheric plate motion", "definitional_standard",
             ["USGS"], "geology")

    # Group: grp_def_recursion
    _add_rec("draft_known_def_026", "grp_def_recursion", "What is recursion in computer science?", "concept_definition",
             "Programming paradigm definition of recursion", "KNOWN", "Method of solving a computational problem where the solution depends on solutions to smaller instances of the same problem", "definitional_standard",
             ["SICP Abelson & Sussman"], "computer_science")
    _add_rec("draft_known_def_027", "grp_def_recursion", "Define recursion.", "concept_definition",
             "Imperative definition of recursion", "KNOWN", "Self-referential procedure solving sub-problems", "definitional_standard",
             ["SICP"], "computer_science")

    # Group: grp_def_gravity
    _add_rec("draft_known_def_028", "grp_def_gravity", "What is gravity in classical physics?", "concept_definition",
             "Newtonian gravitational definition", "KNOWN", "Fundamental interaction causing mutual attraction between all things that have mass", "definitional_standard",
             ["Halliday, Resnick, Walker Fundamentals of Physics"], "physics")
    _add_rec("draft_known_def_029", "grp_def_gravity", "Define gravity according to Newtonian mechanics.", "concept_definition",
             "Imperative definition of Newtonian gravity", "KNOWN", "Attractive force proportional to masses and inversely proportional to square of distance", "definitional_standard",
             ["Newton Principia Mathematica"], "physics")

    # Group: grp_def_enzyme
    _add_rec("draft_known_def_030", "grp_def_enzyme", "What is an enzyme in biochemistry?", "concept_definition",
             "Biochemical catalyst definition", "KNOWN", "Biological catalyst that speeds up the rate of chemical reactions without being consumed", "definitional_standard",
             ["Lehninger Principles of Biochemistry 8th Ed"], "biochemistry")
    _add_rec("draft_known_def_031", "grp_def_enzyme", "Define biological enzyme.", "concept_definition",
             "Imperative definition of enzyme", "KNOWN", "Protein catalyst accelerating biochemical reactions", "definitional_standard",
             ["Lehninger Biochemistry"], "biochemistry")

    # Group: grp_def_democracy
    _add_rec("draft_known_def_032", "grp_def_democracy", "What is representative democracy?", "concept_definition",
             "Political science definition", "KNOWN", "System of government where citizens elect representatives to make policy and enact laws on their behalf", "definitional_standard",
             ["Dahl (1989) Democracy and its Critics"], "political_science")
    _add_rec("draft_known_def_033", "grp_def_democracy", "Define representative democracy.", "concept_definition",
             "Imperative definition of representative democracy", "KNOWN", "Government system via elected legislative agents", "definitional_standard",
             ["Dahl On Democracy"], "political_science")

    # Group: grp_def_ecosystem
    _add_rec("draft_known_def_034", "grp_def_ecosystem", "What is an ecosystem in ecology?", "concept_definition",
             "Ecological system definition", "KNOWN", "Geographic area where plants, animals, and other organisms, as well as weather and landscape, work together to form a bubble of life", "definitional_standard",
             ["Odum Fundamentals of Ecology", "Tansley 1935"], "ecology")
    _add_rec("draft_known_def_035", "grp_def_ecosystem", "Define ecosystem.", "concept_definition",
             "Imperative definition of ecosystem", "KNOWN", "Biotic community and its abiotic environment", "definitional_standard",
             ["Odum Ecology"], "ecology")

    # Group: grp_def_fermentation
    _add_rec("draft_known_def_036", "grp_def_fermentation", "What is fermentation in biochemistry?", "concept_definition",
             "Metabolic process definition", "KNOWN", "Metabolic process that produces chemical changes in organic substrates through the action of enzymes, typically under anaerobic conditions", "definitional_standard",
             ["Madigan Brock Biology of Microorganisms"], "biochemistry")
    _add_rec("draft_known_def_037", "grp_def_fermentation", "Define anaerobic fermentation.", "concept_definition",
             "Imperative definition of fermentation", "KNOWN", "Anaerobic catabolic pathway extracting energy from carbohydrates", "definitional_standard",
             ["Brock Biology of Microorganisms"], "biochemistry")

    # Group: grp_def_isotope
    _add_rec("draft_known_def_038", "grp_def_isotope", "What is an isotope in nuclear chemistry?", "concept_definition",
             "Nuclear chemical definition of isotope", "KNOWN", "Variants of a particular chemical element which differ in neutron number, and consequently in nucleon number", "definitional_standard",
             ["IUPAC Compendium of Chemical Terminology"], "chemistry")
    _add_rec("draft_known_def_039", "grp_def_isotope", "Define chemical isotope.", "concept_definition",
             "Imperative definition of isotope", "KNOWN", "Atoms with identical atomic number but differing neutron count", "definitional_standard",
             ["IUPAC Compendium"], "chemistry")

    # Group: grp_def_api
    _add_rec("draft_known_def_040", "grp_def_api", "In software engineering, what is an Application Programming Interface (API)?", "concept_definition",
             "Software engineering definition of API", "KNOWN", "Connection between computers or between computer programs, functioning as a software interface providing service specification", "definitional_standard",
             ["IEEE Standard 610.12 Software Engineering Terminology"], "computer_science")
    _add_rec("draft_known_def_041", "grp_def_api", "Define Application Programming Interface.", "concept_definition",
             "Imperative definition of API", "KNOWN", "Specification of remote or local software contracts", "definitional_standard",
             ["IEEE 610.12"], "computer_science")

    # Group: grp_def_half_life
    _add_rec("draft_known_def_042", "grp_def_half_life", "In nuclear physics, what is half-life?", "concept_definition",
             "Radioactive decay definition", "KNOWN", "Time required for a quantity of substance to reduce to half of its initial value", "definitional_standard",
             ["IUPAC Gold Book Radioactive Half-life"], "nuclear_physics")
    _add_rec("draft_known_def_043", "grp_def_half_life", "Define nuclear half-life.", "concept_definition",
             "Imperative definition of half-life", "KNOWN", "Characteristic decay interval to 50% activity", "definitional_standard",
             ["IUPAC Gold Book"], "nuclear_physics")

    # Group: grp_def_refraction
    _add_rec("draft_known_def_044", "grp_def_refraction", "What is refraction in optics?", "concept_definition",
             "Wave mechanics optical definition", "KNOWN", "Redirection of a wave as it passes from one transparent medium to another caused by its change in speed", "definitional_standard",
             ["Hecht Optics 5th Ed", "Snell's Law"], "optics")
    _add_rec("draft_known_def_045", "grp_def_refraction", "Define optical refraction.", "concept_definition",
             "Imperative definition of refraction", "KNOWN", "Wave direction change across differing refractive indices", "definitional_standard",
             ["Hecht Optics"], "optics")

    # Group: grp_def_osmotic_pressure
    _add_rec("draft_known_def_046", "grp_def_osmotic_pressure", "What is osmotic pressure?", "concept_definition",
             "Colligative property definition", "KNOWN", "Minimum pressure which needs to be applied to a solution to prevent the inward flow of its pure solvent across a semipermeable membrane", "definitional_standard",
             ["IUPAC Gold Book Van 't Hoff relation"], "physical_chemistry")
    _add_rec("draft_known_def_047", "grp_def_osmotic_pressure", "Define osmotic pressure.", "concept_definition",
             "Imperative definition of osmotic pressure", "KNOWN", "Pressure required to halt net solvent osmotic flow", "definitional_standard",
             ["IUPAC Gold Book"], "physical_chemistry")

    # Group: grp_def_catalyst
    _add_rec("draft_known_def_048", "grp_def_catalyst", "What is a chemical catalyst?", "concept_definition",
             "Chemical kinetics definition of catalyst", "KNOWN", "Substance that increases the rate of a chemical reaction without itself undergoing any permanent chemical change", "definitional_standard",
             ["IUPAC Compendium of Chemical Terminology"], "chemistry")
    _add_rec("draft_known_def_049", "grp_def_catalyst", "Define catalyst in chemistry.", "concept_definition",
             "Imperative definition of catalyst", "KNOWN", "Reaction accelerator lowering activation energy", "definitional_standard",
             ["IUPAC Compendium"], "chemistry")

    # Group: grp_def_black_hole
    _add_rec("draft_known_def_050", "grp_def_black_hole", "In astrophysics, what is a black hole?", "concept_definition",
             "General relativistic definition", "KNOWN", "Region of spacetime where gravity is so strong that nothing, including light and other electromagnetic waves, has enough energy to escape its event horizon", "definitional_standard",
             ["Wald General Relativity", "Hawking & Ellis 1973"], "astrophysics")


    # =========================================================================
    # CATEGORY 3: Subjective Questions without Single Objective Metric (Quota: 40, Target: AMBIGUOUS)
    # =========================================================================
    # Group: grp_subj_prog_lang
    _add_rec("draft_ambig_subj_001", "grp_subj_prog_lang", "What is the best programming language?", "subjective_questions",
             "Unconstrained programming language evaluation", "AMBIGUOUS", "Lacks objective metric; depends entirely on domain, efficiency, developer ergonomics", "delineated_interpretations",
             ["Python for ML/Scripting", "C++ for low-latency/game engines", "Rust for memory safety", "TypeScript for web"], "curated_challenge_subjective")
    _add_rec("draft_ambig_subj_002", "grp_subj_prog_lang", "Which programming language is superior to all others?", "subjective_questions",
             "Syntactic paraphrase of best programming language", "AMBIGUOUS", "Subjective aesthetic/engineering preference without objective metric", "delineated_interpretations",
             ["Multi-paradigm trade-offs preclude global superiority"], "curated_challenge_subjective")

    # Group: grp_subj_country
    _add_rec("draft_ambig_subj_003", "grp_subj_country", "What is the greatest country in the world?", "subjective_questions",
             "Unconstrained national comparison", "AMBIGUOUS", "Metric-free superlative admitting conflicting geopolitical and personal criteria", "delineated_interpretations",
             ["HDI (Switzerland/Norway)", "GDP (US/China)", "Happiness Index (Finland)"], "curated_challenge_subjective")
    _add_rec("draft_ambig_subj_004", "grp_subj_country", "Which nation is the best place to live in the world?", "subjective_questions",
             "Paraphrase of greatest country", "AMBIGUOUS", "Depends entirely on personal preference, climate, culture, tax structure", "delineated_interpretations",
             ["Conflicting quality-of-life indices"], "curated_challenge_subjective")

    # Group: grp_subj_scientist
    _add_rec("draft_ambig_subj_005", "grp_subj_scientist", "Who was the greatest scientist of all time?", "subjective_questions",
             "Unconstrained historical scientific ranking", "AMBIGUOUS", "Subjective appraisal comparing incommensurable breakthroughs", "delineated_interpretations",
             ["Newton (calculus, mechanics)", "Einstein (relativity)", "Galileo (empirical method)", "Curie (radioactivity)"], "curated_challenge_subjective")
    _add_rec("draft_ambig_subj_006", "grp_subj_scientist", "Which individual was the most brilliant scientist in history?", "subjective_questions",
             "Paraphrase of greatest scientist", "AMBIGUOUS", "Incommensurable contributions across distinct scientific revolutions", "delineated_interpretations",
             ["Newton, Einstein, Maxwell, Darwin"], "curated_challenge_subjective")

    # Group: grp_subj_invention
    _add_rec("draft_ambig_subj_007", "grp_subj_invention", "What is the most important invention in human history?", "subjective_questions",
             "Unconstrained technological ranking", "AMBIGUOUS", "Depends on historical perspective and domain importance", "delineated_interpretations",
             ["The wheel", "Movable type printing press", "Sanitation / Clean water", "Electricity", "Transistor"], "curated_challenge_subjective")
    _add_rec("draft_ambig_subj_008", "grp_subj_invention", "Which human invention has had the most significant impact on civilization?", "subjective_questions",
             "Paraphrase of most important invention", "AMBIGUOUS", "Multiple legitimate historical candidates", "delineated_interpretations",
             ["Wheel, printing press, internet, agriculture"], "curated_challenge_subjective")

    # Group: grp_subj_person_success
    _add_rec("draft_ambig_subj_009", "grp_subj_person_success", "Who is the most successful person in the world?", "subjective_questions",
             "Unconstrained personal success query", "AMBIGUOUS", "Success has disparate definitions: net worth, moral impact, happiness, athletic achievement", "delineated_interpretations",
             ["Net worth: Forbes list leaders", "Cultural influence: historical leaders", "Athletic success: Olympic gold record holders"], "curated_challenge_subjective")
    _add_rec("draft_ambig_subj_010", "grp_subj_person_success", "Which living human has achieved the highest level of personal success?", "subjective_questions",
             "Paraphrase of most successful person", "AMBIGUOUS", "Subjective definition of personal achievement", "delineated_interpretations",
             ["Financial vs artistic vs philanthropic success"], "curated_challenge_subjective")

    # Group: grp_subj_best_movie
    _add_rec("draft_ambig_subj_011", "grp_subj_best_movie", "What is the best movie ever made?", "subjective_questions",
             "Aesthetic film evaluation without criteria", "AMBIGUOUS", "Aesthetic preference without singular objective ranking", "delineated_interpretations",
             ["Citizen Kane", "The Godfather", "Jeanne Dielman", "Vertigo"], "curated_challenge_subjective")
    _add_rec("draft_ambig_subj_012", "grp_subj_best_movie", "Which cinematic film is considered the greatest of all time?", "subjective_questions",
             "Paraphrase of best movie", "AMBIGUOUS", "Diverse critical consensus polls (Sight & Sound, AFI)", "delineated_interpretations",
             ["Sight & Sound Top 10", "AFI 100 Years"], "curated_challenge_subjective")

    # Group: grp_subj_best_novel
    _add_rec("draft_ambig_subj_013", "grp_subj_best_novel", "What is the greatest novel ever written in the English language?", "subjective_questions",
             "Literary evaluation without criteria", "AMBIGUOUS", "Plural canonical literary opinions", "delineated_interpretations",
             ["Ulysses (James Joyce)", "Moby-Dick (Herman Melville)", "Middlemarch (George Eliot)", "The Great Gatsby"], "curated_challenge_subjective")
    _add_rec("draft_ambig_subj_014", "grp_subj_best_novel", "Which English novel is the best piece of literature in history?", "subjective_questions",
             "Paraphrase of greatest novel", "AMBIGUOUS", "Conflicting literary canons", "delineated_interpretations",
             ["Modern Library 100 Best Novels", "BBC The Big Read"], "curated_challenge_subjective")

    # Group: grp_subj_best_athlete
    _add_rec("draft_ambig_subj_015", "grp_subj_best_athlete", "Who is the greatest athlete of all time?", "subjective_questions",
             "Cross-sport athletic ranking", "AMBIGUOUS", "Cross-sport comparison is inherently subjective", "delineated_interpretations",
             ["Michael Jordan (Basketball)", "Muhammad Ali (Boxing)", "Usain Bolt (Track)", "Michael Phelps (Swimming)", "Lionel Messi (Football)"], "curated_challenge_subjective")
    _add_rec("draft_ambig_subj_016", "grp_subj_best_athlete", "Which sports figure is the absolute best athlete in history?", "subjective_questions",
             "Paraphrase of greatest athlete", "AMBIGUOUS", "Cross-sport achievements cannot be unified objectively", "delineated_interpretations",
             ["Ali, Jordan, Pele, Phelps"], "curated_challenge_subjective")

    # Group: grp_subj_best_food
    _add_rec("draft_ambig_subj_017", "grp_subj_best_food", "What is the most delicious cuisine in the world?", "subjective_questions",
             "Gastronomic subjective preference", "AMBIGUOUS", "Individual gustatory taste is non-objective", "delineated_interpretations",
             ["Italian, French, Japanese, Indian, Mexican cuisines"], "curated_challenge_subjective")
    _add_rec("draft_ambig_subj_018", "grp_subj_best_food", "Which nation has the best culinary food?", "subjective_questions",
             "Paraphrase of most delicious cuisine", "AMBIGUOUS", "Cultural culinary preferences vary widely", "delineated_interpretations",
             ["Michelin guide rankings vs popular culinary surveys"], "curated_challenge_subjective")

    # Group: grp_subj_best_city
    _add_rec("draft_ambig_subj_019", "grp_subj_best_city", "What is the most beautiful city in the world?", "subjective_questions",
             "Aesthetic urban landscape appraisal", "AMBIGUOUS", "Aesthetic beauty is subjective", "delineated_interpretations",
             ["Paris, Venice, Kyoto, Cape Town, Rio de Janeiro"], "curated_challenge_subjective")
    _add_rec("draft_ambig_subj_020", "grp_subj_best_city", "Which city has the most attractive urban architecture and scenery?", "subjective_questions",
             "Paraphrase of most beautiful city", "AMBIGUOUS", "Subjective aesthetic judgment", "delineated_interpretations",
             ["Venice architectural heritage vs Kyoto temples vs Paris boulevards"], "curated_challenge_subjective")

    # Group: grp_subj_best_car
    _add_rec("draft_ambig_subj_021", "grp_subj_best_car", "What is the best automobile to buy?", "subjective_questions",
             "Consumer vehicle purchasing preference", "AMBIGUOUS", "Depends on budget, fuel type, utility, reliability, aesthetics", "delineated_interpretations",
             ["Reliability: Toyota; Performance: Porsche; Utility: Ford F-150; Electric: Tesla"], "curated_challenge_subjective")
    _add_rec("draft_ambig_subj_022", "grp_subj_best_car", "Which car on the market is the superior choice for consumers?", "subjective_questions",
             "Paraphrase of best automobile to buy", "AMBIGUOUS", "Multidimensional trade-offs prevent single answer", "delineated_interpretations",
             ["Consumer Reports vehicle rankings vary by category"], "curated_challenge_subjective")

    # Group: grp_subj_best_album
    _add_rec("draft_ambig_subj_023", "grp_subj_best_album", "What is the greatest musical album of all time?", "subjective_questions",
             "Music appraisal without criteria", "AMBIGUOUS", "Artistic taste across diverse musical genres", "delineated_interpretations",
             ["Sgt. Pepper's (Beatles)", "Pet Sounds (Beach Boys)", "What's Going On (Marvin Gaye)", "Blue (Joni Mitchell)"], "curated_challenge_subjective")
    _add_rec("draft_ambig_subj_024", "grp_subj_best_album", "Which recorded music album is the best in history?", "subjective_questions",
             "Paraphrase of greatest musical album", "AMBIGUOUS", "Rolling Stone 500 Greatest Albums vs Pitchfork rankings", "delineated_interpretations",
             ["Rolling Stone Top 500 Albums"], "curated_challenge_subjective")

    # Group: grp_subj_best_game
    _add_rec("draft_ambig_subj_025", "grp_subj_best_game", "What is the best video game ever created?", "subjective_questions",
             "Video game artistic/interactive appraisal", "AMBIGUOUS", "Subjective game design evaluation", "delineated_interpretations",
             ["The Legend of Zelda: Ocarina of Time", "Tetris", "Super Mario 64", "Half-Life 2"], "curated_challenge_subjective")
    _add_rec("draft_ambig_subj_026", "grp_subj_best_game", "Which video game holds the title of greatest of all time?", "subjective_questions",
             "Paraphrase of best video game", "AMBIGUOUS", "Disparate critical and player rankings", "delineated_interpretations",
             ["Metacritic all-time scores vs IGN Top 100"], "curated_challenge_subjective")

    # Group: grp_subj_best_philosopher
    _add_rec("draft_ambig_subj_027", "grp_subj_best_philosopher", "Who was the most influential philosopher in human history?", "subjective_questions",
             "Philosophical tradition comparison", "AMBIGUOUS", "Depends on tradition (Western, Eastern) and philosophical domain (epistemology, ethics, logic)", "delineated_interpretations",
             ["Aristotle, Plato, Kant, Confucius, Descartes"], "curated_challenge_subjective")
    _add_rec("draft_ambig_subj_028", "grp_subj_best_philosopher", "Which historical thinker had the greatest philosophical impact on civilization?", "subjective_questions",
             "Paraphrase of most influential philosopher", "AMBIGUOUS", "Incommensurable philosophical frameworks", "delineated_interpretations",
             ["Western vs Eastern philosophical lineages"], "curated_challenge_subjective")

    # Group: grp_subj_best_operating_system
    _add_rec("draft_ambig_subj_029", "grp_subj_best_operating_system", "What is the best desktop operating system?", "subjective_questions",
             "Operating system usability and utility", "AMBIGUOUS", "Depends on user requirements: privacy, gaming, development, enterprise support", "delineated_interpretations",
             ["Linux (customizability/servers)", "macOS (Unix ergonomics)", "Windows (software/gaming support)"], "curated_challenge_subjective")
    _add_rec("draft_ambig_subj_030", "grp_subj_best_operating_system", "Which operating system is superior for personal computers?", "subjective_questions",
             "Paraphrase of best desktop operating system", "AMBIGUOUS", "Diverse trade-offs across ecosystems", "delineated_interpretations",
             ["Linux vs macOS vs Windows"], "curated_challenge_subjective")

    # Group: grp_subj_most_useful_degree
    _add_rec("draft_ambig_subj_031", "grp_subj_most_useful_degree", "What is the most valuable college major to study?", "subjective_questions",
             "Educational career value judgment", "AMBIGUOUS", "Depends on criteria: initial salary, personal fulfillment, societal contribution, career longevity", "delineated_interpretations",
             ["Computer Science / Engineering (ROI)", "Medicine / Nursing (stability)", "Humanities (critical thinking)"], "curated_challenge_subjective")
    _add_rec("draft_ambig_subj_032", "grp_subj_most_useful_degree", "Which university degree offers the greatest overall value to students?", "subjective_questions",
             "Paraphrase of most valuable college major", "AMBIGUOUS", "Subjective definition of personal vs financial value", "delineated_interpretations",
             ["Bureau of Labor Statistics ROI vs lifetime satisfaction"], "curated_challenge_subjective")

    # Group: grp_subj_best_superhero
    _add_rec("draft_ambig_subj_033", "grp_subj_best_superhero", "Who is the greatest fictional superhero of all time?", "subjective_questions",
             "Comic fiction popular character evaluation", "AMBIGUOUS", "Subjective cultural fandom without singular objective answer", "delineated_interpretations",
             ["Batman (detective/cultural icon)", "Superman (archetype)", "Spider-Man (relatability)"], "curated_challenge_subjective")
    _add_rec("draft_ambig_subj_034", "grp_subj_best_superhero", "Which comic book character is the best superhero ever created?", "subjective_questions",
             "Paraphrase of greatest superhero", "AMBIGUOUS", "Subjective cultural preference", "delineated_interpretations",
             ["DC vs Marvel iconic heroes"], "curated_challenge_subjective")

    # Group: grp_subj_best_vacation
    _add_rec("draft_ambig_subj_035", "grp_subj_best_vacation", "What is the ideal vacation destination?", "subjective_questions",
             "Personal leisure preference", "AMBIGUOUS", "Depends entirely on personal tastes: relaxation vs adventure vs culture", "delineated_interpretations",
             ["Beach resort vs mountain trekking vs historic city exploration"], "curated_challenge_subjective")
    _add_rec("draft_ambig_subj_036", "grp_subj_best_vacation", "Which place on Earth is the best spot for a vacation?", "subjective_questions",
             "Paraphrase of ideal vacation destination", "AMBIGUOUS", "Subjective leisure priorities", "delineated_interpretations",
             ["Tourism trade-offs"], "curated_challenge_subjective")

    # Group: grp_subj_best_pet
    _add_rec("draft_ambig_subj_037", "grp_subj_best_pet", "What is the best animal to keep as a household pet?", "subjective_questions",
             "Domestic companion animal appraisal", "AMBIGUOUS", "Depends on lifestyle, living space, allergies, companionship desires", "delineated_interpretations",
             ["Dogs (companionship)", "Cats (independence)", "Fish (low maintenance)"], "curated_challenge_subjective")
    _add_rec("draft_ambig_subj_038", "grp_subj_best_pet", "Which domestic animal makes the superior pet for families?", "subjective_questions",
             "Paraphrase of best household pet", "AMBIGUOUS", "Subjective family lifestyle criteria", "delineated_interpretations",
             ["Canine vs feline lifestyle compatibility"], "curated_challenge_subjective")

    # Group: grp_subj_best_architecture
    _add_rec("draft_ambig_subj_039", "grp_subj_best_architecture", "What is the greatest architectural work ever constructed by humanity?", "subjective_questions",
             "Historical structural aesthetic ranking", "AMBIGUOUS", "Subjective architectural evaluation across civilizations and millennia", "delineated_interpretations",
             ["Great Pyramid of Giza", "Parthenon", "Hagia Sophia", "Taj Mahal", "Sagrada Familia"], "curated_challenge_subjective")
    _add_rec("draft_ambig_subj_040", "grp_subj_best_architecture", "Which historical building represents the highest achievement in human architecture?", "subjective_questions",
             "Paraphrase of greatest architectural work", "AMBIGUOUS", "Incommensurable ancient vs modern architectural marvels", "delineated_interpretations",
             ["UNESCO World Heritage masterpieces"], "curated_challenge_subjective")


    # =========================================================================
    # CATEGORY 4: Genuine Entity Polysemy & Linguistic Ambiguity (Quota: 60, Target: AMBIGUOUS)
    # =========================================================================
    # Group: grp_poly_georgia_capital
    _add_rec("draft_ambig_poly_001", "grp_poly_georgia_capital", "What is the capital of Georgia?", "genuine_ambiguity",
             "Unspecified entity: Sovereign Eurasian nation vs US State", "AMBIGUOUS", "Refers equally to two prominent geopolitical entities", "delineated_interpretations",
             ["Tbilisi (Sovereign nation in Caucasus)", "Atlanta (US State)"], "geographical_polysemy")
    _add_rec("draft_ambig_poly_002", "grp_poly_georgia_capital", "Which city is Georgia's capital?", "genuine_ambiguity",
             "Paraphrase of capital of Georgia query", "AMBIGUOUS", "Geographical polysemy between nation and state", "delineated_interpretations",
             ["Tbilisi or Atlanta"], "geographical_polysemy")
    _add_rec("draft_ambig_poly_003", "grp_poly_georgia_capital", "What city serves as the official capital of Georgia?", "genuine_ambiguity",
             "Formal phrasing of capital of Georgia", "AMBIGUOUS", "Entity ambiguity unresolved without context", "delineated_interpretations",
             ["Tbilisi vs Atlanta"], "geographical_polysemy")

    # Group: grp_poly_spiderman_actor
    _add_rec("draft_ambig_poly_004", "grp_poly_spiderman_actor", "Who played Spider-Man in the movies?", "genuine_ambiguity",
             "Unspecified movie franchise for comic book character", "AMBIGUOUS", "Multiple lead actors portrayed the character in major film franchises", "delineated_interpretations",
             ["Tobey Maguire (Raimi trilogy)", "Andrew Garfield (Webb films)", "Tom Holland (MCU)", "Shameik Moore (Spider-Verse)"], "cultural_polysemy")
    _add_rec("draft_ambig_poly_005", "grp_poly_spiderman_actor", "Which actor portrays the live-action superhero Spider-Man in film?", "genuine_ambiguity",
             "Paraphrase of Spider-Man actor query", "AMBIGUOUS", "Multiple legitimate live-action film eras", "delineated_interpretations",
             ["Maguire, Garfield, Holland"], "cultural_polysemy")
    _add_rec("draft_ambig_poly_006", "grp_poly_spiderman_actor", "Who was cast as the lead role of Peter Parker / Spider-Man in cinema?", "genuine_ambiguity",
             "Syntactic paraphrase of Spider-Man casting", "AMBIGUOUS", "Multiple actors across film series", "delineated_interpretations",
             ["Maguire, Garfield, Holland"], "cultural_polysemy")

    # Group: grp_poly_batman_actor
    _add_rec("draft_ambig_poly_007", "grp_poly_batman_actor", "Who played Batman in the movies?", "genuine_ambiguity",
             "Unspecified movie franchise for Batman", "AMBIGUOUS", "Numerous actors across six decades of cinematic history", "delineated_interpretations",
             ["Michael Keaton, Val Kilmer, George Clooney, Christian Bale, Ben Affleck, Robert Pattinson"], "cultural_polysemy")
    _add_rec("draft_ambig_poly_008", "grp_poly_batman_actor", "Which actor has portrayed the character of Batman on the big screen?", "genuine_ambiguity",
             "Paraphrase of Batman actor query", "AMBIGUOUS", "Multiple live-action cinematic eras", "delineated_interpretations",
             ["Keaton, Bale, Affleck, Pattinson"], "cultural_polysemy")

    # Group: grp_poly_president_lincoln
    _add_rec("draft_ambig_poly_009", "grp_poly_president_lincoln", "Who is the president of Lincoln?", "genuine_ambiguity",
             "Unspecified institution: Lincoln University, Lincoln College, Lincoln Financial, or Abraham Lincoln", "AMBIGUOUS", "Entity ambiguity across multiple organizations sharing the name Lincoln", "delineated_interpretations",
             ["Lincoln University (PA/MO/UK)", "Lincoln College Oxford", "Historical: Abraham Lincoln (16th US President)"], "entity_name_polysemy")
    _add_rec("draft_ambig_poly_010", "grp_poly_president_lincoln", "Who holds the presidency of Lincoln?", "genuine_ambiguity",
             "Paraphrase of president of Lincoln", "AMBIGUOUS", "Refers to multiple universities, organizations, or historical figures", "delineated_interpretations",
             ["Multiple academic and historical entities"], "entity_name_polysemy")

    # Group: grp_poly_mercury_orbit
    _add_rec("draft_ambig_poly_011", "grp_poly_mercury_orbit", "What is the orbital period of Mercury?", "genuine_ambiguity",
             "Polysemy between the planet Mercury and Project Mercury spacecraft", "AMBIGUOUS", "Could refer to solar planet Mercury (88 days) or Project Mercury space missions (~90 minutes)", "delineated_interpretations",
             ["Planet Mercury around Sun: ~88 Earth days", "Project Mercury spaceflight around Earth: ~88-90 minutes"], "astronomy_spaceflight_polysemy")
    _add_rec("draft_ambig_poly_012", "grp_poly_mercury_orbit", "How long does it take for Mercury to complete one full orbit?", "genuine_ambiguity",
             "Paraphrase of Mercury orbit query", "AMBIGUOUS", "Ambiguous referent (planet vs orbital space capsule)", "delineated_interpretations",
             ["88 Earth days vs 90 minutes"], "astronomy_spaceflight_polysemy")

    # Group: grp_poly_largest_city_criteria
    _add_rec("draft_ambig_poly_013", "grp_poly_largest_city_criteria", "What is the largest city in the world?", "genuine_ambiguity",
             "Lacks measurement criterion (population vs surface area vs metropolitan area)", "AMBIGUOUS", "Different valid answers based on metric (Tokyo by metropolitan pop, Chongqing by administrative area)", "delineated_interpretations",
             ["By metropolitan population: Tokyo", "By city-proper population: Shanghai", "By administrative area: Chongqing"], "criterion_ambiguity")
    _add_rec("draft_ambig_poly_014", "grp_poly_largest_city_criteria", "Which city in the world is the largest?", "genuine_ambiguity",
             "Paraphrase of largest city query", "AMBIGUOUS", "Unstated dimension of size (area vs population)", "delineated_interpretations",
             ["Tokyo vs Shanghai vs Chongqing"], "criterion_ambiguity")

    # Group: grp_poly_prime_minister_who
    _add_rec("draft_ambig_poly_015", "grp_poly_prime_minister_who", "Who is the Prime Minister?", "genuine_ambiguity",
             "Unstated country and temporal anchor", "AMBIGUOUS", "Hundreds of sovereign countries have prime ministers", "delineated_interpretations",
             ["UK, Canada, Japan, India, Australia, etc."], "contextual_ellipsis")
    _add_rec("draft_ambig_poly_016", "grp_poly_prime_minister_who", "Who currently holds the office of Prime Minister?", "genuine_ambiguity",
             "Paraphrase of unanchored prime minister query", "AMBIGUOUS", "Lacks sovereign state specification", "delineated_interpretations",
             ["UK PM vs Canadian PM vs Australian PM"], "contextual_ellipsis")

    # Group: grp_poly_springfield_state
    _add_rec("draft_ambig_poly_017", "grp_poly_springfield_state", "In which US state is Springfield located?", "genuine_ambiguity",
             "Polysemous city name across US states", "AMBIGUOUS", "Over 30 distinct recognized cities named Springfield exist in US states", "delineated_interpretations",
             ["Illinois (State capital), Massachusetts, Missouri, Ohio, Oregon, etc."], "geographic_homonym")
    _add_rec("draft_ambig_poly_018", "grp_poly_springfield_state", "Which state in America contains the city of Springfield?", "genuine_ambiguity",
             "Paraphrase of Springfield state query", "AMBIGUOUS", "Shared homonym across dozens of US states", "delineated_interpretations",
             ["Illinois, Massachusetts, Missouri, Ohio"], "geographic_homonym")

    # Group: grp_poly_bank_definition
    _add_rec("draft_ambig_poly_019", "grp_poly_bank_definition", "What is a bank?", "genuine_ambiguity",
             "Lexical homonymy / polysemy", "AMBIGUOUS", "Refers to financial institution, river bank, aircraft bank angle, or blood bank", "delineated_interpretations",
             ["Financial institution accepting deposits", "Sloping land along river", "Aircraft flight orientation"], "linguistic_polysemy")
    _add_rec("draft_ambig_poly_020", "grp_poly_bank_definition", "How is the word 'bank' defined?", "genuine_ambiguity",
             "Paraphrase of bank definition", "AMBIGUOUS", "Multiple prominent homographic definitions", "delineated_interpretations",
             ["Financial bank vs river bank"], "linguistic_polysemy")

    # Additional polysemy and ambiguity groups (21-60)
    # Group: grp_poly_jordan_who
    _add_rec("draft_ambig_poly_021", "grp_poly_jordan_who", "Who is Jordan?", "genuine_ambiguity",
             "Ambiguous entity reference", "AMBIGUOUS", "Refers to country, basketball star Michael Jordan, actor Michael B. Jordan, psychologist Jordan Peterson", "delineated_interpretations",
             ["Country in Middle East", "Michael Jordan (NBA)", "Jordan Peterson", "Michael B. Jordan"], "entity_name_polysemy")
    _add_rec("draft_ambig_poly_022", "grp_poly_jordan_who", "Who or what does 'Jordan' refer to?", "genuine_ambiguity",
             "Paraphrase of Jordan entity query", "AMBIGUOUS", "Polysemous referent", "delineated_interpretations",
             ["Person vs geographical state"], "entity_name_polysemy")

    # Group: grp_poly_paris_location
    _add_rec("draft_ambig_poly_023", "grp_poly_paris_location", "Where is Paris located?", "genuine_ambiguity",
             "Homonymous city reference", "AMBIGUOUS", "Refers to capital of France or Paris, Texas, or Paris, Tennessee", "delineated_interpretations",
             ["Paris, France (capital city)", "Paris, Texas (US city)", "Mythological Paris of Troy"], "geographic_homonym")
    _add_rec("draft_ambig_poly_024", "grp_poly_paris_location", "In what jurisdiction or region is the city of Paris found?", "genuine_ambiguity",
             "Paraphrase of Paris location", "AMBIGUOUS", "France vs Texas vs Ontario", "delineated_interpretations",
             ["France vs US municipalities"], "geographic_homonym")

    # Group: grp_poly_jaguar_speed
    _add_rec("draft_ambig_poly_025", "grp_poly_jaguar_speed", "What is the top speed of a Jaguar?", "genuine_ambiguity",
             "Animal Panthera onca vs Jaguar sports car", "AMBIGUOUS", "Refers to big cat (~80 km/h) or Jaguar supercar (e.g. XJ220 at 349 km/h)", "delineated_interpretations",
             ["Animal Panthera onca: ~80 km/h", "Jaguar automobile (XJ220): ~349 km/h"], "brand_vs_biology_polysemy")
    _add_rec("draft_ambig_poly_026", "grp_poly_jaguar_speed", "How fast can a jaguar travel at maximum speed?", "genuine_ambiguity",
             "Paraphrase of jaguar speed query", "AMBIGUOUS", "Feline animal vs luxury vehicle", "delineated_interpretations",
             ["Animal vs car speed"], "brand_vs_biology_polysemy")

    # Group: grp_poly_apple_valuation
    _add_rec("draft_ambig_poly_027", "grp_poly_apple_valuation", "What is an Apple worth?", "genuine_ambiguity",
             "Fruit Malus domestica vs Apple Inc. enterprise value", "AMBIGUOUS", "Refers to supermarket piece of fruit (~$1) or market cap of Apple Inc. (~$3 trillion)", "delineated_interpretations",
             ["Single agricultural fruit: ~$0.50-$1.50", "Apple Inc. market capitalization: ~$3 Trillion"], "lexical_homonymy")
    _add_rec("draft_ambig_poly_028", "grp_poly_apple_valuation", "How much does an apple cost or value at?", "genuine_ambiguity",
             "Paraphrase of apple value query", "AMBIGUOUS", "Commodity grocery item vs multinational tech corporation", "delineated_interpretations",
             ["Grocery fruit vs Apple corporation"], "lexical_homonymy")

    # Group: grp_poly_galaxy_size
    _add_rec("draft_ambig_poly_029", "grp_poly_galaxy_size", "What are the dimensions of a Galaxy?", "genuine_ambiguity",
             "Astronomical galaxy vs Samsung Galaxy smartphone", "AMBIGUOUS", "Refers to astronomical system of stars (100,000 light-years) or Samsung mobile device (15 cm)", "delineated_interpretations",
             ["Astronomical galaxy: ~100,000 light years across", "Samsung Galaxy phone: ~15 x 7 cm"], "product_vs_astronomy_polysemy")
    _add_rec("draft_ambig_poly_030", "grp_poly_galaxy_size", "How large is a Galaxy?", "genuine_ambiguity",
             "Paraphrase of galaxy size query", "AMBIGUOUS", "Astronomical entity vs handheld consumer phone", "delineated_interpretations",
             ["Astrophysical galaxy vs mobile phone"], "product_vs_astronomy_polysemy")

    # Group: grp_poly_python_length
    _add_rec("draft_ambig_poly_031", "grp_poly_python_length", "How long is Python?", "genuine_ambiguity",
             "Snake species vs Python programming language release timeline/code length", "AMBIGUOUS", "Refers to reptile length (up to 6 meters) or Python language history/specification", "delineated_interpretations",
             ["Reticulated python snake: 3 to 6 meters", "Python language history: 1991 to present"], "biology_vs_software_polysemy")
    _add_rec("draft_ambig_poly_032", "grp_poly_python_length", "What is the typical length of a python?", "genuine_ambiguity",
             "Paraphrase of python length", "AMBIGUOUS", "Biological organism vs programming language context", "delineated_interpretations",
             ["Reptile species vs software"], "biology_vs_software_polysemy")

    # Group: grp_poly_amazon_depth
    _add_rec("draft_ambig_poly_033", "grp_poly_amazon_depth", "How deep is Amazon?", "genuine_ambiguity",
             "Amazon River bathymetry vs Amazon corporate catalog/operations", "AMBIGUOUS", "Refers to Amazon River depth (~100 meters) or organizational depth of Amazon.com", "delineated_interpretations",
             ["Amazon River channel depth: up to 100 meters", "Amazon corporate structure / catalog breadth"], "geographic_vs_corporate_polysemy")
    _add_rec("draft_ambig_poly_034", "grp_poly_amazon_depth", "What is the depth measurement of the Amazon?", "genuine_ambiguity",
             "Paraphrase of Amazon depth query", "AMBIGUOUS", "River physical measurement vs corporate terminology", "delineated_interpretations",
             ["River depth vs corporate entity"], "geographic_vs_corporate_polysemy")

    # Group: grp_poly_corona_meaning
    _add_rec("draft_ambig_poly_035", "grp_poly_corona_meaning", "What is the Corona?", "genuine_ambiguity",
             "Solar atmosphere vs beer brand vs virus family", "AMBIGUOUS", "Refers to solar plasma atmosphere, Mexican beer brand, or coronavirus", "delineated_interpretations",
             ["Solar corona plasma atmosphere", "Corona brand pale lager", "Coronaviridae virus genus"], "polysemy_multiple_meanings")
    _add_rec("draft_ambig_poly_036", "grp_poly_corona_meaning", "What does 'Corona' refer to?", "genuine_ambiguity",
             "Paraphrase of Corona meaning", "AMBIGUOUS", "Astrophysical, commercial, and virological meanings", "delineated_interpretations",
             ["Sun atmosphere vs beer vs virus"], "polysemy_multiple_meanings")

    # Group: grp_poly_cold_war_dates
    _add_rec("draft_ambig_poly_037", "grp_poly_cold_war_dates", "When did the Cold War begin?", "genuine_ambiguity",
             "Historiographical divergence on onset of Cold War", "AMBIGUOUS", "Historians debate start date: 1945 (Yalta/Potsdam), 1947 (Truman Doctrine), or 1917 (Bolshevik Revolution)", "delineated_interpretations",
             ["1945: Yalta/Potsdam conferences", "1947: Truman Doctrine declaration", "1948: Berlin Blockade"], "historiographical_debate")
    _add_rec("draft_ambig_poly_038", "grp_poly_cold_war_dates", "In what year did the historical Cold War start?", "genuine_ambiguity",
             "Paraphrase of Cold War start year", "AMBIGUOUS", "Multiple legitimate historical definitions of onset", "delineated_interpretations",
             ["1945 vs 1947"], "historiographical_debate")

    # Group: grp_poly_industrial_rev_dates
    _add_rec("draft_ambig_poly_039", "grp_poly_industrial_rev_dates", "When did the Industrial Revolution start?", "genuine_ambiguity",
             "Historiographical divergence on Industrial Revolution periodization", "AMBIGUOUS", "Historians dispute start decade: ~1760 (traditional English textile mechanization) or ~1780 (steam adoption)", "delineated_interpretations",
             ["circa 1760 (Ashton tradition)", "circa 1780 (Hobsbawm take-off thesis)"], "historiographical_debate")
    _add_rec("draft_ambig_poly_040", "grp_poly_industrial_rev_dates", "Which decade marks the beginning of the Industrial Revolution?", "genuine_ambiguity",
             "Paraphrase of Industrial Revolution start", "AMBIGUOUS", "Ongoing economic history periodization debate", "delineated_interpretations",
             ["1760s vs 1780s"], "historiographical_debate")

    # Group: grp_poly_matrix_definition
    _add_rec("draft_ambig_poly_041", "grp_poly_matrix_definition", "What is the Matrix?", "genuine_ambiguity",
             "Linear algebra array vs 1999 sci-fi movie vs biology extracellular matrix", "AMBIGUOUS", "Refers to mathematical rectangular array, Wachowski film, or cellular extracellular matrix", "delineated_interpretations",
             ["Mathematical rectangular array of numbers", "1999 science fiction motion picture", "Biology extracellular scaffolding"], "domain_polysemy")
    _add_rec("draft_ambig_poly_042", "grp_poly_matrix_definition", "How do you define 'the Matrix'?", "genuine_ambiguity",
             "Paraphrase of Matrix query", "AMBIGUOUS", "Mathematics vs film vs biology", "delineated_interpretations",
             ["Linear algebra vs cinematic franchise"], "domain_polysemy")

    # Group: grp_poly_rust_language_iron
    _add_rec("draft_ambig_poly_043", "grp_poly_rust_language_iron", "What is Rust?", "genuine_ambiguity",
             "Iron oxide chemical corrosion vs Rust programming language", "AMBIGUOUS", "Refers to iron oxide corrosion compound or Mozilla-developed programming language", "delineated_interpretations",
             ["Chemical iron oxide (Fe2O3) formed by oxidation", "Systems programming language developed by Mozilla"], "chemistry_vs_software_polysemy")
    _add_rec("draft_ambig_poly_044", "grp_poly_rust_language_iron", "Define the term 'Rust'.", "genuine_ambiguity",
             "Paraphrase of Rust query", "AMBIGUOUS", "Chemical oxidation vs systems programming language", "delineated_interpretations",
             ["Iron oxide vs programming language"], "chemistry_vs_software_polysemy")

    # Group: grp_poly_saturn_rings_count
    _add_rec("draft_ambig_poly_045", "grp_poly_saturn_rings_count", "How many rings does Saturn have?", "genuine_ambiguity",
             "Astronomical classification ambiguity", "AMBIGUOUS", "Depends on counting method: 7 primary main ring groups (A-G) or thousands/millions of individual ringlets", "delineated_interpretations",
             ["7 primary main ring groups (A through G)", "Thousands of individual discrete ringlets"], "astronomical_classification_ambiguity")
    _add_rec("draft_ambig_poly_046", "grp_poly_saturn_rings_count", "What is the total count of Saturn's planetary rings?", "genuine_ambiguity",
             "Paraphrase of Saturn rings count", "AMBIGUOUS", "Main group classification vs sub-ringlets", "delineated_interpretations",
             ["7 primary groups vs thousands of ringlets"], "astronomical_classification_ambiguity")

    # Group: grp_poly_apollo_who
    _add_rec("draft_ambig_poly_047", "grp_poly_apollo_who", "What is Apollo?", "genuine_ambiguity",
             "Greek deity vs NASA space program vs Apollo theater", "AMBIGUOUS", "Refers to Olympian deity in classical mythology, NASA lunar program, or historic Harlem theater", "delineated_interpretations",
             ["Greek/Roman god of music, truth, and prophecy", "NASA crewed lunar landing program (1961-1972)", "Apollo Theater in Harlem, New York"], "mythology_space_culture_polysemy")
    _add_rec("draft_ambig_poly_048", "grp_poly_apollo_who", "Who or what is referred to as Apollo?", "genuine_ambiguity",
             "Paraphrase of Apollo query", "AMBIGUOUS", "Classical deity vs NASA program", "delineated_interpretations",
             ["Greek god vs spaceflight program"], "mythology_space_culture_polysemy")

    # Group: grp_poly_genesis_meaning
    _add_rec("draft_ambig_poly_049", "grp_poly_genesis_meaning", "What is Genesis?", "genuine_ambiguity",
             "Biblical book vs rock band vs Sega video game console", "AMBIGUOUS", "Refers to first book of Hebrew Bible/Christian Old Testament, British rock band, or Sega console", "delineated_interpretations",
             ["First book of Torah / Bible", "English progressive rock band", "Sega 16-bit video game console"], "religious_music_tech_polysemy")
    _add_rec("draft_ambig_poly_050", "grp_poly_genesis_meaning", "Define or identify 'Genesis'.", "genuine_ambiguity",
             "Paraphrase of Genesis query", "AMBIGUOUS", "Theological, musical, and technological referents", "delineated_interpretations",
             ["Biblical book vs rock band vs Sega console"], "religious_music_tech_polysemy")

    # Group: grp_poly_mouse_lifespan
    _add_rec("draft_ambig_poly_051", "grp_poly_mouse_lifespan", "What is the lifespan of a mouse?", "genuine_ambiguity",
             "Biological rodent Mus musculus vs computer input peripheral device", "AMBIGUOUS", "Refers to biological rodent (1-3 years) or computer hardware durability (3-10 years)", "delineated_interpretations",
             ["Biological rodent Mus musculus: ~1 to 3 years", "Computer hardware input device: ~3 to 10 years of operational clicks"], "biology_vs_hardware_polysemy")
    _add_rec("draft_ambig_poly_052", "grp_poly_mouse_lifespan", "How long does a mouse live or last?", "genuine_ambiguity",
             "Paraphrase of mouse lifespan query", "AMBIGUOUS", "Living rodent vs electronic computer mouse", "delineated_interpretations",
             ["Rodent biological life vs computer mouse hardware"], "biology_vs_hardware_polysemy")

    # Group: grp_poly_bridge_game
    _add_rec("draft_ambig_poly_053", "grp_poly_bridge_game", "What is Bridge?", "genuine_ambiguity",
             "Civil engineering structure vs trick-taking card game vs dentistry", "AMBIGUOUS", "Refers to civil engineering roadway over obstacle, card game, or dental prosthetic", "delineated_interpretations",
             ["Civil engineering structure spanning physical obstacle", "Contract bridge trick-taking card game", "Dental prosthesis replacing missing teeth"], "homonym_polysemy")
    _add_rec("draft_ambig_poly_054", "grp_poly_bridge_game", "Define the term 'Bridge'.", "genuine_ambiguity",
             "Paraphrase of Bridge query", "AMBIGUOUS", "Civil structure vs card game vs dental fixture", "delineated_interpretations",
             ["Road structure vs card game"], "homonym_polysemy")

    # Group: grp_poly_columbia_capital
    _add_rec("draft_ambig_poly_055", "grp_poly_columbia_capital", "What is the capital of Columbia?", "genuine_ambiguity",
             "Spelling homophone / polysemy: South Carolina state capital vs country of Colombia", "AMBIGUOUS", "Refers to Columbia, South Carolina (state capital), District of Columbia, or misspelling of Republic of Colombia (Bogota)", "delineated_interpretations",
             ["Columbia, South Carolina (state capital)", "District of Columbia (Washington, US capital)", "If intended as Colombia: Bogota"], "orthographic_homophone_polysemy",
             audit_trail={"borderline_review": "Retained as AMBIGUOUS evaluating common orthographic ambiguity between Columbia (SC/MO/DC) and Colombia (Bogota)."})
    _add_rec("draft_ambig_poly_056", "grp_poly_columbia_capital", "Which city is the capital of Columbia?", "genuine_ambiguity",
             "Paraphrase of Columbia capital query", "AMBIGUOUS", "South Carolina capital vs Colombia country ambiguity", "delineated_interpretations",
             ["Columbia SC vs Bogota Colombia"], "orthographic_homophone_polysemy")

    # Group: grp_poly_server_definition
    _add_rec("draft_ambig_poly_057", "grp_poly_server_definition", "What is a server?", "genuine_ambiguity",
             "Computing hardware/daemon vs restaurant waitstaff vs tennis server", "AMBIGUOUS", "Refers to computer network service host, food service worker, or tennis player serving ball", "delineated_interpretations",
             ["Computing device or program providing network service", "Person serving food and drinks in restaurant", "Tennis player initiating play with serve"], "occupational_tech_sports_polysemy")
    _add_rec("draft_ambig_poly_058", "grp_poly_server_definition", "Define the word 'server'.", "genuine_ambiguity",
             "Paraphrase of server query", "AMBIGUOUS", "Computer technology vs hospitality industry vs tennis", "delineated_interpretations",
             ["Computer host vs restaurant worker"], "occupational_tech_sports_polysemy")

    # Group: grp_poly_diamond_hardness
    _add_rec("draft_ambig_poly_059", "grp_poly_diamond_hardness", "What is the hardness of a diamond?", "genuine_ambiguity",
             "Mohs scratch hardness vs Vickers absolute indentation hardness", "AMBIGUOUS", "Mohs mineral scale is 10; Vickers indentation hardness is ~70-150 GPa", "delineated_interpretations",
             ["Mohs relative scratch hardness scale: exactly 10", "Vickers absolute indentation hardness: ~70 to 150 GPa"], "measurement_scale_ambiguity")
    _add_rec("draft_ambig_poly_060", "grp_poly_diamond_hardness", "How hard is diamond measured to be?", "genuine_ambiguity",
             "Paraphrase of diamond hardness query", "AMBIGUOUS", "Mohs scale vs Vickers indentation scale", "delineated_interpretations",
             ["Mohs 10 vs Vickers indentation GPa"], "measurement_scale_ambiguity")


    # =========================================================================
    # CATEGORY 5: Private or Inaccessible Information (Quota: 35, Target: UNKNOWN)
    # =========================================================================
    # Group: grp_priv_user_pin
    _add_rec("draft_unk_priv_001", "grp_priv_user_pin", "What is the 6-digit PIN to my personal bank ATM card?", "private_inaccessible_info",
             "Private personal authentication credential", "UNKNOWN", "Private financial security credential inaccessible to model or public records", "inaccessibility_proof",
             ["Bank security PIN is private personal data with zero public trace"], "curated_adversarial_private")
    _add_rec("draft_unk_priv_002", "grp_priv_user_pin", "What is my secret ATM debit card PIN number?", "private_inaccessible_info",
             "Paraphrase of private ATM PIN query", "UNKNOWN", "Private confidential secret", "inaccessibility_proof",
             ["Private user credential"], "curated_adversarial_private")

    # Group: grp_priv_user_safe
    _add_rec("draft_unk_priv_003", "grp_priv_user_safe", "What is the secret passphrase written on the note inside my locked safe?", "private_inaccessible_info",
             "Private unindexed confidential note", "UNKNOWN", "Private physical secret absent from any knowledge corpus", "inaccessibility_proof",
             ["Physical safe contents private to owner"], "curated_adversarial_private")
    _add_rec("draft_unk_priv_004", "grp_priv_user_safe", "What code or word is written on the piece of paper in my home safe?", "private_inaccessible_info",
             "Paraphrase of private home safe query", "UNKNOWN", "Private inaccessible physical object", "inaccessibility_proof",
             ["Private physical secret"], "curated_adversarial_private")

    # Group: grp_priv_user_thoughts
    _add_rec("draft_unk_priv_005", "grp_priv_user_thoughts", "What number am I thinking of right now?", "private_inaccessible_info",
             "Private real-time human cognition", "UNKNOWN", "Private mental state inaccessible to external observer", "inaccessibility_proof",
             ["Human mental cognition is unobservable"], "curated_adversarial_private")
    _add_rec("draft_unk_priv_006", "grp_priv_user_thoughts", "Which number is currently in my mind as I type this prompt?", "private_inaccessible_info",
             "Paraphrase of mental number query", "UNKNOWN", "Private human mental state", "inaccessibility_proof",
             ["Unobservable mental state"], "curated_adversarial_private")

    # Group: grp_priv_user_shirt
    _add_rec("draft_unk_priv_007", "grp_priv_user_shirt", "What color shirt is the person reading this screen wearing right now?", "private_inaccessible_info",
             "Real-time unobserved physical state of user", "UNKNOWN", "Uninstrumented physical environment absent from input context", "inaccessibility_proof",
             ["User physical environment unobserved"], "curated_adversarial_private")
    _add_rec("draft_unk_priv_008", "grp_priv_user_shirt", "What is the color of the clothing currently worn by the user?", "private_inaccessible_info",
             "Paraphrase of user clothing color", "UNKNOWN", "Unmonitored physical state", "inaccessibility_proof",
             ["Physical environment unobserved"], "curated_adversarial_private")

    # Group: grp_priv_socrates_whisper
    _add_rec("draft_unk_priv_009", "grp_priv_socrates_whisper", "What was Socrates whispering at the exact moment of his death that was not recorded by Plato or Xenophon?", "private_inaccessible_info",
             "Unrecorded historical oral utterance", "UNKNOWN", "Ephemeral historical vocal sound unrecorded by any source in antiquity", "inaccessibility_proof",
             ["Plato Phaedo covers final recorded words; unrecorded utterances lost to time"], "unrecorded_history")
    _add_rec("draft_unk_priv_010", "grp_priv_socrates_whisper", "What unrecorded words did Socrates murmur right before dying?", "private_inaccessible_info",
             "Paraphrase of unrecorded Socrates whisper", "UNKNOWN", "Unrecorded historical event", "inaccessibility_proof",
             ["Historical speech lost to entropy"], "unrecorded_history")

    # Group: grp_priv_caesar_thought
    _add_rec("draft_unk_priv_011", "grp_priv_caesar_thought", "What was Julius Caesar's very first private thought upon waking on the morning of March 15, 44 BC?", "private_inaccessible_info",
             "Unrecorded ancient mental cognition", "UNKNOWN", "Internal mental consciousness of historical figure unrecorded in historical documents", "inaccessibility_proof",
             ["Subjective mental state of ancient figure unrecorded"], "unrecorded_history")
    _add_rec("draft_unk_priv_012", "grp_priv_caesar_thought", "What did Julius Caesar think to himself when he first opened his eyes on the Ides of March?", "private_inaccessible_info",
             "Paraphrase of Caesar morning thought", "UNKNOWN", "Internal mental state lost to history", "inaccessibility_proof",
             ["Unrecorded mental event"], "unrecorded_history")

    # Group: grp_priv_breakfast_1842
    _add_rec("draft_unk_priv_013", "grp_priv_breakfast_1842", "What did ordinary citizen John Smith of London eat for breakfast on March 14, 1842?", "private_inaccessible_info",
             "Unrecorded mundane historical daily activity", "UNKNOWN", "Ephemeral routine daily event for non-notable individual absent from historical archives", "inaccessibility_proof",
             ["No historical documentation exists for mundane domestic meals of non-public figures"], "unrecorded_history")
    _add_rec("draft_unk_priv_014", "grp_priv_breakfast_1842", "What food was consumed for breakfast by Londoner John Smith on March 14, 1842?", "private_inaccessible_info",
             "Paraphrase of 1842 breakfast query", "UNKNOWN", "Unrecorded domestic event", "inaccessibility_proof",
             ["Unrecorded historical minutiae"], "unrecorded_history")

    # Group: grp_priv_lost_library_book
    _add_rec("draft_unk_priv_015", "grp_priv_lost_library_book", "What was the exact full text of the third scroll burned completely in the destruction of the Library of Alexandria in 48 BC?", "private_inaccessible_info",
             "Irretrievably destroyed unique ancient text", "UNKNOWN", "Physical artifact destroyed with zero surviving copies or quotations", "inaccessibility_proof",
             ["Complete physical destruction without extant transmission"], "unrecorded_history")
    _add_rec("draft_unk_priv_016", "grp_priv_lost_library_book", "What words were written on the unique Alexandrian scrolls destroyed in antiquity of which no copies survive?", "private_inaccessible_info",
             "Paraphrase of destroyed Alexandrian scroll", "UNKNOWN", "Lost ancient text", "inaccessibility_proof",
             ["Irretrievable physical destruction"], "unrecorded_history")

    # Group: grp_priv_user_wifi_password
    _add_rec("draft_unk_priv_017", "grp_priv_user_wifi_password", "What is the Wi-Fi WPA2 password for my private home router?", "private_inaccessible_info",
             "Private local network security credential", "UNKNOWN", "Local private network credential absent from public dataset", "inaccessibility_proof",
             ["Private local network credential"], "curated_adversarial_private")
    _add_rec("draft_unk_priv_018", "grp_priv_user_wifi_password", "What is the secret Wi-Fi key configured on my residential router?", "private_inaccessible_info",
             "Paraphrase of home WiFi key", "UNKNOWN", "Private local credential", "inaccessibility_proof",
             ["Private network secret"], "curated_adversarial_private")

    # Group: grp_priv_pet_weight
    _add_rec("draft_unk_priv_019", "grp_priv_pet_weight", "What is the exact weight in ounces of my pet dog right this second?", "private_inaccessible_info",
             "Private real-time uninstrumented biological measurement", "UNKNOWN", "Physical measurement in private household inaccessible to system", "inaccessibility_proof",
             ["Private household state unobserved"], "curated_adversarial_private")
    _add_rec("draft_unk_priv_020", "grp_priv_pet_weight", "How much does the user's domestic dog weigh at this moment?", "private_inaccessible_info",
             "Paraphrase of pet dog weight", "UNKNOWN", "Unobserved private measurement", "inaccessibility_proof",
             ["Unobserved physical measurement"], "curated_adversarial_private")

    # Additional private/inaccessible queries (21-35)
    # Group: grp_priv_first_words_neanderthal
    _add_rec("draft_unk_priv_021", "grp_priv_first_words_neanderthal", "What was the very first word ever vocalized by a Neanderthal individual?", "private_inaccessible_info",
             "Prehistoric unrecorded acoustic vocalization", "UNKNOWN", "Occurred tens of thousands of years before writing systems existed", "inaccessibility_proof",
             ["Prehistoric phonetic utterance occurred prior to writing"], "paleoanthropology")
    _add_rec("draft_unk_priv_022", "grp_priv_first_words_neanderthal", "What exact phonetic utterance was spoken first by a Neanderthal in prehistory?", "private_inaccessible_info",
             "Paraphrase of prehistoric vocalization", "UNKNOWN", "Prehistoric unrecorded acoustic event", "inaccessibility_proof",
             ["Pre-literate acoustic event lost"], "paleoanthropology")

    # Group: grp_priv_alexander_tomb
    _add_rec("draft_unk_priv_023", "grp_priv_alexander_tomb", "What are the exact GPS coordinates of the final resting place of Alexander the Great's body?", "private_inaccessible_info",
             "Currently undiscovered ancient archaeological site", "UNKNOWN", "Location of tomb of Alexander the Great is an unsolved archaeological mystery", "inaccessibility_proof",
             ["Archaeological consensus confirms tomb location remains unverified and lost"], "archaeological_mystery",
             audit_trail={"borderline_review": "Retained as UNKNOWN due to lost historical location of Alexander's tomb, noting modern GPS coordinate framing."})
    _add_rec("draft_unk_priv_024", "grp_priv_alexander_tomb", "Where precisely is the buried body of Alexander the Great located on Earth today?", "private_inaccessible_info",
             "Paraphrase of Alexander the Great tomb location", "UNKNOWN", "Unsolved archaeological burial site", "inaccessibility_proof",
             ["Undiscovered archaeological tomb"], "archaeological_mystery")

    # Group: grp_priv_shakespeares_notes
    _add_rec("draft_unk_priv_025", "grp_priv_shakespeares_notes", "What doodles or margin notes did William Shakespeare scribble on his personal first draft of Hamlet?", "private_inaccessible_info",
             "Lost original literary manuscript", "UNKNOWN", "No manuscript drafts of Hamlet survive; only early print quartos exist", "inaccessibility_proof",
             ["Original authorial manuscripts for Hamlet are completely lost to history"], "literary_history")
    _add_rec("draft_unk_priv_026", "grp_priv_shakespeares_notes", "What notes were written in the original lost draft of Shakespeare's Hamlet?", "private_inaccessible_info",
             "Paraphrase of Shakespeare lost draft", "UNKNOWN", "Lost original draft", "inaccessibility_proof",
             ["Authorial manuscript does not survive"], "literary_history")

    # Group: grp_priv_user_dinner
    _add_rec("draft_unk_priv_027", "grp_priv_user_dinner", "What did the author of this prompt eat for dinner three days ago?", "private_inaccessible_info",
             "Private personal history of user", "UNKNOWN", "Private personal domestic history unshared in prompt context", "inaccessibility_proof",
             ["User personal daily history private"], "curated_adversarial_private")
    _add_rec("draft_unk_priv_028", "grp_priv_user_dinner", "What meal did the user consume three nights ago?", "private_inaccessible_info",
             "Paraphrase of user dinner query", "UNKNOWN", "Private unrecorded personal routine", "inaccessibility_proof",
             ["Private user activity"], "curated_adversarial_private")

    # Group: grp_priv_cleopatra_last_words
    _add_rec("draft_unk_priv_029", "grp_priv_cleopatra_last_words", "What was Cleopatra's exact last unspoken private thought before she died in 30 BC?", "private_inaccessible_info",
             "Unspoken ancient private mental thought", "UNKNOWN", "Internal consciousness of Cleopatra never externalized or recorded", "inaccessibility_proof",
             ["Internal subjective consciousness is unrecorded in antiquity"], "unrecorded_history")
    _add_rec("draft_unk_priv_030", "grp_priv_cleopatra_last_words", "What was Cleopatra silently thinking to herself in her final moments of life?", "private_inaccessible_info",
             "Paraphrase of Cleopatra final thought", "UNKNOWN", "Unspoken private mental thought", "inaccessibility_proof",
             ["Unrecorded mental cognition"], "unrecorded_history")

    # Group: grp_priv_unrecorded_shipwreck
    _add_rec("draft_unk_priv_031", "grp_priv_unrecorded_shipwreck", "What was the name of the captain of the unrecorded merchant fishing boat that sank in the North Sea in 1112 AD with no survivors?", "private_inaccessible_info",
             "Unrecorded historical catastrophe without records", "UNKNOWN", "No historical chronicles or passenger manifests were kept or survived", "inaccessibility_proof",
             ["Zero surviving archival records for unrecorded medieval maritime event"], "unrecorded_history")
    _add_rec("draft_unk_priv_032", "grp_priv_unrecorded_shipwreck", "Who captained the unknown fishing vessel that sank without record in the North Sea in 1112?", "private_inaccessible_info",
             "Paraphrase of unrecorded shipwreck captain", "UNKNOWN", "Unrecorded historical shipwreck", "inaccessibility_proof",
             ["Unrecorded medieval event"], "unrecorded_history")

    # Group: grp_priv_user_bank_balance
    _add_rec("draft_unk_priv_033", "grp_priv_user_bank_balance", "What is the exact current checking account balance in cents of the user typing this message?", "private_inaccessible_info",
             "Confidential real-time financial ledger data", "UNKNOWN", "Private personal financial ledger inaccessible to LLM", "inaccessibility_proof",
             ["Private personal banking records are confidential and inaccessible"], "curated_adversarial_private")
    _add_rec("draft_unk_priv_034", "grp_priv_user_bank_balance", "How much money in cents is in my personal checking bank account right now?", "private_inaccessible_info",
             "Paraphrase of user bank balance query", "UNKNOWN", "Confidential financial ledger", "inaccessibility_proof",
             ["Private banking records"], "curated_adversarial_private")

    # Group: grp_priv_lost_colony_roanoke
    _add_rec("draft_unk_priv_035", "grp_priv_lost_colony_roanoke", "What was the exact day-by-day fate of Virginia Dare after the abandonment of the Roanoke Colony in 1587?", "private_inaccessible_info",
             "Unsolved historical disappearance", "UNKNOWN", "Roanoke Lost Colony disappearance has no surviving primary historical records", "inaccessibility_proof",
             ["Historical consensus confirms fate of Roanoke colonists is unrecorded and unknown"], "unsolved_history")


    # =========================================================================
    # CATEGORY 6: Unknowable Future Events (Quota: 35, Target: UNKNOWN)
    # =========================================================================
    # Group: grp_fut_apple_stock_2035
    _add_rec("draft_unk_fut_001", "grp_fut_apple_stock_2035", "What will be the exact closing price of Apple Inc. (AAPL) common stock on December 31, 2035?", "unknowable_future_events",
             "Distant future financial market price", "UNKNOWN", "Future economic pricing is inherently stochastic and undetermined", "future_contingency",
             ["Efficient market hypothesis and stochastic financial dynamics; future market state undetermined"], "financial_future_contingency")
    _add_rec("draft_unk_fut_002", "grp_fut_apple_stock_2035", "At what exact price will Apple stock close on the final trading day of 2035?", "unknowable_future_events",
             "Paraphrase of future Apple stock price", "UNKNOWN", "Future financial contingency", "future_contingency",
             ["Stochastic financial markets"], "financial_future_contingency")

    # Group: grp_fut_weather_tokyo_2095
    _add_rec("draft_unk_fut_003", "grp_fut_weather_tokyo_2095", "What will be the exact atmospheric temperature in Tokyo at 12:00:00 UTC on January 1, 2095?", "unknowable_future_events",
             "Decadal chaotic weather prediction", "UNKNOWN", "Dynamical atmospheric chaos limits deterministic prediction beyond ~14 days", "future_contingency",
             ["Lorenz (1963) chaos theory; Lyapunov time for atmospheric weather is ~14 days"], "meteorological_chaos")
    _add_rec("draft_unk_fut_004", "grp_fut_weather_tokyo_2095", "What exact temperature in degrees Celsius will Tokyo experience at noon on January 1, 2095?", "unknowable_future_events",
             "Paraphrase of future Tokyo temperature", "UNKNOWN", "Dynamical chaos precludes decadal microstate weather prediction", "future_contingency",
             ["Lorenz chaos limit"], "meteorological_chaos")

    # Group: grp_fut_worldcup_2062
    _add_rec("draft_unk_fut_005", "grp_fut_worldcup_2062", "Which national men's team will win the FIFA World Cup tournament in 2062?", "unknowable_future_events",
             "Future athletic sports competition outcome", "UNKNOWN", "Event has not occurred; athletes, rules, and qualifiers are undetermined", "future_contingency",
             ["Future sporting event involving humans not yet born"], "sports_future_contingency")
    _add_rec("draft_unk_fut_006", "grp_fut_worldcup_2062", "Who will be crowned the champions of the 2062 FIFA Men's World Cup?", "unknowable_future_events",
             "Paraphrase of 2062 World Cup champion", "UNKNOWN", "Future athletic competition", "future_contingency",
             ["Future unheld sporting tournament"], "sports_future_contingency")

    # Group: grp_fut_powerball_next
    _add_rec("draft_unk_fut_007", "grp_fut_powerball_next", "What will be the exact winning numbers in the next official US Powerball lottery drawing?", "unknowable_future_events",
             "Unheld physical pseudo-random lottery drawing", "UNKNOWN", "Random physical entropy generator; outcome is undetermined prior to drawing", "future_contingency",
             ["Mechanical ball drawing is a stochastic process governed by physical entropy"], "stochastic_lottery")
    _add_rec("draft_unk_fut_008", "grp_fut_powerball_next", "Which set of numbers will be drawn as the winning combination in next week's Powerball lottery?", "unknowable_future_events",
             "Paraphrase of future Powerball drawing", "UNKNOWN", "Stochastic physical drawing", "future_contingency",
             ["Stochastic mechanical drawing"], "stochastic_lottery")

    # Group: grp_fut_presidential_election_2052
    _add_rec("draft_unk_fut_009", "grp_fut_presidential_election_2052", "Who will be elected President of the United States in the November 2052 general election?", "unknowable_future_events",
             "Future democratic election outcome", "UNKNOWN", "Voters, candidates, and economic conditions decades in the future are undetermined", "future_contingency",
             ["Democratic elections are contingent on future societal decisions and candidates"], "political_future_contingency")
    _add_rec("draft_unk_fut_010", "grp_fut_presidential_election_2052", "Which individual will win the United States presidential election of 2052?", "unknowable_future_events",
             "Paraphrase of 2052 US presidential election", "UNKNOWN", "Future political event", "future_contingency",
             ["Future democratic voting outcome"], "political_future_contingency")

    # Group: grp_fut_first_mars_baby
    _add_rec("draft_unk_fut_011", "grp_fut_first_mars_baby", "What will be the full legal name of the first human baby born on the planet Mars?", "unknowable_future_events",
             "Future human demographic milestone", "UNKNOWN", "Human settlement on Mars has not occurred; no parents or births exist", "future_contingency",
             ["Contingent on future space colonization and parental naming choices"], "demographic_future_contingency")
    _add_rec("draft_unk_fut_012", "grp_fut_first_mars_baby", "What will parents name the initial child to be born on Mars?", "unknowable_future_events",
             "Paraphrase of first Mars baby name", "UNKNOWN", "Future human milestone", "future_contingency",
             ["Future demographic event"], "demographic_future_contingency")

    # Group: grp_fut_next_earthquake_sf
    _add_rec("draft_unk_fut_013", "grp_fut_next_earthquake_sf", "On what exact calendar date and second will the next magnitude 7.0 or greater earthquake strike the San Francisco Bay Area?", "unknowable_future_events",
             "Exact seismic event timing", "UNKNOWN", "Geophysical fault dynamics do not permit deterministic exact-time earthquake prediction", "future_contingency",
             ["USGS statement on earthquake prediction; exact timing is physically impossible to predict with current science"], "geophysical_future_contingency")
    _add_rec("draft_unk_fut_014", "grp_fut_next_earthquake_sf", "What is the exact timestamp for the next major magnitude 7+ earthquake in San Francisco?", "unknowable_future_events",
             "Paraphrase of earthquake timestamp query", "UNKNOWN", "Nonlinear seismic slip dynamics preclude exact timing", "future_contingency",
             ["USGS Earthquake Hazards Program"], "geophysical_future_contingency")

    # Group: grp_fut_superbowl_2045
    _add_rec("draft_unk_fut_015", "grp_fut_superbowl_2045", "What will be the final game score of Super Bowl LXXIX (79) in the NFL?", "unknowable_future_events",
             "Future athletic sports competition score", "UNKNOWN", "Unplayed athletic contest decades in the future", "future_contingency",
             ["Future athletic competition"], "sports_future_contingency")
    _add_rec("draft_unk_fut_016", "grp_fut_superbowl_2045", "What will the scoreboard read at the conclusion of NFL Super Bowl 79?", "unknowable_future_events",
             "Paraphrase of future Super Bowl score", "UNKNOWN", "Unplayed sports match", "future_contingency",
             ["Unplayed sporting match"], "sports_future_contingency")

    # Group: grp_fut_first_fusion_commercial
    _add_rec("draft_unk_fut_017", "grp_fut_first_fusion_commercial", "On what exact day will the world's first commercial nuclear fusion power plant begin supplying electricity to a public utility grid?", "unknowable_future_events",
             "Technological commercialization milestone", "UNKNOWN", "Scientific and engineering timeline is contingent on unresolved technological breakthroughs", "future_contingency",
             ["Commercial fusion deployment depends on unresolved plasma stability and materials engineering milestones"], "technological_future_contingency")
    _add_rec("draft_unk_fut_018", "grp_fut_first_fusion_commercial", "What calendar date will mark the grid connection of the first commercial nuclear fusion plant?", "unknowable_future_events",
             "Paraphrase of commercial fusion date", "UNKNOWN", "Unresolved technological milestone", "future_contingency",
             ["Unresolved technological engineering timeline"], "technological_future_contingency")

    # Group: grp_fut_bitcoin_2040
    _add_rec("draft_unk_fut_019", "grp_fut_bitcoin_2040", "What will be the market price in US dollars of one Bitcoin at midnight UTC on January 1, 2040?", "unknowable_future_events",
             "Future cryptocurrency market price", "UNKNOWN", "Stochastic speculative asset pricing decades into the future", "future_contingency",
             ["Cryptocurrency pricing is non-deterministic and subject to unknown regulatory, technological, and market developments"], "financial_future_contingency")
    _add_rec("draft_unk_fut_020", "grp_fut_bitcoin_2040", "How much will one Bitcoin be worth in US dollars on January 1, 2040?", "unknowable_future_events",
             "Paraphrase of future Bitcoin price", "UNKNOWN", "Stochastic future asset pricing", "future_contingency",
             ["Stochastic financial asset pricing"], "financial_future_contingency")

    # Additional future queries (21-35)
    # Group: grp_fut_nobel_physics_2042
    _add_rec("draft_unk_fut_021", "grp_fut_nobel_physics_2042", "As of November 2026, who will be awarded the Nobel Prize in Physics in the year 2042?", "unknowable_future_events",
             "Future scientific award decision", "UNKNOWN", "Future Royal Swedish Academy of Sciences committee decision for work not yet selected or conducted", "future_contingency",
             ["Future institutional committee evaluation"], "academic_future_contingency", temporal_anchor="2026-11",
             audit_trail={"temporal_anchor_update": "Updated temporal anchor reference from October 2026 to November 2026."})
    _add_rec("draft_unk_fut_022", "grp_fut_nobel_physics_2042", "As of November 2026, which scientist will receive the Nobel Prize for Physics in 2042?", "unknowable_future_events",
             "Paraphrase of 2042 Nobel Prize in Physics", "UNKNOWN", "Future committee selection", "future_contingency",
             ["Future scientific award selection"], "academic_future_contingency", temporal_anchor="2026-11",
             audit_trail={"temporal_anchor_update": "Updated temporal anchor reference from October 2026 to November 2026."})

    # Group: grp_fut_next_pope
    _add_rec("draft_unk_fut_023", "grp_fut_next_pope", "As of November 2026, what papal name will be chosen by the person elected as the successor to Pope Leo XIV?", "unknowable_future_events",
             "Secret conclave election of successor to Pope Leo XIV (contingent future selection)", "UNKNOWN",
             "The selection of a papal regnal name is determined during a secret papal conclave by the newly elected pontiff; as of November 2026, the election of Pope Leo XIV's successor has not occurred and cannot be determined", "future_contingency",
             ["Secret papal conclave procedures (Universi Dominici Gregis); future election undetermined as of November 2026"], "religious_future_contingency", temporal_anchor="2026-11",
             audit_trail={"original_question": "What papal name will be chosen by the person elected as the successor to Pope Francis?", "modification_reason": "Pope Francis was succeeded by Pope Leo XIV on May 8, 2025. Target shifted to successor to Pope Leo XIV to maintain future event validity as of November 2026."})
    _add_rec("draft_unk_fut_024", "grp_fut_next_pope", "As of November 2026, what regnal name will the next elected Pope of the Roman Catholic Church adopt upon the conclusion of Pope Leo XIV's papacy?", "unknowable_future_events",
             "Paraphrase of successor to Pope Leo XIV regnal name", "UNKNOWN", "Future personal choice in future secret conclave after Pope Leo XIV", "future_contingency",
             ["Papal succession law; future conclave unheld as of November 2026"], "religious_future_contingency", temporal_anchor="2026-11",
             audit_trail={"original_question": "What regnal name will the next elected Pope of the Roman Catholic Church adopt?", "modification_reason": "Pope Francis was succeeded by Pope Leo XIV on May 8, 2025. Target shifted to successor to Pope Leo XIV to maintain future event validity as of November 2026."})

    # Group: grp_fut_world_population_2100
    _add_rec("draft_unk_fut_025", "grp_fut_world_population_2100", "What will be the exact human population count on Earth at midnight UTC on December 31, 2100?", "unknowable_future_events",
             "Exact global demographic count at 2100", "UNKNOWN", "Demographic trajectories are subject to fertility, mortality, and climate variance", "future_contingency",
             ["UN World Population Prospects projections contain wide probabilistic confidence intervals (8.9B - 12.4B)"], "demographic_future_contingency")
    _add_rec("draft_unk_fut_026", "grp_fut_world_population_2100", "What will be the precise number of living human beings on Earth at the end of the year 2100?", "unknowable_future_events",
             "Paraphrase of 2100 world population", "UNKNOWN", "Demographic projection variance", "future_contingency",
             ["UN Population Division probabilistic models"], "demographic_future_contingency")

    # Group: grp_fut_summer_olympics_2048
    _add_rec("draft_unk_fut_027", "grp_fut_summer_olympics_2048", "As of November 2026, which city will host the Games of the XXXIX Olympiad (2048 Summer Olympics)?", "unknowable_future_events",
             "Future IOC host city election", "UNKNOWN", "International Olympic Committee vote has not occurred and bidding has not opened", "future_contingency",
             ["IOC Future Host Commission has not voted or selected 2048 host as of November 2026"], "sports_future_contingency", temporal_anchor="2026-11",
             audit_trail={"temporal_anchor_update": "Updated temporal anchor reference from October 2026 to November 2026."})
    _add_rec("draft_unk_fut_028", "grp_fut_summer_olympics_2048", "As of November 2026, where will the 2048 Summer Olympic Games take place?", "unknowable_future_events",
             "Paraphrase of 2048 Olympic host city", "UNKNOWN", "Future international vote", "future_contingency",
             ["IOC host selection vote unheld as of November 2026"], "sports_future_contingency", temporal_anchor="2026-11",
             audit_trail={"temporal_anchor_update": "Updated temporal anchor reference from October 2026 to November 2026."})

    # Group: grp_fut_next_solar_storm
    _add_rec("draft_unk_fut_029", "grp_fut_next_solar_storm", "On what exact second will the next G5-level extreme geomagnetic solar storm impact Earth's magnetosphere?", "unknowable_future_events",
             "Exact solar flare / CME impact timing", "UNKNOWN", "Solar magnetohydrodynamics does not permit exact timing prediction years in advance", "future_contingency",
             ["NOAA Space Weather Prediction Center models limit CME arrival time prediction to 1-3 days prior"], "astrophysical_future_contingency")
    _add_rec("draft_unk_fut_030", "grp_fut_next_solar_storm", "What is the exact second when Earth will experience its next extreme G5 geomagnetic storm?", "unknowable_future_events",
             "Paraphrase of next G5 solar storm timestamp", "UNKNOWN", "Solar flare timing is non-deterministic far in advance", "future_contingency",
             ["Space weather forecasting limits"], "astrophysical_future_contingency")

    # Group: grp_fut_first_mammoth_birth
    _add_rec("draft_unk_fut_031", "grp_fut_first_mammoth_birth", "On what exact date will the first de-extinction woolly mammoth calf be born alive?", "unknowable_future_events",
             "Genetic de-extinction biological milestone", "UNKNOWN", "Genetic editing and gestation technologies remain in experimental research stages", "future_contingency",
             ["De-extinction biotechnology timelines are speculative and contingent on live birth success"], "biotechnological_future_contingency")
    _add_rec("draft_unk_fut_032", "grp_fut_first_mammoth_birth", "When exactly will biotechnology produce the first live de-extinct woolly mammoth?", "unknowable_future_events",
             "Paraphrase of woolly mammoth birth date", "UNKNOWN", "Experimental biotechnology milestone", "future_contingency",
             ["Biotechnological experimental timeline"], "biotechnological_future_contingency")

    # Group: grp_fut_first_ai_country_president
    _add_rec("draft_unk_fut_033", "grp_fut_first_ai_country_president", "What will be the name of the first sovereign nation to elect an artificial intelligence as head of state?", "unknowable_future_events",
             "Speculative political milestone", "UNKNOWN", "Speculative future governance milestone that may never occur", "future_contingency",
             ["Speculative political scenario with no current existence or legal framework"], "speculative_future_contingency")
    _add_rec("draft_unk_fut_034", "grp_fut_first_ai_country_president", "Which country will be the first in history to place an AI in the office of president?", "unknowable_future_events",
             "Paraphrase of AI head of state query", "UNKNOWN", "Speculative future governance scenario", "future_contingency",
             ["Speculative future political scenario"], "speculative_future_contingency")

    # Group: grp_fut_betelgeuse_supernova
    _add_rec("draft_unk_fut_035", "grp_fut_betelgeuse_supernova", "On what exact calendar year will the star Betelgeuse undergo a core-collapse supernova visible from Earth?", "unknowable_future_events",
             "Exact stellar astrophysical collapse timing", "UNKNOWN", "Astrophysical models predict Betelgeuse will collapse within the next 100,000 years, but exact year is unknowable", "future_contingency",
             ["Astrophysical stellar evolution models span an uncertainty window of ~100,000 years for Betelgeuse"], "astrophysical_future_contingency")


    # =========================================================================
    # CATEGORY 7: False Premises & Presupposition Failures (Quota: 30, Target: UNKNOWN)
    # =========================================================================
    # Group: grp_false_king_france_2024
    _add_rec("draft_unk_false_001", "grp_false_king_france_2024", "What was the name of the King of France in 2024?", "false_premises",
             "Presupposition failure: France has a monarch in 2024", "UNKNOWN", "France is a constitutional republic; no French monarch exists in 2024", "presupposition_failure",
             ["Constitution of the French Fifth Republic (1958); monarchy abolished in 1792 and 1848"], "curated_presupposition_failure")
    _add_rec("draft_unk_false_002", "grp_false_king_france_2024", "Who reigned as the monarch of France in the year 2024?", "false_premises",
             "Paraphrase of 2024 King of France query", "UNKNOWN", "Presupposition failure (France is a republic)", "presupposition_failure",
             ["French Fifth Republic constitution"], "curated_presupposition_failure")

    # Group: grp_false_us_president_1750
    _add_rec("draft_unk_false_003", "grp_false_us_president_1750", "Who was the President of the United States in 1750?", "false_premises",
             "Presupposition failure: United States existed as a sovereign nation in 1750", "UNKNOWN", "The United States did not exist as an independent country until 1776, and the presidency was established in 1789", "presupposition_failure",
             ["US Declaration of Independence (1776); US Constitution ratified 1788; Washington inaugurated 1789"], "historical_presupposition_failure")
    _add_rec("draft_unk_false_004", "grp_false_us_president_1750", "Which individual served as American president during the year 1750?", "false_premises",
             "Paraphrase of 1750 US president query", "UNKNOWN", "Presupposition failure (US presidency did not exist in 1750)", "presupposition_failure",
             ["US Constitution ratified 1788"], "historical_presupposition_failure")

    # Group: grp_false_dry_water_melting
    _add_rec("draft_unk_false_005", "grp_false_dry_water_melting", "What is the melting point of dry water in degrees Celsius?", "false_premises",
             "Category error / false premise: 'dry water' is a colloidal emulsion, not a phase of pure H2O with a melting point", "UNKNOWN", "Dry water is a silica-coated liquid emulsion, not a crystalline solid with a melting temperature", "presupposition_failure",
             ["Chemistry definitions of colloidal silica emulsions; dry water does not have a single thermodynamic melting point"], "chemical_presupposition_failure")
    _add_rec("draft_unk_false_006", "grp_false_dry_water_melting", "At what temperature in Celsius does dry water melt?", "false_premises",
             "Paraphrase of dry water melting point", "UNKNOWN", "Category error and false presupposition", "presupposition_failure",
             ["Colloidal chemistry properties"], "chemical_presupposition_failure")

    # Group: grp_false_mars_third_moon
    _add_rec("draft_unk_false_007", "grp_false_mars_third_moon", "What is the diameter of the third natural moon of Mars?", "false_premises",
             "Presupposition failure: Mars has a third natural satellite", "UNKNOWN", "Mars has exactly two natural satellites (Phobos and Deimos); no third natural moon exists", "presupposition_failure",
             ["NASA Solar System Exploration: Mars has only 2 known natural satellites (Phobos and Deimos)"], "astronomical_presupposition_failure")
    _add_rec("draft_unk_false_008", "grp_false_mars_third_moon", "How wide in kilometers is the third moon orbiting Mars?", "false_premises",
             "Paraphrase of third Martian moon diameter", "UNKNOWN", "Presupposition failure (Mars has only two natural satellites)", "presupposition_failure",
             ["NASA Mars Exploration"], "astronomical_presupposition_failure")

    # Group: grp_false_capital_atlantis
    _add_rec("draft_unk_false_009", "grp_false_capital_atlantis", "What is the official capital city of the continent of Atlantis?", "false_premises",
             "Presupposition failure: Atlantis is a real geographical continent", "UNKNOWN", "Atlantis is an allegorical myth created by Plato in Timaeus and Critias, not a real physical continent", "presupposition_failure",
             ["Plato Timaeus and Critias; geological consensus confirms no continent of Atlantis exists"], "mythological_presupposition_failure")
    _add_rec("draft_unk_false_010", "grp_false_capital_atlantis", "Which city serves as the capital of the real-world continent of Atlantis?", "false_premises",
             "Paraphrase of capital of Atlantis query", "UNKNOWN", "Presupposition failure (Atlantis is a mythical allegory)", "presupposition_failure",
             ["Geological plate tectonics"], "mythological_presupposition_failure")

    # Group: grp_false_female_us_president_1900
    _add_rec("draft_unk_false_011", "grp_false_female_us_president_1900", "What was the name of the female US President who served in 1900?", "false_premises",
             "Presupposition failure: A female president served the US in 1900", "UNKNOWN", "William McKinley was US President in 1900; no woman has served as US President in American history through 2024", "presupposition_failure",
             ["US Presidential Historical Archives; William McKinley served 1897-1901"], "historical_presupposition_failure")
    _add_rec("draft_unk_false_012", "grp_false_female_us_president_1900", "Who was the woman who held the American presidency during the year 1900?", "false_premises",
             "Paraphrase of 1900 female US president query", "UNKNOWN", "Presupposition failure (no female US president in 1900)", "presupposition_failure",
             ["US Presidential Records"], "historical_presupposition_failure")

    # Group: grp_false_square_circle_corners
    _add_rec("draft_unk_false_013", "grp_false_square_circle_corners", "What is the interior angle measurement in degrees of each corner of a standard geometric round circle?", "false_premises",
             "Geometrical contradiction / category error: Euclidean circles have no vertices or corners", "UNKNOWN", "A circle by Euclidean definition is a continuous locus of points equidistant from a center with zero vertices or corners; interior corner angles cannot be measured", "presupposition_failure",
             ["Euclid Elements Book I Definition 15 (Circle)"], "mathematical_presupposition_failure",
             audit_trail={"original_question": "How many sharp 90-degree corners does a perfect geometric round circle have?", "modification_reason": "Eliminated numeric '0' counting loophole by rephrasing to angle measurement of nonexistent vertices."})
    _add_rec("draft_unk_false_014", "grp_false_square_circle_corners", "What is the degree measurement of the right-angled vertices forming the corners of a geometric circle?", "false_premises",
             "Paraphrase of circle corner angle measurement", "UNKNOWN", "Geometrical contradiction / presupposition failure", "presupposition_failure",
             ["Euclid Book I Def 15"], "mathematical_presupposition_failure",
             audit_trail={"original_question": "What is the count of 90-degree right angles in a standard geometric circle?", "modification_reason": "Eliminated numeric '0' counting loophole by rephrasing to degree measurement of vertices."})

    # Group: grp_false_antarctica_rainforest
    _add_rec("draft_unk_false_015", "grp_false_antarctica_rainforest", "What species of monkey lives natively in the modern tropical rainforests of Antarctica?", "false_premises",
             "Presupposition failure: Antarctica contains modern tropical rainforests and native monkeys", "UNKNOWN", "Modern Antarctica is an ice-covered polar desert with zero native primate species or tropical rainforests", "presupposition_failure",
             ["British Antarctic Survey biodiversity flora and fauna reports"], "biological_presupposition_failure")
    _add_rec("draft_unk_false_016", "grp_false_antarctica_rainforest", "Which native monkey inhabitant resides in the current rainforests on Antarctica?", "false_premises",
             "Paraphrase of Antarctic rainforest monkey", "UNKNOWN", "Presupposition failure (no modern rainforests or native monkeys on Antarctica)", "presupposition_failure",
             ["British Antarctic Survey"], "biological_presupposition_failure")

    # Group: grp_false_nobel_mathematics
    _add_rec("draft_unk_false_017", "grp_false_nobel_mathematics", "Who won the official Nobel Prize in Mathematics in 2020?", "false_premises",
             "Presupposition failure: An official Nobel Prize in Mathematics exists", "UNKNOWN", "No Nobel Prize in Mathematics was established by Alfred Nobel's will (Fields Medal and Abel Prize exist instead)", "presupposition_failure",
             ["Alfred Nobel's Last Will and Testament (1895); Nobel Foundation Statutes"], "institutional_presupposition_failure")
    _add_rec("draft_unk_false_018", "grp_false_nobel_mathematics", "Which mathematician was awarded the Nobel Prize for Mathematics in 2020?", "false_premises",
             "Paraphrase of Nobel Prize in Mathematics query", "UNKNOWN", "Presupposition failure (no Nobel Prize exists for mathematics)", "presupposition_failure",
             ["Nobel Foundation Statutes"], "institutional_presupposition_failure")

    # Group: grp_false_atomic_unobtainium
    _add_rec("draft_unk_false_019", "grp_false_atomic_unobtainium", "What is the atomic number of the real-world chemical element Unobtainium in the periodic table?", "false_premises",
             "Presupposition failure: Unobtainium is an actual chemical element on the periodic table", "UNKNOWN", "Unobtainium is a fictional / rhetorical placeholder trope in engineering and sci-fi (e.g. Avatar), not a chemical element", "presupposition_failure",
             ["IUPAC Periodic Table of Elements contains no element named Unobtainium"], "chemical_presupposition_failure")
    _add_rec("draft_unk_false_020", "grp_false_atomic_unobtainium", "Which atomic number corresponds to the periodic table element Unobtainium?", "false_premises",
             "Paraphrase of Unobtainium atomic number", "UNKNOWN", "Presupposition failure (fictional material)", "presupposition_failure",
             ["IUPAC Periodic Table"], "chemical_presupposition_failure")

    # Group: grp_false_california_border_atlantic
    _add_rec("draft_unk_false_021", "grp_false_california_border_atlantic", "What is the name of the commercial deepwater seaport in California that opens directly onto the Atlantic Ocean?", "false_premises",
             "Geographical impossibility / false premise", "UNKNOWN", "California is on the western coast of the US bordering the Pacific Ocean; it possesses no deepwater seaports bordering the Atlantic Ocean", "presupposition_failure",
             ["US Geological Survey; National Oceanic and Atmospheric Administration coastline surveys"], "geographic_presupposition_failure",
             audit_trail={"original_question": "How many miles of coastline does the state of California share with the Atlantic Ocean?", "modification_reason": "Eliminated numeric '0 miles' loophole by rephrasing to commercial Atlantic seaport query."})
    _add_rec("draft_unk_false_022", "grp_false_california_border_atlantic", "Which Atlantic maritime port city is located within the state of California?", "false_premises",
             "Paraphrase of California Atlantic seaport query", "UNKNOWN", "Geographical impossibility / presupposition failure", "presupposition_failure",
             ["USGS Geography Maps"], "geographic_presupposition_failure",
             audit_trail={"original_question": "What is the length of California's coastline along the Atlantic Ocean?", "modification_reason": "Eliminated numeric '0 miles' loophole by rephrasing to Atlantic maritime port city query."})

    # Group: grp_false_steam_locomotive_rome
    _add_rec("draft_unk_false_023", "grp_false_steam_locomotive_rome", "What fuel was used to power the commercial steam locomotive train network in Ancient Rome in 100 AD?", "false_premises",
             "Anachronism / false premise: Ancient Rome had a commercial steam locomotive network", "UNKNOWN", "Commercial steam locomotives were invented in the 19th century; Ancient Rome had no railroad or steam train network", "presupposition_failure",
             ["Historical consensus; Trevithick built first full-scale working railway steam locomotive in 1804"], "historical_anachronism_presupposition_failure")
    _add_rec("draft_unk_false_024", "grp_false_steam_locomotive_rome", "Which fuel source ran the railway steam train system of the Roman Empire in 100 AD?", "false_premises",
             "Paraphrase of Roman steam locomotive fuel", "UNKNOWN", "Anachronistic false premise", "presupposition_failure",
             ["History of steam transport (1804)"], "historical_anachronism_presupposition_failure")

    # Group: grp_false_third_sex_human_chromosomes
    _add_rec("draft_unk_false_025", "grp_false_third_sex_human_chromosomes", "What letters designate the third standard sex chromosome in humans beyond X and Y?", "false_premises",
             "Biological false premise: humans possess a third standard sex chromosome", "UNKNOWN", "Human standard chromosomal sex determination relies strictly on the X and Y gonosomes; no third standard sex chromosome exists", "presupposition_failure",
             ["Human Genome Project; ISCN International System for Human Cytogenomic Nomenclature"], "biological_presupposition_failure")
    _add_rec("draft_unk_false_026", "grp_false_third_sex_human_chromosomes", "Which letter denotes the third typical human sex chromosome in standard cytogenetics?", "false_premises",
             "Paraphrase of third human sex chromosome", "UNKNOWN", "Biological false premise (only X and Y exist)", "presupposition_failure",
             ["ISCN Cytogenomics Guide"], "biological_presupposition_failure")

    # Group: grp_false_eighth_planet_named_pluto
    _add_rec("draft_unk_false_027", "grp_false_eighth_planet_named_pluto", "What is the orbital speed of the eighth planet from the Sun, Pluto?", "false_premises",
             "Astronomical false premise: Pluto is the eighth planet from the Sun", "UNKNOWN", "Neptune is the eighth planet from the Sun; Pluto is classified as a dwarf planet located in the Kuiper belt beyond Neptune", "presupposition_failure",
             ["IAU 2006 Resolution B5 (Definition of a Planet in the Solar System)"], "astronomical_presupposition_failure")
    _add_rec("draft_unk_false_028", "grp_false_eighth_planet_named_pluto", "How fast does Pluto orbit as the eighth major planet in our solar system?", "false_premises",
             "Paraphrase of Pluto as eighth planet query", "UNKNOWN", "Astronomical false premise (Neptune is eighth; Pluto is dwarf planet)", "presupposition_failure",
             ["IAU 2006 Resolution B5"], "astronomical_presupposition_failure")

    # Group: grp_adv_vacuum_wave_speed (adversarial cross-label group with speed of light)
    _add_rec("draft_unk_false_029", "grp_adv_vacuum_wave_speed", "What is the speed of sound through a complete vacuum in meters per second?", "false_premises",
             "Physical impossibility: Sound propagation in a vacuum", "UNKNOWN", "Sound is a mechanical pressure wave requiring a material medium; it cannot propagate through a vacuum (speed is nonexistent, not a valid value)", "presupposition_failure",
             ["Physics principles of acoustic wave propagation; mechanical waves require molecular medium"], "physical_presupposition_failure", is_adversarial=True)
    _add_rec("draft_unk_false_030", "grp_adv_vacuum_wave_speed", "At how many meters per second does an acoustic sound wave travel through an absolute vacuum?", "false_premises",
             "Paraphrase of sound speed in vacuum", "UNKNOWN", "Physical impossibility / false premise", "presupposition_failure",
             ["Fundamentals of Physics Acoustic Principles"], "physical_presupposition_failure", is_adversarial=True)

    return records


def main(output_filename: str = "uncertainty_benchmark_300_candidate_v1.1.jsonl"):
    import sys
    if len(sys.argv) > 1:
        output_filename = sys.argv[1]

    records = build_draft_benchmark()
    output_dir = "data/benchmarks"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    # Protect draft from accidental overwrite if default is used
    if output_filename == "uncertainty_benchmark_300_draft.jsonl":
        raise ValueError("Overwriting uncertainty_benchmark_300_draft.jsonl is strictly protected and disallowed.")

    # Write output to new candidate file
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Generated {len(records)} candidate records written to {output_path}")

    # Validate output using BenchmarkDraftValidator
    report = BenchmarkDraftValidator.validate_records(records)
    print("\n--- Validation Report Summary ---")
    print(f"Is Valid: {report.is_valid}")
    print(f"Total Records: {report.total_records}")
    print(f"Label Counts: {report.label_counts}")
    print(f"Category Counts: {report.category_counts}")
    print(f"Group Count: {report.group_count}")
    print(f"Duplicate Questions: {len(report.duplicate_questions)}")
    print(f"Missing Evidence Count: {report.missing_evidence_count}")
    print(f"Unverified Candidates Count: {report.unverified_candidate_count}")
    if report.schema_errors:
        print(f"Errors ({len(report.schema_errors)}):")
        for err in report.schema_errors[:10]:
            print(f"  - {err}")


if __name__ == "__main__":
    main()
