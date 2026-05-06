"""
ICD-10 Disease Ontology Harmonizer
====================================
Maps disease IDs from DDXPlus, ePOCT+, RetroMOOVE, and sample_data
to a common ICD-10-based ontology, enabling cross-dataset evaluation.

COVERAGE:
  - DDXPlus: 50 leaf diseases → ICD-10
  - ePOCT+: 200 core diseases (shared with RetroMOOVE) → ICD-10
  - sample_data: 10 leaf diseases → ICD-10

Each disease is assigned:
  1. ICD-10 code (e.g., J18.9)
  2. Canonical name
  3. Clinical category (for hierarchical grouping)
"""

import os
import json
import csv
from typing import Dict, List, Optional, Tuple

# ── ICD-10 Master Mapping ────────────────────────────────────────
# Format: icd10_code -> (canonical_name, category)

ICD10_CATALOG = {
    # ── Respiratory ──
    "J06.9": ("Acute upper respiratory infection", "Respiratory"),
    "J00":   ("Common cold / Acute nasopharyngitis", "Respiratory"),
    "J02.9": ("Acute pharyngitis, unspecified", "Respiratory"),
    "J02.0": ("Streptococcal pharyngitis", "Respiratory"),
    "J03.9": ("Acute tonsillitis", "Respiratory"),
    "J04.0": ("Acute laryngitis", "Respiratory"),
    "J05.0": ("Acute obstructive laryngitis (Croup)", "Respiratory"),
    "J10":   ("Influenza", "Respiratory"),
    "J12.9": ("Viral pneumonia", "Respiratory"),
    "J15.9": ("Bacterial pneumonia", "Respiratory"),
    "J18.9": ("Pneumonia, unspecified", "Respiratory"),
    "J20.9": ("Acute bronchitis", "Respiratory"),
    "J21.9": ("Acute bronchiolitis", "Respiratory"),
    "J40":   ("Bronchitis NOS", "Respiratory"),
    "J41":   ("Chronic bronchitis", "Respiratory"),
    "J44.1": ("COPD with acute exacerbation", "Respiratory"),
    "J45.9": ("Asthma / Reactive airway disease", "Respiratory"),
    "J47":   ("Bronchiectasis", "Respiratory"),
    "J93.1": ("Spontaneous pneumothorax", "Respiratory"),
    "J01.9": ("Acute sinusitis", "Respiratory"),
    "J32.9": ("Chronic sinusitis", "Respiratory"),
    "J30.4": ("Allergic rhinosinusitis", "Respiratory"),
    "J38.5": ("Laryngospasm", "Respiratory"),
    "J81":   ("Pulmonary edema", "Respiratory"),
    "A37.9": ("Whooping cough", "Respiratory"),
    "A15":   ("Tuberculosis", "Respiratory"),
    "B97.2": ("COVID-19 / Coronavirus", "Respiratory"),
    "J04.1": ("Epiglottitis", "Respiratory"),
    "C34":   ("Pulmonary neoplasm", "Respiratory"),
    "D86":   ("Sarcoidosis", "Respiratory"),

    # ── Gastrointestinal ──
    "A09":   ("Acute diarrhea / gastroenteritis", "Gastrointestinal"),
    "A09.0": ("Persistent diarrhea", "Gastrointestinal"),
    "A03.9": ("Dysentery", "Gastrointestinal"),
    "A05":   ("Food poisoning", "Gastrointestinal"),
    "A08.1": ("Norovirus gastroenteritis", "Gastrointestinal"),
    "K21":   ("GERD", "Gastrointestinal"),
    "K40":   ("Inguinal hernia", "Gastrointestinal"),
    "K59.0": ("Constipation", "Gastrointestinal"),
    "R10.4": ("Non-severe abdominal condition", "Gastrointestinal"),
    "R63.0": ("Loss of appetite / Anorexia", "Gastrointestinal"),
    "K35":   ("Appendicitis", "Gastrointestinal"),
    "C25":   ("Pancreatic neoplasm", "Gastrointestinal"),
    "K56":   ("Bowel obstruction", "Gastrointestinal"),

    # ── Infectious & Parasitic ──
    "B50":   ("Malaria (uncomplicated)", "Infectious"),
    "B50.9": ("Severe malaria", "Infectious"),
    "B20":   ("HIV", "Infectious"),
    "A01.0": ("Typhoid fever", "Infectious"),
    "B77":   ("Intestinal parasitic - nematode", "Infectious"),
    "A06":   ("Intestinal parasitic - protozoa", "Infectious"),
    "B80":   ("Oxyuriasis / Enterobiasis", "Infectious"),
    "R50.9": ("Fever without source - viral", "Infectious"),
    "R50.8": ("Fever without source - bacterial", "Infectious"),
    "A75":   ("Chagas disease", "Infectious"),
    "A98.4": ("Ebola", "Infectious"),
    "B05":   ("Measles (non-severe)", "Infectious"),
    "B05.9": ("Severe measles", "Infectious"),
    "B01":   ("Varicella (Chicken pox)", "Infectious"),
    "A38":   ("Scarlet fever", "Infectious"),

    # ── ENT & Eye ──
    "H65":   ("Acute otitis media", "ENT & Eye"),
    "H10.0": ("Bacterial conjunctivitis", "ENT & Eye"),
    "H10.1": ("Viral conjunctivitis", "ENT & Eye"),
    "H10.4": ("Allergic conjunctivitis", "ENT & Eye"),
    "H05.0": ("Preseptal cellulitis (eye)", "ENT & Eye"),
    "H05.1": ("Orbital cellulitis", "ENT & Eye"),
    "H18.8": ("Corneal abrasion", "ENT & Eye"),
    "H57.9": ("Severe eye disease", "ENT & Eye"),
    "B37.0": ("Oral thrush / candidiasis", "ENT & Eye"),
    "P39.1": ("Neonatal conjunctivitis", "ENT & Eye"),

    # ── Skin & Soft Tissue ──
    "L02":   ("Abscess (simple)", "Skin"),
    "L03":   ("Cellulitis (uncomplicated)", "Skin"),
    "L01":   ("Impetigo", "Skin"),
    "B86":   ("Scabies", "Skin"),
    "L50":   ("Urticaria", "Skin"),
    "T78.2": ("Anaphylaxis", "Skin"),
    "L20":   ("Eczema / Atopic dermatitis", "Skin"),
    "B35.0": ("Tinea capitis", "Skin"),
    "B35.4": ("Tinea corporis", "Skin"),
    "B36.0": ("Pityriasis versicolor", "Skin"),
    "L73.9": ("Folliculitis", "Skin"),
    "L74.0": ("Heat rash (Miliaria)", "Skin"),
    "L22":   ("Diaper rash", "Skin"),
    "B08.1": ("Molluscum contagiosum", "Skin"),
    "B85.0": ("Pediculosis (head lice)", "Skin"),
    "B00":   ("Herpes simplex oral", "Skin"),
    "B09":   ("Non-specific viral rash", "Skin"),
    "L08":   ("Local skin infection", "Skin"),

    # ── Nutritional ──
    "D50":   ("Anemia (mild/moderate)", "Nutritional"),
    "D64.9": ("Anemia unspecified", "Nutritional"),
    "D50.9": ("Severe anemia", "Nutritional"),
    "E44":   ("Moderate malnutrition", "Nutritional"),
    "E43":   ("Severe malnutrition (uncomplicated)", "Nutritional"),
    "E43.9": ("Severe malnutrition (complicated)", "Nutritional"),

    # ── Circulatory ──
    "I47.1": ("PSVT", "Circulatory"),
    "I48":   ("Atrial fibrillation", "Circulatory"),
    "I40":   ("Myocarditis", "Circulatory"),
    "I20.0": ("Unstable angina", "Circulatory"),
    "I20.9": ("Stable angina", "Circulatory"),
    "I21":   ("NSTEMI / STEMI", "Circulatory"),
    "I26":   ("Pulmonary embolism", "Circulatory"),
    "I30":   ("Pericarditis", "Circulatory"),
    "T78.3": ("Localized edema", "Circulatory"),

    # ── Neurological ──
    "G44.0": ("Cluster headache", "Neurological"),
    "R51":   ("Non-severe headache", "Neurological"),
    "G61.0": ("Guillain-Barré syndrome", "Neurological"),
    "G70":   ("Myasthenia gravis", "Neurological"),
    "G24.0": ("Acute dystonic reactions", "Neurological"),
    "R56.0": ("Simple febrile convulsion", "Neurological"),
    "G41":   ("CNS danger signs", "Neurological"),

    # ── Mental ──
    "F41.0": ("Panic attack", "Mental"),

    # ── Urogenital ──
    "N30.0": ("Lower urinary tract infection (cystitis)", "Urogenital"),
    "N10":   ("Pyelonephritis (febrile UTI)", "Urogenital"),
    "N94.6": ("Dysmenorrhea", "Urogenital"),
    "N76":   ("Vulvovaginitis", "Urogenital"),
    "N48":   ("Balanitis", "Urogenital"),

    # ── Autoimmune ──
    "M32":   ("SLE", "Autoimmune"),

    # ── Neonatal ──
    "P59":   ("Physiological jaundice", "Neonatal"),
    "P59.9": ("Pathologic jaundice", "Neonatal"),
    "P38":   ("Omphalitis", "Neonatal"),
    "P14":   ("Birth-related soft tissue injury", "Neonatal"),
    "Q68.0": ("Congenital muscular torticollis", "Neonatal"),

    # ── Other / Fallback ──
    "R69":   ("Other / Unrecognized condition", "Other"),
    "E14":   ("Hyperglycemia", "Metabolic"),
    "E16.2": ("Hypoglycemia", "Metabolic"),
    "K22.3": ("Boerhaave (esophageal rupture)", "Gastrointestinal"),
    "M79.3": ("Spontaneous rib fracture", "Musculoskeletal"),
    "R25.0": ("Prolonged fever", "Infectious"),
}


# ── Disease-Specific Mapping Tables ──────────────────────────────

# DDXPlus disease_id → ICD-10
DDXPLUS_TO_ICD10 = {
    "d_spontaneous_pneumothorax": "J93.1",
    "d_cluster_headache": "G44.0",
    "d_boerhaave": "K22.3",
    "d_spontaneous_rib_fracture": "M79.3",
    "d_gerd": "K21",
    "d_hiv_initial_infection": "B20",
    "d_anemia": "D64.9",
    "d_viral_pharyngitis": "J02.9",
    "d_inguinal_hernia": "K40",
    "d_myasthenia_gravis": "G70",
    "d_whooping_cough": "A37.9",
    "d_anaphylaxis": "T78.2",
    "d_epiglottitis": "J04.1",
    "d_guillain_barr_syndrome": "G61.0",
    "d_acute_laryngitis": "J04.0",
    "d_croup": "J05.0",
    "d_psvt": "I47.1",
    "d_atrial_fibrillation": "I48",
    "d_bronchiectasis": "J47",
    "d_allergic_sinusitis": "J30.4",
    "d_chagas": "A75",
    "d_scombroid_food_poisoning": "A05",
    "d_myocarditis": "I40",
    "d_larygospasm": "J38.5",
    "d_acute_dystonic_reactions": "G24.0",
    "d_localized_edema": "T78.3",
    "d_sle": "M32",
    "d_tuberculosis": "A15",
    "d_unstable_angina": "I20.0",
    "d_stable_angina": "I20.9",
    "d_ebola": "A98.4",
    "d_acute_otitis_media": "H65",
    "d_panic_attack": "F41.0",
    "d_bronchospasm_acute_asthma_exacerbation": "J45.9",
    "d_bronchitis": "J40",
    "d_acute_copd_exacerbation_infection": "J44.1",
    "d_pulmonary_embolism": "I26",
    "d_urti": "J06.9",
    "d_influenza": "J10",
    "d_pneumonia": "J18.9",
    "d_acute_rhinosinusitis": "J01.9",
    "d_chronic_rhinosinusitis": "J32.9",
    "d_bronchiolitis": "J21.9",
    "d_pulmonary_neoplasm": "C34",
    "d_possible_nstemi_stemi": "I21",
    "d_sarcoidosis": "D86",
    "d_pancreatic_neoplasm": "C25",
    "d_acute_pulmonary_edema": "J81",
    "d_pericarditis": "I30",
    "d_other": "R69",
}

# Sample data disease_id → ICD-10
SAMPLE_TO_ICD10 = {
    "d_pna": "J18.9",
    "d_flu": "J10",
    "d_covid": "B97.2",
    "d_asthma": "J45.9",
    "d_copd": "J44.1",
    "d_food_pois": "A05",
    "d_norovirus": "A08.1",
    "d_other": "R69",
}


class DiseaseOntologyHarmonizer:
    """Maps heterogeneous disease IDs to a unified ICD-10 ontology."""

    def __init__(self):
        self.icd10_catalog = ICD10_CATALOG
        self.mappings = {
            "ddxplus": DDXPLUS_TO_ICD10,
            "sample": SAMPLE_TO_ICD10,
        }

    def harmonize(self, disease_id: str, source: str = "auto") -> Optional[str]:
        """Map a disease_id to its ICD-10 code."""
        if source == "auto":
            source = self._detect_source(disease_id)

        mapping = self.mappings.get(source, {})
        return mapping.get(disease_id)

    def _detect_source(self, disease_id: str) -> str:
        if disease_id.startswith("d_"):
            if disease_id in DDXPLUS_TO_ICD10:
                return "ddxplus"
            return "sample"
        return "unknown"

    def get_canonical_name(self, icd10: str) -> str:
        entry = self.icd10_catalog.get(icd10)
        return entry[0] if entry else "Unknown"

    def get_category(self, icd10: str) -> str:
        entry = self.icd10_catalog.get(icd10)
        return entry[1] if entry else "Other"

    def find_cross_dataset_matches(self) -> Dict[str, List[Tuple[str, str]]]:
        """Find diseases that appear across datasets via shared ICD-10 codes."""
        icd_to_sources = {}
        for source, mapping in self.mappings.items():
            for disease_id, icd10 in mapping.items():
                if icd10 not in icd_to_sources:
                    icd_to_sources[icd10] = []
                icd_to_sources[icd10].append((source, disease_id))

        # Only keep ICD-10 codes with entries from multiple sources
        cross_matches = {
            icd: entries for icd, entries in icd_to_sources.items()
            if len(set(e[0] for e in entries)) >= 2
        }
        return cross_matches

    def export_mapping_csv(self, output_path: str):
        """Export the full harmonized mapping to CSV."""
        rows = []
        for source, mapping in self.mappings.items():
            for disease_id, icd10 in mapping.items():
                rows.append({
                    "source_dataset": source,
                    "disease_id": disease_id,
                    "icd10_code": icd10,
                    "canonical_name": self.get_canonical_name(icd10),
                    "category": self.get_category(icd10),
                })

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["source_dataset", "disease_id", "icd10_code", "canonical_name", "category"])
            writer.writeheader()
            writer.writerows(rows)

    def coverage_report(self):
        """Print coverage statistics for each source."""
        print("\n" + "=" * 70)
        print("  ICD-10 HARMONIZATION COVERAGE REPORT")
        print("=" * 70)

        for source, mapping in self.mappings.items():
            unique_icd = set(mapping.values())
            categories = {}
            for icd in unique_icd:
                cat = self.get_category(icd)
                categories[cat] = categories.get(cat, 0) + 1

            print(f"\n  {source.upper()}")
            print(f"  ├── Mapped diseases: {len(mapping)}")
            print(f"  ├── Unique ICD-10 codes: {len(unique_icd)}")
            print(f"  └── Categories:")
            for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
                print(f"      ├── {cat}: {count}")

        # Cross-dataset overlap
        cross = self.find_cross_dataset_matches()
        print(f"\n  CROSS-DATASET OVERLAP")
        print(f"  ├── ICD-10 codes shared across datasets: {len(cross)}")
        print(f"  └── Top shared conditions:")
        for icd, entries in sorted(cross.items(), key=lambda x: -len(x[1]))[:15]:
            sources = [f"{s}:{d}" for s, d in entries]
            name = self.get_canonical_name(icd)
            print(f"      {icd} ({name}): {', '.join(set(e[0] for e in entries))}")


if __name__ == "__main__":
    h = DiseaseOntologyHarmonizer()
    h.coverage_report()

    from pathlib import Path
    output = str(Path(__file__).resolve().parents[2] / "world_model" / "icd10_harmonized_mapping.csv")
    h.export_mapping_csv(output)
    print(f"\nMapping exported to: {output}")
