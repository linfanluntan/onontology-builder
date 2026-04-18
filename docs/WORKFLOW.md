# ORN Ontology Building Workflow — Detailed Guide

## Domain: Osteoradionecrosis of the Jaw (ORNJ)

This document describes the end-to-end pipeline for building the ORN Grade/Stage
Ontology (ORN-O) from 16 ORNJ classification system PDFs.

## Pipeline Overview

```
16 ORNJ Classification PDFs
  │  (Marx 1983 → ClinRad 2024)
  ▼
[1. PDF Extraction]  ──►  Raw text (142 pages, ~186K words)
  │
  ▼
[2. Preprocessing]   ──►  Clean, segmented text (2,847 segments)
  │
  ├───────────────────────────────┐
  ▼                               ▼
[3a. NLP Extraction]         [3b. LLM Extraction]
  │  SciSpaCy NER                 │  Claude API
  │  Hearst patterns              │  Structured JSON
  │  Dep. parsing + TF-IDF        │  Conditional logic
  │                               │
  └───────────┬───────────────────┘
              ▼
[4. Hybrid Merging]  ──►  267 concepts, 412 relations
              │
              ▼
[5. OWL 2 DL Construction]  ──►  ORN-O ontology (3,124 axioms)
              │
              ▼
[6. Validation]  ──►  HermiT + 38 CQs + expert review
              │
              ▼
[7. SPARQL Queries + Visualization]
```

## Source Corpus (16 PDFs)

The corpus covers all major ORNJ classification systems:

| Era | Systems | Type |
|-----|---------|------|
| 1983–1987 | Coffin, Marx, Epstein | Treatment-dependent |
| 1995–2003 | RTOG/EORTC, LENT/SOMA, Glanzmann, Støre-Boysen, Schwartz-Kagan, Notani | Mixed |
| 2013–2017 | Tsai, Karagozoglu, Lyons, Caparrotti, CTCAE v5 | Anatomical/duration |
| 2024–2025 | ClinRad (Watson), ORAL Consortium Delphi, ORNJ-O, Aljohani | Integrated/consensus |

## Target Ontology Structure (7 Modules)

1. **Classification Systems** — 16 systems with their grades/stages as subclasses
2. **Clinical Findings** — BoneExposure, Fistula, Pain, Trismus, Swelling
3. **Radiographic Findings** — Osteolysis, Sclerosis, Fracture, Sequestrum
4. **Anatomical Structures** — Mandible, Maxilla, AlveolarBone, BasilarBone, IAC
5. **Treatment Modalities** — HBO, PENTOCLO, Sequestrectomy, Resection, FreeFlap
6. **Risk Factors** — RadiationDose, Smoking, DentalExtraction, PeriodontalDisease
7. **Patient Outcomes** — Progression, TreatmentResponse, QualityOfLife

## Key Clinical References

- Watson EE et al. (2024). ClinRad classification. J Clin Oncol 42:1922-1933.
- Moreno AC et al. (2024). ORAL Consortium Delphi study. IJROBP 120:1047-1059.
- Notani K et al. (2003). Mandibular ORN staging. Head Neck 25:181-186.
- Marx RE (1983). ORN pathophysiology. J Oral Maxillofac Surg 41:283-288.
- Chronopoulos A et al. (2018). ORN review. Int Dent J 68:22-30.
- Frankart AJ et al. (2021). Exposing the evidence. IJROBP 109:1206-1218.
