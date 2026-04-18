# 🦴 ORN Ontology Builder

**Build a formal OWL 2 DL ontology for Osteoradionecrosis (ORN) grade and stage classification from PDF literature.**

Automatically extract and harmonize knowledge from 16 published ORNJ classification systems (Marx 1983 → ClinRad 2024) into a machine-readable ontology, answering the ORAL Consortium's call for standardized ORNJ knowledge representation.

---

## ✨ Features

- **PDF Text Extraction** — Native text + OCR fallback for scanned papers (Tesseract)
- **Biomedical NLP** — SciSpaCy NER, UMLS-augmented entity recognition, Hearst patterns
- **LLM-Accelerated Extraction** — Claude API structured extraction of classification criteria
- **Ontology Construction** — Automated OWL 2 DL generation with Owlready2
- **Classification Harmonization** — Cross-system severity mapping (Notani ↔ Marx ↔ ClinRad)
- **SNOMED-CT Alignment** — Maps ORN-O classes to established clinical terminology
- **Reasoning & Validation** — HermiT/Pellet consistency checking + SPARQL competency questions
- **Google Colab Notebooks** — Step-by-step interactive tutorials

## 🏥 Clinical Context

Osteoradionecrosis of the jaw (ORNJ) is a severe complication of radiation therapy for head and neck cancer, affecting up to 15% of patients. With **16+ classification systems** published over four decades, the field suffers from high rates of inability to classify (up to 76%), treatment-dependent vs. treatment-independent staging confusion, and no standardized machine-readable representation.

The **ORAL Consortium** (69 international experts) and **ISOO-MASCC-ASCO** have explicitly called for an ORNJ ontology. This project delivers it.

## 🚀 Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/orn-ontology-builder.git
cd orn-ontology-builder
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
```

```python
from src.pdf_extractor import PDFExtractor
from src.concept_extractor import ConceptExtractor
from src.relation_extractor import RelationExtractor
from src.ontology_builder import OntologyBuilder

documents = PDFExtractor().extract_from_directory("data/ornj_papers/")
concepts = ConceptExtractor().extract(documents)
relations = RelationExtractor().extract(documents, concepts)

builder = OntologyBuilder(iri="http://example.org/orn-ontology")
builder.add_concepts(concepts)
builder.add_relations(relations)
builder.save("output/orn_ontology.owl")
```

## 📚 Key References

- Watson EE et al. (2024). ClinRad classification. *J Clin Oncol* 42:1922-1933.
- Moreno AC et al. (2024). ORAL Consortium Delphi. *IJROBP* 120:1047-1059.
- Notani K et al. (2003). Mandibular ORN staging. *Head Neck* 25:181-186.
- Marx RE (1983). ORN pathophysiology. *J Oral Maxillofac Surg* 41:283-288.

## 📄 License

MIT License
