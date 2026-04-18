# 🧠 PDF Ontology Builder

**Automatically extract structured ontologies from PDF documents using NLP and LLMs.**

Transform unstructured PDF documents into formal OWL/RDF ontologies with a modular Python pipeline. Supports both traditional NLP extraction and LLM-accelerated workflows (Claude, OpenAI-compatible APIs).

---

## ✨ Features

- **PDF Text Extraction** — Native text + OCR fallback (Tesseract)
- **Concept & Term Extraction** — spaCy NER, TF-IDF, noun phrase chunking
- **Relation Extraction** — Hearst patterns, dependency parsing, LLM-based
- **Ontology Construction** — Automated OWL/RDF generation with Owlready2 & rdflib
- **Reasoning & Validation** — HermiT/Pellet integration via Owlready2
- **SPARQL Querying** — Query your ontology with SPARQL
- **Visualization** — Export for WebVOWL, NetworkX graph plots
- **Google Colab Notebooks** — Step-by-step interactive tutorials

## 📁 Repository Structure

```
ontology-builder/
├── README.md
├── requirements.txt
├── setup.py
├── LICENSE
├── .github/
│   └── workflows/
│       └── tests.yml
├── src/
│   ├── __init__.py
│   ├── pdf_extractor.py       # PDF text extraction (native + OCR)
│   ├── preprocessor.py        # Text cleaning & segmentation
│   ├── concept_extractor.py   # NLP-based concept/term extraction
│   ├── relation_extractor.py  # Relation & triple extraction
│   ├── llm_extractor.py       # LLM-accelerated extraction (Claude API)
│   ├── ontology_builder.py    # OWL/RDF ontology construction
│   ├── validator.py           # Reasoning & consistency checks
│   ├── query_engine.py        # SPARQL query interface
│   └── visualizer.py          # Graph visualization utilities
├── notebooks/
│   ├── 01_pdf_extraction.ipynb
│   ├── 02_concept_extraction.ipynb
│   ├── 03_relation_extraction.ipynb
│   ├── 04_ontology_building.ipynb
│   ├── 05_validation_and_querying.ipynb
│   └── 06_full_pipeline_demo.ipynb
├── examples/
│   └── sample_config.yaml
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py
└── docs/
    └── WORKFLOW.md
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/ontology-builder.git
cd ontology-builder
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
```

### Basic Usage

```python
from src.pdf_extractor import PDFExtractor
from src.concept_extractor import ConceptExtractor
from src.relation_extractor import RelationExtractor
from src.ontology_builder import OntologyBuilder

# 1. Extract text from PDFs
extractor = PDFExtractor()
documents = extractor.extract_from_directory("path/to/pdfs/")

# 2. Extract concepts
concept_ext = ConceptExtractor()
concepts = concept_ext.extract(documents)

# 3. Extract relations
relation_ext = RelationExtractor()
relations = relation_ext.extract(documents, concepts)

# 4. Build ontology
builder = OntologyBuilder(iri="http://example.org/my-ontology")
builder.add_concepts(concepts)
builder.add_relations(relations)
builder.save("output/my_ontology.owl")
```

### LLM-Accelerated Pipeline

```python
from src.llm_extractor import LLMExtractor

llm_ext = LLMExtractor(api_key="your-anthropic-key")
knowledge = llm_ext.extract_from_documents(documents)

builder = OntologyBuilder(iri="http://example.org/my-ontology")
builder.from_llm_output(knowledge)
builder.save("output/my_ontology.owl")
```

## 📓 Colab Notebooks

| Notebook | Description | Link |
|----------|-------------|------|
| `01_pdf_extraction` | Extract text from PDFs with native + OCR methods | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#) |
| `02_concept_extraction` | NLP-based concept and term extraction | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#) |
| `03_relation_extraction` | Relation and triple extraction | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#) |
| `04_ontology_building` | Construct OWL ontologies programmatically | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#) |
| `05_validation_and_querying` | Validate with reasoners + SPARQL queries | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#) |
| `06_full_pipeline_demo` | End-to-end pipeline on sample documents | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#) |

> **Note:** Update the badge links with your actual GitHub URLs after pushing the repo.

## ⚙️ Configuration

See `examples/sample_config.yaml` for all options:

```yaml
extraction:
  method: hybrid           # nlp, llm, or hybrid
  llm_provider: anthropic
  llm_model: claude-sonnet-4-20250514
  spacy_model: en_core_web_sm

ontology:
  iri: "http://example.org/ontology"
  format: rdfxml           # rdfxml, turtle, ntriples
  reasoning: true
  reasoner: hermit
```

## 🧪 Testing

```bash
pytest tests/ -v
```

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
