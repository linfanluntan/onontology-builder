# Ontology Building Workflow — Detailed Guide

This document describes the end-to-end pipeline for building OWL ontologies from PDF documents.

## Pipeline Overview

```
PDF Files
  │
  ▼
[1. PDF Extraction]  ──►  Raw text per page
  │
  ▼
[2. Preprocessing]   ──►  Clean, segmented text
  │
  ├───────────────────────────┐
  ▼                           ▼
[3a. NLP Extraction]     [3b. LLM Extraction]
  │  Concepts, Relations      │  Structured JSON
  │                           │
  └───────────┬───────────────┘
              ▼
[4. Ontology Construction]  ──►  OWL/RDF file
              │
              ▼
[5. Validation & Reasoning]  ──►  Consistency report
              │
              ▼
[6. Querying & Visualization]  ──►  SPARQL results, graph plots
```

---

## Step 1: PDF Text Extraction

**Module:** `src/pdf_extractor.py`

The extractor uses PyMuPDF for native text extraction. When a page yields fewer
than `min_text_length` characters (default 50), it falls back to OCR via
Tesseract at 300 DPI.

**Key classes:**
- `PDFExtractor` — main extractor
- `Document` — holds pages, metadata, and full text
- `DocumentPage` — single page with text and extraction method

**Usage:**
```python
extractor = PDFExtractor(ocr_enabled=True, ocr_language="eng")
doc = extractor.extract("paper.pdf")
print(doc.full_text[:500])

# Batch extraction
docs = extractor.extract_from_directory("data/pdfs/", recursive=True)
```

---

## Step 2: Text Preprocessing

**Module:** `src/preprocessor.py`

Cleans raw extracted text:
- Rejoins hyphenated words across line breaks
- Normalizes whitespace
- Removes page numbers and running headers/footers
- Segments text into paragraphs and detects headings

**Output:** List of `TextSegment` objects with metadata (heading, type, source).

---

## Step 3a: NLP-Based Extraction

### Concept Extraction (`src/concept_extractor.py`)

Three complementary strategies:

1. **Named Entity Recognition (NER)** — spaCy identifies organizations,
   people, locations, products, etc.
2. **Noun Phrase Chunking** — extracts multi-word noun phrases as candidate
   concepts.
3. **TF-IDF Ranking** — scores terms by domain specificity across the document
   corpus.

Results are merged, deduplicated, and filtered by minimum frequency.

### Relation Extraction (`src/relation_extractor.py`)

Two strategies:

1. **Hearst Patterns** — regex patterns that capture hypernym (is-a) relations:
   - "X such as Y and Z" → Y isA X, Z isA X
   - "Y and other X" → Y isA X
   - "X, including Y" → Y isA X
   - "X is a type of Y" → X isA Y

2. **Dependency Parsing** — extracts subject-verb-object triples from spaCy
   dependency trees. Confidence is boosted when both subject and object match
   known concepts.

---

## Step 3b: LLM-Accelerated Extraction

**Module:** `src/llm_extractor.py`

Sends text chunks to Claude API with a structured extraction prompt. The model
returns JSON containing:
- **concepts** — with name, label, type, definition, and parent class
- **relations** — subject-predicate-object triples
- **attributes** — data properties with value types

This approach captures more nuanced relations and can infer taxonomy structure
that pattern-based methods miss.

**When to use LLM extraction:**
- Complex or technical domains
- Documents with implicit relationships
- When high-quality taxonomy structure matters
- When NLP-only extraction produces too much noise

**Hybrid approach:** Run both NLP and LLM extraction, then merge results.

---

## Step 4: Ontology Construction

**Module:** `src/ontology_builder.py`

Builds an OWL ontology using Owlready2:

1. Creates OWL classes from extracted concepts
2. Establishes subclass hierarchy from is-a relations
3. Creates object properties from other relations
4. Adds data properties from extracted attributes
5. Optionally populates with instances

**Supported output formats:** RDF/XML, Turtle, N-Triples

---

## Step 5: Validation & Reasoning

**Module:** `src/validator.py`

Runs three validation layers:

1. **Structural checks** — orphan classes, missing labels, empty properties
2. **DL Reasoning** — HermiT or Pellet reasoner checks logical consistency
   and infers new subsumptions
3. **Common issues** — cycle detection in class hierarchy, missing
   domain/range on properties

**Requirements:** Java must be installed for reasoners (HermiT/Pellet).

---

## Step 6: Querying & Visualization

### SPARQL Queries (`src/query_engine.py`)

Load the ontology into an rdflib graph and run SPARQL:

```python
engine = QueryEngine("output/ontology.owl")
classes = engine.get_all_classes()
hierarchy = engine.get_class_hierarchy()
results = engine.query("SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10")
```

### Visualization (`src/visualizer.py`)

- **NetworkX plots** — class hierarchy and property graphs
- **WebVOWL export** — generates Turtle file for the interactive WebVOWL viewer

---

## Tips for Best Results

1. **Start small** — test with 2–3 PDFs before processing a large corpus.
2. **Tune thresholds** — adjust `min_frequency`, `min_confidence`, and
   `max_concepts` based on your domain.
3. **Use a larger spaCy model** — `en_core_web_lg` gives better NER and
   parsing than `en_core_web_sm`.
4. **Provide domain context to the LLM** — the `domain_context` parameter
   significantly improves LLM extraction quality.
5. **Iterate** — review extracted concepts/relations, add stopwords or
   filters, and re-run.
6. **Reuse existing ontologies** — import established upper ontologies or
   domain ontologies as a starting point.
