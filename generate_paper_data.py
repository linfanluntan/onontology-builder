#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║  ORN-O DATA COLLECTION SCRIPT                                   ║
║  Runs the full pipeline on 16 ORNJ PDFs and generates a         ║
║  structured report with real numbers for the journal paper.      ║
║                                                                  ║
║  Usage:                                                          ║
║    python generate_paper_data.py --pdf-dir data/ornj_papers/     ║
║    python generate_paper_data.py --pdf-dir pdfs/ --use-llm       ║
║    python generate_paper_data.py --pdf-dir pdfs/ --use-llm \     ║
║        --api-key sk-ant-...                                      ║
║                                                                  ║
║  Output: output/paper_data_report.txt  (copy-paste into paper)   ║
║          output/orn_ontology.owl       (the ontology)            ║
║          output/orn_ontology.ttl       (Turtle for WebVOWL)      ║
║          output/orn_graph.png          (visualization)            ║
║          output/paper_tables.json      (machine-readable data)   ║
╚══════════════════════════════════════════════════════════════════╝
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

# ── Setup ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("orn-pipeline")

# Add src to path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


def separator(title):
    """Print section separator."""
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}\n")


def subsep(title):
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}\n")


# ═══════════════════════════════════════════════════════════════
# PHASE 1: PDF TEXT EXTRACTION
# ═══════════════════════════════════════════════════════════════
def phase1_extraction(pdf_dir):
    """Extract text from all PDFs. Returns documents list and stats."""
    separator("PHASE 1: PDF TEXT EXTRACTION")

    from src.pdf_extractor import PDFExtractor

    extractor = PDFExtractor(ocr_enabled=True, ocr_language="eng")
    documents = extractor.extract_from_directory(pdf_dir, recursive=False)

    if not documents:
        logger.error(f"No PDFs found in {pdf_dir}")
        sys.exit(1)

    # Collect per-document stats
    doc_stats = []
    total_chars = 0
    total_pages = 0
    total_ocr_pages = 0

    for doc in documents:
        chars = len(doc.full_text)
        total_chars += chars
        total_pages += doc.num_pages

        native_pages = sum(1 for p in doc.pages if p.method == "native")
        ocr_pages = sum(1 for p in doc.pages if p.method == "ocr")
        total_ocr_pages += ocr_pages

        # Estimate quality as % of pages with >50 chars
        good_pages = sum(1 for p in doc.pages if len(p.text.strip()) > 50)
        quality = (good_pages / doc.num_pages * 100) if doc.num_pages > 0 else 0

        doc_stats.append({
            "filename": doc.filename,
            "pages": doc.num_pages,
            "characters": chars,
            "ocr_pages": ocr_pages,
            "ocr_pct": f"{(ocr_pages / doc.num_pages * 100):.0f}%" if doc.num_pages > 0 else "0%",
            "quality": f"{quality:.1f}%",
        })

        logger.info(
            f"  {doc.filename}: {doc.num_pages} pages, {chars:,} chars, "
            f"{ocr_pages} OCR pages ({quality:.0f}% quality)"
        )

    summary = {
        "total_documents": len(documents),
        "total_pages": total_pages,
        "total_characters": total_chars,
        "total_words": len(" ".join(d.full_text for d in documents).split()),
        "total_ocr_pages": total_ocr_pages,
        "ocr_pct_overall": f"{(total_ocr_pages / total_pages * 100):.1f}%" if total_pages > 0 else "0%",
        "doc_stats": doc_stats,
    }

    logger.info(
        f"\nTotal: {len(documents)} docs, {total_pages} pages, "
        f"{total_chars:,} chars, {total_ocr_pages} OCR pages"
    )
    return documents, summary


# ═══════════════════════════════════════════════════════════════
# PHASE 2: TEXT PREPROCESSING
# ═══════════════════════════════════════════════════════════════
def phase2_preprocessing(documents):
    """Preprocess extracted text. Returns segments and stats."""
    separator("PHASE 2: TEXT PREPROCESSING")

    from src.preprocessor import TextPreprocessor

    pp = TextPreprocessor(
        remove_headers_footers=True,
        normalize_whitespace=True,
        fix_hyphenation=True,
        min_segment_length=20,
    )

    all_segments = []
    per_doc_segments = {}

    for doc in documents:
        segments = pp.preprocess(doc.full_text, source_doc=doc.filename)
        all_segments.extend(segments)
        per_doc_segments[doc.filename] = len(segments)
        logger.info(f"  {doc.filename}: {len(segments)} segments")

    headings = [s for s in all_segments if s.segment_type == "heading"]
    paragraphs = [s for s in all_segments if s.segment_type == "paragraph"]

    summary = {
        "total_segments": len(all_segments),
        "headings": len(headings),
        "paragraphs": len(paragraphs),
        "per_doc_segments": per_doc_segments,
        "avg_segment_length": int(
            sum(len(s.text) for s in all_segments) / max(len(all_segments), 1)
        ),
    }

    logger.info(
        f"\nTotal: {len(all_segments)} segments "
        f"({len(headings)} headings, {len(paragraphs)} paragraphs)"
    )
    return all_segments, summary


# ═══════════════════════════════════════════════════════════════
# PHASE 3: CONCEPT EXTRACTION (NLP)
# ═══════════════════════════════════════════════════════════════
def phase3_concepts(segments):
    """Extract concepts via NLP. Returns concepts and stats."""
    separator("PHASE 3: NLP CONCEPT EXTRACTION")

    from src.concept_extractor import ConceptExtractor

    # Try SciSpaCy first, fall back to general model
    try:
        extractor = ConceptExtractor(
            spacy_model="en_core_sci_lg",
            min_frequency=2,
            max_concepts=500,
            tfidf_top_n=200,
        )
        model_used = "en_core_sci_lg"
    except Exception:
        try:
            extractor = ConceptExtractor(
                spacy_model="en_core_web_sm",
                min_frequency=2,
                max_concepts=500,
                tfidf_top_n=200,
            )
            model_used = "en_core_web_sm"
        except Exception as e:
            logger.error(f"No spaCy model available: {e}")
            sys.exit(1)

    logger.info(f"Using spaCy model: {model_used}")
    concepts = extractor.extract(segments)

    # Frequency distribution
    freq_dist = Counter()
    for c in concepts:
        if c.frequency >= 10:
            freq_dist["≥10"] += 1
        elif c.frequency >= 5:
            freq_dist["5-9"] += 1
        elif c.frequency >= 2:
            freq_dist["2-4"] += 1
        else:
            freq_dist["1"] += 1

    summary = {
        "spacy_model": model_used,
        "total_concepts": len(concepts),
        "frequency_distribution": dict(freq_dist),
        "top_30_concepts": [
            {"name": c.name, "label": c.label, "frequency": c.frequency}
            for c in concepts[:30]
        ],
        "all_concept_names": [c.name for c in concepts],
    }

    logger.info(f"\nExtracted {len(concepts)} concepts")
    logger.info(f"Top 10:")
    for c in concepts[:10]:
        logger.info(f"  {c.name:<35} freq={c.frequency}")

    return concepts, summary


# ═══════════════════════════════════════════════════════════════
# PHASE 4: RELATION EXTRACTION (NLP)
# ═══════════════════════════════════════════════════════════════
def phase4_relations(segments, concepts):
    """Extract relations via NLP. Returns relations and stats."""
    separator("PHASE 4: NLP RELATION EXTRACTION")

    from src.relation_extractor import RelationExtractor

    rel_ext = RelationExtractor(
        use_hearst=True,
        use_dependency=True,
        min_confidence=0.3,
    )

    relations = rel_ext.extract(segments, concepts)

    is_a = [r for r in relations if r.relation_type == "is_a"]
    obj_prop = [r for r in relations if r.relation_type == "object_property"]

    conf_dist = Counter()
    for r in relations:
        if r.confidence >= 0.8:
            conf_dist["≥0.8"] += 1
        elif r.confidence >= 0.5:
            conf_dist["0.5-0.79"] += 1
        else:
            conf_dist["0.3-0.49"] += 1

    summary = {
        "total_relations": len(relations),
        "is_a_relations": len(is_a),
        "object_property_relations": len(obj_prop),
        "mean_confidence": round(
            sum(r.confidence for r in relations) / max(len(relations), 1), 3
        ),
        "confidence_distribution": dict(conf_dist),
        "top_20_relations": [
            {
                "subject": r.subject,
                "predicate": r.predicate,
                "object": r.object,
                "confidence": round(r.confidence, 2),
                "type": r.relation_type,
            }
            for r in sorted(relations, key=lambda x: x.confidence, reverse=True)[:20]
        ],
    }

    logger.info(
        f"\nExtracted {len(relations)} relations "
        f"({len(is_a)} is-a, {len(obj_prop)} object property)"
    )
    logger.info(f"Mean confidence: {summary['mean_confidence']}")
    return relations, summary


# ═══════════════════════════════════════════════════════════════
# PHASE 5: LLM EXTRACTION (optional)
# ═══════════════════════════════════════════════════════════════
def phase5_llm(documents, api_key):
    """Extract knowledge via Claude API. Returns LLMKnowledge and stats."""
    separator("PHASE 5: LLM-ACCELERATED EXTRACTION")

    from src.llm_extractor import LLMExtractor

    llm_ext = LLMExtractor(
        api_key=api_key,
        model="claude-sonnet-4-20250514",
        domain_context="osteoradionecrosis of the jaw classification and staging systems",
        chunk_size=3000,
    )

    start_time = time.time()
    knowledge = llm_ext.extract_from_documents(documents)
    elapsed = time.time() - start_time

    summary = {
        "llm_model": "claude-sonnet-4-20250514",
        "llm_concepts": len(knowledge.concepts),
        "llm_relations": len(knowledge.relations),
        "llm_attributes": len(knowledge.attributes),
        "extraction_time_seconds": round(elapsed, 1),
        "top_20_llm_concepts": knowledge.concepts[:20],
        "top_20_llm_relations": knowledge.relations[:20],
    }

    logger.info(
        f"\nLLM extracted: {len(knowledge.concepts)} concepts, "
        f"{len(knowledge.relations)} relations, "
        f"{len(knowledge.attributes)} attributes "
        f"in {elapsed:.0f}s"
    )
    return knowledge, summary


# ═══════════════════════════════════════════════════════════════
# PHASE 6: HYBRID MERGING
# ═══════════════════════════════════════════════════════════════
def phase6_merge(nlp_concepts, nlp_relations, llm_knowledge=None):
    """Merge NLP and LLM results. Returns merged concepts/relations and stats."""
    separator("PHASE 6: HYBRID MERGING")

    nlp_names = {c.name.lower() for c in nlp_concepts}
    both, nlp_only, llm_only = 0, 0, 0

    merged_concepts = list(nlp_concepts)
    merged_relations = list(nlp_relations)

    if llm_knowledge:
        llm_names = {c.get("name", "").lower() for c in llm_knowledge.concepts}
        both = len(nlp_names & llm_names)
        nlp_only = len(nlp_names - llm_names)
        llm_only = len(llm_names - nlp_names)

        # Add LLM-only concepts (simplified — in production you'd use Concept objects)
        from src.concept_extractor import Concept

        for lc in llm_knowledge.concepts:
            name = lc.get("name", "")
            if name.lower() not in nlp_names:
                merged_concepts.append(
                    Concept(
                        name=name,
                        label=lc.get("label", name),
                        parent=lc.get("parent", ""),
                    )
                )

        # Add LLM relations
        from src.relation_extractor import Relation

        existing_keys = {
            (r.subject.lower(), r.predicate.lower(), r.object.lower())
            for r in nlp_relations
        }
        for lr in llm_knowledge.relations:
            key = (
                lr.get("subject", "").lower(),
                lr.get("predicate", "").lower(),
                lr.get("object", "").lower(),
            )
            if key not in existing_keys:
                merged_relations.append(
                    Relation(
                        subject=lr.get("subject", ""),
                        predicate=lr.get("predicate", ""),
                        object=lr.get("object", ""),
                        confidence=0.7,
                    )
                )
    else:
        nlp_only = len(nlp_names)

    total = len(merged_concepts)
    summary = {
        "total_merged_concepts": total,
        "total_merged_relations": len(merged_relations),
        "overlap_both": both,
        "nlp_only": nlp_only,
        "llm_only": llm_only,
        "overlap_pct": f"{(both / max(total, 1) * 100):.1f}%",
        "nlp_only_pct": f"{(nlp_only / max(total, 1) * 100):.1f}%",
        "llm_only_pct": f"{(llm_only / max(total, 1) * 100):.1f}%",
    }

    logger.info(
        f"Merged: {total} concepts, {len(merged_relations)} relations"
    )
    if llm_knowledge:
        logger.info(
            f"Overlap: {both} both ({summary['overlap_pct']}), "
            f"{nlp_only} NLP-only ({summary['nlp_only_pct']}), "
            f"{llm_only} LLM-only ({summary['llm_only_pct']})"
        )
    return merged_concepts, merged_relations, summary


# ═══════════════════════════════════════════════════════════════
# PHASE 7: ONTOLOGY CONSTRUCTION
# ═══════════════════════════════════════════════════════════════
def phase7_ontology(concepts, relations, llm_knowledge, output_dir):
    """Build OWL ontology. Returns builder and stats."""
    separator("PHASE 7: OWL 2 DL ONTOLOGY CONSTRUCTION")

    from src.ontology_builder import OntologyBuilder

    builder = OntologyBuilder(
        iri="http://example.org/orn-ontology",
        name="ORNGradeStageOntology",
    )

    builder.add_concepts(concepts)
    builder.add_relations(relations)

    if llm_knowledge and llm_knowledge.attributes:
        builder.add_data_properties(llm_knowledge.attributes)

    owl_path = os.path.join(output_dir, "orn_ontology.owl")
    builder.save(owl_path, format="rdfxml")

    stats = builder.get_stats()

    # Count axiom types from rdflib
    from rdflib import Graph, RDF, RDFS, OWL

    g = Graph()
    g.parse(owl_path, format="xml")
    total_triples = len(g)

    subclass_axioms = len(list(g.triples((None, RDFS.subClassOf, None))))
    domain_axioms = len(list(g.triples((None, RDFS.domain, None))))
    range_axioms = len(list(g.triples((None, RDFS.range, None))))

    # Compute max hierarchy depth
    def get_depth(cls, visited=None):
        if visited is None:
            visited = set()
        if cls in visited:
            return 0
        visited.add(cls)
        parents = list(g.objects(cls, RDFS.subClassOf))
        if not parents:
            return 0
        return 1 + max(get_depth(p, visited.copy()) for p in parents)

    classes = list(g.subjects(RDF.type, OWL.Class))
    max_depth = max((get_depth(c) for c in classes), default=0)

    file_size = os.path.getsize(owl_path)

    summary = {
        **stats,
        "total_triples": total_triples,
        "subclass_axioms": subclass_axioms,
        "domain_range_axioms": domain_axioms + range_axioms,
        "max_hierarchy_depth": max_depth,
        "file_size_kb": round(file_size / 1024, 1),
        "owl_path": owl_path,
    }

    logger.info(f"\nOntology built:")
    for k, v in summary.items():
        if k not in ("class_names", "owl_path"):
            logger.info(f"  {k}: {v}")

    return builder, summary


# ═══════════════════════════════════════════════════════════════
# PHASE 8: VALIDATION
# ═══════════════════════════════════════════════════════════════
def phase8_validation(builder, owl_path, output_dir):
    """Validate ontology with reasoner and SPARQL. Returns report and stats."""
    separator("PHASE 8: VALIDATION & SPARQL")

    # 8a: Reasoner
    from src.validator import OntologyValidator

    validator = OntologyValidator(reasoner="hermit")
    try:
        report = validator.validate(builder.onto)
        reasoner_result = {
            "is_consistent": report.is_consistent,
            "errors": len(report.errors),
            "warnings": len(report.warnings),
            "inferred_subsumptions": len(report.inferred_subsumptions),
            "error_messages": report.errors[:5],
            "warning_messages": report.warnings[:5],
        }
        logger.info(f"Reasoner: {'CONSISTENT' if report.is_consistent else 'INCONSISTENT'}")
        logger.info(f"  Errors: {len(report.errors)}, Warnings: {len(report.warnings)}")
        logger.info(f"  Inferred subsumptions: {len(report.inferred_subsumptions)}")
    except Exception as e:
        logger.warning(f"Reasoner failed (Java may not be installed): {e}")
        reasoner_result = {
            "is_consistent": "unknown (reasoner unavailable)",
            "errors": 0,
            "warnings": 1,
            "inferred_subsumptions": 0,
            "error_messages": [],
            "warning_messages": [str(e)],
        }

    # 8b: SPARQL queries
    from src.query_engine import QueryEngine

    engine = QueryEngine(owl_path)
    total_triples = engine.count_triples()
    all_classes = engine.get_all_classes()
    hierarchy = engine.get_class_hierarchy()

    sparql_results = {
        "total_triples": total_triples,
        "total_classes_sparql": len(all_classes),
        "hierarchy_edges": len(hierarchy),
        "class_names": [
            c["class"].split("/")[-1].split("#")[-1]
            for c in all_classes
            if "class" in c and c["class"]
        ],
    }

    logger.info(f"\nSPARQL: {total_triples} triples, {len(all_classes)} classes")

    # 8c: Visualization
    try:
        from src.visualizer import OntologyVisualizer

        viz = OntologyVisualizer(owl_path)
        graph_summary = viz.get_summary()
        graph_path = os.path.join(output_dir, "orn_graph.png")
        viz.plot_hierarchy(
            figsize=(18, 14),
            title="ORN-O Ontology Class Hierarchy",
            save_path=graph_path,
        )
        ttl_path = os.path.join(output_dir, "orn_ontology.ttl")
        viz.export_for_webvowl(ttl_path)
        logger.info(f"Graph saved to {graph_path}")
        logger.info(f"Turtle exported to {ttl_path}")
    except Exception as e:
        logger.warning(f"Visualization failed: {e}")
        graph_summary = {"error": str(e)}

    return {
        "reasoner": reasoner_result,
        "sparql": sparql_results,
        "graph": graph_summary,
    }


# ═══════════════════════════════════════════════════════════════
# REPORT GENERATION
# ═══════════════════════════════════════════════════════════════
def generate_report(
    extraction_stats,
    preprocessing_stats,
    concept_stats,
    relation_stats,
    llm_stats,
    merge_stats,
    ontology_stats,
    validation_stats,
    output_dir,
):
    """Generate the copy-paste report for the paper."""
    separator("GENERATING PAPER DATA REPORT")

    report_path = os.path.join(output_dir, "paper_data_report.txt")
    json_path = os.path.join(output_dir, "paper_tables.json")

    lines = []
    lines.append("=" * 72)
    lines.append("  ORN-O PAPER DATA REPORT")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 72)

    # ── TABLE 1: PDF Corpus ───────────────────────────────────
    lines.append("\n\n" + "─" * 72)
    lines.append("TABLE 1: PDF Corpus — Extraction Results")
    lines.append("Copy into: Section 3.2 / 3.3 (Table 1)")
    lines.append("─" * 72)
    lines.append(
        f"{'ID':<5} {'Document':<40} {'Pages':>6} {'Chars':>10} {'OCR%':>6} {'Quality':>8}"
    )
    lines.append("-" * 72)
    for i, ds in enumerate(extraction_stats["doc_stats"]):
        lines.append(
            f"D{i+1:02d}  {ds['filename']:<40} {ds['pages']:>6} "
            f"{ds['characters']:>10,} {ds['ocr_pct']:>6} {ds['quality']:>8}"
        )
    lines.append("-" * 72)
    lines.append(
        f"{'TOTAL':<46} {extraction_stats['total_pages']:>6} "
        f"{extraction_stats['total_characters']:>10,} "
        f"{extraction_stats['ocr_pct_overall']:>6}"
    )
    lines.append(f"\nTotal words: {extraction_stats['total_words']:,}")
    lines.append(f"Total segments after preprocessing: {preprocessing_stats['total_segments']}")
    lines.append(f"  Headings: {preprocessing_stats['headings']}")
    lines.append(f"  Paragraphs: {preprocessing_stats['paragraphs']}")

    # ── SECTION 3.5: Concept Extraction Numbers ──────────────
    lines.append("\n\n" + "─" * 72)
    lines.append("SECTION 3.5: Concept Extraction Numbers")
    lines.append("Copy into: Section 3.5 body text")
    lines.append("─" * 72)
    lines.append(f"spaCy model used: {concept_stats['spacy_model']}")
    lines.append(f"Total candidate concepts before filtering: [check extractor log]")
    lines.append(f"Total concepts after frequency filtering: {concept_stats['total_concepts']}")
    lines.append(f"Frequency distribution: {concept_stats['frequency_distribution']}")
    lines.append(f"\nTop 30 concepts:")
    for c in concept_stats["top_30_concepts"]:
        lines.append(f"  {c['name']:<40} label=\"{c['label']}\"  freq={c['frequency']}")

    # ── SECTION 3.6: Relation Extraction Numbers ─────────────
    lines.append("\n\n" + "─" * 72)
    lines.append("SECTION 3.6: Relation Extraction Numbers")
    lines.append("Copy into: Section 3.6 body text")
    lines.append("─" * 72)
    lines.append(f"Total relations: {relation_stats['total_relations']}")
    lines.append(f"  is-a (Hearst patterns): {relation_stats['is_a_relations']}")
    lines.append(f"  Object properties (dependency): {relation_stats['object_property_relations']}")
    lines.append(f"Mean confidence: {relation_stats['mean_confidence']}")
    lines.append(f"Confidence distribution: {relation_stats['confidence_distribution']}")
    lines.append(f"\nTop 20 relations (by confidence):")
    for r in relation_stats["top_20_relations"]:
        lines.append(
            f"  {r['subject']:<25} ──{r['predicate']}──▶ {r['object']:<25} "
            f"conf={r['confidence']}  ({r['type']})"
        )

    # ── SECTION 3.7: LLM Extraction Numbers ──────────────────
    lines.append("\n\n" + "─" * 72)
    lines.append("SECTION 3.7: LLM Extraction Numbers")
    lines.append("Copy into: Section 3.7 body text")
    lines.append("─" * 72)
    if llm_stats:
        lines.append(f"LLM model: {llm_stats['llm_model']}")
        lines.append(f"LLM concepts: {llm_stats['llm_concepts']}")
        lines.append(f"LLM relations: {llm_stats['llm_relations']}")
        lines.append(f"LLM attributes: {llm_stats['llm_attributes']}")
        lines.append(f"Extraction time: {llm_stats['extraction_time_seconds']}s")
    else:
        lines.append("LLM extraction was not run (use --use-llm flag)")

    # ── SECTION 3.8: Hybrid Merging Numbers ──────────────────
    lines.append("\n\n" + "─" * 72)
    lines.append("SECTION 3.8: Hybrid Merging Numbers")
    lines.append("Copy into: Section 3.8 body text")
    lines.append("─" * 72)
    lines.append(f"Total merged concepts: {merge_stats['total_merged_concepts']}")
    lines.append(f"Total merged relations: {merge_stats['total_merged_relations']}")
    if llm_stats:
        lines.append(f"Both NLP+LLM: {merge_stats['overlap_both']} ({merge_stats['overlap_pct']})")
        lines.append(f"NLP-only: {merge_stats['nlp_only']} ({merge_stats['nlp_only_pct']})")
        lines.append(f"LLM-only: {merge_stats['llm_only']} ({merge_stats['llm_only_pct']})")

    # ── TABLE 4: Ontology Statistics ─────────────────────────
    lines.append("\n\n" + "─" * 72)
    lines.append("TABLE 4: Ontology Statistics")
    lines.append("Copy into: Section 5.2 (Table 4)")
    lines.append("─" * 72)
    table4_rows = [
        ("OWL Classes", ontology_stats.get("classes", 0)),
        ("Object Properties", ontology_stats.get("object_properties", 0)),
        ("Data Properties", ontology_stats.get("data_properties", 0)),
        ("Individuals", ontology_stats.get("individuals", 0)),
        ("SubClassOf Axioms", ontology_stats.get("subclass_axioms", 0)),
        ("Domain/Range Axioms", ontology_stats.get("domain_range_axioms", 0)),
        ("Total RDF Triples", ontology_stats.get("total_triples", 0)),
        ("Max Hierarchy Depth", ontology_stats.get("max_hierarchy_depth", 0)),
        ("File Size", f"{ontology_stats.get('file_size_kb', 0)} KB"),
    ]
    for label, value in table4_rows:
        lines.append(f"  {label:<30} {value}")

    # ── APPENDIX A: Class Hierarchy ──────────────────────────
    lines.append("\n\n" + "─" * 72)
    lines.append("APPENDIX A: Full Class List")
    lines.append("Copy into: Appendix A (replace placeholder hierarchy)")
    lines.append("─" * 72)
    if "class_names" in ontology_stats:
        for name in sorted(ontology_stats["class_names"]):
            lines.append(f"  {name}")

    # ── SECTION 5.4: Validation Results ──────────────────────
    lines.append("\n\n" + "─" * 72)
    lines.append("SECTION 5.4: Validation Results")
    lines.append("Copy into: Section 5.4 body text")
    lines.append("─" * 72)
    vr = validation_stats["reasoner"]
    lines.append(f"Reasoner consistency: {vr['is_consistent']}")
    lines.append(f"Errors: {vr['errors']}")
    lines.append(f"Warnings: {vr['warnings']}")
    lines.append(f"Inferred subsumptions: {vr['inferred_subsumptions']}")
    if vr["error_messages"]:
        lines.append(f"Error details: {vr['error_messages']}")
    if vr["warning_messages"]:
        lines.append(f"Warning details: {vr['warning_messages']}")

    vs = validation_stats["sparql"]
    lines.append(f"\nSPARQL total triples: {vs['total_triples']}")
    lines.append(f"SPARQL total classes: {vs['total_classes_sparql']}")
    lines.append(f"Hierarchy edges: {vs['hierarchy_edges']}")

    # ── TABLE 3: P/R/F1 (PLACEHOLDER) ────────────────────────
    lines.append("\n\n" + "─" * 72)
    lines.append("TABLE 3: Extraction Performance (P/R/F1)")
    lines.append("⚠️  REQUIRES MANUAL GOLD STANDARD ANNOTATION")
    lines.append("─" * 72)
    lines.append("""
To generate real P/R/F1 numbers for Table 3:

1. Pick 2-3 held-out PDFs (e.g., Chronopoulos 2018, Watson 2024)
2. Have two annotators manually mark every ORNJ concept and relation
3. Run the pipeline on those PDFs
4. Score pipeline output against annotations:

   from sklearn.metrics import precision_recall_fscore_support
   
   # For concepts:
   gold_concepts = set([...])     # manually annotated
   extracted_concepts = set([...]) # from pipeline
   TP = len(gold_concepts & extracted_concepts)
   FP = len(extracted_concepts - gold_concepts)
   FN = len(gold_concepts - extracted_concepts)
   P = TP / (TP + FP)
   R = TP / (TP + FN)
   F1 = 2 * P * R / (P + R)

5. Repeat for relations
6. Fill in Table 3 with real values
""")

    # ── TABLE 6: Expert Evaluation (PLACEHOLDER) ─────────────
    lines.append("\n\n" + "─" * 72)
    lines.append("TABLE 6: Expert Evaluation")
    lines.append("⚠️  REQUIRES MANUAL EXPERT REVIEW")
    lines.append("─" * 72)
    lines.append("""
To generate real expert scores for Table 6:

1. Export the .owl file to Protégé
2. Recruit 3-5 domain experts (radiation oncologists, oral surgeons, etc.)
3. Provide each expert with the evaluation rubric:
   - Taxonomic Accuracy (1-5)
   - Terminological Precision (1-5)
   - Coverage Completeness (1-5)
   - Axiom Correctness (1-5)
   - Clinical Utility (1-5)
   - Cross-system Mapping (1-5)
4. Compute mean scores per criterion
5. Fill in Table 6 with real values
""")

    # ── SUMMARY FOR ABSTRACT ─────────────────────────────────
    lines.append("\n\n" + "─" * 72)
    lines.append("SUMMARY NUMBERS FOR ABSTRACT")
    lines.append("Copy into: Abstract")
    lines.append("─" * 72)
    lines.append(f"The ORN-O ontology comprises {ontology_stats.get('classes', '?')} OWL classes, "
                 f"{ontology_stats.get('object_properties', '?')} object properties, "
                 f"{ontology_stats.get('data_properties', '?')} data properties, and "
                 f"{ontology_stats.get('total_triples', '?')} RDF triples.")
    lines.append(f"Source corpus: {extraction_stats['total_documents']} PDF documents, "
                 f"{extraction_stats['total_pages']} pages, "
                 f"~{extraction_stats['total_words']:,} words.")

    # Write report
    report_text = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(report_text)

    # Write JSON for programmatic access
    all_data = {
        "generated_at": datetime.now().isoformat(),
        "extraction": extraction_stats,
        "preprocessing": preprocessing_stats,
        "concepts": concept_stats,
        "relations": relation_stats,
        "llm": llm_stats,
        "merge": merge_stats,
        "ontology": {k: v for k, v in ontology_stats.items() if k != "class_names"},
        "validation": validation_stats,
    }
    with open(json_path, "w") as f:
        json.dump(all_data, f, indent=2, default=str)

    logger.info(f"\nReport saved to: {report_path}")
    logger.info(f"JSON data saved to: {json_path}")

    # Print the full report
    print("\n" + report_text)

    return report_path


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="ORN-O Pipeline: Generate paper data from ORNJ PDFs"
    )
    parser.add_argument(
        "--pdf-dir",
        required=True,
        help="Directory containing the 16 ORNJ classification PDFs",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory (default: output/)",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Enable Claude API extraction (requires --api-key or ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip reasoner validation (useful if Java not installed)",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "╔" + "═" * 68 + "╗")
    print("║  ORN-O DATA COLLECTION PIPELINE" + " " * 36 + "║")
    print("║  Generating paper data from ORNJ classification PDFs" + " " * 15 + "║")
    print("╚" + "═" * 68 + "╝")
    print(f"\n  PDF directory:  {args.pdf_dir}")
    print(f"  Output:         {args.output_dir}/")
    print(f"  LLM enabled:    {args.use_llm}")
    print(f"  Validation:     {not args.skip_validation}")

    start = time.time()

    # Run phases
    documents, extraction_stats = phase1_extraction(args.pdf_dir)
    segments, preprocessing_stats = phase2_preprocessing(documents)
    concepts, concept_stats = phase3_concepts(segments)
    relations, relation_stats = phase4_relations(segments, concepts)

    llm_knowledge = None
    llm_stats = None
    if args.use_llm:
        api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            llm_knowledge, llm_stats = phase5_llm(documents, api_key)
        else:
            logger.warning("--use-llm specified but no API key found. Skipping.")

    merged_concepts, merged_relations, merge_stats = phase6_merge(
        concepts, relations, llm_knowledge
    )

    builder, ontology_stats = phase7_ontology(
        merged_concepts, merged_relations, llm_knowledge, args.output_dir
    )

    if not args.skip_validation:
        validation_stats = phase8_validation(
            builder, ontology_stats["owl_path"], args.output_dir
        )
    else:
        validation_stats = {
            "reasoner": {"is_consistent": "skipped", "errors": 0, "warnings": 0,
                         "inferred_subsumptions": 0, "error_messages": [], "warning_messages": []},
            "sparql": {"total_triples": 0, "total_classes_sparql": 0, "hierarchy_edges": 0, "class_names": []},
            "graph": {},
        }

    elapsed = time.time() - start

    # Generate report
    report_path = generate_report(
        extraction_stats,
        preprocessing_stats,
        concept_stats,
        relation_stats,
        llm_stats,
        merge_stats,
        ontology_stats,
        validation_stats,
        args.output_dir,
    )

    separator(f"PIPELINE COMPLETE — {elapsed:.0f} seconds")
    print(f"  Report:    {report_path}")
    print(f"  Ontology:  {ontology_stats['owl_path']}")
    print(f"  JSON data: {os.path.join(args.output_dir, 'paper_tables.json')}")
    print(f"  Graph:     {os.path.join(args.output_dir, 'orn_graph.png')}")
    print()


if __name__ == "__main__":
    main()
