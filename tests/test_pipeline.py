"""Basic tests for the ontology builder pipeline."""

import pytest
import tempfile
import os

from src.preprocessor import TextPreprocessor, TextSegment
from src.concept_extractor import ConceptExtractor, Concept
from src.relation_extractor import RelationExtractor, Relation
from src.ontology_builder import OntologyBuilder


# ── Preprocessor Tests ────────────────────────────────────────

class TestPreprocessor:

    def test_clean_whitespace(self):
        pp = TextPreprocessor()
        text = "Hello    world\n\n\n\n\nTest"
        cleaned = pp._clean(text)
        assert "    " not in cleaned
        assert "\n\n\n" not in cleaned

    def test_fix_hyphenation(self):
        pp = TextPreprocessor()
        text = "This is a hyphen-\nated word."
        cleaned = pp._clean(text)
        assert "hyphenated" in cleaned

    def test_segment_paragraphs(self):
        pp = TextPreprocessor(min_segment_length=5)
        text = "First paragraph here.\n\nSecond paragraph here."
        segments = pp.preprocess(text)
        assert len(segments) == 2

    def test_heading_detection(self):
        pp = TextPreprocessor()
        assert pp._is_heading("INTRODUCTION")
        assert pp._is_heading("1.2 Methods and Materials")
        assert pp._is_heading("Chapter 3 Results")
        assert not pp._is_heading("This is a normal paragraph with many words.")


# ── Concept Extractor Tests ───────────────────────────────────

class TestConceptExtractor:

    @pytest.fixture
    def extractor(self):
        return ConceptExtractor(min_frequency=1, max_concepts=100)

    def test_normalize_name(self):
        assert ConceptExtractor._normalize_name("  The  Big  Dog  ") == "Big Dog"

    def test_to_class_name(self):
        assert ConceptExtractor._to_class_name("machine learning") == "MachineLearning"
        assert ConceptExtractor._to_class_name("NLP") == "Nlp"

    def test_extract_returns_concepts(self, extractor):
        segments = [
            TextSegment(text="Artificial intelligence and machine learning are transforming healthcare. "
                            "Deep learning models process medical images effectively."),
            TextSegment(text="Machine learning algorithms improve drug discovery. "
                            "Artificial intelligence helps diagnose diseases."),
        ]
        concepts = extractor.extract(segments)
        assert len(concepts) > 0
        assert all(isinstance(c, Concept) for c in concepts)


# ── Relation Extractor Tests ──────────────────────────────────

class TestRelationExtractor:

    @pytest.fixture
    def extractor(self):
        return RelationExtractor(min_confidence=0.0)

    def test_hearst_such_as(self, extractor):
        text = "Animals such as Dogs, Cats, and Birds are common pets."
        relations = extractor._hearst_patterns(text)
        is_a_rels = [r for r in relations if r.relation_type == "is_a"]
        assert len(is_a_rels) > 0

    def test_dependency_triples(self, extractor):
        text = "Aspirin treats headaches effectively."
        relations = extractor._dependency_triples(text, set())
        assert len(relations) > 0

    def test_to_predicate_name(self):
        assert RelationExtractor._to_predicate_name("treats") == "treats"
        assert RelationExtractor._to_predicate_name("is-used-for") == "isusedfor"


# ── Ontology Builder Tests ────────────────────────────────────

class TestOntologyBuilder:

    def test_create_ontology(self):
        builder = OntologyBuilder(iri="http://test.org/onto")
        stats = builder.get_stats()
        assert stats["classes"] == 0

    def test_add_concepts(self):
        builder = OntologyBuilder(iri="http://test.org/onto2")
        concepts = [
            Concept(name="Drug", label="Drug"),
            Concept(name="Disease", label="Disease"),
        ]
        builder.add_concepts(concepts)
        stats = builder.get_stats()
        assert stats["classes"] == 2
        assert "Drug" in stats["class_names"]

    def test_add_relations(self):
        builder = OntologyBuilder(iri="http://test.org/onto3")
        concepts = [
            Concept(name="Drug", label="Drug"),
            Concept(name="Disease", label="Disease"),
        ]
        builder.add_concepts(concepts)

        relations = [
            Relation(subject="Drug", predicate="treats", object="Disease"),
        ]
        builder.add_relations(relations)
        stats = builder.get_stats()
        assert stats["object_properties"] >= 1

    def test_save_and_load(self):
        builder = OntologyBuilder(iri="http://test.org/onto4")
        concepts = [
            Concept(name="Animal", label="Animal"),
            Concept(name="Dog", label="Dog", parent="Animal"),
        ]
        builder.add_concepts(concepts)

        with tempfile.NamedTemporaryFile(suffix=".owl", delete=False) as f:
            filepath = f.name

        try:
            builder.save(filepath)
            assert os.path.exists(filepath)
            assert os.path.getsize(filepath) > 0
        finally:
            os.unlink(filepath)

    def test_from_llm_output(self):
        from src.llm_extractor import LLMKnowledge
        knowledge = LLMKnowledge(
            concepts=[
                {"name": "Vehicle", "label": "Vehicle", "parent": "", "definition": "A means of transport"},
                {"name": "Car", "label": "Car", "parent": "Vehicle", "definition": "A road vehicle"},
            ],
            relations=[
                {"subject": "Car", "predicate": "hasEngine", "object": "Engine"},
            ],
            attributes=[
                {"concept": "Car", "attribute": "hasColor", "value_type": "string"},
            ],
        )
        builder = OntologyBuilder(iri="http://test.org/onto5")
        builder.from_llm_output(knowledge)
        stats = builder.get_stats()
        assert stats["classes"] >= 2


# ── Integration Test ──────────────────────────────────────────

class TestIntegration:

    def test_nlp_pipeline_end_to_end(self):
        """End-to-end test: text -> concepts -> relations -> ontology -> save."""
        text = (
            "Aspirin is a drug that treats headaches and inflammation. "
            "Ibuprofen is another drug that reduces pain. "
            "Drugs such as Aspirin, Ibuprofen, and Acetaminophen are analgesics. "
            "Headaches are a type of pain disorder."
        )

        # Preprocess
        pp = TextPreprocessor(min_segment_length=5)
        segments = pp.preprocess(text)
        assert len(segments) > 0

        # Extract concepts
        ce = ConceptExtractor(min_frequency=1)
        concepts = ce.extract(segments)
        assert len(concepts) > 0

        # Extract relations
        re_ = RelationExtractor(min_confidence=0.0)
        relations = re_.extract(segments, concepts)

        # Build ontology
        builder = OntologyBuilder(iri="http://test.org/integration")
        builder.add_concepts(concepts)
        if relations:
            builder.add_relations(relations)

        stats = builder.get_stats()
        assert stats["classes"] > 0

        # Save
        with tempfile.NamedTemporaryFile(suffix=".owl", delete=False) as f:
            filepath = f.name
        try:
            builder.save(filepath)
            assert os.path.getsize(filepath) > 0
        finally:
            os.unlink(filepath)
