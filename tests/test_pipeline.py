"""Tests for the ORN ontology builder pipeline with ORNJ domain data."""

import pytest
import tempfile
import os

from src.preprocessor import TextPreprocessor, TextSegment
from src.concept_extractor import ConceptExtractor, Concept
from src.relation_extractor import RelationExtractor, Relation
from src.ontology_builder import OntologyBuilder


# ── ORNJ Domain Test Data ─────────────────────────────────

ORNJ_TEXT_MARX = """
Marx proposed a three-stage system for osteoradionecrosis in 1983.
Stage I patients exhibit exposed bone in a field of radiation that has
failed to heal for at least 6 months. Stage I patients receive 30
sessions of hyperbaric oxygen therapy at 2.4 atmospheres.
Stage II involves non-responsive disease requiring surgery.
Stage III involves pathological fracture or orocutaneous fistula
requiring resection and reconstruction with free flap.
"""

ORNJ_TEXT_NOTANI = """
Notani et al. divided cases of mandibular osteoradionecrosis into
three grades based on the extent of the lesion. Grade I is defined
as ORN confined to the alveolar bone. Grade II is defined as ORN
limited to the alveolar bone and/or the mandible above the level
of the inferior alveolar canal. Grade III is ORN extending below
the inferior alveolar canal with or without pathological fracture
or orocutaneous fistula.
"""

ORNJ_TEXT_CLINRAD = """
The ClinRad classification system considers both clinical and
radiographic features. Grade 0 represents minor bone spicules.
Grade 1 involves alveolar bone necrosis without bone exposure.
Grade 2 involves alveolar bone necrosis with bone exposure or fistula.
Grade 3 involves basilar bone involvement or maxillary sinus involvement.
Grade 4 involves pathological fracture or orocutaneous fistula.
Treatments such as HBO, PENTOCLO, sequestrectomy, and free flap
reconstruction are recommended based on grade severity.
"""


# ── Preprocessor Tests ────────────────────────────────────

class TestPreprocessor:

    def test_segment_clinical_text(self):
        pp = TextPreprocessor(min_segment_length=10)
        segments = pp.preprocess(ORNJ_TEXT_NOTANI)
        assert len(segments) > 0

    def test_fix_hyphenation(self):
        pp = TextPreprocessor()
        text = "osteoradio-\nnecrosis of the mandible"
        cleaned = pp._clean(text)
        assert "osteoradionecrosis" in cleaned

    def test_heading_detection(self):
        pp = TextPreprocessor()
        assert pp._is_heading("CLASSIFICATION SYSTEMS")
        assert pp._is_heading("3.2 Clinical Staging")
        assert not pp._is_heading("Notani et al. divided cases of mandibular osteoradionecrosis into three grades.")


# ── Concept Extractor Tests ───────────────────────────────

class TestConceptExtractor:

    @pytest.fixture
    def extractor(self):
        return ConceptExtractor(min_frequency=1, max_concepts=200)

    def test_extract_ornj_concepts(self, extractor):
        pp = TextPreprocessor(min_segment_length=10)
        segments = pp.preprocess(ORNJ_TEXT_NOTANI + "\n\n" + ORNJ_TEXT_CLINRAD)
        concepts = extractor.extract(segments)
        assert len(concepts) > 0
        assert all(isinstance(c, Concept) for c in concepts)

    def test_to_class_name(self):
        assert ConceptExtractor._to_class_name("bone exposure") == "BoneExposure"
        assert ConceptExtractor._to_class_name("pathological fracture") == "PathologicalFracture"
        assert ConceptExtractor._to_class_name("inferior alveolar canal") == "InferiorAlveolarCanal"


# ── Relation Extractor Tests ──────────────────────────────

class TestRelationExtractor:

    @pytest.fixture
    def extractor(self):
        return RelationExtractor(min_confidence=0.0)

    def test_hearst_patterns_ornj(self, extractor):
        text = "Treatments such as HBO, PENTOCLO, and sequestrectomy are used for ORN."
        relations = extractor._hearst_patterns(text)
        is_a_rels = [r for r in relations if r.relation_type == "is_a"]
        assert len(is_a_rels) > 0

    def test_dependency_triples_ornj(self, extractor):
        text = "Osteoradionecrosis causes pathological fracture."
        relations = extractor._dependency_triples(text, set())
        assert len(relations) > 0


# ── Ontology Builder Tests ────────────────────────────────

class TestOntologyBuilder:

    def test_build_ornj_ontology(self):
        builder = OntologyBuilder(iri="http://test.org/orn-test")

        # ORNJ classification concepts
        concepts = [
            Concept(name="ORNJ", label="Osteoradionecrosis of the Jaw"),
            Concept(name="ClassificationSystem", label="Classification System"),
            Concept(name="NotaniClassification", label="Notani Classification", parent="ClassificationSystem"),
            Concept(name="NotaniGradeI", label="Notani Grade I", parent="NotaniClassification"),
            Concept(name="NotaniGradeII", label="Notani Grade II", parent="NotaniClassification"),
            Concept(name="NotaniGradeIII", label="Notani Grade III", parent="NotaniClassification"),
            Concept(name="ClinicalFinding", label="Clinical Finding"),
            Concept(name="BoneExposure", label="Bone Exposure", parent="ClinicalFinding"),
            Concept(name="PathologicalFracture", label="Pathological Fracture", parent="ClinicalFinding"),
            Concept(name="OrocutaneousFistula", label="Orocutaneous Fistula", parent="ClinicalFinding"),
            Concept(name="AnatomicalStructure", label="Anatomical Structure"),
            Concept(name="AlveolarBone", label="Alveolar Bone", parent="AnatomicalStructure"),
            Concept(name="InferiorAlveolarCanal", label="Inferior Alveolar Canal", parent="AnatomicalStructure"),
            Concept(name="TreatmentModality", label="Treatment Modality"),
            Concept(name="HyperbaricOxygenTherapy", label="Hyperbaric Oxygen Therapy", parent="TreatmentModality"),
        ]
        builder.add_concepts(concepts)

        # Classification relations
        relations = [
            Relation(subject="NotaniGradeI", predicate="hasAffectedStructure", object="AlveolarBone"),
            Relation(subject="NotaniGradeIII", predicate="hasClinicalFinding", object="PathologicalFracture"),
            Relation(subject="NotaniGradeIII", predicate="hasClinicalFinding", object="OrocutaneousFistula"),
            Relation(subject="HyperbaricOxygenTherapy", predicate="isRecommendedFor", object="ORNJ"),
        ]
        builder.add_relations(relations)

        stats = builder.get_stats()
        assert stats["classes"] >= 14
        assert stats["object_properties"] >= 2
        assert "NotaniGradeI" in stats["class_names"]
        assert "BoneExposure" in stats["class_names"]

    def test_save_ornj_ontology(self):
        builder = OntologyBuilder(iri="http://test.org/orn-save")
        concepts = [
            Concept(name="ORNJ", label="Osteoradionecrosis"),
            Concept(name="MarxStageI", label="Marx Stage I"),
            Concept(name="MarxStageII", label="Marx Stage II"),
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
                {"name": "ClinRadClassification", "label": "ClinRad Classification", "parent": "ClassificationSystem", "definition": "Integrated clinical-radiographic ORNJ staging"},
                {"name": "ClinRadGrade0", "label": "ClinRad Grade 0", "parent": "ClinRadClassification", "definition": "Minor bone spicule"},
                {"name": "ClinRadGrade4", "label": "ClinRad Grade 4", "parent": "ClinRadClassification", "definition": "Pathological fracture or fistula"},
            ],
            relations=[
                {"subject": "ClinRadGrade4", "predicate": "hasClinicalFinding", "object": "PathologicalFracture"},
            ],
            attributes=[
                {"concept": "ClinRadGrade0", "attribute": "hasDescription", "value_type": "string"},
            ],
        )
        builder = OntologyBuilder(iri="http://test.org/orn-llm")
        builder.from_llm_output(knowledge)
        stats = builder.get_stats()
        assert stats["classes"] >= 3


# ── Integration Test ──────────────────────────────────────

class TestIntegration:

    def test_ornj_pipeline_end_to_end(self):
        """End-to-end: ORNJ clinical text → concepts → relations → ontology."""
        text = ORNJ_TEXT_NOTANI + "\n\n" + ORNJ_TEXT_CLINRAD + "\n\n" + ORNJ_TEXT_MARX

        pp = TextPreprocessor(min_segment_length=10)
        segments = pp.preprocess(text)
        assert len(segments) > 0

        ce = ConceptExtractor(min_frequency=1)
        concepts = ce.extract(segments)
        assert len(concepts) > 0

        re_ = RelationExtractor(min_confidence=0.0)
        relations = re_.extract(segments, concepts)

        builder = OntologyBuilder(iri="http://test.org/orn-integration")
        builder.add_concepts(concepts)
        if relations:
            builder.add_relations(relations)

        stats = builder.get_stats()
        assert stats["classes"] > 0

        with tempfile.NamedTemporaryFile(suffix=".owl", delete=False) as f:
            filepath = f.name
        try:
            builder.save(filepath)
            assert os.path.getsize(filepath) > 0
        finally:
            os.unlink(filepath)
