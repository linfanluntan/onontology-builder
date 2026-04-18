"""
Concept & Term Extraction Module
Extracts domain concepts, named entities, and key terms from text using NLP.
"""

import logging
from collections import Counter
from dataclasses import dataclass, field

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


@dataclass
class Concept:
    """A domain concept extracted from text."""
    name: str
    label: str  # Human-readable label
    concept_type: str = "class"  # class, instance, attribute
    frequency: int = 1
    source_texts: list[str] = field(default_factory=list)
    synonyms: list[str] = field(default_factory=list)
    parent: str = ""  # Parent concept for taxonomy

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, Concept) and self.name == other.name


class ConceptExtractor:
    """Extract concepts and terms from text segments using NLP."""

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        min_frequency: int = 2,
        max_concepts: int = 500,
        tfidf_top_n: int = 200,
        entity_types: list[str] | None = None,
    ):
        self.nlp = spacy.load(spacy_model)
        self.min_frequency = min_frequency
        self.max_concepts = max_concepts
        self.tfidf_top_n = tfidf_top_n
        self.entity_types = entity_types or [
            "ORG", "PERSON", "GPE", "PRODUCT", "EVENT",
            "WORK_OF_ART", "LAW", "NORP", "FAC",
        ]

    def extract(self, segments) -> list[Concept]:
        """Extract concepts from a list of TextSegment objects."""
        texts = [seg.text for seg in segments]
        full_text = " ".join(texts)

        # 1. Named Entity Recognition
        ner_concepts = self._extract_entities(full_text)

        # 2. Noun phrase extraction
        np_concepts = self._extract_noun_phrases(full_text)

        # 3. TF-IDF key terms
        tfidf_terms = self._extract_tfidf_terms(texts)

        # 4. Merge and deduplicate
        all_concepts = self._merge_concepts(ner_concepts, np_concepts, tfidf_terms)

        # 5. Filter by frequency
        filtered = [
            c for c in all_concepts if c.frequency >= self.min_frequency
        ]

        # 6. Sort by frequency and cap
        filtered.sort(key=lambda c: c.frequency, reverse=True)
        filtered = filtered[: self.max_concepts]

        logger.info(f"Extracted {len(filtered)} concepts (from {len(all_concepts)} candidates)")
        return filtered

    def _extract_entities(self, text: str) -> list[Concept]:
        """Extract named entities using spaCy NER."""
        doc = self.nlp(text)
        entity_counts = Counter()
        entity_types_map = {}

        for ent in doc.ents:
            if ent.label_ in self.entity_types:
                name = ent.text.strip()
                if len(name) > 2:
                    normalized = self._normalize_name(name)
                    entity_counts[normalized] += 1
                    entity_types_map[normalized] = ent.label_

        concepts = []
        for name, count in entity_counts.items():
            concepts.append(Concept(
                name=self._to_class_name(name),
                label=name,
                concept_type="class",
                frequency=count,
            ))
        return concepts

    def _extract_noun_phrases(self, text: str) -> list[Concept]:
        """Extract noun phrases using spaCy chunking."""
        doc = self.nlp(text)
        np_counts = Counter()

        for chunk in doc.noun_chunks:
            # Filter out pronouns, determiners-only, very short
            text_clean = chunk.text.strip()
            if len(text_clean) > 3 and chunk.root.pos_ in ("NOUN", "PROPN"):
                normalized = self._normalize_name(text_clean)
                if len(normalized) > 2:
                    np_counts[normalized] += 1

        concepts = []
        for name, count in np_counts.most_common(self.max_concepts):
            concepts.append(Concept(
                name=self._to_class_name(name),
                label=name,
                concept_type="class",
                frequency=count,
            ))
        return concepts

    def _extract_tfidf_terms(self, texts: list[str]) -> list[Concept]:
        """Extract important terms using TF-IDF scoring."""
        if len(texts) < 2:
            return []

        vectorizer = TfidfVectorizer(
            max_features=self.tfidf_top_n,
            ngram_range=(1, 3),
            stop_words="english",
            min_df=2 if len(texts) > 5 else 1,
            max_df=0.9,
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
        except ValueError:
            return []

        feature_names = vectorizer.get_feature_names_out()

        # Sum TF-IDF scores across all documents for each term
        scores = tfidf_matrix.sum(axis=0).A1
        ranked = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)

        concepts = []
        for term, score in ranked[: self.tfidf_top_n]:
            if len(term) > 2:
                concepts.append(Concept(
                    name=self._to_class_name(term),
                    label=term,
                    concept_type="class",
                    frequency=int(score * 10),  # Approximate frequency from score
                ))
        return concepts

    def _merge_concepts(self, *concept_lists) -> list[Concept]:
        """Merge multiple concept lists, combining duplicates."""
        merged = {}
        for concepts in concept_lists:
            for concept in concepts:
                key = concept.name.lower()
                if key in merged:
                    merged[key].frequency += concept.frequency
                    if concept.label not in merged[key].synonyms:
                        merged[key].synonyms.append(concept.label)
                else:
                    merged[key] = concept
        return list(merged.values())

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize a concept name for deduplication."""
        import re
        name = re.sub(r"\s+", " ", name).strip()
        # Remove leading determiners
        name = re.sub(r"^(the|a|an)\s+", "", name, flags=re.IGNORECASE)
        return name

    @staticmethod
    def _to_class_name(name: str) -> str:
        """Convert a natural language name to a PascalCase class name."""
        import re
        # Remove special characters
        name = re.sub(r"[^a-zA-Z0-9\s]", "", name)
        parts = name.strip().split()
        return "".join(word.capitalize() for word in parts if word)
