"""
Relation Extraction Module
Extracts semantic relations (triples) from text using patterns and dependency parsing.
"""

import re
import logging
from dataclasses import dataclass, field

import spacy

logger = logging.getLogger(__name__)


@dataclass
class Relation:
    """A semantic relation (triple) between two concepts."""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source_text: str = ""
    relation_type: str = "object_property"  # object_property, data_property, is_a

    def __hash__(self):
        return hash((self.subject, self.predicate, self.object))

    def __eq__(self, other):
        return (
            isinstance(other, Relation)
            and self.subject == other.subject
            and self.predicate == other.predicate
            and self.object == other.object
        )

    def as_tuple(self):
        return (self.subject, self.predicate, self.object)


# Hearst patterns for hypernym (is-a) extraction
HEARST_PATTERNS = [
    # "X such as Y, Z, and W"
    (
        r"(\b[A-Z][a-z]+(?:\s+[a-z]+)*)\s+such\s+as\s+((?:[A-Z][a-z]+(?:\s+[a-z]+)*(?:,\s*)?)+(?:\s+and\s+[A-Z][a-z]+(?:\s+[a-z]+)*)?)",
        "hypernym_first",
    ),
    # "Y, Z, and other X"
    (
        r"((?:[A-Z][a-z]+(?:\s+[a-z]+)*(?:,\s*)?)+(?:\s+and\s+)?)\s+(?:and\s+)?other\s+(\b[a-z]+(?:\s+[a-z]+)*)",
        "hypernym_last",
    ),
    # "X, including Y and Z"
    (
        r"(\b[A-Z][a-z]+(?:\s+[a-z]+)*)\s*,?\s+including\s+((?:[A-Z][a-z]+(?:\s+[a-z]+)*(?:,\s*)?)+(?:\s+and\s+[A-Z][a-z]+(?:\s+[a-z]+)*)?)",
        "hypernym_first",
    ),
    # "X, especially Y"
    (
        r"(\b[A-Z][a-z]+(?:\s+[a-z]+)*)\s*,?\s+especially\s+(\b[A-Z][a-z]+(?:\s+[a-z]+)*)",
        "hypernym_first",
    ),
    # "X is a (type|kind|form) of Y"
    (
        r"(\b[A-Z][a-z]+(?:\s+[a-z]+)*)\s+is\s+a\s+(?:type|kind|form|sort)\s+of\s+(\b[a-z]+(?:\s+[a-z]+)*)",
        "hypernym_last",
    ),
]


class RelationExtractor:
    """Extract semantic relations from text using multiple strategies."""

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        use_hearst: bool = True,
        use_dependency: bool = True,
        min_confidence: float = 0.3,
    ):
        self.nlp = spacy.load(spacy_model)
        self.use_hearst = use_hearst
        self.use_dependency = use_dependency
        self.min_confidence = min_confidence

    def extract(self, segments, concepts: list | None = None) -> list[Relation]:
        """Extract relations from text segments."""
        concept_names = set()
        if concepts:
            for c in concepts:
                concept_names.add(c.label.lower())
                concept_names.add(c.name.lower())

        all_relations = []

        for seg in segments:
            text = seg.text if hasattr(seg, "text") else str(seg)

            if self.use_hearst:
                hearst_rels = self._hearst_patterns(text)
                all_relations.extend(hearst_rels)

            if self.use_dependency:
                dep_rels = self._dependency_triples(text, concept_names)
                all_relations.extend(dep_rels)

        # Deduplicate
        unique = list(set(all_relations))

        # Filter by confidence
        filtered = [r for r in unique if r.confidence >= self.min_confidence]

        logger.info(
            f"Extracted {len(filtered)} relations "
            f"({len(all_relations)} raw, {len(unique)} unique)"
        )
        return filtered

    def _hearst_patterns(self, text: str) -> list[Relation]:
        """Extract is-a (hypernym) relations using Hearst patterns."""
        relations = []
        for pattern, direction in HEARST_PATTERNS:
            for match in re.finditer(pattern, text):
                if direction == "hypernym_first":
                    hypernym = match.group(1).strip()
                    hyponyms_raw = match.group(2).strip()
                else:
                    hyponyms_raw = match.group(1).strip()
                    hypernym = match.group(2).strip()

                # Split comma/and-separated hyponyms
                hyponyms = re.split(r",\s*|\s+and\s+", hyponyms_raw)
                for hypo in hyponyms:
                    hypo = hypo.strip()
                    if hypo and len(hypo) > 1:
                        relations.append(Relation(
                            subject=self._to_class_name(hypo),
                            predicate="isA",
                            object=self._to_class_name(hypernym),
                            confidence=0.8,
                            source_text=match.group(0),
                            relation_type="is_a",
                        ))
        return relations

    def _dependency_triples(
        self, text: str, concept_names: set[str]
    ) -> list[Relation]:
        """Extract SVO triples from dependency parse."""
        doc = self.nlp(text)
        relations = []

        for sent in doc.sents:
            for token in sent:
                # Look for verbs with subject and object
                if token.pos_ != "VERB":
                    continue

                subjects = []
                objects = []

                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        subjects.append(self._get_full_span(child))
                    elif child.dep_ in ("dobj", "attr", "pobj"):
                        objects.append(self._get_full_span(child))
                    elif child.dep_ == "prep":
                        # Prepositional object: "X verb prep Y"
                        for pobj in child.children:
                            if pobj.dep_ == "pobj":
                                objects.append(self._get_full_span(pobj))

                for subj in subjects:
                    for obj in objects:
                        if len(subj) > 1 and len(obj) > 1:
                            # Boost confidence if both are known concepts
                            conf = 0.5
                            if concept_names:
                                if subj.lower() in concept_names:
                                    conf += 0.2
                                if obj.lower() in concept_names:
                                    conf += 0.2

                            relations.append(Relation(
                                subject=self._to_class_name(subj),
                                predicate=self._to_predicate_name(token.lemma_),
                                object=self._to_class_name(obj),
                                confidence=conf,
                                source_text=sent.text.strip(),
                                relation_type="object_property",
                            ))

        return relations

    @staticmethod
    def _get_full_span(token) -> str:
        """Get the full noun phrase for a token by traversing its subtree."""
        subtree = list(token.subtree)
        # Filter to relevant POS tags
        relevant = [
            t for t in subtree
            if t.pos_ in ("NOUN", "PROPN", "ADJ", "NUM")
            or t.dep_ == "compound"
        ]
        if relevant:
            start = min(t.i for t in relevant)
            end = max(t.i for t in relevant) + 1
            return token.doc[start:end].text
        return token.text

    @staticmethod
    def _to_class_name(name: str) -> str:
        name = re.sub(r"[^a-zA-Z0-9\s]", "", name)
        parts = name.strip().split()
        return "".join(word.capitalize() for word in parts if word)

    @staticmethod
    def _to_predicate_name(verb: str) -> str:
        verb = re.sub(r"[^a-zA-Z]", "", verb)
        return verb.lower() if verb else "relatedTo"
