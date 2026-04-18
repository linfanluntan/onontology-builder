"""
Ontology Builder Module
Constructs OWL/RDF ontologies from extracted concepts and relations using Owlready2.
"""

import os
import logging
from dataclasses import dataclass

from owlready2 import (
    get_ontology,
    Thing,
    ObjectProperty,
    DataProperty,
    FunctionalProperty,
    types as owl_types,
)

logger = logging.getLogger(__name__)

# Map string type names to Python/OWL data types
DATATYPE_MAP = {
    "string": str,
    "str": str,
    "integer": int,
    "int": int,
    "float": float,
    "boolean": bool,
    "bool": bool,
}


class OntologyBuilder:
    """Build OWL ontologies programmatically from extracted knowledge."""

    def __init__(
        self,
        iri: str = "http://example.org/ontology",
        name: str = "MyOntology",
    ):
        self.iri = iri.rstrip("/") + "/"
        self.onto = get_ontology(self.iri)
        self.name = name
        self._classes = {}
        self._properties = {}

    def add_concepts(self, concepts) -> None:
        """Add Concept objects as OWL classes."""
        with self.onto:
            for concept in concepts:
                name = concept.name if hasattr(concept, "name") else str(concept)
                label = concept.label if hasattr(concept, "label") else name
                parent_name = concept.parent if hasattr(concept, "parent") else ""

                parent_cls = self._classes.get(parent_name, Thing)
                cls = self._get_or_create_class(name, parent_cls)

                # Add label annotation
                cls.label = [label]

                # Add synonyms as altLabel if available
                if hasattr(concept, "synonyms"):
                    for syn in concept.synonyms:
                        cls.comment.append(f"synonym: {syn}")

        logger.info(f"Added {len(concepts)} concept classes")

    def add_relations(self, relations) -> None:
        """Add Relation objects as OWL object properties and axioms."""
        with self.onto:
            for rel in relations:
                subj_name = rel.subject if hasattr(rel, "subject") else rel.get("subject", "")
                pred_name = rel.predicate if hasattr(rel, "predicate") else rel.get("predicate", "")
                obj_name = rel.object if hasattr(rel, "object") else rel.get("object", "")
                rel_type = getattr(rel, "relation_type", rel.get("relation_type", "object_property"))

                if not (subj_name and pred_name and obj_name):
                    continue

                subj_cls = self._get_or_create_class(subj_name)
                obj_cls = self._get_or_create_class(obj_name)

                if rel_type == "is_a":
                    # Make subject a subclass of object
                    if obj_cls not in subj_cls.is_a:
                        subj_cls.is_a.append(obj_cls)
                else:
                    # Create object property
                    prop = self._get_or_create_property(
                        pred_name, subj_cls, obj_cls
                    )

        logger.info(f"Added {len(relations)} relations")

    def add_data_properties(self, attributes: list[dict]) -> None:
        """Add data properties (attributes) to classes."""
        with self.onto:
            for attr in attributes:
                concept_name = attr.get("concept", "")
                attr_name = attr.get("attribute", "")
                value_type = attr.get("value_type", "string")

                if not (concept_name and attr_name):
                    continue

                cls = self._get_or_create_class(concept_name)
                py_type = DATATYPE_MAP.get(value_type, str)

                # Create data property
                prop_name = attr_name
                if prop_name not in self._properties:
                    prop = type(
                        prop_name,
                        (DataProperty,),
                        {
                            "namespace": self.onto,
                            "domain": [cls],
                            "range": [py_type],
                        },
                    )
                    self._properties[prop_name] = prop

        logger.info(f"Added {len(attributes)} data properties")

    def add_instances(self, instances: list[dict]) -> None:
        """Add instances (individuals) to the ontology.

        Each instance dict should have: class_name, instance_name, and
        optionally properties (dict of property_name -> value).
        """
        with self.onto:
            for inst in instances:
                class_name = inst.get("class_name", "")
                inst_name = inst.get("instance_name", "")
                props = inst.get("properties", {})

                if not (class_name and inst_name):
                    continue

                cls = self._get_or_create_class(class_name)
                individual = cls(inst_name)

                for prop_name, value in props.items():
                    if hasattr(individual, prop_name):
                        setattr(individual, prop_name, value)

    def from_llm_output(self, knowledge) -> None:
        """Build ontology from LLMKnowledge output."""
        # Add concepts
        with self.onto:
            for c in knowledge.concepts:
                name = c.get("name", "")
                parent = c.get("parent", "")
                definition = c.get("definition", "")

                if not name:
                    continue

                parent_cls = self._classes.get(parent, Thing)
                cls = self._get_or_create_class(name, parent_cls)
                cls.label = [c.get("label", name)]
                if definition:
                    cls.comment = [definition]

        # Add relations
        self.add_relations(knowledge.relations)

        # Add attributes
        if knowledge.attributes:
            self.add_data_properties(knowledge.attributes)

    def save(self, filepath: str, format: str = "rdfxml") -> None:
        """Save the ontology to a file."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        self.onto.save(file=filepath, format=format)
        logger.info(f"Ontology saved to {filepath}")

    def get_stats(self) -> dict:
        """Return ontology statistics."""
        classes = list(self.onto.classes())
        props = list(self.onto.object_properties()) + list(
            self.onto.data_properties()
        )
        individuals = list(self.onto.individuals())
        return {
            "classes": len(classes),
            "object_properties": len(list(self.onto.object_properties())),
            "data_properties": len(list(self.onto.data_properties())),
            "individuals": len(individuals),
            "class_names": [c.name for c in classes],
        }

    # ── Internal helpers ──────────────────────────────────────────

    def _get_or_create_class(self, name: str, parent=None):
        """Get existing class or create a new one."""
        if name in self._classes:
            return self._classes[name]

        parent = parent or Thing
        with self.onto:
            cls = type(name, (parent,), {"namespace": self.onto})
        self._classes[name] = cls
        return cls

    def _get_or_create_property(self, name: str, domain_cls, range_cls):
        """Get existing object property or create a new one."""
        if name in self._properties:
            return self._properties[name]

        with self.onto:
            prop = type(
                name,
                (ObjectProperty,),
                {
                    "namespace": self.onto,
                    "domain": [domain_cls],
                    "range": [range_cls],
                },
            )
        self._properties[name] = prop
        return prop
