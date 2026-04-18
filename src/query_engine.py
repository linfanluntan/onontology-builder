"""
SPARQL Query Engine
Query OWL ontologies using SPARQL via rdflib.
"""

import logging
from pathlib import Path

from rdflib import Graph, Namespace, RDF, RDFS, OWL

logger = logging.getLogger(__name__)


class QueryEngine:
    """SPARQL query interface for ontologies."""

    def __init__(self, ontology_path: str | None = None):
        self.graph = Graph()
        if ontology_path:
            self.load(ontology_path)

    def load(self, filepath: str) -> None:
        """Load an ontology file (OWL/RDF/Turtle) into the graph."""
        filepath = str(Path(filepath).resolve())
        fmt = self._detect_format(filepath)
        self.graph.parse(filepath, format=fmt)
        logger.info(
            f"Loaded ontology: {len(self.graph)} triples from {filepath}"
        )

    def query(self, sparql: str) -> list[dict]:
        """Execute a SPARQL query and return results as list of dicts."""
        results = self.graph.query(sparql)
        rows = []
        for row in results:
            row_dict = {}
            for var in results.vars:
                val = getattr(row, str(var), None)
                row_dict[str(var)] = str(val) if val else None
            rows.append(row_dict)
        return rows

    def get_all_classes(self) -> list[dict]:
        """Get all OWL classes with labels."""
        sparql = """
        SELECT ?class ?label WHERE {
            ?class a owl:Class .
            OPTIONAL { ?class rdfs:label ?label }
        }
        ORDER BY ?class
        """
        return self.query(sparql)

    def get_class_hierarchy(self) -> list[dict]:
        """Get the full class hierarchy (subclass relations)."""
        sparql = """
        SELECT ?child ?parent WHERE {
            ?child rdfs:subClassOf ?parent .
            FILTER(?parent != owl:Thing)
            FILTER(isURI(?parent))
            FILTER(isURI(?child))
        }
        ORDER BY ?parent ?child
        """
        return self.query(sparql)

    def get_properties_for_class(self, class_uri: str) -> list[dict]:
        """Get all properties whose domain is the given class."""
        sparql = f"""
        SELECT ?property ?range WHERE {{
            ?property rdfs:domain <{class_uri}> .
            OPTIONAL {{ ?property rdfs:range ?range }}
        }}
        """
        return self.query(sparql)

    def get_instances(self, class_uri: str | None = None) -> list[dict]:
        """Get all instances, optionally filtered by class."""
        if class_uri:
            sparql = f"""
            SELECT ?instance ?class WHERE {{
                ?instance a <{class_uri}> .
                ?instance a ?class .
            }}
            """
        else:
            sparql = """
            SELECT ?instance ?class WHERE {
                ?instance a ?class .
                ?class a owl:Class .
            }
            """
        return self.query(sparql)

    def get_triples_about(self, subject_uri: str) -> list[dict]:
        """Get all triples where the given URI is the subject."""
        sparql = f"""
        SELECT ?predicate ?object WHERE {{
            <{subject_uri}> ?predicate ?object .
        }}
        """
        return self.query(sparql)

    def count_triples(self) -> int:
        """Count total triples in the graph."""
        return len(self.graph)

    def custom_query(self, sparql: str) -> list[dict]:
        """Run any custom SPARQL query. Alias for query()."""
        return self.query(sparql)

    @staticmethod
    def _detect_format(filepath: str) -> str:
        """Detect RDF serialization format from file extension."""
        ext = Path(filepath).suffix.lower()
        formats = {
            ".owl": "xml",
            ".rdf": "xml",
            ".xml": "xml",
            ".ttl": "turtle",
            ".turtle": "turtle",
            ".nt": "nt",
            ".n3": "n3",
            ".jsonld": "json-ld",
        }
        return formats.get(ext, "xml")
