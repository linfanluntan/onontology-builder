"""
Ontology Visualization Module
Generate visual representations of ontology graphs using NetworkX and Matplotlib.
"""

import logging
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt
from rdflib import Graph, RDF, RDFS, OWL

logger = logging.getLogger(__name__)


class OntologyVisualizer:
    """Visualize ontology structures as graphs."""

    def __init__(self, ontology_path: str | None = None):
        self.rdf_graph = Graph()
        self.nx_graph = nx.DiGraph()
        if ontology_path:
            self.load(ontology_path)

    def load(self, filepath: str) -> None:
        """Load ontology and build NetworkX graph."""
        ext = Path(filepath).suffix.lower()
        fmt = "xml" if ext in (".owl", ".rdf", ".xml") else "turtle"
        self.rdf_graph.parse(filepath, format=fmt)
        self._build_nx_graph()

    def _build_nx_graph(self) -> None:
        """Convert RDF graph to NetworkX directed graph."""
        self.nx_graph = nx.DiGraph()

        # Add class hierarchy edges
        for s, p, o in self.rdf_graph.triples((None, RDFS.subClassOf, None)):
            if s != OWL.Thing and o != OWL.Thing:
                child = self._short_name(s)
                parent = self._short_name(o)
                if child and parent:
                    self.nx_graph.add_edge(child, parent, relation="subClassOf")

        # Add object property edges
        for prop in self.rdf_graph.subjects(RDF.type, OWL.ObjectProperty):
            prop_name = self._short_name(prop)
            for domain in self.rdf_graph.objects(prop, RDFS.domain):
                for range_ in self.rdf_graph.objects(prop, RDFS.range):
                    d = self._short_name(domain)
                    r = self._short_name(range_)
                    if d and r:
                        self.nx_graph.add_edge(d, r, relation=prop_name)

        # Add isolated classes (no edges)
        for cls in self.rdf_graph.subjects(RDF.type, OWL.Class):
            name = self._short_name(cls)
            if name and name not in self.nx_graph:
                self.nx_graph.add_node(name)

    def plot_hierarchy(
        self,
        figsize: tuple = (16, 12),
        save_path: str | None = None,
        title: str = "Ontology Class Hierarchy",
    ) -> None:
        """Plot the class hierarchy as a tree-like graph."""
        if not self.nx_graph.nodes:
            logger.warning("No nodes to plot.")
            return

        fig, ax = plt.subplots(figsize=figsize)

        # Use hierarchical layout if possible
        try:
            # Filter to only subClassOf edges for hierarchy
            hierarchy = nx.DiGraph()
            for u, v, data in self.nx_graph.edges(data=True):
                if data.get("relation") == "subClassOf":
                    hierarchy.add_edge(u, v)
            for node in self.nx_graph.nodes:
                if node not in hierarchy:
                    hierarchy.add_node(node)

            if hierarchy.edges:
                pos = nx.spring_layout(hierarchy, k=2, iterations=50, seed=42)
            else:
                pos = nx.spring_layout(self.nx_graph, k=2, iterations=50, seed=42)
        except Exception:
            pos = nx.spring_layout(self.nx_graph, k=2, iterations=50, seed=42)

        # Color nodes by degree
        degrees = dict(self.nx_graph.degree())
        node_colors = [degrees.get(n, 0) for n in self.nx_graph.nodes]

        nx.draw_networkx_nodes(
            self.nx_graph,
            pos,
            node_size=800,
            node_color=node_colors,
            cmap=plt.cm.YlOrRd,
            alpha=0.9,
            ax=ax,
        )

        # Draw edges with labels
        edge_colors = []
        for u, v, data in self.nx_graph.edges(data=True):
            if data.get("relation") == "subClassOf":
                edge_colors.append("#888888")
            else:
                edge_colors.append("#4A90D9")

        nx.draw_networkx_edges(
            self.nx_graph,
            pos,
            edge_color=edge_colors,
            arrows=True,
            arrowsize=15,
            alpha=0.7,
            ax=ax,
        )

        nx.draw_networkx_labels(
            self.nx_graph, pos, font_size=8, font_weight="bold", ax=ax
        )

        # Edge labels
        edge_labels = {
            (u, v): data.get("relation", "")
            for u, v, data in self.nx_graph.edges(data=True)
            if data.get("relation") != "subClassOf"
        }
        if edge_labels:
            nx.draw_networkx_edge_labels(
                self.nx_graph, pos, edge_labels, font_size=6, ax=ax
            )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def export_for_webvowl(self, save_path: str) -> None:
        """Export the ontology as a Turtle file for use with WebVOWL."""
        self.rdf_graph.serialize(destination=save_path, format="turtle")
        logger.info(
            f"Exported Turtle file for WebVOWL: {save_path}\n"
            f"Upload to http://www.visualdataweb.de/webvowl/ to visualize."
        )

    def get_summary(self) -> dict:
        """Return graph summary statistics."""
        return {
            "nodes": self.nx_graph.number_of_nodes(),
            "edges": self.nx_graph.number_of_edges(),
            "node_list": list(self.nx_graph.nodes),
            "connected_components": (
                nx.number_weakly_connected_components(self.nx_graph)
                if self.nx_graph.is_directed()
                else nx.number_connected_components(self.nx_graph)
            ),
        }

    @staticmethod
    def _short_name(uri) -> str:
        """Extract short name from a URI."""
        uri_str = str(uri)
        if "#" in uri_str:
            return uri_str.split("#")[-1]
        if "/" in uri_str:
            return uri_str.split("/")[-1]
        return uri_str
