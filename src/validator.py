"""
Ontology Validation Module
Validates ontology consistency using OWL reasoners and custom checks.
"""

import logging
from dataclasses import dataclass, field

from owlready2 import sync_reasoner_pellet, sync_reasoner_hermit

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Results of ontology validation."""
    is_consistent: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    inferred_subsumptions: list[tuple[str, str]] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def summary(self) -> str:
        status = "CONSISTENT" if self.is_consistent else "INCONSISTENT"
        lines = [
            f"Validation Report: {status}",
            f"  Errors:   {len(self.errors)}",
            f"  Warnings: {len(self.warnings)}",
            f"  Inferred subsumptions: {len(self.inferred_subsumptions)}",
        ]
        if self.errors:
            lines.append("\nErrors:")
            for e in self.errors:
                lines.append(f"  - {e}")
        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)


class OntologyValidator:
    """Validate and reason over OWL ontologies."""

    def __init__(self, reasoner: str = "hermit"):
        """
        Args:
            reasoner: 'hermit' or 'pellet'
        """
        self.reasoner = reasoner

    def validate(self, ontology) -> ValidationReport:
        """Run full validation on an Owlready2 ontology."""
        report = ValidationReport()

        # 1. Structural checks
        self._check_structure(ontology, report)

        # 2. Run reasoner
        self._run_reasoner(ontology, report)

        # 3. Check for common issues
        self._check_common_issues(ontology, report)

        # 4. Collect stats
        report.stats = {
            "classes": len(list(ontology.classes())),
            "object_properties": len(list(ontology.object_properties())),
            "data_properties": len(list(ontology.data_properties())),
            "individuals": len(list(ontology.individuals())),
        }

        return report

    def _check_structure(self, ontology, report: ValidationReport) -> None:
        """Check basic structural issues."""
        classes = list(ontology.classes())

        if not classes:
            report.warnings.append("Ontology has no classes defined.")
            return

        # Check for orphan classes (no parent except Thing)
        from owlready2 import Thing
        orphans = []
        for cls in classes:
            parents = [p for p in cls.is_a if p != Thing and hasattr(p, "name")]
            if not parents and cls.name != "Thing":
                orphans.append(cls.name)

        if len(orphans) > 10:
            report.warnings.append(
                f"{len(orphans)} classes have no parent (direct subclass of Thing). "
                f"Consider organizing into a hierarchy."
            )

        # Check for classes with no label
        no_label = [c.name for c in classes if not c.label]
        if no_label:
            report.warnings.append(
                f"{len(no_label)} classes have no rdfs:label annotation."
            )

    def _run_reasoner(self, ontology, report: ValidationReport) -> None:
        """Run the DL reasoner to check consistency."""
        try:
            with ontology:
                if self.reasoner == "pellet":
                    sync_reasoner_pellet(infer_property_values=True)
                else:
                    sync_reasoner_hermit(infer_property_values=True)

            report.is_consistent = True

            # Collect inferred subsumptions
            for cls in ontology.classes():
                for parent in cls.is_a:
                    if hasattr(parent, "name"):
                        report.inferred_subsumptions.append(
                            (cls.name, parent.name)
                        )

            logger.info(f"Reasoner ({self.reasoner}): ontology is consistent")

        except Exception as e:
            error_msg = str(e)
            if "Inconsistent" in error_msg or "inconsistent" in error_msg:
                report.is_consistent = False
                report.errors.append(f"Ontology is inconsistent: {error_msg}")
            else:
                report.warnings.append(
                    f"Reasoner failed (may need Java installed): {error_msg}"
                )
            logger.warning(f"Reasoner error: {e}")

    def _check_common_issues(self, ontology, report: ValidationReport) -> None:
        """Check for common ontology design issues."""
        # Check for cycles in subclass hierarchy
        classes = list(ontology.classes())
        for cls in classes:
            visited = set()
            if self._has_cycle(cls, visited):
                report.errors.append(
                    f"Cycle detected in class hierarchy involving: {cls.name}"
                )

        # Check for properties without domain/range
        for prop in ontology.object_properties():
            if not prop.domain:
                report.warnings.append(
                    f"Object property '{prop.name}' has no domain."
                )
            if not prop.range:
                report.warnings.append(
                    f"Object property '{prop.name}' has no range."
                )

    @staticmethod
    def _has_cycle(cls, visited: set, path: set | None = None) -> bool:
        """Detect cycles in the class hierarchy using DFS."""
        if path is None:
            path = set()

        if cls.name in path:
            return True
        if cls.name in visited:
            return False

        visited.add(cls.name)
        path.add(cls.name)

        for parent in cls.is_a:
            if hasattr(parent, "name") and hasattr(parent, "is_a"):
                if OntologyValidator._has_cycle(parent, visited, path.copy()):
                    return True

        return False
