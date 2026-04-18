"""
LLM-Accelerated Extraction Module for ORNJ Domain
Uses Claude API to extract structured ORNJ classification knowledge
(grades, stages, clinical/radiographic criteria, treatment recommendations)
from document text for ontology construction.
"""

import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """You are an expert biomedical knowledge engineer specializing in head and neck cancer complications. Your task is to extract structured ontological knowledge about osteoradionecrosis of the jaw (ORNJ) classification systems from the provided text.

Extract the following and return ONLY valid JSON (no markdown, no backticks):

{
  "concepts": [
    {
      "name": "PascalCaseName",
      "label": "Human Readable Label",
      "type": "class",
      "definition": "Brief clinical definition",
      "parent": "ParentClassName or empty string",
      "source_system": "Name of the classification system if applicable"
    }
  ],
  "relations": [
    {
      "subject": "SubjectClassName",
      "predicate": "camelCaseRelation",
      "object": "ObjectClassName",
      "description": "Brief description of the relation"
    }
  ],
  "attributes": [
    {
      "concept": "ClassName",
      "attribute": "attributeName",
      "value_type": "string|integer|float|boolean|date",
      "description": "Brief description"
    }
  ]
}

Domain-specific guidelines:
- Identify ORNJ classification/grading/staging systems and their individual grades/stages as separate classes.
- Capture conditional classification criteria as relations, e.g.:
  - "Grade III is defined as ORN below the inferior alveolar canal" → NotaniGradeIII hasCriterion BelowInferiorAlveolarCanal
  - "Stage I responds to HBO alone" → MarxStageI hasRecommendedTreatment HyperbaricOxygenTherapy
- Distinguish clinical findings (bone exposure, fistula, pain) from radiographic findings (osteolysis, sclerosis, fracture on imaging).
- Capture anatomical landmarks: alveolar bone, basilar bone, inferior alveolar canal, mandible, maxilla, maxillary sinus.
- Identify risk factors: radiation dose, smoking, dental extraction timing, periodontal status.
- Use PascalCase for class names, camelCase for relations/attributes.
- Only extract knowledge explicitly stated or strongly implied in the text.
- Map to SNOMED-CT concepts where obvious (e.g., osteoradionecrosis → SNOMED 399098001).
"""


@dataclass
class LLMKnowledge:
    """Structured knowledge extracted by the LLM."""
    concepts: list[dict] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)
    attributes: list[dict] = field(default_factory=list)
    raw_responses: list[str] = field(default_factory=list)


class LLMExtractor:
    """Extract ORNJ ontological knowledge using Claude API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        chunk_size: int = 3000,
        domain_context: str = "osteoradionecrosis of the jaw classification and staging systems",
    ):
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")

        self.model = model
        self.max_tokens = max_tokens
        self.chunk_size = chunk_size
        self.domain_context = domain_context

    def extract_from_text(self, text: str) -> LLMKnowledge:
        """Extract knowledge from a single text block."""
        chunks = self._chunk_text(text, self.chunk_size)
        knowledge = LLMKnowledge()

        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i + 1}/{len(chunks)}...")
            result = self._call_api(chunk)
            if result:
                knowledge.concepts.extend(result.get("concepts", []))
                knowledge.relations.extend(result.get("relations", []))
                knowledge.attributes.extend(result.get("attributes", []))

        knowledge.concepts = self._deduplicate_concepts(knowledge.concepts)
        knowledge.relations = self._deduplicate_relations(knowledge.relations)

        logger.info(
            f"LLM extracted: {len(knowledge.concepts)} concepts, "
            f"{len(knowledge.relations)} relations, "
            f"{len(knowledge.attributes)} attributes"
        )
        return knowledge

    def extract_from_documents(self, documents) -> LLMKnowledge:
        """Extract knowledge from multiple Document objects."""
        all_knowledge = LLMKnowledge()

        for doc in documents:
            logger.info(f"Processing document: {doc.filename}")
            k = self.extract_from_text(doc.full_text)
            all_knowledge.concepts.extend(k.concepts)
            all_knowledge.relations.extend(k.relations)
            all_knowledge.attributes.extend(k.attributes)

        all_knowledge.concepts = self._deduplicate_concepts(all_knowledge.concepts)
        all_knowledge.relations = self._deduplicate_relations(all_knowledge.relations)

        return all_knowledge

    def _call_api(self, text_chunk: str) -> dict | None:
        """Call Claude API to extract knowledge from a text chunk."""
        user_content = f"Domain context: {self.domain_context}\n\nExtract ORNJ ontological knowledge from this text:\n\n{text_chunk}"

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=EXTRACTION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
            raw = response.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw.rsplit("```", 1)[0]
            return json.loads(raw.strip())
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return None

    @staticmethod
    def _chunk_text(text: str, chunk_size: int) -> list[str]:
        """Split text into chunks at paragraph boundaries."""
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk += "\n\n" + para

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text]

    @staticmethod
    def _deduplicate_concepts(concepts: list[dict]) -> list[dict]:
        seen = {}
        for c in concepts:
            key = c.get("name", "").lower()
            if key and key not in seen:
                seen[key] = c
        return list(seen.values())

    @staticmethod
    def _deduplicate_relations(relations: list[dict]) -> list[dict]:
        seen = set()
        unique = []
        for r in relations:
            key = (
                r.get("subject", "").lower(),
                r.get("predicate", "").lower(),
                r.get("object", "").lower(),
            )
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique
