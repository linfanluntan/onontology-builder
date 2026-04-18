"""
LLM-Accelerated Extraction Module
Uses Claude API to extract structured knowledge (concepts, relations, taxonomy)
from document text for ontology construction.
"""

import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """You are an expert knowledge engineer. Your task is to extract structured ontological knowledge from the provided text.

Extract the following and return ONLY valid JSON (no markdown, no backticks):

{
  "concepts": [
    {
      "name": "PascalCaseName",
      "label": "Human Readable Label",
      "type": "class",
      "definition": "Brief definition",
      "parent": "ParentClassName or empty string"
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

Guidelines:
- Use PascalCase for class names, camelCase for relations/attributes.
- Identify is-a (taxonomy), part-of, causal, functional, and associative relations.
- Be precise and domain-specific. Prefer established terminology.
- Include parent classes to form a taxonomy hierarchy.
- Only extract knowledge that is explicitly stated or strongly implied in the text.
"""


@dataclass
class LLMKnowledge:
    """Structured knowledge extracted by the LLM."""
    concepts: list[dict] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)
    attributes: list[dict] = field(default_factory=list)
    raw_responses: list[str] = field(default_factory=list)


class LLMExtractor:
    """Extract ontological knowledge using Claude API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        chunk_size: int = 3000,
        domain_context: str = "",
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

        # Deduplicate
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

        # Final deduplication
        all_knowledge.concepts = self._deduplicate_concepts(all_knowledge.concepts)
        all_knowledge.relations = self._deduplicate_relations(all_knowledge.relations)

        return all_knowledge

    def _call_api(self, text_chunk: str) -> dict | None:
        """Call Claude API to extract knowledge from a text chunk."""
        user_content = f"Extract ontological knowledge from this text:\n\n{text_chunk}"
        if self.domain_context:
            user_content = (
                f"Domain context: {self.domain_context}\n\n{user_content}"
            )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=EXTRACTION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
            raw = response.content[0].text
            # Parse JSON, stripping any accidental markdown fences
            raw = raw.strip()
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
        """Split text into chunks, breaking at paragraph boundaries."""
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
