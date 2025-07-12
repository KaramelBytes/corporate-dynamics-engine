from __future__ import annotations
"""Quality validation for AI-generated content adhering to free-tier POC rules."""

from typing import Dict, Any, List

VALIDATION_RULES: Dict[str, Any] = {
    "content_length": {
        "scenario_min": 100,
        "scenario_max": 800,
        "dialogue_min": 20,
        "dialogue_max": 300,
    },
    "required_json_fields": {
        "scenario": [
            "description",
            "available_actions",
            "stakeholder_context",
        ],
        "dialogue": [
            "speaker",
            "content",
            "tone",
            "relationship_impact",
        ],
    },
    "forbidden_content": [
        "placeholder",
        "lorem ipsum",
        "todo",
        "tbd",
        "example.com",
        "[insert",
        "coming soon",
        "under construction",
    ],
    "corporate_authenticity": {
        "required_corporate_terms": [
            "meeting",
            "team",
            "project",
            "budget",
            "deadline",
        ],
        "stakeholder_consistency": True,
        "scenario_realism": True,
    },
}


class QualityValidator:  # noqa: D101, pylint: disable=too-few-public-methods
    def __init__(self, rules: Dict[str, Any] | None = None):
        self.rules = rules or VALIDATION_RULES

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def validate(self, payload: Dict[str, Any]) -> bool:  # noqa: D401
        """Validate AI output payload (scenario or dialogue).

        Expects a key ``type`` in payload (``scenario`` or ``dialogue``).
        Returns True if all checks pass, otherwise False.
        """
        payload_type = payload.get("type")
        if payload_type not in ("scenario", "dialogue"):
            return False

        # Core checks
        return all(
            [
                self._check_length(payload_type, payload),
                self._check_required_fields(payload_type, payload),
                self._check_forbidden_content(payload),
                self._check_corporate_authenticity(payload),
            ]
        )

    # ------------------------------------------------------------------
    # Internal rule checks
    # ------------------------------------------------------------------
    def _check_length(self, payload_type: str, payload: Dict[str, Any]) -> bool:
        content = payload.get("content", "") if payload_type == "dialogue" else payload.get("description", "")
        length = len(content)
        min_key = f"{payload_type}_min"
        max_key = f"{payload_type}_max"
        limits = self.rules["content_length"]
        return limits[min_key] <= length <= limits[max_key]

    def _check_required_fields(self, payload_type: str, payload: Dict[str, Any]) -> bool:
        required = self.rules["required_json_fields"][payload_type]
        return all(field in payload for field in required)

    def _check_forbidden_content(self, payload: Dict[str, Any]) -> bool:
        text_blob = " ".join(map(str, payload.values())).lower()
        return not any(forbidden in text_blob for forbidden in self.rules["forbidden_content"])

    def _check_corporate_authenticity(self, payload: Dict[str, Any]) -> bool:
        if not self.rules["corporate_authenticity"]["scenario_realism"]:
            return True
        text_blob = " ".join(map(str, payload.values())).lower()
        required_terms = self.rules["corporate_authenticity"]["required_corporate_terms"]
        return all(term in text_blob for term in required_terms)
