"""Owner-safe Persona projection with no transport or storage dependencies."""
from __future__ import annotations


def projection_level(value: object, low: float, high: float) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    if number >= high:
        return "high"
    if number <= low:
        return "low"
    return "medium"


def safe_persona_projection(state: dict, updated_at: str = "", profile_id: str = "") -> dict:
    """Reduce Persona state to approved labels and exclude private narrative."""
    affect = state.get("affect", {}) if isinstance(state, dict) else {}
    relationship = state.get("relationship", {}) if isinstance(state, dict) else {}
    mood_label = str(affect.get("mood_label") or "warm_neutral")[:60]
    drive_labels = []
    if float(affect.get("longing") or 0.0) >= 0.45:
        drive_labels.append("connection")
    if float(affect.get("protective_drive") or 0.0) >= 0.62:
        drive_labels.append("protection")
    if float(affect.get("arousal") or 0.0) >= 0.62:
        drive_labels.append("expression")
    if not drive_labels:
        drive_labels.append("companionship")

    affinity = float(relationship.get("affinity") or 0.0)
    trust = float(relationship.get("trust") or 0.0)
    defensiveness = float(relationship.get("defensiveness") or 0.0)
    if affinity >= 0.78 and trust >= 0.72:
        weather = "close_and_secure"
    elif defensiveness >= 0.45:
        weather = "careful"
    else:
        weather = "steady"

    return {
        "profile_id": str(state.get("profile_id") or profile_id),
        "affect": {
            "label": mood_label,
            "intensity": projection_level(affect.get("arousal"), 0.24, 0.62),
            "valence": round(float(affect.get("valence") or 0.0), 3),
            "arousal": round(float(affect.get("arousal") or 0.0), 3),
            "source": "ombre_affect",
        },
        "drive": {
            "labels": drive_labels[:4],
            "longing": projection_level(affect.get("longing"), 0.20, 0.45),
            "protective": projection_level(affect.get("protective_drive"), 0.35, 0.62),
            "source": "ombre_affect",
        },
        "relationship": {
            "weather": weather,
            "closeness": projection_level(affinity, 0.45, 0.78),
            "trust": projection_level(trust, 0.45, 0.72),
            "source": "ombre_relationship",
        },
        "updated_at": str(updated_at or ""),
    }
