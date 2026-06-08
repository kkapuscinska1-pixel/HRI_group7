import os
import json
import re
from typing import Dict, Any, List
from config import PROFILES_DIR, SESSIONS_DIR


class ProfileManager:
    """Handles all profile file I/O and data transformations."""

    @staticmethod
    def ensure_dirs() -> None:
        os.makedirs(PROFILES_DIR, exist_ok=True)
        os.makedirs(SESSIONS_DIR, exist_ok=True)

    @staticmethod
    def make_participant_id(first: str, last: str) -> str:
        """Converts 'John Smith' → 'john_smith'"""
        raw = f"{first.strip().lower()}_{last.strip().lower()}"
        return re.sub(r"[^a-z0-9_]", "", raw)

    @staticmethod
    def empty_profile(pid: str) -> Dict[str, Any]:
        return {
            "participant_id": pid,
            "preferred_name": None,
            "enjoyed_topics": [],
            "liked_activities": [],
            "people_or_pets": [],
            "communication_preferences": {
                "question_type": None,
                "needs_extra_time": None,
                "helpful_supports": []
            },
            "topics_to_avoid": [],
            "future_conversation_suggestions": [],
            "session_notes": {
                "what_went_well": [],
                "what_was_hard": [],
                "support_that_helped": [],
                "possible_stt_or_transcription_issues": []
            },
            "suggested_next_scenario": {
                "scenario": None,
                "reason": None
            }
        }

    @staticmethod
    def load(pid: str) -> Dict[str, Any]:
        """Load from disk. Returns blank profile on missing file or JSON corruption."""
        ProfileManager.ensure_dirs()
        path = os.path.join(PROFILES_DIR, f"{pid}.json")

        default_profile = ProfileManager.empty_profile(pid)

        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for key, default_val in default_profile.items():
                    if key not in data:
                        data[key] = default_val

                print(f"[PROFILE] Loaded profile for '{pid}'.")
                return data
            except (json.JSONDecodeError, OSError) as exc:
                print(
                    f"[PROFILE] Warning — corrupt profile ({exc}). Starting fresh.")
        else:
            print(
                f"[PROFILE] No profile found for '{pid}'. Creating new profile.")

        return default_profile

    @staticmethod
    def save(pid: str, data: Dict[str, Any]) -> None:
        """Write profile. Logs warning on failure rather than crashing."""
        ProfileManager.ensure_dirs()
        path = os.path.join(PROFILES_DIR, f"{pid}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"[PROFILE] Saved → {path}")
        except OSError as exc:
            print(f"[PROFILE] Error saving: {exc}")

    @staticmethod
    def merge_updates(existing: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """Safely merge LLM-generated post-session data into the stored profile."""
        # Merge lists without duplicates
        for field in ("enjoyed_topics", "avoided_topics", "liked_activities", "people_or_pets", "future_conversation_suggestions"):
            incoming = updates.get(field) or []
            if isinstance(incoming, list):
                current = existing.get(field, [])
                seen = set(current)
                existing[field] = current + \
                    [t for t in incoming if t not in seen]

        # Merge dicts safely
        for dict_field in ("communication_preferences", "session_notes", "suggested_next_scenario"):
            incoming = updates.get(dict_field) or {}
            if isinstance(incoming, dict):
                current = existing.get(dict_field, {})
                for k, v in incoming.items():
                    if isinstance(v, list):
                        current_list = current.get(k, [])
                        seen = set(current_list)
                        current[k] = current_list + \
                            [i for i in v if i not in seen]
                    elif v is not None and v != "":
                        current[k] = v
                existing[dict_field] = current

        # Merge string/null overrides
        if updates.get("preferred_name"):
            existing["preferred_name"] = updates["preferred_name"]

        return existing

    @staticmethod
    def build_memory_block(profile: Dict[str, Any]) -> str:
        """Convert stored profile data into a plain-English paragraph for the system prompt."""
        p = profile
        lines: List[str] = []

        if p.get("preferred_name"):
            lines.append(
                f"The user's preferred name is {p['preferred_name']}.")

        comms = p.get("communication_preferences", {})
        if comms.get("question_type"):
            lines.append(f"They prefer {comms['question_type']} questions.")
        if comms.get("needs_extra_time"):
            lines.append("They need extra time to process questions.")

        if p.get("enjoyed_topics"):
            lines.append(
                f"Topics they enjoy: {', '.join(p['enjoyed_topics'])}.")
        if p.get("topics_to_avoid"):
            lines.append(
                f"Topics to avoid: {', '.join(p['topics_to_avoid'])}.")

        next_scen = p.get("suggested_next_scenario", {})
        if next_scen.get("scenario"):
            lines.append(
                f"Suggested focus for today: {next_scen['scenario']} ({next_scen.get('reason', '')})")

        if not lines:
            return ""
        return "\nPERSONALIZATION MEMORY (from previous sessions):\n" + "\n".join(lines) + "\n"
