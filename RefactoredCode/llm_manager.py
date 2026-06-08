import json
import time
from typing import Tuple, List, Dict, Any, Optional
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
from config import LLM_MODEL, LLM_TIMEOUT_SECS, LLM_RETRY_DELAYS
from prompts import LLM_FALLBACK_SPEECH, PROFILE_EXTRACTION_PROMPT


class LLMManager:
    """Wraps all OpenAI API calls. Runs synchronously in Twisted worker threads."""

    def __init__(self) -> None:
        self.client = OpenAI()

    def sync_get_response(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[str, List[str], Dict[str, Any], Optional[str]]:
        """
        Synchronous LLM call with exponential-backoff retry.
        Returns (robot_speech, gestures_list, metadata_dict, error_or_None).
        """
        attempts = 1 + len(LLM_RETRY_DELAYS)
        for attempt in range(attempts):
            try:
                response = self.client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=messages,
                    response_format={"type": "json_object"},
                    timeout=LLM_TIMEOUT_SECS,
                )
                content = response.choices[0].message.content or "{}"
                parsed = json.loads(content)

                speech = parsed.get("text", "").strip()

                gestures = parsed.get("gesture", ["BlocklyWaveRightArm"])
                if isinstance(gestures, str):
                    gestures = [gestures]
                if not isinstance(gestures, list) or not gestures:
                    gestures = ["BlocklyWaveRightArm"]

                metadata = parsed.get("metadata", {})

                if not speech:
                    raise ValueError("'text' empty in LLM JSON")

                return speech, gestures, metadata, None

            except (APIError, APIConnectionError, APITimeoutError) as exc:
                err = f"API error attempt {attempt + 1}: {exc}"
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                err = f"Parse error attempt {attempt + 1}: {exc}"
            except Exception as exc:
                err = f"Unexpected error attempt {attempt + 1}: {exc}"

            print(f"[LLM] {err}")

            if attempt < len(LLM_RETRY_DELAYS):
                delay = LLM_RETRY_DELAYS[attempt]
                print(f"[LLM] Retrying in {delay}s...")
                time.sleep(delay)

        return LLM_FALLBACK_SPEECH, ["BlocklyWaveRightArm"], {"pace": "slow"}, err

    def sync_extract_profile(
        self, conversation: List[Dict[str, str]]
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """Extracts nested profile updates post-session."""
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": PROFILE_EXTRACTION_PROMPT},
                    {"role": "user", "content": json.dumps(
                        conversation, ensure_ascii=False)}
                ],
                response_format={"type": "json_object"},
                timeout=LLM_TIMEOUT_SECS,
            )
            content = response.choices[0].message.content or "{}"
            return json.loads(content), None
        except Exception as exc:
            err = f"Profile extraction failed: {exc}"
            print(f"[PROFILE] {err}")
            return {}, err
