# llama_prompt.py
from __future__ import annotations

import json
from typing import Any, Dict


def build_system_prompt() -> str:
    return (
        "You are a professional English pronunciation coach.\n"
        "The learner is a Chinese speaker learning English pronunciation.\n"
        "You will receive structured pronunciation analysis data.\n"
        "Your job is to provide practical pronunciation feedback.\n\n"
        "Rules:\n"
        "- Focus on pronunciation, not grammar.\n"
        "- Be specific and practical.\n"
        "- If data is incomplete, be cautious.\n"
        "- Output JSON only.\n"
    )


def build_user_prompt(analysis_result: Dict[str, Any]) -> str:
    return (
        "Analyze the learner's pronunciation based on the following structured data.\n"
        "Return JSON only.\n\n"
        f"{json.dumps(analysis_result, ensure_ascii=False, indent=2)}"
    )


def build_output_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "overall_comment": {"type": "string"},
            "pronunciation_summary_cn": {"type": "string"},
            "main_problems": {
                "type": "array",
                "items": {"type": "string"},
            },
            "correction_tips": {
                "type": "array",
                "items": {"type": "string"},
            },
            "practice_words": {
                "type": "array",
                "items": {"type": "string"},
            },
            "practice_sentence": {"type": "string"},
            "encouragement": {"type": "string"},
        },
        "required": [
            "overall_comment",
            "pronunciation_summary_cn",
            "main_problems",
            "correction_tips",
            "practice_words",
            "practice_sentence",
            "encouragement",
        ],
        "additionalProperties": False,
    }


def build_chat_messages(analysis_result: Dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": build_user_prompt(analysis_result)},
    ]