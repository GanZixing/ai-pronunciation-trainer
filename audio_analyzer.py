# audio_analyzer.py
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional


def normalize_text(text: str) -> str:
    """Lowercase and keep only letters, apostrophes, and spaces."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def simple_similarity(a: str, b: str) -> float:
    """Return a similarity ratio in [0, 1]."""
    return SequenceMatcher(None, a, b).ratio()


def tokenize_words(text: str) -> List[str]:
    text = normalize_text(text)
    return text.split() if text else []


def classify_issue(expected: str, actual: str) -> str:
    """
    Very simple heuristic issue classifier.
    This is only a first version and can be upgraded later.
    """
    if expected == actual:
        return "ok"

    # Common pronunciation-sensitive words for a first-pass rule.
    if expected == "three" and actual in {"free", "tree", "full", "flee"}:
        return "possible /θ/ pronunciation issue"
    if expected.endswith("s") and not actual.endswith("s"):
        return "possible final consonant omission"
    if len(actual) > len(expected) + 2:
        return "possible insertion or recognition drift"
    if len(actual) + 2 < len(expected):
        return "possible omission"
    return "word mismatch"


@dataclass
class WordAnalysis:
    expected_word: str
    actual_word: Optional[str]
    score: float
    issue: str


@dataclass
class ProsodySummary:
    speech_rate: str
    pause_ok: Optional[bool]
    stress_ok: Optional[bool]
    intonation_ok: Optional[bool]


@dataclass
class AnalysisResult:
    target_text: str
    recognized_text: str
    normalized_target: str
    normalized_recognized: str
    overall_score: int
    similarity_score: float
    word_count_target: int
    word_count_recognized: int
    word_analysis: List[Dict[str, Any]]
    pronunciation_hypotheses: List[str]
    prosody: Dict[str, Any]
    raw_features: Dict[str, Any]


class AudioAnalyzer:
    """
    First-pass pronunciation analysis aggregator.

    Current version:
    - text normalization
    - global similarity
    - word-level comparison
    - simple rule-based hypotheses

    Later you can extend this with:
    - phoneme alignment
    - Whisper/WhisperX timestamps
    - MFA alignment
    - pitch / formant / energy features
    """

    def analyze(
        self,
        target_text: str,
        recognized_text: str,
        similarity_score: Optional[float] = None,
        raw_features: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raw_features = raw_features or {}

        normalized_target = normalize_text(target_text)
        normalized_recognized = normalize_text(recognized_text)

        if similarity_score is None:
            similarity_score = simple_similarity(normalized_target, normalized_recognized)

        target_words = tokenize_words(target_text)
        recognized_words = tokenize_words(recognized_text)

        word_analysis: List[WordAnalysis] = []
        max_len = max(len(target_words), len(recognized_words))

        for i in range(max_len):
            expected_word = target_words[i] if i < len(target_words) else ""
            actual_word = recognized_words[i] if i < len(recognized_words) else None

            if expected_word and actual_word:
                score = simple_similarity(expected_word, actual_word)
                issue = classify_issue(expected_word, actual_word)
            elif expected_word and actual_word is None:
                score = 0.0
                issue = "missing word"
            else:
                # Extra recognized word
                expected_word = ""
                score = 0.0
                issue = "extra word"

            word_analysis.append(
                WordAnalysis(
                    expected_word=expected_word,
                    actual_word=actual_word,
                    score=round(score, 3),
                    issue=issue,
                )
            )

        pronunciation_hypotheses = self._build_hypotheses(word_analysis, similarity_score)
        prosody = self._build_prosody_summary(raw_features)

        result = AnalysisResult(
            target_text=target_text,
            recognized_text=recognized_text,
            normalized_target=normalized_target,
            normalized_recognized=normalized_recognized,
            overall_score=int(round(similarity_score * 100)),
            similarity_score=round(similarity_score, 4),
            word_count_target=len(target_words),
            word_count_recognized=len(recognized_words),
            word_analysis=[asdict(x) for x in word_analysis],
            pronunciation_hypotheses=pronunciation_hypotheses,
            prosody=asdict(prosody),
            raw_features=raw_features,
        )

        return asdict(result)

    def _build_hypotheses(
        self,
        word_analysis: List[WordAnalysis],
        similarity_score: float,
    ) -> List[str]:
        hypotheses: List[str] = []

        low_score_words = [w for w in word_analysis if w.expected_word and w.score < 0.5]
        if similarity_score < 0.6:
            hypotheses.append("overall pronunciation accuracy is low")
        elif similarity_score < 0.8:
            hypotheses.append("overall pronunciation is understandable but unstable")
        else:
            hypotheses.append("overall pronunciation is relatively good")

        if any("possible /θ/" in w.issue for w in word_analysis):
            hypotheses.append("possible dental fricative issue, such as /θ/")

        if any("final consonant" in w.issue for w in word_analysis):
            hypotheses.append("possible weak final consonants")

        if len(low_score_words) >= 2:
            hypotheses.append("multiple words differ significantly from the target")

        if not hypotheses:
            hypotheses.append("no obvious issue detected from text comparison alone")

        return hypotheses

    def _build_prosody_summary(self, raw_features: Dict[str, Any]) -> ProsodySummary:
        """
        Use existing raw features if available.
        If not provided, return placeholders.
        """
        speech_rate_value = raw_features.get("speech_rate_wpm")
        if speech_rate_value is None:
            speech_rate = "unknown"
        elif speech_rate_value < 90:
            speech_rate = "slow"
        elif speech_rate_value <= 160:
            speech_rate = "normal"
        else:
            speech_rate = "fast"

        return ProsodySummary(
            speech_rate=speech_rate,
            pause_ok=raw_features.get("pause_ok"),
            stress_ok=raw_features.get("stress_ok"),
            intonation_ok=raw_features.get("intonation_ok"),
        )


if __name__ == "__main__":
    analyzer = AudioAnalyzer()

    demo = analyzer.analyze(
        target_text="I have three dogs.",
        recognized_text="I have a full lid lock.",
        similarity_score=0.5128205128,
        raw_features={
            "speech_rate_wpm": 92,
            "pause_ok": True,
            "stress_ok": False,
            "intonation_ok": False,
        },
    )

    import json
    print(json.dumps(demo, ensure_ascii=False, indent=2))