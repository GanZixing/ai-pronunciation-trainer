# training_data_store.py
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TrainingSample:
    sample_id: str
    timestamp: str
    speaker_id: Optional[str]
    target_text: str
    recognized_text: str
    audio_path: Optional[str]
    similarity_score: Optional[float]
    analysis_result: Dict[str, Any]
    llm_feedback: Optional[Dict[str, Any]] = None
    human_feedback: Optional[Dict[str, Any]] = None
    tags: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


class TrainingDataStore:
    def __init__(
        self,
        base_dir: str = "data",
        samples_file: str = "samples.jsonl",
    ) -> None:
        self.base_dir = Path(base_dir)
        self.audio_dir = self.base_dir / "audio"
        self.samples_path = self.base_dir / samples_file

        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.samples_path.touch(exist_ok=True)

    def generate_sample_id(self) -> str:
        now = datetime.now()
        prefix = now.strftime("%Y%m%d_%H%M%S")
        counter = 1

        while True:
            sample_id = f"{prefix}_{counter:03d}"
            if not self._sample_exists(sample_id):
                return sample_id
            counter += 1

    def _sample_exists(self, sample_id: str) -> bool:
        if not self.samples_path.exists():
            return False

        with self.samples_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if item.get("sample_id") == sample_id:
                        return True
                except json.JSONDecodeError:
                    continue
        return False

    def copy_audio_file(self, source_audio_path: str, sample_id: str) -> str:
        src = Path(source_audio_path)
        if not src.exists():
            raise FileNotFoundError(f"Audio file not found: {source_audio_path}")

        ext = src.suffix if src.suffix else ".wav"
        dst = self.audio_dir / f"{sample_id}{ext}"
        shutil.copy2(src, dst)
        return str(dst.as_posix())

    def save_sample(
        self,
        target_text: str,
        recognized_text: str,
        analysis_result: Dict[str, Any],
        similarity_score: Optional[float] = None,
        speaker_id: Optional[str] = None,
        source_audio_path: Optional[str] = None,
        copy_audio: bool = True,
        llm_feedback: Optional[Dict[str, Any]] = None,
        human_feedback: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        sample_id = self.generate_sample_id()
        timestamp = datetime.now().isoformat(timespec="seconds")

        audio_path = None
        if source_audio_path:
            audio_path = (
                self.copy_audio_file(source_audio_path, sample_id)
                if copy_audio
                else source_audio_path
            )

        sample = TrainingSample(
            sample_id=sample_id,
            timestamp=timestamp,
            speaker_id=speaker_id,
            target_text=target_text,
            recognized_text=recognized_text,
            audio_path=audio_path,
            similarity_score=similarity_score,
            analysis_result=analysis_result,
            llm_feedback=llm_feedback,
            human_feedback=human_feedback,
            tags=tags or ["unreviewed"],
            extra=extra or {},
        )

        self._append_jsonl(asdict(sample))
        return asdict(sample)

    def _append_jsonl(self, item: Dict[str, Any]) -> None:
        with self.samples_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def load_all_samples(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        if not self.samples_path.exists():
            return results

        with self.samples_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        return results

    def find_by_sample_id(self, sample_id: str) -> Optional[Dict[str, Any]]:
        for item in self.load_all_samples():
            if item.get("sample_id") == sample_id:
                return item
        return None


if __name__ == "__main__":
    # demo
    demo_analysis = {
        "overall_score": 52,
        "pronunciation_hypotheses": [
            "overall pronunciation accuracy is low",
            "possible dental fricative issue, such as /θ/",
            "multiple words differ significantly from the target"
        ],
        "word_analysis": [
            {"expected_word": "i", "actual_word": "i", "score": 1.0, "issue": "ok"},
            {"expected_word": "have", "actual_word": "have", "score": 1.0, "issue": "ok"},
            {"expected_word": "three", "actual_word": "full", "score": 0.0, "issue": "possible /θ/ pronunciation issue"},
            {"expected_word": "dogs", "actual_word": "lid", "score": 0.25, "issue": "word mismatch"},
        ],
        "prosody": {
            "speech_rate": "slow",
            "pause_ok": True,
            "stress_ok": False,
            "intonation_ok": False,
        }
    }

    store = TrainingDataStore()
    sample = store.save_sample(
        target_text="I have three dogs.",
        recognized_text="I have a full lid lock.",
        analysis_result=demo_analysis,
        similarity_score=0.5128,
        speaker_id="user_001",
        source_audio_path=None,  # 这里可以换成真实音频路径
        tags=["unreviewed", "demo"]
    )

    print(json.dumps(sample, ensure_ascii=False, indent=2))