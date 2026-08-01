import json
import os
import re


def extract_speaker_id(audio_path):
    """Extract VoiceBank speaker IDs such as p226 from an audio path."""
    name = os.path.basename(str(audio_path))
    match = re.match(r"(p\d+)", name, flags=re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return os.path.splitext(name)[0].split("_")[0].lower()


def build_speaker_map_from_jsons(*json_paths):
    speakers = set()
    for json_path in json_paths:
        if not json_path or not os.path.exists(json_path):
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        for item in items:
            clean_path = item.get("clean") if isinstance(item, dict) else item
            if clean_path:
                speakers.add(extract_speaker_id(clean_path))
    return {speaker: idx for idx, speaker in enumerate(sorted(speakers))}
