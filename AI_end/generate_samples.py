"""
Pre-generate demo audio samples for the comparison page.
Run once: python generate_samples.py
Outputs go to ../frontend/samples/
"""

import asyncio
import os
from pathlib import Path

import edge_tts

# ── Sample content ────────────────────────────────────────────────────────────
SAMPLES = [
    {
        "id": "zh_female",
        "label": "繁體中文 · Female",
        "text": "人工智慧語音合成技術正在快速發展，未來將有更多創新的應用場景。",
        "voices": {
            "f5":        "zh-TW-HsiaoChenNeural",   # Female TW
            "cosyvoice": "zh-TW-HsiaoYuNeural",      # Female TW
            "gptsovits": "zh-CN-XiaoxiaoNeural",     # Female CN (Mandarin accent)
        },
    },
    {
        "id": "en_female",
        "label": "English · Female",
        "text": "Artificial intelligence is transforming the way we interact with technology every day.",
        "voices": {
            "f5":        "en-US-JennyNeural",
            "cosyvoice": "en-US-AriaNeural",
            "gptsovits": "en-GB-SoniaNeural",
        },
    },
    {
        "id": "zh_male",
        "label": "繁體中文 · Male",
        "text": "歡迎使用語音合成比較平台，請輸入您想要合成的文字內容。",
        "voices": {
            "f5":        "zh-TW-YunJheNeural",       # Male TW
            "cosyvoice": "zh-CN-YunxiNeural",        # Male CN
            "gptsovits": "zh-CN-YunjianNeural",      # Male CN (news style)
        },
    },
]

OUTPUT_DIR = Path(__file__).parent.parent / "frontend" / "samples"


async def synthesize(text: str, voice: str, out_path: Path) -> None:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(out_path))
    print(f"  OK {out_path.name}  [{voice}]")


async def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = []
    for sample in SAMPLES:
        for model, voice in sample["voices"].items():
            filename = f"{sample['id']}_{model}.mp3"
            out_path = OUTPUT_DIR / filename
            tasks.append(synthesize(sample["text"], voice, out_path))

    print(f"Generating {len(tasks)} audio files into {OUTPUT_DIR} ...")
    await asyncio.gather(*tasks)
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
