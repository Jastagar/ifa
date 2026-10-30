"""Fine-tune Ifa's speech-to-text on your own voice with LoRA.

Uses the dataset that the launcher's Transcripts tab freezes (stt_data/datasets/<time>/train and test).
Only small adapter matrices are trained; the Whisper model itself stays as it is. The adapter is kept only if
it beats the current model on the test clips, which training never sees.

    python -m scripts.finetune_stt                  # newest frozen dataset, Ifa's current model
    python -m scripts.finetune_stt --epochs 6 --data stt_data/datasets/20261030-2100

Then set IFA_WHISPER_ADAPTER to the printed folder (launcher: Listening) and restart Ifa.
Stop Ifa first: training needs the GPU.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "stt_data"
MIN_TRAIN_CLIPS = 100


def log(message: str) -> None:
    print(f"[finetune] {message}", flush=True)


def load_split(folder: Path) -> list[tuple[np.ndarray, str]]:
    import soundfile as sf

    meta = folder / "metadata.csv"
    if not meta.exists():
        return []
    clips = []
    with open(meta, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            audio, rate = sf.read(folder / row["file_name"], dtype="float32")
            if rate != 16_000:
                raise ValueError(f"{row['file_name']} is {rate} Hz; Ifa saves 16 kHz clips")
            clips.append((audio if audio.ndim == 1 else audio.mean(axis=1), row["transcription"]))
    return clips


def normalize_text(text: str) -> list[str]:
    return re.sub(r"[^\w\s']", " ", text.lower()).split()


def word_errors(ref: list[str], hyp: list[str]) -> int:
    """Word-level edit distance (substitutions + insertions + deletions)."""
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (r != h))
    return row[-1]


def wer(refs: list[str], hyps: list[str]) -> float:
    words = sum(len(normalize_text(r)) for r in refs)
    errors = sum(word_errors(normalize_text(r), normalize_text(h)) for r, h in zip(refs, hyps))
    return errors / max(words, 1)


def pick_device(choice: str) -> str:
    import torch

    if choice != "auto":
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Whisper:
    """The model plus everything needed to batch audio and text the same way Ifa transcribes."""

    def __init__(self, name: str, device: str, language: str) -> None:
        import torch
        from transformers import AutoProcessor, WhisperForConditionalGeneration

        self.torch = torch
        self.device = device
        self.language = language
        self.processor = AutoProcessor.from_pretrained(name)
        self.processor.tokenizer.set_prefix_tokens(language=language, task="transcribe")
        # fp32 weights; on CUDA the forward pass runs in bf16 autocast (fits a 16 GB card for Whisper-turbo).
        self.model = WhisperForConditionalGeneration.from_pretrained(name, torch_dtype=torch.float32).to(device)
        self.autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if device == "cuda" else None

    def features(self, audios: list[np.ndarray]):
        out = self.processor.feature_extractor(audios, sampling_rate=16_000, return_tensors="pt")
        return out.input_features.to(self.device)

    def labels(self, texts: list[str]):
        ids = self.processor.tokenizer(texts, padding=True, return_tensors="pt")
        labels = ids.input_ids.masked_fill(ids.attention_mask.eq(0), -100)
        start = self.model.config.decoder_start_token_id
        if (labels[:, 0] == start).all():  # the model adds the start token itself
            labels = labels[:, 1:]
        return labels.to(self.device)

    def transcribe(self, clips: list[tuple[np.ndarray, str]], batch: int) -> list[str]:
        self.model.eval()
        out = []
        with self.torch.no_grad():
            for i in range(0, len(clips), batch):
                feats = self.features([a for a, _ in clips[i:i + batch]])
                ctx = self.autocast() if self.autocast else _null()
                with ctx:
                    ids = self.model.generate(input_features=feats, language=self.language, task="transcribe",
                                              max_new_tokens=128)
                out += [t.strip() for t in self.processor.batch_decode(ids, skip_special_tokens=True)]
        return out


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def newest_dataset() -> Path:
    found = sorted(p for p in (DATA_ROOT / "datasets").glob("*") if (p / "train" / "metadata.csv").exists())
    if not found:
        sys.exit("No frozen dataset yet: review clips in the launcher's Transcripts tab, then press Freeze dataset.")
    return found[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, help="frozen dataset folder (default: the newest)")
    parser.add_argument("--base", default=os.environ.get("IFA_WHISPER_MODEL") or "Oriserve/Whisper-Hindi2Hinglish-Apex")
    parser.add_argument("--language", default=os.environ.get("IFA_WHISPER_LANGUAGE") or "en")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank: bigger learns more, overfits sooner")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0, help="stop early (for a quick smoke test)")
    args = parser.parse_args()

    data = args.data or newest_dataset()
    train, test = load_split(data / "train"), load_split(data / "test")
    log(f"dataset {data}: {len(train)} train, {len(test)} test clips")
    if not test:
        sys.exit("The dataset has no test clips (every 5th reviewed clip). Review more, then freeze again.")
    if len(train) < MIN_TRAIN_CLIPS:
        log(f"only {len(train)} training clips; {MIN_TRAIN_CLIPS}+ gives a real improvement. Running anyway.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    log(f"loading {args.base} on {device}")
    w = Whisper(args.base, device, args.language)
    w.torch.manual_seed(args.seed)

    refs = [t for _, t in test]
    started = time.monotonic()
    baseline = w.transcribe(test, args.batch)
    base_wer = wer(refs, baseline)
    log(f"baseline word error rate on test: {base_wer:.1%} ({time.monotonic() - started:.0f}s)")

    from peft import LoraConfig, get_peft_model

    # LoRA on every attention projection, in the encoder (how you sound) and the decoder (how you phrase things).
    w.model = get_peft_model(w.model, LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]))
    trainable = sum(p.numel() for p in w.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in w.model.parameters())
    log(f"training {trainable:,} of {total:,} parameters ({trainable / total:.2%})")

    optim = w.torch.optim.AdamW([p for p in w.model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    steps_per_epoch = (len(train) + args.batch - 1) // args.batch
    total_steps = steps_per_epoch * args.epochs
    sched = w.torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=args.lr, total_steps=max(total_steps, 1), pct_start=0.1)

    out = DATA_ROOT / "adapters" / time.strftime("%Y%m%d-%H%M%S")
    best_wer, best_epoch, best_hyps, step = base_wer, 0, baseline, 0
    for epoch in range(1, args.epochs + 1):
        w.model.train()
        order = list(range(len(train)))
        random.shuffle(order)
        losses = []
        for i in range(0, len(order), args.batch):
            batch = [train[j] for j in order[i:i + args.batch]]
            feats, labels = w.features([a for a, _ in batch]), w.labels([t for _, t in batch])
            with (w.autocast() if w.autocast else _null()):
                loss = w.model(input_features=feats, labels=labels).loss
            loss.backward()
            w.torch.nn.utils.clip_grad_norm_(w.model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad()
            losses.append(float(loss))
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        hyps = w.transcribe(test, args.batch)
        score = wer(refs, hyps)
        log(f"epoch {epoch}/{args.epochs}: train loss {np.mean(losses):.3f}, test word error rate {score:.1%}")
        if score < best_wer:
            best_wer, best_epoch, best_hyps = score, epoch, hyps
            w.model.save_pretrained(out)
        if args.max_steps and step >= args.max_steps:
            break

    log(f"baseline {base_wer:.1%} -> best {best_wer:.1%}" + (f" (epoch {best_epoch})" if best_epoch else ""))
    changed = [(r, b, h) for r, b, h in zip(refs, baseline, best_hyps) if b != h][:8]
    for ref, before, after in changed:
        print(f"  said:   {ref}\n  before: {before}\n  after:  {after}\n")
    if not best_epoch:
        log("the adapter never beat the current model, so nothing was saved. Collect more clips and try again.")
        return 1
    (out / "report.json").write_text(json.dumps({
        "base_model": args.base, "dataset": str(data), "train_clips": len(train), "test_clips": len(test),
        "baseline_wer": base_wer, "adapter_wer": best_wer, "epoch": best_epoch,
        "rank": args.rank, "lr": args.lr}, indent=2), encoding="utf-8")
    log(f"saved the adapter to {out}")
    log(f"to use it: set IFA_WHISPER_ADAPTER={out} (launcher: Listening), then restart Ifa")
    return 0


if __name__ == "__main__":
    sys.exit(main())
