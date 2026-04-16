#!/usr/bin/env python3
"""V2 Step 1: Extract hidden states from decoder LLMs.

Usage:
    python scripts/run_v2_extract.py --model meta-llama/Llama-3.1-8B-Instruct
    python scripts/run_v2_extract.py --model all          # run all primary models
    python scripts/run_v2_extract.py --model all --pooling mean  # alternate pooling
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hidden_states import (
    HiddenStateExtractor, ExtractionConfig, PRIMARY_MODELS,
    format_prompt, save_hidden_states,
)
from src.stimuli import LANGUAGES

# --- Load stimuli ---

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "stimuli"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "hidden_states"


def load_tier_stimuli():
    """Load all three tiers of stimuli."""
    tiers = {}

    # Tier 1: existing computational ops
    with open(DATA_DIR / "computational.json") as f:
        tier1 = json.load(f)
    tiers[1] = tier1

    # Tier 2: multi-step algorithms
    with open(DATA_DIR / "tier2_multistep.json") as f:
        tiers[2] = json.load(f)

    # Tier 3: compositional operations
    with open(DATA_DIR / "tier3_compositional.json") as f:
        tiers[3] = json.load(f)

    return tiers


def extract_for_model(model_name: str, pooling: str = "last", device: str = "cuda"):
    """Run full extraction pipeline for one model."""
    config = ExtractionConfig(
        model_name=model_name,
        device=device,
        pooling=pooling,
    )
    print(f"\n{'='*60}")
    print(f"Extracting: {model_name} (pooling={pooling})")
    print(f"{'='*60}")

    extractor = HiddenStateExtractor(config)
    tiers = load_tier_stimuli()

    for tier_num, ops in tiers.items():
        print(f"\n--- Tier {tier_num} ({len(ops)} operations) ---")

        # Extract NL descriptions per language
        for lang in LANGUAGES:
            texts = []
            op_ids = []
            for op in ops:
                desc = op.get("descriptions", {}).get(lang)
                if desc:
                    texts.append(format_prompt(desc, model_name, is_code=False))
                    op_ids.append(op["id"])

            if not texts:
                print(f"  {lang}: no descriptions, skipping")
                continue

            print(f"  {lang}: {len(texts)} descriptions")
            states = extractor.extract_batch(texts)
            save_hidden_states(
                states, RESULTS_DIR, model_name,
                tier=tier_num, modality="nl", lang=lang,
                metadata={"pooling": pooling, "n_layers": extractor.n_layers,
                           "hidden_dim": extractor.hidden_dim, "op_ids": op_ids},
            )

        # Extract code snippets
        code_texts = []
        code_ids = []
        for op in ops:
            code = op.get("code")
            if not code:
                # Tier 1 uses CODE_EQUIVALENTS from code_alignment.py
                from src.code_alignment import CODE_EQUIVALENTS
                code = CODE_EQUIVALENTS.get(op["id"])
            if code:
                code_texts.append(format_prompt(code, model_name, is_code=True))
                code_ids.append(op["id"])

        if code_texts:
            print(f"  code: {len(code_texts)} snippets")
            code_states = extractor.extract_batch(code_texts)
            save_hidden_states(
                code_states, RESULTS_DIR, model_name,
                tier=tier_num, modality="code",
                metadata={"pooling": pooling, "op_ids": code_ids},
            )

    print(f"\nDone: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="V2: Extract hidden states from LLMs")
    parser.add_argument("--model", type=str, default="all",
                        help="Model name or 'all' for all primary models")
    parser.add_argument("--pooling", type=str, default="last",
                        choices=["last", "mean"])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.model == "all":
        models = PRIMARY_MODELS
    else:
        models = [args.model]

    for model_name in models:
        extract_for_model(model_name, pooling=args.pooling, device=args.device)


if __name__ == "__main__":
    main()
