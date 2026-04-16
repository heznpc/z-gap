"""V2 Hidden State extraction from decoder LLMs.

Extracts per-layer representations from open-weight models
(Llama, CodeLlama, Qwen, DeepSeek) for layer-wise Z analysis.
"""

import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ExtractionConfig:
    model_name: str
    device: str = "cuda"
    dtype: str = "float16"
    pooling: str = "last"          # "last" or "mean"
    save_dtype: str = "float16"    # storage precision

    @property
    def torch_dtype(self):
        return {"float16": torch.float16, "bfloat16": torch.bfloat16,
                "float32": torch.float32}[self.dtype]

    @property
    def np_dtype(self):
        return {"float16": np.float16, "float32": np.float32}[self.save_dtype]


# --- Primary models (per experiment design) ---

PRIMARY_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
]

SECONDARY_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "bigcode/starcoder2-15b",
    "Qwen/Qwen3-Coder-Next",
]


class HiddenStateExtractor:
    """Extract per-layer hidden states from a decoder LLM."""

    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=config.torch_dtype,
            output_hidden_states=True,
            trust_remote_code=True,
        ).to(config.device)
        self.model.eval()
        self.n_layers = self.model.config.num_hidden_layers
        self.hidden_dim = self.model.config.hidden_size

    def extract_single(self, text: str) -> np.ndarray:
        """Extract hidden states for one text.

        Returns: array of shape (n_layers+1, hidden_dim)
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.config.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # outputs.hidden_states: tuple of (n_layers+1) tensors
        # each shape (1, seq_len, hidden_dim)
        layers = []
        for layer_h in outputs.hidden_states:
            if self.config.pooling == "last":
                vec = layer_h[0, -1, :]
            elif self.config.pooling == "mean":
                mask = inputs["attention_mask"][0].unsqueeze(-1).float()
                vec = (layer_h[0] * mask).sum(dim=0) / mask.sum(dim=0)
            else:
                raise ValueError(f"Unknown pooling: {self.config.pooling}")
            layers.append(vec.cpu().float().numpy())

        return np.stack(layers)  # (n_layers+1, hidden_dim)

    def extract_batch(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        """Extract hidden states for a batch of texts.

        Returns: array of shape (n_texts, n_layers+1, hidden_dim)
        """
        from tqdm import tqdm

        all_results = []
        for i in tqdm(range(0, len(texts), batch_size),
                      desc=f"Extracting {self.config.model_name.split('/')[-1]}"):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=2048
            ).to(self.config.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            attn_mask = inputs["attention_mask"]  # (B, seq_len)

            for b_idx in range(len(batch)):
                layers = []
                for layer_h in outputs.hidden_states:
                    if self.config.pooling == "last":
                        # Find actual last token (before padding)
                        seq_len = attn_mask[b_idx].sum().item()
                        vec = layer_h[b_idx, seq_len - 1, :]
                    elif self.config.pooling == "mean":
                        mask = attn_mask[b_idx].unsqueeze(-1).float()
                        vec = (layer_h[b_idx] * mask).sum(dim=0) / mask.sum(dim=0)
                    else:
                        raise ValueError(f"Unknown pooling: {self.config.pooling}")
                    layers.append(vec.cpu().float().numpy())
                all_results.append(np.stack(layers))

        return np.stack(all_results)  # (n_texts, n_layers+1, hidden_dim)


def format_prompt(text: str, model_name: str, is_code: bool = False) -> str:
    """Format input text according to model's expected template.

    Instruct models use chat template; base models use raw text.
    """
    model_lower = model_name.lower()

    # Base models: raw text
    if "instruct" not in model_lower and "chat" not in model_lower:
        return text

    # Llama-style instruct
    if "llama" in model_lower:
        return (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{text}<|eot_id|>"
        )

    # Qwen-style instruct
    if "qwen" in model_lower:
        return f"<|im_start|>user\n{text}<|im_end|>"

    # DeepSeek-style
    if "deepseek" in model_lower:
        return f"User: {text}\n\nAssistant:"

    # Fallback
    return text


def save_hidden_states(
    states: np.ndarray,
    output_dir: Path,
    model_name: str,
    tier: int,
    modality: str,  # "nl" or "code"
    lang: str | None = None,
    metadata: dict | None = None,
    save_dtype: str = "float16",
):
    """Save extracted hidden states to disk.

    Directory structure:
        results/hidden_states/{model_name}/
            tier{N}_{modality}_{lang}.npz
            metadata.json
    """
    safe_name = model_name.split("/")[-1]
    model_dir = output_dir / safe_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Filename
    if modality == "nl" and lang:
        fname = f"tier{tier}_{modality}_{lang}.npz"
    else:
        fname = f"tier{tier}_{modality}.npz"

    # Convert dtype for storage
    dtype = {"float16": np.float16, "float32": np.float32}[save_dtype]
    np.savez_compressed(model_dir / fname, states=states.astype(dtype))

    # Save/update metadata
    meta_path = model_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {"model": model_name, "files": {}}

    meta["files"][fname] = {
        "shape": list(states.shape),
        "dtype": save_dtype,
        "modality": modality,
        "tier": tier,
        "lang": lang,
    }
    if metadata:
        meta.update(metadata)

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def load_hidden_states(
    output_dir: Path,
    model_name: str,
    tier: int,
    modality: str,
    lang: str | None = None,
) -> np.ndarray:
    """Load hidden states from disk."""
    safe_name = model_name.split("/")[-1]
    model_dir = output_dir / safe_name

    if modality == "nl" and lang:
        fname = f"tier{tier}_{modality}_{lang}.npz"
    else:
        fname = f"tier{tier}_{modality}.npz"

    return np.load(model_dir / fname)["states"]
