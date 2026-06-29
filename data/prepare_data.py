"""FusionLLM data prep: thin shim over the universal pipeline (custom 64K BPE)."""
import argparse
import sys
from pathlib import Path

import yaml


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LLM_ROOT = _PROJECT_ROOT.parent.parent  # .../CoreProjects/
for _p in (_PROJECT_ROOT, _LLM_ROOT):
    _p = str(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)


FUSIONLLM_TOKENIZER_NAME = "fusionllm-bpe-64k"
FUSIONLLM_VOCAB_SIZE = 64_000
FUSIONLLM_EOS_TOKEN_ID = 0
FUSIONLLM_PAD_TOKEN_ID = 2
FUSIONLLM_BOS_TOKEN_ID = 1


def _ensure_fusionllm_data_config(project_root: Path) -> Path:
    """Materialise a project-local data_config.yaml with FusionLLM's vocab."""
    from shared_data.config import UNIVERSAL_DATA_CONFIG_PATH
    from shared_data.common import load_yaml

    out_path = project_root / "data" / "data_config.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(UNIVERSAL_DATA_CONFIG_PATH)
    cfg["pipeline"]["tokenizer"]["name"] = FUSIONLLM_TOKENIZER_NAME
    cfg["pipeline"]["tokenizer"]["vocab_size"] = FUSIONLLM_VOCAB_SIZE
    cfg["pipeline"]["tokenizer"]["eos_token_id"] = FUSIONLLM_EOS_TOKEN_ID
    cfg["pipeline"]["tokenizer"]["pad_token_id"] = FUSIONLLM_PAD_TOKEN_ID
    cfg["_generator"] = "FusionLLM/data/prepare_data.py"
    cfg["_tokenizer_family"] = "fusionllm-bpe-64k"

    text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    out_path.write_text(text, encoding="utf-8")
    return out_path


def _apply_fusionllm_defaults() -> Path:
    from shared_data.config import UNIVERSAL_TOTAL_TOKENS
    print(f"[data/fusionllm] universal corpus: {UNIVERSAL_TOTAL_TOKENS:,} tokens")
    print(f"[data/fusionllm] tokenizer: {FUSIONLLM_TOKENIZER_NAME} "
          f"(vocab={FUSIONLLM_VOCAB_SIZE:,}, EOS={FUSIONLLM_EOS_TOKEN_ID})")
    print(f"[data/fusionllm] shard size: 50,000,000 tokens (uint32)")
    print(f"[data/fusionllm] NOTE: train the custom 64K BPE first via "
          f"`python -m shared_data.scripts.train_tokenizer --vocab-size 64000` "
          f"and set `data_config.yaml:tokenizer.path`.")
    return _ensure_fusionllm_data_config(Path(__file__).resolve().parents[1])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="FusionLLM data prep (delegates to universal pipeline)",
    )
    parser.add_argument("--stage", choices=["pretrain"], default="pretrain")
    parser.add_argument("--mixture", default=None)
    parser.add_argument("--data-config", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--skip-tokenize", action="store_true")
    parser.add_argument("--skip-pack", action="store_true")
    args = parser.parse_args()

    project_data_config = _apply_fusionllm_defaults()

    from shared_data.config import UNIVERSAL_MIXTURE_PATH
    from shared_data.prepare_data import run_pipeline

    return run_pipeline(
        mixture_path=Path(args.mixture) if args.mixture else UNIVERSAL_MIXTURE_PATH,
        data_config_path=Path(args.data_config) if args.data_config else project_data_config,
        source=args.source,
        skip_download=args.skip_download,
        skip_clean=args.skip_clean,
        skip_tokenize=args.skip_tokenize,
        skip_pack=args.skip_pack,
        data_root=Path(args.data_root) if args.data_root else None,
    )


if __name__ == "__main__":
    sys.exit(main())