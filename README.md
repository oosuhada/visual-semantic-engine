# Visual Semantic Engine

Local-first semantic image indexing and retrieval built directly on the OpenCLIP training and inference codebase.

This repository starts from a **clean source snapshot** rather than a fork. It contains no upstream Git history. The original OpenCLIP implementation remains available in `src/open_clip`, while the original OpenAI CLIP Python package is preserved under `third_party/openai_clip` for compatibility and study.

The new layer in this repository turns that model stack into one practical workflow:

```text
Image Directory
    ↓
OpenCLIP / OpenAI-pretrained CLIP weights
    ↓
Normalized image embeddings
    ↓
Persistent local semantic index
    ↓
Natural-language or image query
    ↓
Ranked matching files
```

## What is added here

- `visual-semantic index`: recursively index a local image directory.
- `visual-semantic search --text`: search images with natural language.
- `visual-semantic search --image`: find visually/semantically similar images.
- Persistent NumPy + JSON index format with model metadata.
- OpenAI CLIP weights through OpenCLIP's `pretrained=openai` path, avoiding two parallel runtime implementations.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

visual-semantic index ./photos --output ./artifacts/my-index

visual-semantic search ./artifacts/my-index \
  --text "a dark wine bar with warm ambient lighting" \
  --top-k 10
```

The default model is `ViT-B-32` with the original OpenAI pretrained weights. Any model/tag supported by the bundled OpenCLIP code can be selected with `--model` and `--pretrained`.

## Image-query search

```bash
visual-semantic search ./artifacts/my-index \
  --image ./query.jpg \
  --top-k 10
```

## Why this is not another CLIP reimplementation

The encoder, tokenizer, transforms, pretrained-weight loading, and model definitions are used directly from the imported OpenCLIP source snapshot. The custom code only owns indexing, persistence, and retrieval behavior.

## Source lineage

See [`THIRD_PARTY_NOTICE.md`](THIRD_PARTY_NOTICE.md). Original license files and copyright notices are retained.

## Large artifacts

Model weights, generated indexes, Hugging Face caches, and local datasets are intentionally excluded from Git.

