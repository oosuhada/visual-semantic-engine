# Third-party source notice

This repository intentionally uses source snapshots without importing upstream Git history.

| Project | Source snapshot | Role |
| --- | --- | --- |
| OpenCLIP | `mlfoundations/open_clip` at `e0c00c8` | Primary model, training, tokenizer, transform, and checkpoint stack |
| OpenAI CLIP | `openai/CLIP` at `d05afc4` | Original CLIP Python package retained in `third_party/openai_clip` |

The OpenCLIP MIT license remains in the repository root. The OpenAI CLIP MIT license is retained at `third_party/openai_clip/LICENSE`.

The custom `visual_semantic_engine` package is an added application layer and does not claim authorship over the bundled upstream implementations.

