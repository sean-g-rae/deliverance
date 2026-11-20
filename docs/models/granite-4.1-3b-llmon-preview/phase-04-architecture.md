# Phase 4: Granite 4.1 Architecture

## Sources

- Checkpoint config: `docs/models/granite-4.1-3b-llmon-preview/config.json`
- Hugging Face Granite config: https://github.com/huggingface/transformers/blob/main/src/transformers/models/granite/configuration_granite.py
- Hugging Face Granite implementation: https://github.com/huggingface/transformers/blob/main/src/transformers/models/granite/modeling_granite.py
- Existing Reckoning Granite 4.0 hybrid implementation: `core/src/main/java/io/teknek/reckoning/model/granitemoehybrid/`
- Local Transformers checkout: `/ai-code/transformers`

## Confirmed Model Family

The checkpoint declares `model_type: granite` and `architectures: [GraniteForCausalLM]`. It is a dense decoder-only transformer and is not the existing `GraniteMoeHybrid` model. LLMON is tokenizer/chat-template behavior and does not add a model block.

## Confirmed Shape And Scaling Invariants

| Invariant | Value |
| --- | ---: |
| Hidden size | 2560 |
| Intermediate size | 8192 |
| Decoder layers | 40 |
| Query heads | 40 |
| Key/value heads | 8 |
| Head size | 64 |
| GQA group size | 5 |
| Vocabulary size | 100352 |
| Context length | 131072 |
| RoPE theta | 10000000 |
| RMSNorm epsilon | 1e-5 |
| Embedding multiplier | 12.0 |
| Attention multiplier | 0.015625 |
| Residual multiplier | 0.22 |
| Logits scaling | 10.0 |

The checkpoint uses bias-free attention and MLP projections, SiLU gated MLP, untied word embeddings, and `use_cache: false` in the config metadata. The Transformers implementation applies the embedding multiplier before the decoder, the residual multiplier after attention and MLP sublayers, the attention multiplier as attention scaling, and divides final logits by `logits_scaling`.

## Local Transformers Test Scan

The local checkout contains the generic Granite suite at:

```text
tests/models/granite/test_modeling_granite.py
```

Portable test candidates:

- Generic config construction and common config tests.
- Integer `embedding_multiplier` and `logits_scaling` save/load round-trip.
- Tiny model forward with hidden-state shape assertions.
- Generic generation/model mixin coverage where Reckoning has an equivalent.

The slow real-checkpoint tests use `ibm/PowerLM-3b`, not `ibm-granite/granite-4.1-3b-llmon-preview`. No `LLMON` or `granite-4.1-3b-llmon-preview` references were found in the local Transformers source or tests. CUDA/XPU/pipeline/offload tests are not direct Reckoning port targets.

LLMON-specific behavior remains sourced from the checkpoint chat template and the separate LLMON repository fixture documented in Phase 3.

## Reckoning Port Boundary

The existing Granite hybrid classes cannot be reused as the model implementation because they include Mamba and MoE layer behavior. The dense port should introduce a separate Granite package and model class, then register `GRANITE` in both config parsing and model construction.

## Delivered In This Phase

- Added `GraniteConfig` with the checkpoint's scaling, bias, RoPE, cache, and architecture fields.
- Registered `GRANITE` for config parsing in `AutoModelForCausaLm`.
- Added `GraniteConfigTest`, which parses the real dossier config and asserts representative dimensions, GQA, token IDs, scaling values, and architecture metadata.
- Ported the Transformers integer multiplier round-trip characterization for `embedding_multiplier` and `logits_scaling`.
- Added the initial dense `GraniteModel` TensorRef path using the existing causal attention, MLP, RMSNorm, and `TransformerBlock2` components.
- Registered `GRANITE` config parsing and model construction in `AutoModelForCausaLm`.

Validation:

```text
Tests run: 2, Failures: 0, Errors: 0
```

## Implementation Check

The core module compiles with the initial Granite model class. A tiny forward/hidden-state test is not claimed yet; tensor loading, tiny forward parity, and logits validation remain outstanding.

## Next Phase

Implement the dense Granite decoder model and its real checkpoint tensor-name mapping. Before generation support, add a tiny synthetic checkpoint using the real Granite tensor names and assert representative forward shapes.
