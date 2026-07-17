# MLX wiring in mesh-llm

`skippy-engine-mlx` is a workspace member and the Apple-Silicon serving engine
used by the host runtime's optional `mlx` feature. It supports both whole-model
OpenAI serving and explicit mesh-managed dense-Llama stage chains. Current
status and evidence are in `SERVE_INTEGRATION_STATUS.md` and
`STAGED_EXECUTION.md`.

## Dependency and publication shape

The crate manifest uses ordinary crates.io requirements:

- `safemlx = 0.1.3`
- `safemlx-lm = 0.4.1`
- `safemlx-lm-utils = 0.1.4`

The repository root patches those packages to public safemlx commit
`4e53c5e`. That commit contains the loader/model behavior certified by this
draft; the published releases do not yet contain every required correction.
Using `[patch.crates-io]` keeps the workspace reproducibly pinned without
putting illegal git dependencies in the published crate manifest.

`skippy-engine-mlx` is therefore part of `scripts/publish-crates.sh` after its
workspace dependencies and before `mesh-llm-host-runtime`. Both
`repo-consistency publish-crates` and `repo-consistency release-targets` must
pass when this dependency shape changes.

## Feature and platform gating

- MLX code and native dependencies are behind the crate's `mlx` feature.
- Host integration is behind `mesh-llm-host-runtime/mlx` and macOS target
  dependencies.
- Capability advertisement additionally requires `target_arch = "aarch64"`.
- Enabling MLX also enables the dynamic llama native runtime. MLX and static
  llama.cpp both expose GGUF C symbols, so keeping llama.cpp in a separate
  dynamic link unit avoids duplicate-symbol failures.
- An MLX-only start may proceed without an installed llama native runtime.
  Actual GGUF/Skippy loading still checks and reports that requirement.

Use the dedicated recipes so the complete Xcode Metal toolchain is selected
and the generated library is copied beside the executable:

```bash
just mlx-build
just mlx-release-build
```

Both produce `mesh-llm` and a sibling `mlx.metallib`; the build verifies that
the copied resource is byte-identical.

## Whole-model selection

On an MLX build, a resolved SafeTensors model routes to `MlxModelHandle` before
the GGUF path. The automatic weight policy is intentionally conservative:

- eligible unquantized dense families request affine-4/group-64 at load time;
- Inkling and Nemotron-H load their native representation because their routed
  rank-3 experts do not support that transform;
- checkpoints already declaring quantization/compression load natively;
- explicit `mlx-serve` users may choose auto, none, affine4, affine8, or mxfp4.

The model is exposed through the shared `openai-frontend` router. The HTTP
listener binds before runtime readiness is reported.

## Split selection

Explicit `mesh-llm serve --split` uses metadata-only checkpoint description,
additive MLX peer capability, ordinary resource-aware topology planning, exact
per-stage HTTP tensor ranges, recipe-keyed affine artifacts, and the existing
Skippy binary stage wire. The coordinator retains tokenizer/config sidecars but
does not download model tensor payloads.

Automatic split selection and partial adapters beyond dense Llama remain
future work. Inkling and Nemotron-H whole-model support must not be confused
with certified partial-stage support.
