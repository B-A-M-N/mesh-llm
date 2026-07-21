# Suffix N-gram Proposer

## Purpose

The suffix N-gram proposer is a request-local prompt-lookup source for staged
speculative decoding. It extends a native-MTP prefix by retrieving the
continuation that followed the longest exact suffix seen earlier in the prompt
or target-committed output.

The proposer is designed for coding-agent and tool-loop traffic with large
verbatim regions: file re-emission with a localized edit, repeated structured
configuration, and continuations grounded in earlier tool output. It is not a
semantic model and makes no proposal when the request contains no sufficiently
long exact match.

The target model remains authoritative. Suffix tokens are candidates only;
verification commits the matching prefix and replaces the first mismatch with
the target token. The implementation refinements in this document were made
after maintainer review.[^review-changes]

## Position in the Hybrid Pipeline

Native MTP and suffix retrieval have complementary roles:

```text
target-committed history ──► incremental exact-seed index
          │                              │
          ├──► native MTP prefix ────────┤ read-only lookup query
          │                              ▼
          │                       copied suffix tail
          │                              │
          └──────────────► [MTP prefix | suffix tail]
                                      │
                                      ▼
                              target verification
                                      │
                                      ▼
                            newly committed history
```

1. Native MTP predicts the first uncertain or changed token from model state.
2. The suffix proposer treats that MTP output as a read-only continuation of
   target-committed history.
3. If the extended suffix matches an earlier request position, the proposer
   copies the tokens that followed that position.
4. The target verifies the MTP prefix and suffix tail as one candidate while
   preserving their boundary for acceptance telemetry.

The MTP continuation never mutates suffix history. Only prompt tokens and
target-committed generation tokens enter the index.

## Lookup Algorithm

Configuration supplies:

- `ngram_min`: minimum exact suffix match; at least 3;
- `ngram_max`: maximum backward comparison window; at most 64;
- `ngram_max_proposal_tokens`: independent output-token budget.

The index seed length is `min(ngram_min, 8)`. Each target-committed token that
completes a seed adds its end position to an `AHashMap<SeedKey, Vec<u32>>`.
`SeedKey` stores the exact seed tokens rather than a precomputed fingerprint,
so different token sequences never share an entry merely because their
fingerprints collide.

For each proposal:

1. Incrementally append newly committed tokens and their completed seeds.
2. Build the current seed from committed history plus the optional read-only
   MTP continuation.
3. Visit every earlier position indexed by that exact seed.
4. Compare tokens backwards, up to `ngram_max`, and retain the longest match.
   Equal-length matches prefer the most recent position.
5. Stay silent when the longest match is shorter than `ngram_min`.
6. Copy the tokens following the selected position. Draft length is bounded by
   the caller budget, `ngram_max_proposal_tokens`, twice the match length, and
   the historical continuation available.

Index construction is O(committed tokens), normal synchronization is O(newly
committed tokens), and lookup is O(candidate positions multiplied by the
bounded comparison window). A divergence or request reset rebuilds the index.

## Why AHashMap

The lookup operates on small fixed-size token keys and is called on the decode
hot path. `AHashMap` provides a fast randomized map on the supported platforms
without relying on a hand-written fingerprint or a SIMD-only implementation.
Exact `SeedKey` equality remains the correctness boundary; target verification
remains the output-correctness boundary.

The map choice should be revisited only if profiling shows hashing rather than
candidate comparison or model execution is material. Telemetry exposes lookup
and synchronization time for that purpose.

## Configuration

Suffix is currently a mesh-config-selected native-MTP extension:

```toml
[models.speculative]
strategy = "mtp"
ngram_proposer = "suffix"
ngram_min = 5
ngram_max = 32
ngram_max_proposal_tokens = 48
extension_initial_tokens = 2
extension_max_tokens = 48
extension_tail_backoff_proposals = 2
verify_window_min_tokens = 1
verify_window_max_tokens = 32
verify_window_pipeline_depth = 2
```

It is not yet a valid `model-package.json` proposer type. Package preflight and
package strategy resolution continue to accept `ngram-simple` and
`ngram-cache` only.

The cache and suffix limits intentionally differ. Cache uses llama.cpp's
stateful lookup with a match window no larger than four tokens. Suffix may use
a much longer exact match. For both proposers, proposal output length is
independent from match-window length.

| Proposer | State | Match horizon | Candidate source | Hybrid MTP role |
|---|---|---:|---|---|
| Simple | None | Up to 4 tokens | Accepted token history | Standalone fallback |
| Cache | Request-local llama.cpp cache | Up to 4 tokens | Recent cache match | MTP sidecar |
| Suffix | Request-local exact-seed index | Up to 64 tokens | Longest earlier exact suffix | MTP sidecar |

## Telemetry

Request summaries and response timings identify the configured proposer and
report:

- proposal attempts, hits, and proposed tokens;
- suffix match-length sum and maximum;
- candidates examined;
- incrementally appended tokens and rebuilds;
- synchronization and lookup microseconds;
- native-MTP tokens, N-gram tail tokens, and accepted counts;
- N-gram tail rejections and sidecar backoff.

Prompt text, token IDs, candidate tokens, filesystem paths, endpoints, and raw
node identities are not exported.

## Correctness Invariants

- The index is scoped to one request.
- Only target-committed history mutates the index.
- An MTP continuation is read-only.
- A divergent history clears and rebuilds the index.
- Candidate selection compares exact tokens; hashes do not establish matches.
- Verification, not retrieval, determines committed output.
- A suffix rejection must not count as a native-MTP rejection when the MTP
  prefix was accepted.

## Expected Workloads and Failure Modes

Long exact repeats can provide enough future tokens to keep a staged
verification pipeline occupied. Novel prose or novel code normally produces no
suffix proposal. Short or common matches can select the wrong continuation;
`ngram_min`, sidecar backoff, bounded verification windows, and stale-work
telemetry control that cost.

Retaining all exact-seed positions avoids silently dropping an older long
prompt match. Highly repetitive inputs may therefore increase candidate
comparison work, which is visible through candidate and lookup-time telemetry.

## Benchmark Contract

A reportable comparison requires:

- an MTP-capable model on at least a two-stage split;
- MTP-only, simple, cache, and suffix arms on the same topology;
- release builds, identical sampling and verification settings, and sequential
  arms on shared hardware;
- long coding/edit workloads plus a low-overlap control;
- warmups, per-sample artifacts, wall throughput, server throughput,
  acceptance, output hashes, and finish reasons;
- pipeline occupancy, simultaneous stage compute, downstream wait, and stale
  work when claiming latency hiding;
- explicit activation evidence showing the suffix proposer produced tokens.

The expected outcome is a hypothesis until measured. Acceptance rate alone is
not the success criterion: a lower-acceptance arm may still win throughput when
its longer horizon keeps distributed stages busy.

[^review-changes]: Follow-up changes requested by [@i386](https://github.com/i386) replaced the original FNV fingerprint and eight-position buckets with exact `SeedKey` entries in `AHashMap`, made history synchronization incremental, separated proposal budget from match length, tightened suffix validation, corrected source attribution, and required reproducible benchmark evidence. These changes preserve the original prompt-lookup design while making its behavior measurable on long coding workloads.
