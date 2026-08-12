---
title: "KV Disk Cache"
---

# KV Disk Cache

Mesh can keep the KV cache for a long shared prompt prefix on disk, so the same
prefix does not have to be prefilled again after it falls out of memory — or
after the node restarts.

This matters most for agent workloads. An agent sends the same system prompt and
tool schemas on every turn with a different tail. Prefill cost grows roughly
quadratically with prefix length, while restoring a saved prefix is linear in
bytes, so **the bigger the shared prefix, the better this pays off**.

Measured on a 2-node split serving Qwen3-8B with a ~16.9k-token agent prompt:

| Scenario | Time to first token |
|---|---|
| Cold, nothing cached | 31.0s |
| Same prefix, new tail, later session | 1.3s |
| First request after restarting both nodes | 1.5s |

## Turning it on

The disk cache is **off by default**. It uses real disk space and writes model
state to it, so it is opt-in.

```sh
# Enable with the default budget
SKIPPY_KV_DISK_TIER=1 mesh-llm serve --model <model-ref>

# Or set an explicit budget, in MiB
SKIPPY_KV_DISK_TIER_MIB=8192 mesh-llm serve --model <model-ref>
```

| Variable | Effect |
|---|---|
| `SKIPPY_KV_DISK_TIER=1` | Enable with the default budget |
| `SKIPPY_KV_DISK_TIER_MIB=<mib>` | Enable with an explicit budget |
| `SKIPPY_KV_DISK_TIER_DIR=<path>` | Store the cache somewhere other than the default |

The budget is a **whole-node total**, shared across every model the node is
serving. It does not multiply by the number of loaded models.

By default the cache lives under `~/.mesh-llm/kv-cache/`. Put it on your fastest
local disk. Do not put it on a network filesystem: the cache relies on
memory-mapping files and on exclusive local file locking.

## When it will not turn on

Mesh declines to enable the disk cache, and says why on stderr, when:

- **There is no content digest for the model.** A cached prefix must be tied to
  the exact weights that produced it. A display name is not enough — two
  different GGUFs can be served under one name — so without a
  `manifest_sha256` or `source_model_sha256` the cache stays off rather than
  risk serving a prefix computed from different weights.
- **There is not enough free disk space.**
- **Another Mesh instance already owns the cache directory.** Only one process
  may use a cache directory at a time.

None of these stop the node serving. You simply get today's behaviour: every
prefix is recomputed.

## What is safe about it

A cached prefix is only reused when it was produced by an identical setup. The
cache key covers the model weights, the KV cache dtypes, flash attention mode,
the CPU/GPU layer split, the backend device, and which layers this stage owns.
Change any of those and old entries are ignored, not reinterpreted.

Every entry also carries checksums over both its bytes and the metadata that
describes how to interpret them. An entry that fails verification is deleted and
the request falls back to a normal prefill. A cache miss is cheap; a wrong
restore would be silently wrong output, so the cache always chooses the miss.

Interrupted writes and stale files are cleaned up the next time the node starts.

## Clearing it

Stop the node and delete the directory:

```sh
rm -rf ~/.mesh-llm/kv-cache
```

Everything in it is regenerable. Deleting it costs you a slow first request and
nothing else.

## Details

For the on-disk format, integrity guarantees, and versioning rules, see
[KV disk tier on-disk format](https://github.com/Mesh-LLM/mesh-llm/blob/main/docs/skippy/KV_DISK_TIER_FORMAT.md).
