# meshllm-native-runtime-darwin-aarch64-metal

This artifact contains MeshLLM native runtime shared libraries for:

- target: `aarch64-apple-darwin`
- backend: `metal`
- flavor: `metal`
- MeshLLM version: `0.72.1`
- Skippy ABI: `0.1.38`

`mesh-llm runtime install` reads `manifest.json`, verifies the archive
checksum from `native-runtimes.json`, installs the artifact into the
versioned native runtime cache, and loads these libraries before Skippy starts.
