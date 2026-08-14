// Stage materializer for D0 memory envelope sweep.
//
// Uses Skippy FFI directly to materialize [0,N) stage artifacts from the
// Qwen3.5-122B-A10B layer package. Avoids Rust native-runtime loading.
//
// Usage:
//   d0-materialize <pkg_dir> <layer_end> <out_path>
//
// Examples:
//   d0-materialize /mnt/mesh-models/Qwen3.5-122B-A10B-Q3_K_XL-layer-package 4 /tmp/d0-stage-0_4.gguf
//   d0-materialize /mnt/mesh-models/Qwen3.5-122B-A10B-Q3_K_XL-layer-package 30 /tmp/d0-stage-0_30.gguf
//
// Exit codes:
//   0 = stage materialized + verified
//   1 = materialization failed
//   2 = verification failed

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>
#include <string>
#include "skippy.h"

static long now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long)(ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <pkg_dir> <layer_end> <out_path>\n", argv[0]);
        return 2;
    }
    const char *pkg = argv[1];
    int layer_end = atoi(argv[2]);
    const char *out_path = argv[3];
    int layer_start = 0;
    int layer_count = layer_end - layer_start;

    fprintf(stderr, "D0-MATERIALIZE: pkg=%s range=[0,%d) count=%d out=%s\n",
            pkg, layer_end, layer_count, out_path);

    // Collect individual layer GGUF paths + shared parts
    std::vector<std::string> layer_paths;
    char path[4096];
    
    // Add shared/metadata.gguf first (required for model metadata)
    snprintf(path, sizeof(path), "%s/shared/metadata.gguf", pkg);
    {
        FILE *f = fopen(path, "r");
        if (f) {
            fclose(f);
            layer_paths.push_back(std::string(path));
        } else {
            fprintf(stderr, "D0-MATERIALIZE-WARN missing %s\n", path);
        }
    }
    
    // Add layers
    for (int i = layer_start; i < layer_end; i++) {
        snprintf(path, sizeof(path), "%s/layers/layer-%03d.gguf", pkg, i);
        FILE *f = fopen(path, "r");
        if (f) {
            fclose(f);
            layer_paths.push_back(std::string(path));
        } else {
            fprintf(stderr, "D0-MATERIALIZE-WARN missing layer %d at %s\n", i, path);
        }
    }
    
    // Add shared/embeddings.gguf (required for include_embeddings=true)
    snprintf(path, sizeof(path), "%s/shared/embeddings.gguf", pkg);
    {
        FILE *f = fopen(path, "r");
        if (f) {
            fclose(f);
            layer_paths.push_back(std::string(path));
        } else {
            fprintf(stderr, "D0-MATERIALIZE-WARN missing %s\n", path);
        }
    }
    
    // Add shared/output.gguf (optional, but included for completeness)
    snprintf(path, sizeof(path), "%s/shared/output.gguf", pkg);
    {
        FILE *f = fopen(path, "r");
        if (f) {
            fclose(f);
            layer_paths.push_back(std::string(path));
        } else {
            fprintf(stderr, "D0-MATERIALIZE-WARN missing %s\n", path);
        }
    }

    // Count actual layer files (exclude shared parts)
    size_t actual_layers = 0;
    for (const auto &p : layer_paths) {
        if (p.find("/layers/") != std::string::npos) actual_layers++;
    }
    if ((int)actual_layers != layer_count) {
        fprintf(stderr, "D0-MATERIALIZE-FAIL expected %d layers, found %zu\n",
                layer_count, actual_layers);
        return 1;
    }

    fprintf(stderr, "D0-MATERIALIZE: found %zu layer files\n", layer_paths.size());

    // Build argv array for skippy_write_gguf_from_parts
    std::vector<const char *> parts;
    for (const auto &p : layer_paths) {
        parts.push_back(p.c_str());
    }

    skippy_error *err = nullptr;
    long t_write_start = now_ms();
    skippy_status st = skippy_write_gguf_from_parts(
        parts.data(), parts.size(), out_path, &err);
    long t_write = now_ms() - t_write_start;

    if (st != SKIPPY_STATUS_OK) {
        fprintf(stderr, "D0-MATERIALIZE-FAIL write status=%d msg=%s\n",
                st, err ? err->message : "<null>");
        return 1;
    }

    fprintf(stderr, "D0-MATERIALIZE: wrote %s in %ldms\n", out_path, t_write);

    // Verify: reopen the stage and check metadata
    skippy_model_info *vinfo = nullptr;
    st = skippy_model_info_open(out_path, &vinfo, &err);
    if (st != SKIPPY_STATUS_OK) {
        fprintf(stderr, "D0-MATERIALIZE-VERIFY-FAIL cannot reopen stage status=%d msg=%s\n",
                st, err ? err->message : "<null>");
        return 2;
    }

    size_t vcount = 0;
    skippy_model_info_tensor_count(vinfo, &vcount, &err);
    fprintf(stderr, "D0-MATERIALIZE-VERIFY: stage_tensors=%zu path=%s\n", vcount, out_path);

    skippy_model_info_free(vinfo, &err);

    // Output JSON summary
    FILE *out = fopen(out_path, "r+b");
    long file_size = 0;
    if (out) {
        fseek(out, 0, SEEK_END);
        file_size = ftell(out);
        fclose(out);
        fprintf(stderr, "D0-MATERIALIZE-OK path=%s size=%ld bytes (%.2f GiB) write_ms=%ld\n",
                out_path, file_size, (double)file_size / (1024.0*1024.0*1024.0), t_write);
    } else {
        fprintf(stderr, "D0-MATERIALIZE-OK path=%s write_ms=%ld\n", out_path, t_write);
    }

    // Output machine-readable JSON to stdout
    printf("{\n");
    printf("  \"source\": \"%s\",\n", pkg);
    printf("  \"output\": \"%s\",\n", out_path);
    printf("  \"layer_start\": %d,\n", layer_start);
    printf("  \"layer_end\": %d,\n", layer_end);
    printf("  \"layer_count\": %d,\n", layer_count);
    printf("  \"tensor_count\": %zu,\n", vcount);
    printf("  \"write_ms\": %ld\n", t_write);
    printf("}\n");

    return 0;
}
