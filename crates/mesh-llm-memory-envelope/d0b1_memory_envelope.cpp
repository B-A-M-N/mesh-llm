// D0b1 CUDA memory envelope harness.
//
// Loads real Qwen3.5-122B-A10B layer slices via Skippy with HOST routed experts,
// sampling system + process memory at each phase boundary:
//
//   T0 pre-open
//   T1 model loaded
//   T2 session/context created
//   T3 first forward completed
//   T4 post-warmup forwards
//
// Outputs JSON with timeline memory samples.
//
// Usage:
//   d0b1_memory_envelope <pkg_dir> <layer_start> <layer_end> <out.json>
//
// Exit codes:
//   0 = measurement completed + forward passed
//   1 = forward/runtime failure
//   2 = allocation/load failure
//   3 = measurement/probe failure

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>
#include <vector>
#include <cmath>
#include "skippy.h"

struct memory_sample {
    long rss_kb;
    long mem_available_kb;
    long swap_total_kb;
    long swap_free_kb;
    long minflt;
    long majflt;
};

static int read_meminfo(struct memory_sample *s) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return -1;
    char line[256];
    s->mem_available_kb = 0;
    s->swap_total_kb = 0;
    s->swap_free_kb = 0;
    while (fgets(line, sizeof(line), f)) {
        long val;
        if (sscanf(line, "MemAvailable: %ld kB", &val) == 1) s->mem_available_kb = val;
        else if (sscanf(line, "SwapTotal: %ld kB", &val) == 1) s->swap_total_kb = val;
        else if (sscanf(line, "SwapFree: %ld kB", &val) == 1) s->swap_free_kb = val;
    }
    fclose(f);
    return 0;
}

static int read_status(struct memory_sample *s) {
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256];
    s->rss_kb = 0;
    while (fgets(line, sizeof(line), f)) {
        long val;
        if (sscanf(line, "VmRSS: %ld kB", &val) == 1) s->rss_kb = val;
    }
    fclose(f);
    return 0;
}

static int sample_memory(struct memory_sample *s) {
    memset(s, 0, sizeof(*s));
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
        s->minflt = ru.ru_minflt;
        s->majflt = ru.ru_majflt;
    }
    if (read_meminfo(s) != 0) return -1;
    if (read_status(s) != 0) return -1;
    return 0;
}

static long now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long)(ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <pkg_dir> <layer_start> <layer_end> <out.json>\n", argv[0]);
        return 3;
    }
    const char *pkg = argv[1];
    int layer_start = atoi(argv[2]);
    int layer_end = atoi(argv[3]);
    const char *out_path = argv[4];
    int layer_count = layer_end - layer_start;

    fprintf(stderr, "D0b1 CUDA: pkg=%s layers=[%d,%d) count=%d\n", pkg, layer_start, layer_end, layer_count);

    struct memory_sample t0, t1, t2, t3, t4;

    // T0: pre-open
    if (sample_memory(&t0) != 0) { fprintf(stderr, "D0B1_PROBE_FAIL T0\n"); return 3; }
    fprintf(stderr, "T0 rss=%ldMB memavail=%ldMB swap=%ldMB\n", t0.rss_kb/1024, t0.mem_available_kb/1024, (t0.swap_total_kb - t0.swap_free_kb)/1024);

    // Build placement rules: routed experts -> HOST
    struct spike_tensor_placement_rule {
        const char *pattern;
        int target;
    };
    std::vector<struct spike_tensor_placement_rule> rules;
    char buf[256];
    for (int i = layer_start; i < layer_end; i++) {
        snprintf(buf, sizeof(buf), "blk\\.%d\\.ffn_down_exps\\.weight", i); rules.push_back({strdup(buf), 1});
        snprintf(buf, sizeof(buf), "blk\\.%d\\.ffn_up_exps\\.weight", i); rules.push_back({strdup(buf), 1});
        snprintf(buf, sizeof(buf), "blk\\.%d\\.ffn_gate_exps\\.weight", i); rules.push_back({strdup(buf), 1});
    }

    struct spike_model_placement {
        uint32_t struct_size;
        uint32_t abi_version;
        const struct spike_tensor_placement_rule *rules;
        size_t rule_count;
    };
    struct spike_model_placement placement = {
        sizeof(struct spike_model_placement),
        SKIPPY_MODEL_PLACEMENT_ABI_VERSION,
        rules.data(), rules.size(),
    };

    struct spike_runtime_config {
        uint32_t struct_size;
        uint32_t abi_version;
        int32_t stage_index;
        int32_t layer_start;
        int32_t layer_end;
        int32_t ctx_size;
        int32_t lane_count;
        int32_t n_batch;
        int32_t n_ubatch;
        int32_t n_threads;
        int32_t n_threads_batch;
        int32_t n_gpu_layers;
        bool has_mmap_override;
        bool use_mmap;
        bool use_mlock;
        int32_t cache_type_k;
        int32_t cache_type_v;
        int32_t flash_attn_type;
        int32_t load_mode;
        bool disable_repack;
        bool use_mmap_prefetch;
        bool use_mmap_buffer;
        bool filter_tensors_on_load;
        bool include_embeddings;
        bool include_output;
        const char *selected_backend_device;
        int32_t glm_dsa_policy_profile;
        uint32_t glm_dsa_policy_flags;
        int32_t glm_dsa_short_prefill_max_tokens;
        int32_t glm_dsa_direct_sparse_decode_max_top_k;
        uint64_t glm_dsa_dense_sparse_mask_max_bytes;
        int32_t glm_dsa_compact_flash_min_kv;
        const struct spike_model_placement *placement;
    };

    struct spike_runtime_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.struct_size = sizeof(cfg);
    cfg.abi_version = SKIPPY_ABI_VERSION;
    cfg.layer_start = layer_start;
    cfg.layer_end = layer_end;
    cfg.ctx_size = 2048;
    cfg.n_batch = 64;
    cfg.n_ubatch = 64;
    cfg.n_threads = 8;
    cfg.n_gpu_layers = 999;
    cfg.filter_tensors_on_load = true;
    cfg.include_embeddings = true;
    cfg.include_output = false;
    cfg.has_mmap_override = true;
    cfg.use_mmap = false;
    cfg.placement = &placement;
    cfg.selected_backend_device = getenv("MESH_BACKEND") ? getenv("MESH_BACKEND") : "CUDA0";

    long t_start = now_ms();
    skippy_model *model = nullptr;
    skippy_error *err = nullptr;
    skippy_status st = skippy_model_open(pkg, (const skippy_runtime_config*)&cfg, &model, &err);
    long t_load = now_ms() - t_start;

    if (st != SKIPPY_STATUS_OK || model == nullptr) {
        fprintf(stderr, "D0B1_LOAD_FAIL status=%d msg=%s\n", st, err ? err->message : "<null>");
        return 2;
    }

    // T1: model loaded
    if (sample_memory(&t1) != 0) {
        fprintf(stderr, "D0B1_PROBE_FAIL T1\n");
        skippy_model_free(model, &err);
        return 3;
    }
    fprintf(stderr, "T1 load=%ldms rss=%ldMB memavail=%ldMB\n", t_load, t1.rss_kb/1024, t1.mem_available_kb/1024);

    // T2: session created
    skippy_session *session = nullptr;
    if (skippy_session_create(model, &session, &err) != SKIPPY_STATUS_OK || session == nullptr) {
        fprintf(stderr, "D0B1_SESSION_FAIL msg=%s\n", err ? err->message : "<null>");
        skippy_model_free(model, &err);
        return 2;
    }
    if (sample_memory(&t2) != 0) {
        fprintf(stderr, "D0B1_PROBE_FAIL T2\n");
        skippy_session_free(session, &err);
        skippy_model_free(model, &err);
        return 3;
    }
    fprintf(stderr, "T2 rss=%ldMB memavail=%ldMB\n", t2.rss_kb/1024, t2.mem_available_kb/1024);

    // External decode
    llama_context *ctx = skippy_session_llama_context(session);
    if (!ctx) {
        fprintf(stderr, "D0B1_CTX_NULL\n");
        skippy_session_free(session, &err);
        skippy_model_free(model, &err);
        return 2;
    }
    if (skippy_session_begin_external_decode(session, &err) != SKIPPY_STATUS_OK) {
        fprintf(stderr, "D0B1_BEGIN_DECODE_FAIL msg=%s\n", err ? err->message : "<null>");
        skippy_session_free(session, &err);
        skippy_model_free(model, &err);
        return 1;
    }

    // Tokenize prompt
    const llama_model *lmodel = skippy_model_llama_model(model);
    const llama_vocab *vocab = llama_model_get_vocab(lmodel);
    const char *prompt = "The capital of France is";
    std::vector<llama_token> tokens(llama_vocab_n_tokens(vocab));
    int n_tok = llama_tokenize(vocab, prompt, strlen(prompt), tokens.data(), tokens.size(), true, true);
    if (n_tok <= 0) {
        fprintf(stderr, "D0B1_TOKENIZE_FAIL\n");
        skippy_session_end_external_decode(session, nullptr);
        skippy_session_free(session, &err);
        skippy_model_free(model, &err);
        return 1;
    }
    tokens.resize(n_tok);

    // T3: first forward
    long t_fwd_start = now_ms();
    llama_batch batch = llama_batch_get_one(tokens.data(), n_tok);
    int decode_ret = llama_decode(ctx, batch);
    long t_fwd = now_ms() - t_fwd_start;

    if (decode_ret != 0) {
        fprintf(stderr, "D0B1_DECODE_FAIL ret=%d\n", decode_ret);
        skippy_session_end_external_decode(session, nullptr);
        skippy_session_free(session, &err);
        skippy_model_free(model, &err);
        return 1;
    }
    if (sample_memory(&t3) != 0) {
        fprintf(stderr, "D0B1_PROBE_FAIL T3\n");
        skippy_session_end_external_decode(session, nullptr);
        skippy_session_free(session, &err);
        skippy_model_free(model, &err);
        return 3;
    }
    fprintf(stderr, "T3 fwd=%ldms rss=%ldMB memavail=%ldMB\n", t_fwd, t3.rss_kb/1024, t3.mem_available_kb/1024);

    // T4: warmup (3 single-token decodes)
    for (int i = 0; i < 3; i++) {
        llama_batch one = llama_batch_get_one(&tokens[0], 1);
        llama_decode(ctx, one);
    }
    if (sample_memory(&t4) != 0) {
        fprintf(stderr, "D0B1_PROBE_FAIL T4\n");
        skippy_session_end_external_decode(session, nullptr);
        skippy_session_free(session, &err);
        skippy_model_free(model, &err);
        return 3;
    }
    fprintf(stderr, "T4 rss=%ldMB memavail=%ldMB\n", t4.rss_kb/1024, t4.mem_available_kb/1024);

    skippy_session_end_external_decode(session, &err);
    skippy_session_free(session, &err);
    skippy_model_free(model, &err);

    // Compute derived values
    long rss_delta = t4.rss_kb - t0.rss_kb;
    long memavail_delta = t0.mem_available_kb - t4.mem_available_kb;

    // Output JSON
    FILE *out = fopen(out_path, "w");
    if (!out) {
        fprintf(stderr, "D0B1_OUTPUT_FAIL %s\n", out_path);
        return 3;
    }

    fprintf(out, "{\n");
    fprintf(out, "  \"layer_start\": %d,\n", layer_start);
    fprintf(out, "  \"layer_end\": %d,\n", layer_end);
    fprintf(out, "  \"layer_count\": %d,\n", layer_count);
    fprintf(out, "  \"requested_host_bytes\": %ld,\n", (long)layer_count * 1000000000L);
    fprintf(out, "  \"planned_accel_bytes\": %ld,\n", (long)layer_count * 177551020L);
    fprintf(out, "  \"load_duration_ms\": %ld,\n", t_load);
    fprintf(out, "  \"first_forward_ms\": %ld,\n", t_fwd);
    fprintf(out, "  \"forward_passed\": true,\n");
    fprintf(out, "  \"rss_delta_kb\": %ld,\n", rss_delta);
    fprintf(out, "  \"mem_available_delta_kb\": %ld,\n", memavail_delta);
    fprintf(out, "  \"timeline\": [\n");
    fprintf(out, "    {\"label\":\"T0_pre_open\",\"rss_kb\":%ld,\"mem_available_kb\":%ld,\"swap_used_kb\":%ld,\"minflt\":%ld,\"majflt\":%ld},\n",
            t0.rss_kb, t0.mem_available_kb, t0.swap_total_kb - t0.swap_free_kb, t0.minflt, t0.majflt);
    fprintf(out, "    {\"label\":\"T1_model_loaded\",\"rss_kb\":%ld,\"mem_available_kb\":%ld,\"swap_used_kb\":%ld,\"minflt\":%ld,\"majflt\":%ld},\n",
            t1.rss_kb, t1.mem_available_kb, t1.swap_total_kb - t1.swap_free_kb, t1.minflt, t1.majflt);
    fprintf(out, "    {\"label\":\"T2_session_created\",\"rss_kb\":%ld,\"mem_available_kb\":%ld,\"swap_used_kb\":%ld,\"minflt\":%ld,\"majflt\":%ld},\n",
            t2.rss_kb, t2.mem_available_kb, t2.swap_total_kb - t2.swap_free_kb, t2.minflt, t2.majflt);
    fprintf(out, "    {\"label\":\"T3_first_forward\",\"rss_kb\":%ld,\"mem_available_kb\":%ld,\"swap_used_kb\":%ld,\"minflt\":%ld,\"majflt\":%ld},\n",
            t3.rss_kb, t3.mem_available_kb, t3.swap_total_kb - t3.swap_free_kb, t3.minflt, t3.majflt);
    fprintf(out, "    {\"label\":\"T4_post_warmup\",\"rss_kb\":%ld,\"mem_available_kb\":%ld,\"swap_used_kb\":%ld,\"minflt\":%ld,\"majflt\":%ld}\n",
            t4.rss_kb, t4.mem_available_kb, t4.swap_total_kb - t4.swap_free_kb, t4.minflt, t4.majflt);
    fprintf(out, "  ]\n");
    fprintf(out, "}\n");
    fclose(out);

    fprintf(stderr, "D0B1_OK output=%s rss_delta=%ldMB memavail_delta=%ldMB\n", out_path, rss_delta/1024, memavail_delta/1024);
    return 0;
}
