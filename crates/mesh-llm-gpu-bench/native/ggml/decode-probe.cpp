#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#if defined(MESH_LLM_GGML_PROBE_METAL)
#include "ggml-metal.h"
#endif

#if defined(MESH_LLM_GGML_PROBE_CUDA)
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr int WARMUP_RUNS = 3;
constexpr int TIMED_RUNS = 12;
constexpr int GRAPH_WARMUP_RUNS = 1;
constexpr int GRAPH_TIMED_RUNS = 3;
constexpr int SOURCE_BOUNDARY_TIMED_RUNS = 7;
constexpr int64_t MAX_MODEL_SHAPED_MOE_EXPERTS = 128;
constexpr int64_t DEEP_LLAMA_GRAPH_LAYERS[] = {4, 8};
constexpr int GRAPH_FEATURE_ATTENTION_Q_NORM = 1 << 0;
constexpr int GRAPH_FEATURE_ATTENTION_K_NORM = 1 << 1;
constexpr int GRAPH_FEATURE_ATTENTION_POST_NORM = 1 << 2;
constexpr int GRAPH_FEATURE_FFN_POST_NORM = 1 << 3;

enum ProbeBackend {
    PROBE_BACKEND_METAL = 0,
    PROBE_BACKEND_CUDA = 1,
    PROBE_BACKEND_HIP = 2,
};

enum ProbeDepth {
    PROBE_DEPTH_STANDARD = 0,
    PROBE_DEPTH_DEEP = 1,
};

enum ProbeTensorType {
    PROBE_TENSOR_Q4_K = 0,
    PROBE_TENSOR_Q6_K = 1,
    PROBE_TENSOR_Q8_0 = 2,
    PROBE_TENSOR_F16 = 3,
};

struct GraphInventoryBucket {
    std::string family;
    int64_t ggml_op;
    uint64_t ggml_type;
    uint64_t node_count;
    uint64_t element_count;
    uint64_t output_bytes;
    uint64_t src0_bytes;
    uint64_t src1_bytes;
    std::array<int64_t, 4> ne;
};

struct ProbeResult {
    std::string name;
    std::string tensor_type;
    int64_t rows;
    int64_t cols;
    int64_t graph_node_count;
    std::vector<GraphInventoryBucket> graph_inventory;
    double effective_gbps;
    double tflops;
    double elapsed_ms;
    double min_elapsed_ms;
    double max_elapsed_ms;
    double spread_pct;
    int graph_features;
    int runs;
};

struct ProbeShape {
    const char * suffix;
    int64_t rows;
    int64_t cols;
};

struct SamplerProbeResult {
    double history_us_per_token;
    double vocab_us_per_token;
    int64_t history_tokens;
    int64_t vocab_tokens;
    int runs;
};

struct SourceSamplerCandidate {
    int32_t id;
    float logit;
    float p;
};

struct ScheduledGraph {
    ggml_backend_sched_t sched;
    ggml_backend_t cpu_backend;
};

struct EncodedWeightCache {
    std::unordered_map<std::string, std::vector<uint8_t>> encoded_by_shape;
};

constexpr ProbeShape DECODE_SHAPES[] = {
    {"square_2048", 2048, 2048},
    {"square_4096", 4096, 4096},
    {"ffn_up_4096_12288", 12288, 4096},
    {"ffn_down_12288_4096", 4096, 12288},
    {"expert_2048_128", 128, 2048},
};

constexpr ProbeShape LLAMA_GRAPH_SHAPES[] = {
    {"768_2048", 768, 2048},
    {"1024_4096", 1024, 4096},
    {"2048_6144", 2048, 6144},
    {"2560_9728", 2560, 9728},
    {"4096_12288", 4096, 12288},
};

constexpr ProbeShape LLAMA_GQA_GRAPH_SHAPES[] = {
    {"2048_kv1024_6144", 2048, 6144},
    {"2560_kv1024_9728", 2560, 9728},
    {"4096_kv1024_12288", 4096, 12288},
};

char * copy_c_string(const std::string & value) {
    char * out = static_cast<char *>(std::malloc(value.size() + 1));
    if (out != nullptr) {
        std::memcpy(out, value.c_str(), value.size() + 1);
    }
    return out;
}

void set_error(char ** error_out, const std::string & message) {
    if (error_out != nullptr) {
        *error_out = copy_c_string(message);
    }
}

ggml_backend_t init_backend(int backend_kind) {
    switch (backend_kind) {
        case PROBE_BACKEND_METAL:
#if defined(MESH_LLM_GGML_PROBE_METAL)
            return ggml_backend_metal_init();
#else
            return nullptr;
#endif
        case PROBE_BACKEND_CUDA:
#if defined(MESH_LLM_GGML_PROBE_CUDA)
            return ggml_backend_cuda_init(0);
#else
            return nullptr;
#endif
        case PROBE_BACKEND_HIP:
#if defined(MESH_LLM_GGML_PROBE_CUDA)
            return ggml_backend_cuda_init(0);
#else
            return nullptr;
#endif
        default:
            return nullptr;
    }
}

enum ggml_type probe_tensor_type(int tensor_type_kind) {
    switch (tensor_type_kind) {
        case PROBE_TENSOR_Q4_K:
            return GGML_TYPE_Q4_K;
        case PROBE_TENSOR_Q6_K:
            return GGML_TYPE_Q6_K;
        case PROBE_TENSOR_Q8_0:
            return GGML_TYPE_Q8_0;
        case PROBE_TENSOR_F16:
            return GGML_TYPE_F16;
        default:
            return GGML_TYPE_COUNT;
    }
}

bool quantized_row_width_supported(enum ggml_type type, int64_t cols) {
    const int64_t block_size = ggml_blck_size(type);
    return block_size > 0 && cols > 0 && cols % block_size == 0;
}

bool matrix_shape_supported(enum ggml_type type, int64_t rows, int64_t cols) {
    return rows > 0 && quantized_row_width_supported(type, cols);
}

bool dense_llama_shape_supported(
    enum ggml_type type,
    int64_t hidden,
    int64_t kv_width,
    int64_t ffn) {
    // GGML quantized tensor storage is row-blocked along ne[0]. Real GGUF
    // loading enforces `ne[0] % ggml_blck_size(type) == 0`; a model can still
    // be valid because mixed-quant files choose compatible tensor types per
    // tensor. The synthetic validator probes ask a stricter question by forcing
    // one candidate type across an entire llama-shaped block. When a small
    // model has widths such as hidden=576 and kv_width=192, an all-Q6_K graph is
    // not a source-plausible GGUF tensor layout even if the model file contains
    // other Q6_K tensors. Reject those probe shapes before graph construction so
    // backend planners never see an impossible quantized tensor.
    return matrix_shape_supported(type, hidden, hidden)
        && matrix_shape_supported(type, kv_width, hidden)
        && matrix_shape_supported(type, ffn, hidden)
        && matrix_shape_supported(type, hidden, ffn);
}

bool moe_shape_supported(
    enum ggml_type type,
    int64_t hidden,
    int64_t expert_width) {
    return matrix_shape_supported(type, expert_width, hidden)
        && matrix_shape_supported(type, hidden, expert_width);
}

const char * probe_tensor_type_name(int tensor_type_kind) {
    switch (tensor_type_kind) {
        case PROBE_TENSOR_Q4_K:
            return "q4_k";
        case PROBE_TENSOR_Q6_K:
            return "q6_k";
        case PROBE_TENSOR_Q8_0:
            return "q8_0";
        case PROBE_TENSOR_F16:
            return "f16";
        default:
            return "unknown";
    }
}

std::vector<float> deterministic_f32(int64_t count, uint32_t salt) {
    std::vector<float> values(static_cast<size_t>(count));
    uint32_t state = 0x9e3779b9u ^ salt;
    for (int64_t i = 0; i < count; ++i) {
        state = state * 1664525u + 1013904223u;
        const float centered = static_cast<float>((state >> 8) & 0xffffu) / 32768.0f - 1.0f;
        values[static_cast<size_t>(i)] = centered * 0.125f;
    }
    return values;
}

int64_t graph_node_count(ggml_cgraph * graph) {
    return graph != nullptr ? static_cast<int64_t>(ggml_graph_n_nodes(graph)) : 0;
}

bool graph_name_contains(const char * name, const char * needle) {
    return name != nullptr && needle != nullptr && std::strstr(name, needle) != nullptr;
}

bool graph_is_attention_weight_name(const char * name) {
    return graph_name_contains(name, "attn_q") || graph_name_contains(name, "attn_k") ||
        graph_name_contains(name, "attn_v") || graph_name_contains(name, "attn_output") ||
        graph_name_contains(name, "attn_qkv") || graph_name_contains(name, "attn_out");
}

bool graph_is_feed_forward_weight_name(const char * name) {
    return graph_name_contains(name, "ffn_gate") || graph_name_contains(name, "ffn_up") ||
        graph_name_contains(name, "ffn_down");
}

const char * graph_inventory_family(const ggml_tensor * node) {
    if (node == nullptr) {
        return "unknown";
    }
    const char * name = node->name;
    const char * src0_name = node->src[0] != nullptr ? node->src[0]->name : nullptr;
    if (node->op == GGML_OP_MUL_MAT_ID) {
        return "moe_matmul_id";
    }
    if (graph_name_contains(name, "ffn_moe") || graph_name_contains(name, "expert")) {
        return "moe_runtime";
    }
    if (node->op == GGML_OP_MUL_MAT) {
        if (graph_name_contains(name, "Qcur") || graph_name_contains(name, "Kcur") ||
            graph_name_contains(name, "Vcur") || graph_name_contains(name, "wqkv") ||
            graph_name_contains(name, "kqv") || graph_name_contains(name, "attn") ||
            graph_is_attention_weight_name(src0_name)) {
            return "attention_matmul";
        }
        if (graph_name_contains(name, "ffn_") || graph_is_feed_forward_weight_name(src0_name)) {
            return "ffn_matmul";
        }
        if (graph_name_contains(name, "result_output") || graph_name_contains(name, "output") ||
            graph_name_contains(name, "logits")) {
            return "output_matmul";
        }
        return "matmul";
    }
    if (graph_name_contains(name, "kq") || graph_name_contains(name, "fattn") ||
        node->op == GGML_OP_FLASH_ATTN_EXT || node->op == GGML_OP_SOFT_MAX) {
        return "attention_runtime";
    }
    if (graph_name_contains(name, "cache") || graph_name_contains(name, "kv") ||
        graph_name_contains(name, "k_idxs") || graph_name_contains(name, "v_idxs")) {
        return "kv_cache";
    }
    if (graph_name_contains(name, "ffn_")) {
        return "ffn_runtime";
    }
    if (graph_name_contains(name, "norm") || node->op == GGML_OP_RMS_NORM ||
        node->op == GGML_OP_NORM) {
        return "normalization";
    }
    if (graph_name_contains(name, "result_output") || graph_name_contains(name, "logits")) {
        return "output_runtime";
    }
    return "other";
}

std::vector<GraphInventoryBucket> collect_graph_inventory(ggml_cgraph * graph) {
    std::vector<GraphInventoryBucket> buckets;
    if (graph == nullptr) {
        return buckets;
    }
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const ggml_tensor * node = ggml_graph_node(graph, i);
        if (node == nullptr) {
            continue;
        }
        const char * family = graph_inventory_family(node);
        auto bucket = std::find_if(
            buckets.begin(),
            buckets.end(),
            [&](const GraphInventoryBucket & existing) {
                return existing.ggml_op == static_cast<int64_t>(node->op) &&
                    existing.ggml_type == static_cast<uint64_t>(node->type) &&
                    existing.family == family &&
                    existing.ne == std::array<int64_t, 4>{node->ne[0], node->ne[1], node->ne[2], node->ne[3]};
            });
        if (bucket == buckets.end()) {
            GraphInventoryBucket created{};
            created.family = family;
            created.ggml_op = static_cast<int64_t>(node->op);
            created.ggml_type = static_cast<uint64_t>(node->type);
            for (int dim = 0; dim < 4; ++dim) {
                created.ne[dim] = node->ne[dim];
            }
            buckets.push_back(created);
            bucket = std::prev(buckets.end());
        }
        bucket->node_count++;
        bucket->element_count += static_cast<uint64_t>(std::max<int64_t>(0, ggml_nelements(node)));
        bucket->output_bytes += static_cast<uint64_t>(ggml_nbytes(node));
        if (node->src[0] != nullptr) {
            bucket->src0_bytes += static_cast<uint64_t>(ggml_nbytes(node->src[0]));
        }
        if (node->src[1] != nullptr) {
            bucket->src1_bytes += static_cast<uint64_t>(ggml_nbytes(node->src[1]));
        }
    }
    return buckets;
}

bool kq_mask_type_supported(enum ggml_type type) {
    return type == GGML_TYPE_F16 || type == GGML_TYPE_F32;
}

int64_t source_decode_position_for_run(int64_t context_tokens, int run_index) {
    const int64_t context = std::max<int64_t>(1, context_tokens);
    return std::max<int64_t>(0, context - 1 - static_cast<int64_t>(run_index & 3));
}

std::vector<uint8_t> empty_kq_mask_bytes(const ggml_tensor * mask) {
    if (mask == nullptr || !kq_mask_type_supported(mask->type)) {
        return {};
    }
    return std::vector<uint8_t>(ggml_nbytes(mask), 0);
}

void fill_source_shaped_kq_mask_bytes(
    const ggml_tensor * mask,
    std::vector<uint8_t> & encoded,
    int64_t active_position,
    int run_index) {
    if (mask == nullptr || encoded.empty()) {
        return;
    }
    const int64_t elements = ggml_nelements(mask);
    if (elements <= 0 || encoded.size() < ggml_nbytes(mask)) {
        return;
    }

    const int64_t keep_until = std::clamp<int64_t>(active_position, 0, elements - 1);
    const int64_t perturb_cell = std::clamp<int64_t>(keep_until - (run_index & 3), 0, elements - 1);
    const float perturb = static_cast<float>((run_index & 7) + 1) * 0.0001f;

    // llama.cpp rebuilds KQ mask tensors in `set_input_kq_mask()` before each
    // decode graph submit. That work is source-visible even though the exact
    // backend kernels are not part of the transformer matmuls: the host walks
    // the active KV cells, writes keep/drop values, then uploads the whole mask
    // input. Tiny models are dominated by these fixed per-token source-boundary
    // costs, so mutating a single byte of a prebuilt mask would make the probe
    // claim confidence it has not earned. This loop intentionally follows the
    // same coarse source shape, while still avoiding model-observed tok/s or
    // backend-name constants.
    //
    // The extra sequence/cell bookkeeping below mirrors the shape of
    // llama.cpp's current `set_input_kq_mask_impl`: initialize the sequence
    // position table, create per-stream reuse maps/vectors, then walk KV cells
    // deciding keep/drop from sequence membership and position. For a synthetic
    // one-token decode the data are simple, but the source-visible work is not
    // free. Paying for it in the submission probe is especially important for
    // tiny models where this CPU-side path can rival the submitted matmul graph.
    std::array<int32_t, 1024> seq_pos_min{};
    seq_pos_min.fill(std::numeric_limits<int32_t>::max());
    seq_pos_min[0] = static_cast<int32_t>(keep_until);
    std::unordered_map<int32_t, uint32_t> seq_srct;
    std::unordered_map<int32_t, std::vector<uint32_t>> seq_idxs;
    seq_srct.reserve(1);
    seq_idxs.reserve(1);
    auto & idxs = seq_idxs[0];
    idxs.reserve(static_cast<size_t>(std::min<int64_t>(elements, 64)));
    seq_srct[0] = 0;

    if (mask->type == GGML_TYPE_F16) {
        auto * values = reinterpret_cast<ggml_fp16_t *>(encoded.data());
        const ggml_fp16_t keep = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t drop = ggml_fp32_to_fp16(-INFINITY);
        for (int64_t i = 0; i < elements; ++i) {
            const bool cell_is_empty = i > keep_until;
            const bool same_sequence = !cell_is_empty;
            const int32_t pos = static_cast<int32_t>(std::min<int64_t>(i, keep_until));
            if (same_sequence && pos + 32 >= seq_pos_min[0]) {
                idxs.push_back(static_cast<uint32_t>(i));
            }
            values[i] = same_sequence && pos <= static_cast<int32_t>(keep_until) ? keep : drop;
        }
        values[perturb_cell] = ggml_fp32_to_fp16(perturb);
        volatile size_t source_bookkeeping_sink = idxs.size() + seq_srct.size();
        (void) source_bookkeeping_sink;
        return;
    }

    if (mask->type == GGML_TYPE_F32) {
        auto * values = reinterpret_cast<float *>(encoded.data());
        for (int64_t i = 0; i < elements; ++i) {
            const bool cell_is_empty = i > keep_until;
            const bool same_sequence = !cell_is_empty;
            const int32_t pos = static_cast<int32_t>(std::min<int64_t>(i, keep_until));
            if (same_sequence && pos + 32 >= seq_pos_min[0]) {
                idxs.push_back(static_cast<uint32_t>(i));
            }
            values[i] = same_sequence && pos <= static_cast<int32_t>(keep_until) ? 0.0f : -INFINITY;
        }
        values[perturb_cell] = perturb;
        volatile size_t source_bookkeeping_sink = idxs.size() + seq_srct.size();
        (void) source_bookkeeping_sink;
    }
}

struct SourceDecodeBookkeepingScratch {
    std::vector<uint64_t> input_descriptor_hashes;
    std::vector<uint32_t> rollback_cells;
    std::vector<int32_t> output_ids;
    std::vector<int64_t> out_ids;
    std::vector<int32_t> seq_output_count;
    std::vector<int32_t> seq_pos_max_rm;
    std::vector<int32_t> token_history;
};

void run_source_decode_bookkeeping(
    ggml_cgraph * graph,
    int64_t layers,
    int64_t context_tokens,
    int64_t hidden,
    int64_t vocab,
    int run_index,
    SourceDecodeBookkeepingScratch & scratch,
    volatile uint64_t & sink) {
    const int64_t context = std::max<int64_t>(1, context_tokens);
    const int64_t active_position = source_decode_position_for_run(context, run_index);

    // This intentionally models source-visible work around a reused
    // `llama_context::decode()` call, not a backend kernel:
    //
    // - output bookkeeping scans output flags and computes buffer offsets;
    // - memory-context preparation finds a KV slot and preserves enough cell
    //   state to roll back on failure;
    // - graph reuse walks input descriptors before `set_inputs()`;
    // - sampled decode keeps per-sequence output ids around logits extraction.
    //
    // The real objects live inside llama.cpp, so this probe cannot call them
    // without loading the GGUF model. Instead it mirrors the memory-write shape
    // from `llama_context::decode()`, `llama_kv_cache::prepare()`,
    // `llm_graph_result::can_reuse()`, and Skippy's token-history append using
    // only GGUF-derived dimensions. Keep reusable scratch outside this helper:
    // llama.cpp reuses batch/output/session storage across generated tokens too,
    // so per-token heap allocation would be the wrong source model.
    scratch.seq_output_count.assign(1024, 0);
    scratch.seq_output_count[0] = 1;

    scratch.seq_pos_max_rm.assign(1024, -1);

    const uint32_t graph_nodes =
        graph != nullptr ? static_cast<uint32_t>(std::max(0, ggml_graph_n_nodes(graph))) : 0;
    uint64_t local = static_cast<uint64_t>(run_index + 1);
    local += static_cast<uint64_t>(std::max<int64_t>(1, vocab));
    local += static_cast<uint64_t>(std::max<int64_t>(1, hidden));
    local += static_cast<uint64_t>(scratch.seq_output_count[0]);

    const uint32_t input_descriptors = graph_nodes > 0
        ? std::min<uint32_t>(graph_nodes, static_cast<uint32_t>(layers * 8 + 16))
        : static_cast<uint32_t>(layers * 8 + 16);
    scratch.input_descriptor_hashes.resize(input_descriptors);
    for (uint32_t i = 0; i < input_descriptors; ++i) {
        const uint64_t expected_tokens = 1;
        const uint64_t expected_outputs = 1;
        const uint64_t shape_hash =
            (static_cast<uint64_t>(i + 1) * 1315423911ULL) ^
            static_cast<uint64_t>(context) ^
            (static_cast<uint64_t>(active_position) << 7);
        scratch.input_descriptor_hashes[i] = shape_hash + expected_tokens + expected_outputs;
        local ^= scratch.input_descriptor_hashes[i];
    }

    scratch.rollback_cells.clear();
    scratch.rollback_cells.reserve(static_cast<size_t>(std::min<int64_t>(context, layers + 4)));
    uint32_t head_cur = static_cast<uint32_t>(active_position % context);
    const uint32_t n_test = 1;
    uint32_t n_tested = 0;
    while (n_tested < static_cast<uint32_t>(std::min<int64_t>(context, layers + 4))) {
        if (head_cur + n_test > static_cast<uint32_t>(context)) {
            head_cur = 0;
            continue;
        }
        scratch.rollback_cells.push_back(head_cur);
        local += head_cur;
        head_cur++;
        n_tested++;
    }

    for (uint32_t cell : scratch.rollback_cells) {
        scratch.seq_pos_max_rm[0] = std::max(scratch.seq_pos_max_rm[0], static_cast<int32_t>(cell));
    }

    scratch.output_ids.resize(1);
    scratch.out_ids.resize(1);
    scratch.output_ids[0] = -1;
    scratch.out_ids[0] = 0;
    for (size_t i = 0; i < scratch.out_ids.size(); ++i) {
        const int64_t out_id = scratch.out_ids[i];
        scratch.output_ids[static_cast<size_t>(out_id)] = static_cast<int32_t>(i);
        local += static_cast<uint64_t>(scratch.output_ids[static_cast<size_t>(out_id)] + 1);
    }

    if (scratch.token_history.capacity() < static_cast<size_t>(context)) {
        scratch.token_history.reserve(static_cast<size_t>(context));
    }
    if (scratch.token_history.size() >= static_cast<size_t>(context)) {
        scratch.token_history.clear();
    }
    scratch.token_history.push_back(static_cast<int32_t>((active_position + run_index) & 0x7fffffff));
    local += static_cast<uint64_t>(scratch.token_history.back());

    const size_t logits_bytes =
        static_cast<size_t>(std::max<int64_t>(1, vocab)) * sizeof(float);
    const size_t output_id_bytes = sizeof(int32_t);
    local += static_cast<uint64_t>(logits_bytes + output_id_bytes);
    local += static_cast<uint64_t>(scratch.seq_pos_max_rm[0] + 1);
    sink += local;
}

bool graph_supported_by_backend(ggml_backend_t backend, ggml_cgraph * graph) {
    // Probe timings are only useful to model-fit when they measure the same
    // backend boundary that llama.cpp would use for a source-visible decode
    // graph. Checking only the final output tensor is too weak for synthetic
    // graphs that include Flash Attention, KV-cache writes, RoPE, MoE routing,
    // or logits handoff nodes: the final tensor can look supportable while an
    // interior op is unsupported, delegated differently, or rejected by the
    // backend planner. In that case the honest result is "no probe evidence",
    // not a partial timing that model-fit could mistake for hardware truth.
    if (backend == nullptr || graph == nullptr) {
        return false;
    }
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_tensor * node = ggml_graph_node(graph, i);
        if (node == nullptr) {
            return false;
        }
        if (!ggml_backend_supports_op(backend, node)) {
            return false;
        }
    }
    return true;
}

bool source_sampler_candidate_desc(
    const SourceSamplerCandidate & left,
    const SourceSamplerCandidate & right) {
    return left.logit > right.logit;
}

double median(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    if (values.size() % 2 == 1) {
        return values[middle];
    }
    return (values[middle - 1] + values[middle]) * 0.5;
}

void source_sampler_top_k(std::vector<SourceSamplerCandidate> & candidates, int32_t k, bool & sorted) {
    if (k <= 0 || candidates.empty()) {
        return;
    }
    const size_t keep = std::min<size_t>(static_cast<size_t>(k), candidates.size());
    if (!sorted) {
        // Keep this intentionally close to llama.cpp's `llama_sampler_top_k_impl`.
        // For the default top_k=40 path, llama.cpp uses `std::partial_sort`
        // over the full vocabulary candidate array. A cheaper nth-element
        // surrogate under-measures tiny models because sampler work dominates
        // once the transformer graph is only a few milliseconds.
        std::partial_sort(
            candidates.begin(),
            candidates.begin() + static_cast<std::ptrdiff_t>(keep),
            candidates.end(),
            source_sampler_candidate_desc);
        sorted = true;
    }
    candidates.resize(keep);
}

void source_sampler_softmax(std::vector<SourceSamplerCandidate> & candidates, bool sorted) {
    if (candidates.empty()) {
        return;
    }
    float max_logit = candidates[0].logit;
    if (!sorted) {
        for (size_t i = 1; i < candidates.size(); ++i) {
            max_logit = std::max(max_logit, candidates[i].logit);
        }
    }
    float total = 0.0f;
    for (auto & candidate : candidates) {
        candidate.p = std::exp(candidate.logit - max_logit);
        total += candidate.p;
    }
    if (total <= 0.0f) {
        return;
    }
    for (auto & candidate : candidates) {
        candidate.p /= total;
    }
}

void source_sampler_top_p(std::vector<SourceSamplerCandidate> & candidates, float top_p, bool & sorted) {
    if (top_p >= 1.0f || top_p <= 0.0f || candidates.empty()) {
        return;
    }
    source_sampler_softmax(candidates, sorted);
    if (!sorted) {
        std::partial_sort(
            candidates.begin(),
            candidates.end(),
            candidates.end(),
            source_sampler_candidate_desc);
        sorted = true;
    }
    float cumulative = 0.0f;
    size_t keep = candidates.size();
    for (size_t i = 0; i < candidates.size(); ++i) {
        cumulative += candidates[i].p;
        if (cumulative >= top_p) {
            keep = i + 1;
            break;
        }
    }
    candidates.resize(std::max<size_t>(keep, 1));
}

void source_sampler_min_p(std::vector<SourceSamplerCandidate> & candidates, float min_p, bool & sorted) {
    if (min_p <= 0.0f || candidates.empty()) {
        return;
    }
    if (!sorted) {
        std::partial_sort(
            candidates.begin(),
            candidates.end(),
            candidates.end(),
            source_sampler_candidate_desc);
        sorted = true;
    }
    const float min_logit = candidates[0].logit + std::log(min_p);
    size_t keep = 1;
    for (; keep < candidates.size(); ++keep) {
        if (candidates[keep].logit < min_logit) {
            break;
        }
    }
    candidates.resize(std::max<size_t>(keep, 1));
}

void source_sampler_temperature(std::vector<SourceSamplerCandidate> & candidates, float temperature) {
    if (temperature <= 0.0f) {
        return;
    }
    for (auto & candidate : candidates) {
        candidate.logit /= temperature;
    }
}

int32_t source_sampler_dist(std::vector<SourceSamplerCandidate> & candidates, bool sorted, std::mt19937 & rng) {
    if (candidates.empty()) {
        return -1;
    }
    float max_logit = candidates[0].logit;
    if (!sorted) {
        for (size_t i = 1; i < candidates.size(); ++i) {
            max_logit = std::max(max_logit, candidates[i].logit);
        }
    }
    double total = 0.0;
    for (auto & candidate : candidates) {
        candidate.p = std::exp(candidate.logit - max_logit);
        total += candidate.p;
    }
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    const double target = total * dist(rng);
    double running = 0.0;
    for (auto & candidate : candidates) {
        running += candidate.p;
        candidate.p = total > 0.0 ? static_cast<float>(candidate.p / total) : 0.0f;
        if (running >= target) {
            return candidate.id;
        }
    }
    return candidates.back().id;
}

double measure_source_sampler_history_once(int64_t history_tokens) {
    std::vector<int32_t> tokens;
    tokens.reserve(static_cast<size_t>(history_tokens));
    for (int64_t i = 0; i < history_tokens; ++i) {
        tokens.push_back(static_cast<int32_t>((i * 1103 + 17) & 0xffff));
    }
    std::unordered_map<int32_t, int32_t> accepted;
    accepted.reserve(static_cast<size_t>(std::min<int64_t>(history_tokens, 65536)));
    uint64_t state = 0;
    const auto started = std::chrono::steady_clock::now();
    for (const int32_t token : tokens) {
        // Default Skippy chat sampling has top-k/top-p/min-p/temp/dist in the
        // chain and no repeat penalties. Most `llama_sampler_accept()` calls are
        // therefore cheap no-ops, but the source path still walks token history
        // before first-token sampling. This loop keeps a deterministic
        // stateful accept-shaped lower bound without charging model-specific
        // prompt content.
        const int32_t count = ++accepted[token];
        state = state * 1099511628211ull + static_cast<uint32_t>(token) + static_cast<uint32_t>(count);
    }
    const auto finished = std::chrono::steady_clock::now();
    volatile uint64_t sink = state;
    (void) sink;
    return std::chrono::duration<double>(finished - started).count()
        * 1000000.0
        / static_cast<double>(std::max<int64_t>(history_tokens, 1));
}

double measure_source_sampler_vocab_once(int64_t vocab_tokens) {
    const std::vector<float> logits = deterministic_f32(vocab_tokens, 947);
    std::mt19937 rng(0x5eed1234u);
    const auto started = std::chrono::steady_clock::now();
    std::vector<SourceSamplerCandidate> candidates;
    candidates.reserve(static_cast<size_t>(vocab_tokens));
    for (int64_t token = 0; token < vocab_tokens; ++token) {
        candidates.push_back({
            static_cast<int32_t>(token),
            logits[static_cast<size_t>(token)],
            0.0f,
        });
    }
    bool sorted = false;
    source_sampler_top_k(candidates, 40, sorted);
    source_sampler_top_p(candidates, 0.95f, sorted);
    source_sampler_min_p(candidates, 0.05f, sorted);
    source_sampler_temperature(candidates, 0.8f);
    const int32_t selected = source_sampler_dist(candidates, sorted, rng);
    const auto finished = std::chrono::steady_clock::now();
    volatile int32_t sink = selected;
    (void) sink;
    return std::chrono::duration<double>(finished - started).count()
        * 1000000.0
        / static_cast<double>(std::max<int64_t>(vocab_tokens, 1));
}

bool run_source_sampler_probe(
    int64_t vocab_tokens,
    int64_t history_tokens,
    SamplerProbeResult & result) {
    if (vocab_tokens <= 0 || history_tokens <= 0) {
        return false;
    }
    for (int i = 0; i < WARMUP_RUNS; ++i) {
        (void) measure_source_sampler_history_once(history_tokens);
        (void) measure_source_sampler_vocab_once(vocab_tokens);
    }
    std::vector<double> history_samples;
    std::vector<double> vocab_samples;
    history_samples.reserve(TIMED_RUNS);
    vocab_samples.reserve(TIMED_RUNS);
    for (int i = 0; i < TIMED_RUNS; ++i) {
        history_samples.push_back(measure_source_sampler_history_once(history_tokens));
        vocab_samples.push_back(measure_source_sampler_vocab_once(vocab_tokens));
    }
    result = {
        median(history_samples),
        median(vocab_samples),
        history_tokens,
        vocab_tokens,
        TIMED_RUNS,
    };
    return result.history_us_per_token > 0.0 && result.vocab_us_per_token > 0.0;
}

std::vector<uint8_t> encode_weights(
    enum ggml_type type,
    const std::vector<float> & weights,
    int64_t rows,
    int64_t cols) {
    const size_t encoded_bytes = ggml_row_size(type, cols) * rows;
    std::vector<uint8_t> encoded(encoded_bytes);
    if (type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(
            weights.data(),
            reinterpret_cast<ggml_fp16_t *>(encoded.data()),
            static_cast<int64_t>(weights.size()));
        return encoded;
    }
    ggml_quantize_chunk(type, weights.data(), encoded.data(), 0, rows, cols, nullptr);
    return encoded;
}

std::string encoded_weight_cache_key(enum ggml_type type, int64_t rows, int64_t cols) {
    return std::to_string(static_cast<int>(type))
        + ":"
        + std::to_string(rows)
        + "x"
        + std::to_string(cols);
}

const std::vector<uint8_t> & cached_encoded_weights(
    EncodedWeightCache & cache,
    enum ggml_type type,
    int64_t rows,
    int64_t cols) {
    const std::string key = encoded_weight_cache_key(type, rows, cols);
    auto existing = cache.encoded_by_shape.find(key);
    if (existing != cache.encoded_by_shape.end()) {
        return existing->second;
    }
    // The synthetic probes measure graph topology, backend scheduling, and
    // quantized kernel traffic. They do not test numerical accuracy. Repeated
    // model-shaped graphs contain the same small set of tensor shapes in every
    // layer, so quantizing fresh random weights for each layer only measures
    // CPU-side benchmark setup. Reusing one deterministic encoded blob per
    // `(type, rows, cols)` preserves tensor type, byte size, layout, and GGML op
    // support while keeping validation time proportional to the graph we time,
    // not to redundant host quantization.
    std::vector<float> weight_f32 = deterministic_f32(rows * cols, 17);
    auto inserted = cache.encoded_by_shape.emplace(
        key,
        encode_weights(type, weight_f32, rows, cols));
    return inserted.first->second;
}

ProbeResult make_probe_result(
    const std::string & name,
    const std::string & tensor_type,
    int64_t rows,
    int64_t cols,
    int64_t graph_node_count,
    double effective_gbps,
    double tflops,
    double median_seconds,
    const std::vector<double> & seconds,
    int graph_features,
    int runs) {
    const auto [min_it, max_it] = std::minmax_element(seconds.begin(), seconds.end());
    const double elapsed_ms = median_seconds * 1000.0;
    const double min_elapsed_ms = seconds.empty() ? elapsed_ms : *min_it * 1000.0;
    const double max_elapsed_ms = seconds.empty() ? elapsed_ms : *max_it * 1000.0;
    const double spread_pct =
        elapsed_ms > 0.0 ? ((max_elapsed_ms - min_elapsed_ms) / elapsed_ms) * 100.0 : 0.0;
    return ProbeResult{
        name,
        tensor_type,
        rows,
        cols,
        graph_node_count,
        std::vector<GraphInventoryBucket>{},
        effective_gbps,
        tflops,
        elapsed_ms,
        min_elapsed_ms,
        max_elapsed_ms,
        spread_pct,
        graph_features,
        runs,
    };
}

bool run_probe(
    ggml_backend_t backend,
    enum ggml_type type,
    const char * name,
    const char * tensor_type,
    const ProbeShape & shape,
    ProbeResult & result) {
    if (!matrix_shape_supported(type, shape.rows, shape.cols)) {
        return false;
    }
    const size_t context_bytes = ggml_tensor_overhead() * 8 + ggml_graph_overhead();
    ggml_init_params params{};
    params.mem_size = context_bytes;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }

    ggml_tensor * weights = ggml_new_tensor_2d(ctx, type, shape.cols, shape.rows);
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, shape.cols, 1);
    ggml_tensor * output = ggml_mul_mat(ctx, weights, input);
    ggml_set_name(weights, "ggml_decode_probe_weights");
    ggml_set_name(input, "ggml_decode_probe_input");
    ggml_set_name(output, "ggml_decode_probe_output");
    ggml_set_output(output);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    if (!graph_supported_by_backend(backend, graph)) {
        ggml_free(ctx);
        return false;
    }

    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    if (cpu_backend == nullptr) {
        ggml_free(ctx);
        return false;
    }
    ggml_backend_t backends[] = { backend, cpu_backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends,
        nullptr,
        2,
        GGML_DEFAULT_GRAPH_SIZE,
        false,
        true);
    if (sched == nullptr) {
        ggml_backend_free(cpu_backend);
        ggml_free(ctx);
        return false;
    }
    if (!ggml_backend_sched_alloc_graph(sched, graph)) {
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu_backend);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> weight_f32 = deterministic_f32(shape.rows * shape.cols, 17);
    std::vector<uint8_t> weight_encoded = encode_weights(type, weight_f32, shape.rows, shape.cols);
    std::vector<float> input_f32 = deterministic_f32(shape.cols, 29);
    ggml_backend_tensor_set(weights, weight_encoded.data(), 0, weight_encoded.size());
    ggml_backend_tensor_set(input, input_f32.data(), 0, input_f32.size() * sizeof(float));
    ggml_backend_synchronize(backend);

    auto compute_once = [&]() -> double {
        const auto started = std::chrono::steady_clock::now();
        enum ggml_status status = ggml_backend_sched_graph_compute_async(sched, graph);
        ggml_backend_sched_synchronize(sched);
        const auto finished = std::chrono::steady_clock::now();
        if (status != GGML_STATUS_SUCCESS) {
            return 0.0;
        }
        return std::chrono::duration<double>(finished - started).count();
    };

    for (int i = 0; i < WARMUP_RUNS; ++i) {
        if (compute_once() <= 0.0) {
            ggml_backend_sched_free(sched);
            ggml_backend_free(cpu_backend);
            ggml_free(ctx);
            return false;
        }
    }

    std::vector<double> seconds;
    seconds.reserve(TIMED_RUNS);
    for (int i = 0; i < TIMED_RUNS; ++i) {
        const double elapsed = compute_once();
        if (elapsed <= 0.0) {
            ggml_backend_sched_free(sched);
            ggml_backend_free(cpu_backend);
            ggml_free(ctx);
            return false;
        }
        seconds.push_back(elapsed);
    }

    // Decode-kernel probes feed the model-fit tok/s estimator, whose
    // validation target is median steady decode throughput. Use median elapsed
    // for this reusable kernel slope. The top-level streaming-memory benchmark
    // still reports p50/p90 separately; this probe has one effective bandwidth
    // field, and using p90/max here made metadata-only fit estimates
    // systematically conservative when one backend scheduler sample was slow.
    const double median_seconds = median(seconds);
    if (!std::isfinite(median_seconds) || median_seconds <= 0.0) {
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu_backend);
        ggml_free(ctx);
        return false;
    }
    const double bytes = static_cast<double>(weight_encoded.size())
        + static_cast<double>(input_f32.size() * sizeof(float))
        + static_cast<double>(shape.rows * sizeof(float));
    const double flops = 2.0 * static_cast<double>(shape.rows) * static_cast<double>(shape.cols);
    const double effective_gbps = bytes / median_seconds / 1e9;
    const double tflops = flops / median_seconds / 1e12;
    const double elapsed_ms = median_seconds * 1000.0;
    if (!std::isfinite(effective_gbps) || !std::isfinite(tflops) || !std::isfinite(elapsed_ms)) {
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu_backend);
        ggml_free(ctx);
        return false;
    }
    result = make_probe_result(
        name,
        tensor_type,
        shape.rows,
        shape.cols,
        graph_node_count(graph),
        effective_gbps,
        tflops,
        median_seconds,
        seconds,
        0,
        TIMED_RUNS);
    result.graph_inventory = collect_graph_inventory(graph);

    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu_backend);
    ggml_free(ctx);
    return true;
}

ScheduledGraph alloc_sched_for_graph(ggml_backend_t backend, ggml_cgraph * graph) {
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    if (cpu_backend == nullptr) {
        return ScheduledGraph{nullptr, nullptr};
    }
    ggml_backend_t backends[] = { backend, cpu_backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends,
        nullptr,
        2,
        GGML_DEFAULT_GRAPH_SIZE,
        false,
        true);
    if (sched == nullptr) {
        ggml_backend_free(cpu_backend);
        return ScheduledGraph{nullptr, nullptr};
    }
    if (!ggml_backend_sched_alloc_graph(sched, graph)) {
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu_backend);
        return ScheduledGraph{nullptr, nullptr};
    }
    return ScheduledGraph{sched, cpu_backend};
}

void free_scheduled_graph(ScheduledGraph scheduled) {
    if (scheduled.sched != nullptr) {
        ggml_backend_sched_free(scheduled.sched);
    }
    if (scheduled.cpu_backend != nullptr) {
        ggml_backend_free(scheduled.cpu_backend);
    }
}

bool compute_graph_timed(
    ggml_cgraph * graph,
    ScheduledGraph scheduled,
    ProbeResult & result,
    const std::string & name,
    const std::string & tensor_type,
    int64_t rows,
    int64_t cols,
    double bytes,
    double flops,
    int graph_features = 0,
    int warmup_runs = WARMUP_RUNS,
    int timed_runs = TIMED_RUNS) {
    if (scheduled.sched == nullptr) {
        return false;
    }
    auto compute_once = [&]() -> double {
        const auto started = std::chrono::steady_clock::now();
        enum ggml_status status = ggml_backend_sched_graph_compute_async(scheduled.sched, graph);
        ggml_backend_sched_synchronize(scheduled.sched);
        const auto finished = std::chrono::steady_clock::now();
        if (status != GGML_STATUS_SUCCESS) {
            return 0.0;
        }
        return std::chrono::duration<double>(finished - started).count();
    };

    for (int i = 0; i < warmup_runs; ++i) {
        if (compute_once() <= 0.0) {
            free_scheduled_graph(scheduled);
            return false;
        }
    }

    std::vector<double> seconds;
    seconds.reserve(timed_runs);
    for (int i = 0; i < timed_runs; ++i) {
        const double elapsed = compute_once();
        if (elapsed <= 0.0) {
            free_scheduled_graph(scheduled);
            return false;
        }
        seconds.push_back(elapsed);
    }

    // Same rationale as the isolated matvec probe above: this field is consumed
    // as a median decode-performance predictor, not as a worst-case latency
    // guardrail. Keep operational watchdogs outside the scoring model.
    const double median_seconds = median(seconds);
    if (!std::isfinite(median_seconds) || median_seconds <= 0.0) {
        free_scheduled_graph(scheduled);
        return false;
    }
    const double effective_gbps = bytes / median_seconds / 1e9;
    const double tflops = flops / median_seconds / 1e12;
    const double elapsed_ms = median_seconds * 1000.0;
    if (!std::isfinite(effective_gbps) || !std::isfinite(tflops) || !std::isfinite(elapsed_ms)) {
        free_scheduled_graph(scheduled);
        return false;
    }
    result = make_probe_result(
        name,
        tensor_type,
        rows,
        cols,
        graph_node_count(graph),
        effective_gbps,
        tflops,
        median_seconds,
        seconds,
        graph_features,
        timed_runs);
    result.graph_inventory = collect_graph_inventory(graph);

    free_scheduled_graph(scheduled);
    return true;
}

bool compute_graph_output_handoff_timed(
    ggml_backend_t backend,
    ggml_cgraph * graph,
    ggml_tensor * input,
    ggml_tensor * positions,
    ggml_tensor * logits,
    const std::vector<ggml_tensor *> & key_indices,
    const std::vector<ggml_tensor *> & value_indices,
    const std::vector<ggml_tensor *> & masks,
    ScheduledGraph scheduled,
    ProbeResult & result,
    const std::string & name,
    const std::string & tensor_type,
    int64_t rows,
    int64_t cols,
    int64_t context_tokens,
    double bytes,
    double flops,
    int graph_features = 0,
    int warmup_runs = WARMUP_RUNS,
    int timed_runs = TIMED_RUNS) {
    if (scheduled.sched == nullptr || logits == nullptr) {
        return false;
    }

    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    ggml_backend_buffer_type_t output_buft = ggml_backend_cpu_buffer_type();
    if (device != nullptr) {
        ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(device);
        if (host_buft != nullptr) {
            output_buft = host_buft;
        }
    }

    const size_t output_bytes = static_cast<size_t>(std::max<int64_t>(1, rows)) * sizeof(float);
    ggml_backend_buffer_t output_buffer = ggml_backend_buft_alloc_buffer(output_buft, output_bytes);
    if (output_buffer == nullptr) {
        free_scheduled_graph(scheduled);
        return false;
    }
    ggml_backend_buffer_clear(output_buffer, 0);
    float * output_base = static_cast<float *>(ggml_backend_buffer_get_base(output_buffer));
    if (output_base == nullptr) {
        ggml_backend_buffer_free(output_buffer);
        free_scheduled_graph(scheduled);
        return false;
    }

    const int64_t input_elements = input != nullptr ? ggml_nelements(input) : 0;
    const bool input_is_f32 = input != nullptr && input->type == GGML_TYPE_F32;
    const bool input_is_i32 = input != nullptr && input->type == GGML_TYPE_I32;
    std::vector<float> input_f32 = input_is_f32 && input_elements > 0
        ? deterministic_f32(input_elements, 4201)
        : std::vector<float>{};
    std::vector<int32_t> input_i32(input_is_i32 && input_elements > 0 ? 1 : 0, 0);
    std::vector<int32_t> position_i32(positions != nullptr ? 1 : 0, 0);
    const int64_t row_count = std::max<int64_t>(1, context_tokens);
    std::vector<int64_t> index_value(1, 0);
    std::vector<std::vector<uint8_t>> mask_inputs;
    mask_inputs.reserve(masks.size());
    for (size_t i = 0; i < masks.size(); ++i) {
        mask_inputs.push_back(empty_kq_mask_bytes(masks[i]));
    }

    volatile int32_t best_token_sink = 0;
    volatile float input_sink = 0.0f;
    auto compute_once = [&](int run_index) -> double {
        // Real llama.cpp decode does not repeatedly submit a byte-identical
        // graph. `llm_graph_result::set_inputs()` changes token/input tensors
        // and the KV write row advances as generation grows. A synthetic probe
        // that replays the same input/KV index can accidentally measure a
        // backend's best reuse path instead of the source-visible sampled decode
        // boundary. Mutate the input and row indices before starting the timer:
        // the dedicated submission probe charges input population; this probe
        // only needs fresh graph data so the timed graph+logits-sync path cannot
        // look artificially cached.
        //
        // llama.cpp's `llm_graph_input_attn_kv::set_input()` also rewrites the
        // KQ mask input (`set_input_kq_mask`) before decode. Leaving the mask
        // byte-identical let tiny synthetic graphs exercise a backend reuse path
        // that the real llama.cpp sampled-token boundary does not take. Keep
        // mask writes outside the timer for the same reason as token/KV input
        // writes: setup is charged by the submission probe, while this probe
        // times graph drain, logits visibility, and CPU logits scan.
        if (input_is_f32 && !input_f32.empty()) {
            const size_t changed =
                static_cast<size_t>(run_index % std::max<int64_t>(input_elements, 1));
            input_f32[changed] += 0.0001f;
            input_sink += input_f32[0];
            ggml_backend_tensor_set(input, input_f32.data(), 0, input_f32.size() * sizeof(float));
        } else if (input_is_i32 && !input_i32.empty()) {
            input_i32[0] = static_cast<int32_t>(run_index % std::max<int64_t>(rows, 1));
            input_sink += static_cast<float>(input_i32[0]);
            ggml_backend_tensor_set(input, input_i32.data(), 0, input_i32.size() * sizeof(int32_t));
        }
        if (!key_indices.empty() || !value_indices.empty()) {
            index_value[0] = source_decode_position_for_run(row_count, run_index);
            if (positions != nullptr && !position_i32.empty()) {
                position_i32[0] = static_cast<int32_t>(index_value[0]);
                ggml_backend_tensor_set(positions, position_i32.data(), 0, sizeof(int32_t));
            }
            for (ggml_tensor * key_index : key_indices) {
                ggml_backend_tensor_set(key_index, index_value.data(), 0, sizeof(int64_t));
            }
            for (ggml_tensor * value_index : value_indices) {
                ggml_backend_tensor_set(value_index, index_value.data(), 0, sizeof(int64_t));
            }
        }
        for (size_t i = 0; i < masks.size(); ++i) {
            auto & encoded = mask_inputs[i];
            if (!encoded.empty()) {
                fill_source_shaped_kq_mask_bytes(masks[i], encoded, index_value[0], run_index);
                ggml_backend_tensor_set(masks[i], encoded.data(), 0, encoded.size());
            }
        }
        const auto started = std::chrono::steady_clock::now();
        enum ggml_status status = ggml_backend_sched_graph_compute_async(scheduled.sched, graph);
        if (status != GGML_STATUS_SUCCESS) {
            return 0.0;
        }
        ggml_backend_t logits_backend = ggml_backend_sched_get_tensor_backend(scheduled.sched, logits);
        if (logits_backend == nullptr) {
            return 0.0;
        }
        // This is the source-shaped boundary used by llama.cpp sampled decode:
        // the token graph is submitted asynchronously, logits are requested from
        // the backend output tensor, and the context is synchronized before the
        // CPU sampler can scan candidates. Timing this together with the full
        // token graph is important for tiny models, where graph drain/output
        // handoff can be the dominant per-token term. The dimensions still come
        // only from GGUF metadata; no observed model throughput is used here.
        ggml_backend_tensor_get_async(logits_backend, logits, output_base, 0, output_bytes);
        ggml_backend_sched_synchronize(scheduled.sched);
        int32_t best = 0;
        float best_logit = -std::numeric_limits<float>::infinity();
        for (int32_t token = 0; token < static_cast<int32_t>(rows); ++token) {
            if (output_base[token] > best_logit) {
                best_logit = output_base[token];
                best = token;
            }
        }
        best_token_sink = best;
        const auto finished = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(finished - started).count();
    };

    for (int i = 0; i < warmup_runs; ++i) {
        if (compute_once(i) <= 0.0) {
            ggml_backend_buffer_free(output_buffer);
            free_scheduled_graph(scheduled);
            return false;
        }
    }

    std::vector<double> seconds;
    seconds.reserve(timed_runs);
    for (int i = 0; i < timed_runs; ++i) {
        const double elapsed = compute_once(warmup_runs + i);
        if (elapsed <= 0.0) {
            ggml_backend_buffer_free(output_buffer);
            free_scheduled_graph(scheduled);
            return false;
        }
        seconds.push_back(elapsed);
    }
    (void) best_token_sink;
    (void) input_sink;

    const double median_seconds = median(seconds);
    if (!std::isfinite(median_seconds) || median_seconds <= 0.0) {
        ggml_backend_buffer_free(output_buffer);
        free_scheduled_graph(scheduled);
        return false;
    }
    const double effective_gbps = (bytes + static_cast<double>(output_bytes)) / median_seconds / 1e9;
    const double tflops = flops / median_seconds / 1e12;
    const double elapsed_ms = median_seconds * 1000.0;
    if (!std::isfinite(effective_gbps) || !std::isfinite(tflops) || !std::isfinite(elapsed_ms)) {
        ggml_backend_buffer_free(output_buffer);
        free_scheduled_graph(scheduled);
        return false;
    }
    result = make_probe_result(
        name,
        tensor_type,
        rows,
        cols,
        graph_node_count(graph),
        effective_gbps,
        tflops,
        median_seconds,
        seconds,
        graph_features,
        timed_runs);
    result.graph_inventory = collect_graph_inventory(graph);

    ggml_backend_buffer_free(output_buffer);
    free_scheduled_graph(scheduled);
    return true;
}

bool compute_graph_submission_timed(
    ggml_backend_t backend,
    ggml_cgraph * graph,
    ggml_tensor * input,
    ggml_tensor * positions,
    ggml_tensor * logits,
    const std::vector<ggml_tensor *> & key_indices,
    const std::vector<ggml_tensor *> & value_indices,
    const std::vector<ggml_tensor *> & masks,
    ScheduledGraph scheduled,
    ProbeResult & result,
    const std::string & name,
    const std::string & tensor_type,
    int64_t rows,
    int64_t cols,
    int64_t layers,
    int64_t context_tokens,
    double bytes,
    double flops,
    int graph_features = 0,
    int warmup_runs = WARMUP_RUNS,
    int timed_runs = TIMED_RUNS) {
    if (scheduled.sched == nullptr || input == nullptr || logits == nullptr) {
        return false;
    }

    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    ggml_backend_buffer_type_t output_buft = ggml_backend_cpu_buffer_type();
    if (device != nullptr) {
        ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(device);
        if (host_buft != nullptr) {
            output_buft = host_buft;
        }
    }

    const size_t output_bytes = static_cast<size_t>(std::max<int64_t>(1, rows)) * sizeof(float);
    ggml_backend_buffer_t output_buffer = ggml_backend_buft_alloc_buffer(output_buft, output_bytes);
    if (output_buffer == nullptr) {
        free_scheduled_graph(scheduled);
        return false;
    }
    ggml_backend_buffer_clear(output_buffer, 0);
    float * output_base = static_cast<float *>(ggml_backend_buffer_get_base(output_buffer));
    if (output_base == nullptr) {
        ggml_backend_buffer_free(output_buffer);
        free_scheduled_graph(scheduled);
        return false;
    }

    const int64_t input_elements = ggml_nelements(input);
    const bool input_is_f32 = input->type == GGML_TYPE_F32;
    const bool input_is_i32 = input->type == GGML_TYPE_I32;
    std::vector<float> input_f32 = input_is_f32 ? deterministic_f32(input_elements, 2203) : std::vector<float>{};
    std::vector<int32_t> input_i32(input_is_i32 && input_elements > 0 ? 1 : 0, 0);
    std::vector<int32_t> position_i32(positions != nullptr ? 1 : 0, 0);
    std::vector<int64_t> index_value(1, std::max<int64_t>(0, context_tokens - 1));
    std::vector<std::vector<uint8_t>> mask_inputs;
    mask_inputs.reserve(masks.size());
    for (size_t i = 0; i < masks.size(); ++i) {
        mask_inputs.push_back(empty_kq_mask_bytes(masks[i]));
    }
    volatile float input_sink = 0.0f;
    volatile uint64_t source_decode_sink = 0;
    SourceDecodeBookkeepingScratch source_decode_scratch;
    auto submit_once = [&](int run_index) -> double {
        // This probe intentionally times the source-visible submission half of
        // sampled decode, not graph completion. In llama.cpp, a reused decode
        // step calls `llm_graph_result::set_inputs()`, submits
        // `ggml_backend_sched_graph_compute_async()`, then schedules async
        // logits extraction before `llama_decode()` returns. The sampler call
        // that follows waits for graph/logits visibility. That is why the
        // synchronization below happens after the timestamp: it drains work so
        // the next sample is clean, but it is not part of the decode-call
        // submission interval we are trying to model.
        const auto started = std::chrono::steady_clock::now();
        run_source_decode_bookkeeping(
            graph,
            layers,
            context_tokens,
            cols,
            rows,
            run_index,
            source_decode_scratch,
            source_decode_sink);
        if (input_is_f32 && !input_f32.empty()) {
            input_f32[static_cast<size_t>(run_index % std::max<int64_t>(input_elements, 1))] += 0.0001f;
            input_sink += input_f32[0];
            ggml_backend_tensor_set(input, input_f32.data(), 0, input_f32.size() * sizeof(float));
        } else if (input_is_i32 && !input_i32.empty()) {
            input_i32[0] = static_cast<int32_t>(run_index % std::max<int64_t>(rows, 1));
            input_sink += static_cast<float>(input_i32[0]);
            ggml_backend_tensor_set(input, input_i32.data(), 0, input_i32.size() * sizeof(int32_t));
        }
        index_value[0] = source_decode_position_for_run(context_tokens, run_index);
        if (positions != nullptr && !position_i32.empty()) {
            position_i32[0] = static_cast<int32_t>(index_value[0]);
            ggml_backend_tensor_set(positions, position_i32.data(), 0, sizeof(int32_t));
        }
        for (ggml_tensor * key_index : key_indices) {
            ggml_backend_tensor_set(key_index, index_value.data(), 0, sizeof(int64_t));
        }
        for (ggml_tensor * value_index : value_indices) {
            ggml_backend_tensor_set(value_index, index_value.data(), 0, sizeof(int64_t));
        }
        for (size_t i = 0; i < masks.size(); ++i) {
            auto & encoded = mask_inputs[i];
            if (!encoded.empty()) {
                fill_source_shaped_kq_mask_bytes(masks[i], encoded, index_value[0], run_index);
                ggml_backend_tensor_set(masks[i], encoded.data(), 0, encoded.size());
            }
        }
        enum ggml_status status = ggml_backend_sched_graph_compute_async(scheduled.sched, graph);
        if (status != GGML_STATUS_SUCCESS) {
            return 0.0;
        }
        ggml_backend_t logits_backend = ggml_backend_sched_get_tensor_backend(scheduled.sched, logits);
        if (logits_backend == nullptr) {
            return 0.0;
        }
        ggml_backend_tensor_get_async(logits_backend, logits, output_base, 0, output_bytes);
        const auto submitted = std::chrono::steady_clock::now();
        ggml_backend_sched_synchronize(scheduled.sched);
        return std::chrono::duration<double>(submitted - started).count();
    };

    for (int i = 0; i < warmup_runs; ++i) {
        if (submit_once(i) <= 0.0) {
            ggml_backend_buffer_free(output_buffer);
            free_scheduled_graph(scheduled);
            return false;
        }
    }

    std::vector<double> seconds;
    seconds.reserve(timed_runs);
    for (int i = 0; i < timed_runs; ++i) {
        const double elapsed = submit_once(warmup_runs + i);
        if (elapsed <= 0.0) {
            ggml_backend_buffer_free(output_buffer);
            free_scheduled_graph(scheduled);
            return false;
        }
        seconds.push_back(elapsed);
    }
    (void) input_sink;
    (void) source_decode_sink;

    const double median_seconds = median(seconds);
    if (!std::isfinite(median_seconds) || median_seconds <= 0.0) {
        ggml_backend_buffer_free(output_buffer);
        free_scheduled_graph(scheduled);
        return false;
    }
    const double submitted_bytes = static_cast<double>(input_f32.size() * sizeof(float))
        + static_cast<double>(input_i32.size() * sizeof(int32_t))
        + static_cast<double>(position_i32.size() * sizeof(int32_t))
        + static_cast<double>((key_indices.size() + value_indices.size()) * sizeof(int64_t))
        + std::accumulate(
            mask_inputs.begin(),
            mask_inputs.end(),
            0.0,
            [](double acc, const std::vector<uint8_t> & encoded) {
                return acc + static_cast<double>(encoded.size());
            })
        + static_cast<double>(output_bytes);
    const double effective_gbps = submitted_bytes / median_seconds / 1e9;
    const double tflops = flops / median_seconds / 1e12;
    if (!std::isfinite(effective_gbps) || !std::isfinite(tflops)) {
        ggml_backend_buffer_free(output_buffer);
        free_scheduled_graph(scheduled);
        return false;
    }
    result = make_probe_result(
        name,
        tensor_type,
        rows,
        cols,
        graph_node_count(graph),
        effective_gbps,
        tflops,
        median_seconds,
        seconds,
        graph_features,
        timed_runs);
    result.graph_inventory = collect_graph_inventory(graph);

    (void) bytes;
    ggml_backend_buffer_free(output_buffer);
    free_scheduled_graph(scheduled);
    return true;
}

bool compute_graph_source_input_timed(
    ggml_cgraph * graph,
    ggml_tensor * input,
    ggml_tensor * positions,
    const std::vector<ggml_tensor *> & key_indices,
    const std::vector<ggml_tensor *> & value_indices,
    const std::vector<ggml_tensor *> & masks,
    ScheduledGraph scheduled,
    ProbeResult & result,
    const std::string & name,
    const std::string & tensor_type,
    int64_t rows,
    int64_t cols,
    int64_t layers,
    int64_t context_tokens,
    double flops,
    int graph_features = 0,
    int warmup_runs = WARMUP_RUNS,
    int timed_runs = TIMED_RUNS) {
    if (scheduled.sched == nullptr || input == nullptr) {
        return false;
    }

    const int64_t input_elements = ggml_nelements(input);
    const bool input_is_f32 = input->type == GGML_TYPE_F32;
    const bool input_is_i32 = input->type == GGML_TYPE_I32;
    std::vector<float> input_f32 = input_is_f32 ? deterministic_f32(input_elements, 2303) : std::vector<float>{};
    std::vector<int32_t> input_i32(input_is_i32 && input_elements > 0 ? 1 : 0, 0);
    std::vector<int32_t> position_i32(positions != nullptr ? 1 : 0, 0);
    std::vector<int64_t> index_value(1, std::max<int64_t>(0, context_tokens - 1));
    std::vector<std::vector<uint8_t>> mask_inputs;
    mask_inputs.reserve(masks.size());
    for (size_t i = 0; i < masks.size(); ++i) {
        mask_inputs.push_back(empty_kq_mask_bytes(masks[i]));
    }

    volatile float input_sink = 0.0f;
    volatile uint64_t source_decode_sink = 0;
    SourceDecodeBookkeepingScratch source_decode_scratch;
    auto input_once = [&](int run_index) -> double {
        // This is the part of `llama_context::decode()` that happens before the
        // backend graph is submitted. It is deliberately narrower than the
        // submission probe: no `ggml_backend_sched_graph_compute_async()`, no
        // async logits request, and no scheduler synchronization. That makes it
        // usable as a non-overlapping lower bound beside a full-token handoff
        // probe. The work itself mirrors source-visible llama.cpp tasks:
        // output/session bookkeeping, graph-input compatibility checks, token
        // and position input updates, KV write indices, and KQ mask rebuilds.
        const auto started = std::chrono::steady_clock::now();
        run_source_decode_bookkeeping(
            graph,
            layers,
            context_tokens,
            cols,
            rows,
            run_index,
            source_decode_scratch,
            source_decode_sink);
        if (input_is_f32 && !input_f32.empty()) {
            input_f32[static_cast<size_t>(run_index % std::max<int64_t>(input_elements, 1))] += 0.0001f;
            input_sink += input_f32[0];
            ggml_backend_tensor_set(input, input_f32.data(), 0, input_f32.size() * sizeof(float));
        } else if (input_is_i32 && !input_i32.empty()) {
            input_i32[0] = static_cast<int32_t>(run_index % std::max<int64_t>(rows, 1));
            input_sink += static_cast<float>(input_i32[0]);
            ggml_backend_tensor_set(input, input_i32.data(), 0, input_i32.size() * sizeof(int32_t));
        }
        index_value[0] = source_decode_position_for_run(context_tokens, run_index);
        if (positions != nullptr && !position_i32.empty()) {
            position_i32[0] = static_cast<int32_t>(index_value[0]);
            ggml_backend_tensor_set(positions, position_i32.data(), 0, sizeof(int32_t));
        }
        for (ggml_tensor * key_index : key_indices) {
            ggml_backend_tensor_set(key_index, index_value.data(), 0, sizeof(int64_t));
        }
        for (ggml_tensor * value_index : value_indices) {
            ggml_backend_tensor_set(value_index, index_value.data(), 0, sizeof(int64_t));
        }
        for (size_t i = 0; i < masks.size(); ++i) {
            auto & encoded = mask_inputs[i];
            if (!encoded.empty()) {
                fill_source_shaped_kq_mask_bytes(masks[i], encoded, index_value[0], run_index);
                ggml_backend_tensor_set(masks[i], encoded.data(), 0, encoded.size());
            }
        }
        const auto finished = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(finished - started).count();
        enum ggml_status status = ggml_backend_sched_graph_compute_async(scheduled.sched, graph);
        if (status != GGML_STATUS_SUCCESS) {
            return 0.0;
        }
        ggml_backend_sched_synchronize(scheduled.sched);
        return elapsed;
    };

    for (int i = 0; i < warmup_runs; ++i) {
        if (input_once(i) <= 0.0) {
            return false;
        }
    }

    std::vector<double> seconds;
    seconds.reserve(timed_runs);
    for (int i = 0; i < timed_runs; ++i) {
        const double elapsed = input_once(warmup_runs + i);
        if (elapsed <= 0.0) {
            return false;
        }
        seconds.push_back(elapsed);
    }
    (void) input_sink;
    (void) source_decode_sink;

    const double median_seconds = median(seconds);
    if (!std::isfinite(median_seconds) || median_seconds <= 0.0) {
        return false;
    }
    const double input_bytes = static_cast<double>(input_f32.size() * sizeof(float))
        + static_cast<double>(input_i32.size() * sizeof(int32_t))
        + static_cast<double>(position_i32.size() * sizeof(int32_t))
        + static_cast<double>((key_indices.size() + value_indices.size()) * sizeof(int64_t))
        + std::accumulate(
            mask_inputs.begin(),
            mask_inputs.end(),
            0.0,
            [](double acc, const std::vector<uint8_t> & encoded) {
                return acc + static_cast<double>(encoded.size());
            });
    const double effective_gbps = input_bytes / median_seconds / 1e9;
    const double tflops = flops / median_seconds / 1e12;
    if (!std::isfinite(effective_gbps) || !std::isfinite(tflops)) {
        return false;
    }
    result = make_probe_result(
        name,
        tensor_type,
        rows,
        cols,
        graph_node_count(graph),
        effective_gbps,
        tflops,
        median_seconds,
        seconds,
        graph_features,
        timed_runs);
    result.graph_inventory = collect_graph_inventory(graph);
    return true;
}

bool compute_graph_source_sampled_timed(
    ggml_backend_t backend,
    ggml_cgraph * graph,
    ggml_tensor * input,
    ggml_tensor * positions,
    ggml_tensor * logits,
    const std::vector<ggml_tensor *> & key_indices,
    const std::vector<ggml_tensor *> & value_indices,
    const std::vector<ggml_tensor *> & masks,
    ScheduledGraph scheduled,
    ProbeResult & result,
    const std::string & name,
    const std::string & tensor_type,
    int64_t rows,
    int64_t cols,
    int64_t layers,
    int64_t context_tokens,
    double bytes,
    double flops,
    int graph_features = 0,
    int warmup_runs = WARMUP_RUNS,
    int timed_runs = TIMED_RUNS) {
    if (scheduled.sched == nullptr || input == nullptr || logits == nullptr) {
        return false;
    }

    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    ggml_backend_buffer_type_t output_buft = ggml_backend_cpu_buffer_type();
    if (device != nullptr) {
        ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(device);
        if (host_buft != nullptr) {
            output_buft = host_buft;
        }
    }

    const size_t output_bytes = static_cast<size_t>(std::max<int64_t>(1, rows)) * sizeof(float);
    ggml_backend_buffer_t output_buffer = ggml_backend_buft_alloc_buffer(output_buft, output_bytes);
    if (output_buffer == nullptr) {
        free_scheduled_graph(scheduled);
        return false;
    }
    ggml_backend_buffer_clear(output_buffer, 0);
    float * output_base = static_cast<float *>(ggml_backend_buffer_get_base(output_buffer));
    if (output_base == nullptr) {
        ggml_backend_buffer_free(output_buffer);
        free_scheduled_graph(scheduled);
        return false;
    }

    const int64_t input_elements = ggml_nelements(input);
    const bool input_is_f32 = input->type == GGML_TYPE_F32;
    const bool input_is_i32 = input->type == GGML_TYPE_I32;
    std::vector<float> input_f32 = input_is_f32 ? deterministic_f32(input_elements, 3203) : std::vector<float>{};
    std::vector<int32_t> input_i32(input_is_i32 && input_elements > 0 ? 1 : 0, 0);
    std::vector<int32_t> position_i32(positions != nullptr ? 1 : 0, 0);
    std::vector<int64_t> index_value(1, std::max<int64_t>(0, context_tokens - 1));
    std::vector<std::vector<uint8_t>> mask_inputs;
    mask_inputs.reserve(masks.size());
    for (size_t i = 0; i < masks.size(); ++i) {
        mask_inputs.push_back(empty_kq_mask_bytes(masks[i]));
    }

    volatile int32_t best_token_sink = 0;
    volatile float input_sink = 0.0f;
    volatile uint64_t source_decode_sink = 0;
    SourceDecodeBookkeepingScratch source_decode_scratch;
    auto sample_once = [&](int run_index) -> double {
        // This is the broadest synthetic source-boundary probe. It intentionally
        // starts where llama.cpp's reused sampled decode token starts to become
        // visible to source code: batch/graph bookkeeping, graph input updates,
        // scheduler submission, async logits extraction, scheduler
        // synchronization, and a CPU scan of the vocab row. Tiny models expose
        // this boundary because their matmul graph is cheap; measuring it as one
        // interval avoids fitting separate backend constants or borrowing
        // observed model tok/s.
        const auto started = std::chrono::steady_clock::now();
        run_source_decode_bookkeeping(
            graph,
            layers,
            context_tokens,
            cols,
            rows,
            run_index,
            source_decode_scratch,
            source_decode_sink);
        if (input_is_f32 && !input_f32.empty()) {
            input_f32[static_cast<size_t>(run_index % std::max<int64_t>(input_elements, 1))] += 0.0001f;
            input_sink += input_f32[0];
            ggml_backend_tensor_set(input, input_f32.data(), 0, input_f32.size() * sizeof(float));
        } else if (input_is_i32 && !input_i32.empty()) {
            input_i32[0] = static_cast<int32_t>(run_index % std::max<int64_t>(rows, 1));
            input_sink += static_cast<float>(input_i32[0]);
            ggml_backend_tensor_set(input, input_i32.data(), 0, input_i32.size() * sizeof(int32_t));
        }
        index_value[0] = source_decode_position_for_run(context_tokens, run_index);
        if (positions != nullptr && !position_i32.empty()) {
            position_i32[0] = static_cast<int32_t>(index_value[0]);
            ggml_backend_tensor_set(positions, position_i32.data(), 0, sizeof(int32_t));
        }
        for (ggml_tensor * key_index : key_indices) {
            ggml_backend_tensor_set(key_index, index_value.data(), 0, sizeof(int64_t));
        }
        for (ggml_tensor * value_index : value_indices) {
            ggml_backend_tensor_set(value_index, index_value.data(), 0, sizeof(int64_t));
        }
        for (size_t i = 0; i < masks.size(); ++i) {
            auto & encoded = mask_inputs[i];
            if (!encoded.empty()) {
                fill_source_shaped_kq_mask_bytes(masks[i], encoded, index_value[0], run_index);
                ggml_backend_tensor_set(masks[i], encoded.data(), 0, encoded.size());
            }
        }
        enum ggml_status status = ggml_backend_sched_graph_compute_async(scheduled.sched, graph);
        if (status != GGML_STATUS_SUCCESS) {
            return 0.0;
        }
        ggml_backend_t logits_backend = ggml_backend_sched_get_tensor_backend(scheduled.sched, logits);
        if (logits_backend == nullptr) {
            return 0.0;
        }
        ggml_backend_tensor_get_async(logits_backend, logits, output_base, 0, output_bytes);
        ggml_backend_sched_synchronize(scheduled.sched);
        int32_t best = 0;
        float best_logit = -std::numeric_limits<float>::infinity();
        for (int32_t token = 0; token < static_cast<int32_t>(rows); ++token) {
            if (output_base[token] > best_logit) {
                best_logit = output_base[token];
                best = token;
            }
        }
        best_token_sink = best;
        const auto finished = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(finished - started).count();
    };

    for (int i = 0; i < warmup_runs; ++i) {
        if (sample_once(i) <= 0.0) {
            ggml_backend_buffer_free(output_buffer);
            free_scheduled_graph(scheduled);
            return false;
        }
    }

    std::vector<double> seconds;
    seconds.reserve(timed_runs);
    for (int i = 0; i < timed_runs; ++i) {
        const double elapsed = sample_once(warmup_runs + i);
        if (elapsed <= 0.0) {
            ggml_backend_buffer_free(output_buffer);
            free_scheduled_graph(scheduled);
            return false;
        }
        seconds.push_back(elapsed);
    }
    (void) best_token_sink;
    (void) input_sink;
    (void) source_decode_sink;

    const double median_seconds = median(seconds);
    if (!std::isfinite(median_seconds) || median_seconds <= 0.0) {
        ggml_backend_buffer_free(output_buffer);
        free_scheduled_graph(scheduled);
        return false;
    }
    const double effective_gbps = (bytes + static_cast<double>(output_bytes)) / median_seconds / 1e9;
    const double tflops = flops / median_seconds / 1e12;
    if (!std::isfinite(effective_gbps) || !std::isfinite(tflops)) {
        ggml_backend_buffer_free(output_buffer);
        free_scheduled_graph(scheduled);
        return false;
    }
    result = make_probe_result(
        name,
        tensor_type,
        rows,
        cols,
        graph_node_count(graph),
        effective_gbps,
        tflops,
        median_seconds,
        seconds,
        graph_features,
        timed_runs);
    result.graph_inventory = collect_graph_inventory(graph);

    ggml_backend_buffer_free(output_buffer);
    free_scheduled_graph(scheduled);
    return true;
}

bool set_encoded_weights(
    ggml_tensor * tensor,
    enum ggml_type type,
    int64_t rows,
    int64_t cols,
    EncodedWeightCache & cache,
    double & bytes,
    double & flops) {
    const std::vector<uint8_t> & encoded = cached_encoded_weights(cache, type, rows, cols);
    // Real GGUF files can mix tensor types per tensor, while these probes ask a
    // stricter question: "can the backend execute this whole model-shaped graph
    // if every matmul weight uses this candidate type?"  For very small or
    // unusual widths, especially K-quants, GGML may allocate fewer bytes for the
    // synthetic tensor than ggml_quantize_chunk produces for the same logical
    // rows/cols.  Calling ggml_backend_tensor_set in that state trips GGML's
    // bounds assertion and aborts the validator.  Treat it as an unsupported
    // synthetic probe shape instead; the estimator can then fall back to another
    // source of evidence without pretending this graph was measured.
    if (encoded.size() > ggml_nbytes(tensor)) {
        return false;
    }
    ggml_backend_tensor_set(tensor, encoded.data(), 0, encoded.size());
    bytes += static_cast<double>(encoded.size());
    flops += 2.0 * static_cast<double>(rows) * static_cast<double>(cols);
    return true;
}

bool set_active_encoded_weights(
    ggml_tensor * tensor,
    enum ggml_type type,
    int64_t rows,
    int64_t cols,
    EncodedWeightCache & cache,
    double & bytes,
    double & flops) {
    // MoE probes route deterministically to expert ids 0..experts_used-1. The
    // full expert tensor must still exist so the GGML_OP_MUL_MAT_ID graph has
    // the same 3D expert-pool shape as llama.cpp, but initializing every unused
    // expert makes deep l4/l8 validation spend minutes in CPU-side quantization
    // before the timed graph even starts. Populate only the contiguous active
    // expert rows that the synthetic ids can read. The byte/flop counters here
    // also describe active traffic, not resident model size, matching the
    // model-fit active-expert accounting.
    const std::vector<uint8_t> & encoded = cached_encoded_weights(cache, type, rows, cols);
    if (encoded.size() > ggml_nbytes(tensor)) {
        return false;
    }
    ggml_backend_tensor_set(tensor, encoded.data(), 0, encoded.size());
    bytes += static_cast<double>(encoded.size());
    flops += 2.0 * static_cast<double>(rows) * static_cast<double>(cols);
    return true;
}

bool set_f32_weights(
    ggml_tensor * tensor,
    int64_t rows,
    int64_t cols,
    uint32_t salt,
    double & bytes,
    double & flops) {
    std::vector<float> weights = deterministic_f32(rows * cols, salt);
    ggml_backend_tensor_set(tensor, weights.data(), 0, weights.size() * sizeof(float));
    bytes += static_cast<double>(weights.size() * sizeof(float));
    flops += 2.0 * static_cast<double>(rows) * static_cast<double>(cols);
    return true;
}

ggml_tensor * rms_norm_attention_projection(
    ggml_context * ctx,
    ggml_tensor * projection,
    int64_t total_width,
    int64_t norm_head_width) {
    if (norm_head_width > 0 && norm_head_width < total_width && total_width % norm_head_width == 0) {
        // llama.cpp's Q/K norm tensors are not normalized over the flattened
        // residual width. `build_qkv()` leaves Q and K shaped as
        // `[head_dim, heads, tokens]`, and `build_norm()` applies RMSNorm over
        // `ne[0]`, i.e. one attention head at a time. That matters for the
        // synthetic fit probe because backend kernels and graph scheduling see
        // many small head-width norms, not one hidden-width norm. Use the GGUF
        // head width when it is available, then reshape back to the flattened
        // vector shape consumed by this compact attention proxy.
        ggml_tensor * shaped = ggml_reshape_3d(
            ctx,
            projection,
            norm_head_width,
            total_width / norm_head_width,
            1);
        ggml_tensor * normed = ggml_rms_norm(ctx, shaped, 1e-5f);
        return ggml_reshape_2d(ctx, normed, total_width, 1);
    }
    return ggml_rms_norm(ctx, projection, 1e-5f);
}

bool run_llama_graph_probe(
    ggml_backend_t backend,
    enum ggml_type type,
    const char * name,
    const char * tensor_type,
    int64_t hidden,
    int64_t kv_width,
    int64_t ffn,
    int64_t repeat_layers,
    int graph_features,
    int64_t norm_head_width,
    ProbeResult & result) {
    const int64_t layers = std::max<int64_t>(1, repeat_layers);
    if (!dense_llama_shape_supported(type, hidden, kv_width, ffn)) {
        return false;
    }
    const bool use_q_norm = (graph_features & GRAPH_FEATURE_ATTENTION_Q_NORM) != 0;
    const bool use_k_norm = (graph_features & GRAPH_FEATURE_ATTENTION_K_NORM) != 0;
    const bool use_post_attention_norm = (graph_features & GRAPH_FEATURE_ATTENTION_POST_NORM) != 0;
    const bool use_post_ffn_norm = (graph_features & GRAPH_FEATURE_FFN_POST_NORM) != 0;
    const size_t context_bytes =
        ggml_tensor_overhead() * static_cast<size_t>(80 * layers) + ggml_graph_overhead();
    ggml_init_params params{};
    params.mem_size = context_bytes;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, 1);
    ggml_tensor * output = input;
    ggml_cgraph * graph = ggml_new_graph(ctx);
    for (int64_t layer = 0; layer < layers; ++layer) {
        // Keep the synthetic dense block close to llama.cpp's source graph, not
        // just to its resident tensor bytes. `build_llama()` normalizes the
        // residual stream before attention, adds the attention result back into
        // the residual stream, normalizes again before FFN, then adds the FFN
        // result. Earlier probes skipped the RMSNorm/residual structure and
        // therefore measured mostly quantized matvec throughput. That was
        // enough on some Metal rows but overestimated CUDA rows where the
        // quantized matvecs are very fast and the surrounding graph work is no
        // longer hidden under memory traffic.
        ggml_tensor * attn_input = use_post_attention_norm ? output : ggml_rms_norm(ctx, output, 1e-5f);
        ggml_tensor * q = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, hidden), attn_input);
        ggml_tensor * k = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, kv_width), attn_input);
        ggml_tensor * v = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, kv_width), attn_input);
        if (use_q_norm) {
            q = rms_norm_attention_projection(ctx, q, hidden, norm_head_width);
        }
        if (use_k_norm) {
            k = rms_norm_attention_projection(ctx, k, kv_width, norm_head_width);
        }
        // llama.cpp's `llm_graph_context::build_attn()` does not leave Q/K/V
        // as anonymous dependencies under the final attention output. It calls
        // `ggml_build_forward_expand()` on the Q, K, and V projection nodes
        // before building attention, with an in-source note that this prevents
        // reordering and reduces graph splits. That scheduler shape matters for
        // decode probes: if this synthetic graph only expands from the final
        // output, Metal/CUDA can schedule a graph that is source-plausible at
        // the op level but not the graph llama.cpp actually submits. Keep the
        // probe source-shaped at the graph-boundary level too.
        ggml_build_forward_expand(graph, q);
        ggml_build_forward_expand(graph, k);
        ggml_build_forward_expand(graph, v);
        // The output projection consumes the result of attention, not a literal
        // elementwise blend of the raw Q/K/V projection vectors. We use Q as the
        // hidden-width proxy for that attention result and keep K/V scheduled
        // through the explicit graph expansion above. Earlier probes used
        // `q + k + v` when `kv_width == hidden`; that serialized the output
        // projection behind synthetic elementwise dependencies that are not in
        // llama.cpp's decode graph and made full-width attention probes too
        // pessimistic on Metal. KV-cache read traffic is accounted separately in
        // model-fit's metadata estimator, where it can scale with workload
        // prompt/context length without making this graph probe allocate a
        // production-sized cache.
        ggml_tensor * attn = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, hidden), q);
        ggml_tensor * ffn_input;
        ggml_tensor * residual_after_attn;
        if (use_post_attention_norm) {
            ggml_tensor * attn_norm = ggml_rms_norm(ctx, attn, 1e-5f);
            ffn_input = ggml_add(ctx, output, attn_norm);
            residual_after_attn = ffn_input;
        } else {
            residual_after_attn = ggml_add(ctx, output, attn);
            ffn_input = ggml_rms_norm(ctx, residual_after_attn, 1e-5f);
        }
        ggml_tensor * gate = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, ffn), ffn_input);
        ggml_tensor * up = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, ffn), ffn_input);
        // llama.cpp's dense SWIGLU FFN path (`build_ffn(..., LLM_FFN_SILU,
        // LLM_FFN_PAR, ...)`) does not lower gate/up activation to a plain
        // elementwise multiply. It uses the source-visible GGML_SWIGLU_SPLIT op
        // before the down projection. The decode probe needs the same graph shape
        // because Metal/CUDA schedule and sometimes fuse these skinny activation
        // nodes differently from an isolated `ggml_mul`.
        ggml_tensor * gated = ggml_swiglu_split(ctx, gate, up);
        ggml_tensor * down = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, ffn, hidden), gated);
        if (use_post_ffn_norm) {
            down = ggml_rms_norm(ctx, down, 1e-5f);
            output = ggml_add(ctx, residual_after_attn, down);
        } else {
            output = ggml_add(ctx, residual_after_attn, down);
        }
        // For GQA/MQA the K/V projections are narrower than the hidden residual
        // stream. We still need those projections to be scheduled because
        // llama.cpp explicitly expands Q, K, and V before it builds attention.
        // The `ggml_build_forward_expand(graph, k/v)` calls above already do
        // that. Do not attach synthetic `sum(k)` / `sum(v)` dependencies to the
        // final output: those reductions are not part of llama decode and they
        // can dominate small Metal graph probes, which would make the metadata
        // estimator learn the benchmark artifact instead of the source graph.
    }
    ggml_set_name(input, "ggml_decode_llama_graph_input");
    ggml_set_name(output, "ggml_decode_llama_graph_output");
    ggml_set_output(output);

    ggml_build_forward_expand(graph, output);
    if (!graph_supported_by_backend(backend, graph)) {
        ggml_free(ctx);
        return false;
    }

    ScheduledGraph scheduled = alloc_sched_for_graph(backend, graph);
    if (scheduled.sched == nullptr) {
        ggml_free(ctx);
        return false;
    }

    double bytes = 0.0;
    double flops = 0.0;
    std::vector<float> input_f32 = deterministic_f32(hidden, 101);
    ggml_backend_tensor_set(input, input_f32.data(), 0, input_f32.size() * sizeof(float));
    bytes += static_cast<double>(input_f32.size() * sizeof(float));
    EncodedWeightCache weight_cache;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        if (t->type != type || t->op != GGML_OP_NONE) {
            continue;
        }
        const int64_t rows = t->ne[1];
        const int64_t cols = t->ne[0];
        if (!set_encoded_weights(t, type, rows, cols, weight_cache, bytes, flops)) {
            free_scheduled_graph(scheduled);
            ggml_free(ctx);
            return false;
        }
    }
    bytes += static_cast<double>(layers * (4 * hidden + ffn) * sizeof(float));
    ggml_backend_synchronize(backend);

    const bool ok = compute_graph_timed(
        graph,
        scheduled,
        result,
        name,
        tensor_type,
        ffn,
        hidden,
        bytes,
        flops,
        graph_features,
        GRAPH_WARMUP_RUNS,
        GRAPH_TIMED_RUNS);
    ggml_free(ctx);
    return ok;
}

bool set_f16_tensor(
    ggml_tensor * tensor,
    int64_t elements,
    uint32_t salt,
    double & bytes) {
    std::vector<float> values = deterministic_f32(elements, salt);
    std::vector<uint8_t> encoded = encode_weights(GGML_TYPE_F16, values, elements, 1);
    if (encoded.size() > ggml_nbytes(tensor)) {
        return false;
    }
    ggml_backend_tensor_set(tensor, encoded.data(), 0, encoded.size());
    bytes += static_cast<double>(encoded.size());
    return true;
}

bool set_f32_tensor(
    ggml_tensor * tensor,
    int64_t elements,
    uint32_t salt,
    double & bytes) {
    std::vector<float> values = deterministic_f32(elements, salt);
    const size_t encoded_bytes = values.size() * sizeof(float);
    if (encoded_bytes > ggml_nbytes(tensor)) {
        return false;
    }
    ggml_backend_tensor_set(tensor, values.data(), 0, encoded_bytes);
    bytes += static_cast<double>(encoded_bytes);
    return true;
}

bool set_i64_tensor(
    ggml_tensor * tensor,
    int64_t elements,
    int64_t value,
    double & bytes) {
    std::vector<int64_t> values(static_cast<size_t>(elements), value);
    ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(int64_t));
    bytes += static_cast<double>(values.size() * sizeof(int64_t));
    return true;
}

bool set_i32_tensor(
    ggml_tensor * tensor,
    int64_t elements,
    int32_t value,
    double & bytes) {
    std::vector<int32_t> values(static_cast<size_t>(elements), value);
    ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(int32_t));
    bytes += static_cast<double>(values.size() * sizeof(int32_t));
    return true;
}

bool run_attention_runtime_probe(
    ggml_backend_t backend,
    const char * name,
    int64_t head_dim,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t context_tokens,
    int64_t repeat_layers,
    ProbeResult & result) {
    const int64_t layers = std::max<int64_t>(1, repeat_layers);
    if (head_dim <= 0 || query_heads <= 0 || kv_heads <= 0 || context_tokens <= 0) {
        return false;
    }
    if (query_heads % kv_heads != 0) {
        return false;
    }
    const size_t context_bytes =
        ggml_tensor_overhead() * static_cast<size_t>(16 * layers) + ggml_graph_overhead();
    ggml_init_params params{};
    params.mem_size = context_bytes;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }

    ggml_tensor * input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, 1, query_heads, 1);
    ggml_tensor * cur = input;
    std::vector<ggml_tensor *> keys;
    std::vector<ggml_tensor *> values;
    std::vector<ggml_tensor *> masks;
    keys.reserve(static_cast<size_t>(layers));
    values.reserve(static_cast<size_t>(layers));
    masks.reserve(static_cast<size_t>(layers));
    for (int64_t layer = 0; layer < layers; ++layer) {
        ggml_tensor * key = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, context_tokens, kv_heads, 1);
        ggml_tensor * value = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, context_tokens, kv_heads, 1);
        ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, context_tokens, 1, 1, 1);
        ggml_tensor * attn = ggml_flash_attn_ext(
            ctx,
            cur,
            key,
            value,
            mask,
            1.0f / std::sqrt(static_cast<float>(head_dim)),
            0.0f,
            0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
        cur = ggml_permute(ctx, attn, 0, 2, 1, 3);
        keys.push_back(key);
        values.push_back(value);
        masks.push_back(mask);
    }
    ggml_set_name(input, "ggml_decode_flash_attn_runtime_input");
    ggml_set_name(cur, "ggml_decode_flash_attn_runtime_output");
    ggml_set_output(cur);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, cur);
    if (!graph_supported_by_backend(backend, graph)) {
        ggml_free(ctx);
        return false;
    }

    ScheduledGraph scheduled = alloc_sched_for_graph(backend, graph);
    if (scheduled.sched == nullptr) {
        ggml_free(ctx);
        return false;
    }

    double bytes = 0.0;
    double flops = 0.0;
    std::vector<float> input_f32 = deterministic_f32(head_dim * query_heads, 613);
    ggml_backend_tensor_set(input, input_f32.data(), 0, input_f32.size() * sizeof(float));
    bytes += static_cast<double>(input_f32.size() * sizeof(float));
    for (size_t layer = 0; layer < keys.size(); ++layer) {
        const int64_t kv_elements = head_dim * context_tokens * kv_heads;
        const int64_t mask_elements = context_tokens;
        if (!set_f16_tensor(keys[layer], kv_elements, 631 + static_cast<uint32_t>(layer), bytes)
            || !set_f16_tensor(values[layer], kv_elements, 733 + static_cast<uint32_t>(layer), bytes)
            || !set_f16_tensor(masks[layer], mask_elements, 839 + static_cast<uint32_t>(layer), bytes)) {
            free_scheduled_graph(scheduled);
            ggml_free(ctx);
            return false;
        }
        flops += 4.0 * static_cast<double>(query_heads)
            * static_cast<double>(context_tokens)
            * static_cast<double>(head_dim);
    }
    bytes += static_cast<double>(layers * head_dim * query_heads * sizeof(float));
    ggml_backend_synchronize(backend);

    const bool ok = compute_graph_timed(
        graph,
        scheduled,
        result,
        name,
        "runtime",
        context_tokens,
        head_dim,
        bytes,
        flops,
        0,
        GRAPH_WARMUP_RUNS,
        GRAPH_TIMED_RUNS);
    ggml_free(ctx);
    return ok;
}

bool run_logits_readback_probe(
    ggml_backend_t backend,
    const char * name,
    int64_t vocab,
    ProbeResult & result) {
    // This probe models the source-visible logits handoff shape, not another
    // transformer matmul. llama.cpp's decode graph can leave accelerator work
    // queued until logits are requested by the CPU-side sampler. The fit crate
    // already has separate probes for transformer blocks and output projection;
    // here we allocate a vocab-sized F32 tensor on the selected backend, read it
    // back through ggml_backend_tensor_get, and perform the same kind of
    // vocabulary scan a greedy sampler must do. Keeping this as a direct backend
    // tensor probe avoids constructing an artificial graph whose scheduler
    // behavior would be more about the probe than the logits handoff we need to
    // charge.
    if (vocab <= 0) {
        return false;
    }
    const size_t context_bytes = ggml_tensor_overhead() * 4 + ggml_graph_overhead();
    ggml_init_params params{};
    params.mem_size = context_bytes;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }

    ggml_tensor * logits = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, vocab);
    ggml_set_name(logits, "ggml_decode_logits_readback");
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        ggml_free(ctx);
        return false;
    }

    std::vector<float> logits_f32 = deterministic_f32(vocab, 947);
    ggml_backend_tensor_set(logits, logits_f32.data(), 0, logits_f32.size() * sizeof(float));
    ggml_backend_synchronize(backend);

    std::vector<float> cpu_logits(static_cast<size_t>(vocab));
    volatile int32_t best_token_sink = 0;
    auto read_once = [&]() -> double {
        const auto started = std::chrono::steady_clock::now();
        ggml_backend_tensor_get(logits, cpu_logits.data(), 0, cpu_logits.size() * sizeof(float));
        int32_t best = 0;
        float best_logit = -std::numeric_limits<float>::infinity();
        for (int32_t token = 0; token < static_cast<int32_t>(vocab); ++token) {
            if (cpu_logits[static_cast<size_t>(token)] > best_logit) {
                best_logit = cpu_logits[static_cast<size_t>(token)];
                best = token;
            }
        }
        best_token_sink = best;
        const auto finished = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(finished - started).count();
    };

    for (int i = 0; i < WARMUP_RUNS; ++i) {
        if (read_once() <= 0.0) {
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            return false;
        }
    }

    std::vector<double> seconds;
    seconds.reserve(TIMED_RUNS);
    for (int i = 0; i < TIMED_RUNS; ++i) {
        const double elapsed = read_once();
        if (elapsed <= 0.0) {
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            return false;
        }
        seconds.push_back(elapsed);
    }
    (void) best_token_sink;

    const double median_seconds = median(seconds);
    if (!std::isfinite(median_seconds) || median_seconds <= 0.0) {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }
    const double bytes = static_cast<double>(cpu_logits.size() * sizeof(float));
    const double effective_gbps = bytes / median_seconds / 1e9;
    const double elapsed_ms = median_seconds * 1000.0;
    if (!std::isfinite(effective_gbps) || !std::isfinite(elapsed_ms)) {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }
    result = make_probe_result(
        name,
        "runtime",
        vocab,
        1,
        0,
        effective_gbps,
        0.0,
        median_seconds,
        seconds,
        0,
        TIMED_RUNS);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return true;
}

bool run_logits_sync_probe(
    ggml_backend_t backend,
    const char * name,
    int64_t vocab,
    ProbeResult & result) {
    // llama.cpp's sampled decode path eventually calls `llama_get_logits_ith()`.
    // That accessor synchronizes the context before the sampler can see logits.
    // A plain tensor readback probe measures the CPU-visible copy/scan once the
    // tensor is ready; it does not measure a queued backend graph becoming
    // ready at the logits boundary. This probe submits a tiny vocab-shaped
    // graph, then immediately asks for the output tensor and scans it. It is
    // deliberately not a transformer graph and it does not use model weights:
    // the only GGUF-scaled fact is vocabulary size.
    if (vocab <= 0) {
        return false;
    }
    const size_t context_bytes = ggml_tensor_overhead() * 8 + ggml_graph_overhead();
    ggml_init_params params{};
    params.mem_size = context_bytes;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }

    ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, vocab);
    ggml_tensor * bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, vocab);
    ggml_tensor * logits = ggml_add(ctx, input, bias);
    ggml_set_name(input, "ggml_decode_logits_sync_input");
    ggml_set_name(bias, "ggml_decode_logits_sync_bias");
    ggml_set_name(logits, "ggml_decode_logits_sync");
    ggml_set_output(logits);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, logits);
    if (!graph_supported_by_backend(backend, graph)) {
        ggml_free(ctx);
        return false;
    }

    ScheduledGraph scheduled = alloc_sched_for_graph(backend, graph);
    if (scheduled.sched == nullptr) {
        ggml_free(ctx);
        return false;
    }

    std::vector<float> logits_f32 = deterministic_f32(vocab, 1949);
    ggml_backend_tensor_set(input, logits_f32.data(), 0, logits_f32.size() * sizeof(float));
    std::vector<float> bias_f32(static_cast<size_t>(vocab), 0.0f);
    ggml_backend_tensor_set(bias, bias_f32.data(), 0, bias_f32.size() * sizeof(float));
    ggml_backend_synchronize(backend);

    std::vector<float> cpu_logits(static_cast<size_t>(vocab));
    volatile int32_t best_token_sink = 0;
    auto sync_once = [&]() -> double {
        const auto started = std::chrono::steady_clock::now();
        enum ggml_status status = ggml_backend_sched_graph_compute_async(scheduled.sched, graph);
        if (status != GGML_STATUS_SUCCESS) {
            return 0.0;
        }
        ggml_backend_sched_synchronize(scheduled.sched);
        ggml_backend_tensor_get(logits, cpu_logits.data(), 0, cpu_logits.size() * sizeof(float));
        int32_t best = 0;
        float best_logit = -std::numeric_limits<float>::infinity();
        for (int32_t token = 0; token < static_cast<int32_t>(vocab); ++token) {
            if (cpu_logits[static_cast<size_t>(token)] > best_logit) {
                best_logit = cpu_logits[static_cast<size_t>(token)];
                best = token;
            }
        }
        best_token_sink = best;
        const auto finished = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(finished - started).count();
    };

    for (int i = 0; i < WARMUP_RUNS; ++i) {
        if (sync_once() <= 0.0) {
            free_scheduled_graph(scheduled);
            ggml_free(ctx);
            return false;
        }
    }

    std::vector<double> seconds;
    seconds.reserve(TIMED_RUNS);
    for (int i = 0; i < TIMED_RUNS; ++i) {
        const double elapsed = sync_once();
        if (elapsed <= 0.0) {
            free_scheduled_graph(scheduled);
            ggml_free(ctx);
            return false;
        }
        seconds.push_back(elapsed);
    }
    (void) best_token_sink;

    const double median_seconds = median(seconds);
    if (!std::isfinite(median_seconds) || median_seconds <= 0.0) {
        free_scheduled_graph(scheduled);
        ggml_free(ctx);
        return false;
    }
    const double bytes = static_cast<double>(cpu_logits.size() * sizeof(float));
    const double effective_gbps = bytes / median_seconds / 1e9;
    const double elapsed_ms = median_seconds * 1000.0;
    if (!std::isfinite(effective_gbps) || !std::isfinite(elapsed_ms)) {
        free_scheduled_graph(scheduled);
        ggml_free(ctx);
        return false;
    }
    result = make_probe_result(
        name,
        "runtime",
        vocab,
        1,
        graph_node_count(graph),
        effective_gbps,
        0.0,
        median_seconds,
        seconds,
        0,
        TIMED_RUNS);
    result.graph_inventory = collect_graph_inventory(graph);

    free_scheduled_graph(scheduled);
    ggml_free(ctx);
    return true;
}

bool run_logits_output_handoff_probe(
    ggml_backend_t backend,
    const char * name,
    int64_t vocab,
    ProbeResult & result) {
    // This is intentionally shaped after the llama.cpp sampled decode boundary,
    // not after a generic tensor copy benchmark. In `llama_context::decode()`,
    // the graph is submitted, then llama.cpp extracts the vocab-sized logits row
    // with `ggml_backend_tensor_get_async()` into an output buffer allocated from
    // the model output device's host buffer type when one exists, falling back to
    // a CPU buffer otherwise. Later, `llama_get_logits_ith()` synchronizes the
    // context before Skippy's greedy sampler scans the row. Tiny models can spend
    // a material fraction of token time here because their transformer matmuls are
    // cheap; treating this source-visible handoff as "just bandwidth" hides that
    // miss class. The only GGUF-scaled input is vocabulary size.
    if (vocab <= 0) {
        return false;
    }
    const size_t context_bytes = ggml_tensor_overhead() * 8 + ggml_graph_overhead();
    ggml_init_params params{};
    params.mem_size = context_bytes;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }

    ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, vocab);
    ggml_tensor * bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, vocab);
    ggml_tensor * logits = ggml_add(ctx, input, bias);
    ggml_set_name(input, "ggml_decode_logits_output_handoff_input");
    ggml_set_name(bias, "ggml_decode_logits_output_handoff_bias");
    ggml_set_name(logits, "ggml_decode_logits_output_handoff");
    ggml_set_output(logits);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, logits);
    if (!graph_supported_by_backend(backend, graph)) {
        ggml_free(ctx);
        return false;
    }

    ScheduledGraph scheduled = alloc_sched_for_graph(backend, graph);
    if (scheduled.sched == nullptr) {
        ggml_free(ctx);
        return false;
    }

    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    ggml_backend_buffer_type_t output_buft = ggml_backend_cpu_buffer_type();
    if (device != nullptr) {
        ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(device);
        if (host_buft != nullptr) {
            output_buft = host_buft;
        }
    }
    const size_t output_bytes = static_cast<size_t>(vocab) * sizeof(float);
    ggml_backend_buffer_t output_buffer = ggml_backend_buft_alloc_buffer(output_buft, output_bytes);
    if (output_buffer == nullptr) {
        free_scheduled_graph(scheduled);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_buffer_clear(output_buffer, 0);
    float * output_base = static_cast<float *>(ggml_backend_buffer_get_base(output_buffer));
    if (output_base == nullptr) {
        ggml_backend_buffer_free(output_buffer);
        free_scheduled_graph(scheduled);
        ggml_free(ctx);
        return false;
    }

    std::vector<float> logits_f32 = deterministic_f32(vocab, 3907);
    ggml_backend_tensor_set(input, logits_f32.data(), 0, logits_f32.size() * sizeof(float));
    std::vector<float> bias_f32(static_cast<size_t>(vocab), 0.0f);
    ggml_backend_tensor_set(bias, bias_f32.data(), 0, bias_f32.size() * sizeof(float));
    ggml_backend_synchronize(backend);

    volatile int32_t best_token_sink = 0;
    auto handoff_once = [&]() -> double {
        const auto started = std::chrono::steady_clock::now();
        enum ggml_status status = ggml_backend_sched_graph_compute_async(scheduled.sched, graph);
        if (status != GGML_STATUS_SUCCESS) {
            return 0.0;
        }
        ggml_backend_t logits_backend = ggml_backend_sched_get_tensor_backend(scheduled.sched, logits);
        if (logits_backend == nullptr) {
            return 0.0;
        }
        ggml_backend_tensor_get_async(logits_backend, logits, output_base, 0, output_bytes);
        ggml_backend_sched_synchronize(scheduled.sched);
        int32_t best = 0;
        float best_logit = -std::numeric_limits<float>::infinity();
        for (int32_t token = 0; token < static_cast<int32_t>(vocab); ++token) {
            if (output_base[token] > best_logit) {
                best_logit = output_base[token];
                best = token;
            }
        }
        best_token_sink = best;
        const auto finished = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(finished - started).count();
    };

    for (int i = 0; i < WARMUP_RUNS; ++i) {
        if (handoff_once() <= 0.0) {
            ggml_backend_buffer_free(output_buffer);
            free_scheduled_graph(scheduled);
            ggml_free(ctx);
            return false;
        }
    }

    std::vector<double> seconds;
    seconds.reserve(TIMED_RUNS);
    for (int i = 0; i < TIMED_RUNS; ++i) {
        const double elapsed = handoff_once();
        if (elapsed <= 0.0) {
            ggml_backend_buffer_free(output_buffer);
            free_scheduled_graph(scheduled);
            ggml_free(ctx);
            return false;
        }
        seconds.push_back(elapsed);
    }
    (void) best_token_sink;

    const double median_seconds = median(seconds);
    if (!std::isfinite(median_seconds) || median_seconds <= 0.0) {
        ggml_backend_buffer_free(output_buffer);
        free_scheduled_graph(scheduled);
        ggml_free(ctx);
        return false;
    }
    const double bytes = static_cast<double>(output_bytes);
    const double effective_gbps = bytes / median_seconds / 1e9;
    const double elapsed_ms = median_seconds * 1000.0;
    if (!std::isfinite(effective_gbps) || !std::isfinite(elapsed_ms)) {
        ggml_backend_buffer_free(output_buffer);
        free_scheduled_graph(scheduled);
        ggml_free(ctx);
        return false;
    }
    result = make_probe_result(
        name,
        "runtime",
        vocab,
        1,
        graph_node_count(graph),
        effective_gbps,
        0.0,
        median_seconds,
        seconds,
        0,
        TIMED_RUNS);
    result.graph_inventory = collect_graph_inventory(graph);

    ggml_backend_buffer_free(output_buffer);
    free_scheduled_graph(scheduled);
    ggml_free(ctx);
    return true;
}

bool run_dense_sampled_token_probe(
    ggml_backend_t backend,
    enum ggml_type type,
    const char * name,
    const char * tensor_type,
    int64_t hidden,
    int64_t kv_width,
    int64_t ffn,
    int64_t vocab,
    int64_t repeat_layers,
    int graph_features,
    int64_t norm_head_width,
    ProbeResult & result) {
    const int64_t layers = std::max<int64_t>(1, repeat_layers);
    if (vocab <= 0 || !dense_llama_shape_supported(type, hidden, kv_width, ffn)
        || !matrix_shape_supported(type, vocab, hidden)) {
        return false;
    }
    const bool use_q_norm = (graph_features & GRAPH_FEATURE_ATTENTION_Q_NORM) != 0;
    const bool use_k_norm = (graph_features & GRAPH_FEATURE_ATTENTION_K_NORM) != 0;
    const bool use_post_attention_norm = (graph_features & GRAPH_FEATURE_ATTENTION_POST_NORM) != 0;
    const bool use_post_ffn_norm = (graph_features & GRAPH_FEATURE_FFN_POST_NORM) != 0;
    const size_t context_bytes =
        ggml_tensor_overhead() * static_cast<size_t>(84 * layers + 16) + ggml_graph_overhead();
    ggml_init_params params{};
    params.mem_size = context_bytes;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, 1);
    ggml_tensor * output = input;
    ggml_cgraph * graph = ggml_new_graph(ctx);
    for (int64_t layer = 0; layer < layers; ++layer) {
        ggml_tensor * attn_input = use_post_attention_norm ? output : ggml_rms_norm(ctx, output, 1e-5f);
        ggml_tensor * q = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, hidden), attn_input);
        ggml_tensor * k = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, kv_width), attn_input);
        ggml_tensor * v = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, kv_width), attn_input);
        if (use_q_norm) {
            q = rms_norm_attention_projection(ctx, q, hidden, norm_head_width);
        }
        if (use_k_norm) {
            k = rms_norm_attention_projection(ctx, k, kv_width, norm_head_width);
        }
        ggml_build_forward_expand(graph, q);
        ggml_build_forward_expand(graph, k);
        ggml_build_forward_expand(graph, v);
        ggml_tensor * attn = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, hidden), q);
        ggml_tensor * ffn_input;
        ggml_tensor * residual_after_attn;
        if (use_post_attention_norm) {
            ggml_tensor * attn_norm = ggml_rms_norm(ctx, attn, 1e-5f);
            ffn_input = ggml_add(ctx, output, attn_norm);
            residual_after_attn = ffn_input;
        } else {
            residual_after_attn = ggml_add(ctx, output, attn);
            ffn_input = ggml_rms_norm(ctx, residual_after_attn, 1e-5f);
        }
        ggml_tensor * gate = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, ffn), ffn_input);
        ggml_tensor * up = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, ffn), ffn_input);
        ggml_tensor * gated = ggml_swiglu_split(ctx, gate, up);
        ggml_tensor * down = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, ffn, hidden), gated);
        if (use_post_ffn_norm) {
            down = ggml_rms_norm(ctx, down, 1e-5f);
        }
        output = ggml_add(ctx, residual_after_attn, down);
    }
    ggml_tensor * logits = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, vocab), output);
    ggml_set_name(input, "ggml_decode_sampled_token_input");
    ggml_set_name(logits, "ggml_decode_sampled_token_logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(graph, logits);
    if (!graph_supported_by_backend(backend, graph)) {
        ggml_free(ctx);
        return false;
    }

    ScheduledGraph scheduled = alloc_sched_for_graph(backend, graph);
    if (scheduled.sched == nullptr) {
        ggml_free(ctx);
        return false;
    }

    double bytes = 0.0;
    double flops = 0.0;
    std::vector<float> input_f32 = deterministic_f32(hidden, 101);
    ggml_backend_tensor_set(input, input_f32.data(), 0, input_f32.size() * sizeof(float));
    bytes += static_cast<double>(input_f32.size() * sizeof(float));
    EncodedWeightCache weight_cache;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        if (t->type != type || t->op != GGML_OP_NONE) {
            continue;
        }
        const int64_t rows = t->ne[1];
        const int64_t cols = t->ne[0];
        if (!set_encoded_weights(t, type, rows, cols, weight_cache, bytes, flops)) {
            free_scheduled_graph(scheduled);
            ggml_free(ctx);
            return false;
        }
    }
    bytes += static_cast<double>((layers * (4 * hidden + ffn) + vocab) * sizeof(float));
    std::vector<float> cpu_logits(static_cast<size_t>(vocab));
    volatile int32_t best_token_sink = 0;
    ggml_backend_synchronize(backend);

    auto compute_and_sample_once = [&]() -> double {
        const auto started = std::chrono::steady_clock::now();
        enum ggml_status status = ggml_backend_sched_graph_compute_async(scheduled.sched, graph);
        if (status != GGML_STATUS_SUCCESS) {
            return 0.0;
        }
        ggml_backend_tensor_get(logits, cpu_logits.data(), 0, cpu_logits.size() * sizeof(float));
        int32_t best = 0;
        float best_logit = -std::numeric_limits<float>::infinity();
        for (int32_t token = 0; token < static_cast<int32_t>(vocab); ++token) {
            if (cpu_logits[static_cast<size_t>(token)] > best_logit) {
                best_logit = cpu_logits[static_cast<size_t>(token)];
                best = token;
            }
        }
        best_token_sink = best;
        const auto finished = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(finished - started).count();
    };

    for (int i = 0; i < GRAPH_WARMUP_RUNS; ++i) {
        if (compute_and_sample_once() <= 0.0) {
            free_scheduled_graph(scheduled);
            ggml_free(ctx);
            return false;
        }
    }

    std::vector<double> seconds;
    seconds.reserve(GRAPH_TIMED_RUNS);
    for (int i = 0; i < GRAPH_TIMED_RUNS; ++i) {
        const double elapsed = compute_and_sample_once();
        if (elapsed <= 0.0) {
            free_scheduled_graph(scheduled);
            ggml_free(ctx);
            return false;
        }
        seconds.push_back(elapsed);
    }
    (void) best_token_sink;

    const double median_seconds = median(seconds);
    if (!std::isfinite(median_seconds) || median_seconds <= 0.0) {
        free_scheduled_graph(scheduled);
        ggml_free(ctx);
        return false;
    }
    const double effective_gbps = bytes / median_seconds / 1e9;
    const double tflops = flops / median_seconds / 1e12;
    const double elapsed_ms = median_seconds * 1000.0;
    if (!std::isfinite(effective_gbps) || !std::isfinite(tflops) || !std::isfinite(elapsed_ms)) {
        free_scheduled_graph(scheduled);
        ggml_free(ctx);
        return false;
    }
    result = make_probe_result(
        name,
        tensor_type,
        vocab,
        hidden,
        graph_node_count(graph),
        effective_gbps,
        tflops,
        median_seconds,
        seconds,
        graph_features,
        GRAPH_TIMED_RUNS);
    result.graph_inventory = collect_graph_inventory(graph);

    free_scheduled_graph(scheduled);
    ggml_free(ctx);
    return true;
}

bool run_dense_full_token_probe(
    ggml_backend_t backend,
    enum ggml_type block_type,
    enum ggml_type output_type,
    const char * name,
    const char * tensor_type,
    int64_t hidden,
    int64_t kv_width,
    int64_t ffn,
    int64_t vocab,
    int64_t repeat_layers,
    int graph_features,
    int64_t norm_head_width,
    int64_t head_dim,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t context_tokens,
    int64_t active_context_tokens,
    bool include_output_handoff,
    bool measure_submission,
    bool measure_source_sampled,
    bool measure_source_input,
    ProbeResult & result) {
    const int64_t layers = std::max<int64_t>(1, repeat_layers);
    const int64_t query_width = head_dim * query_heads;
    const int64_t kv_capacity_tokens = std::max<int64_t>(1, context_tokens);
    const int64_t n_kv = std::clamp<int64_t>(
        std::max<int64_t>(1, active_context_tokens),
        1,
        kv_capacity_tokens);
    if (vocab <= 0 || head_dim <= 0 || query_heads <= 0 || kv_heads <= 0 || context_tokens <= 0
        || kv_width != head_dim * kv_heads
        || !matrix_shape_supported(block_type, query_width, hidden)
        || !matrix_shape_supported(block_type, kv_width, hidden)
        || !matrix_shape_supported(block_type, hidden, query_width)
        || !matrix_shape_supported(block_type, ffn, hidden)
        || !matrix_shape_supported(block_type, hidden, ffn)
        || !matrix_shape_supported(output_type, vocab, hidden)) {
        return false;
    }
    const bool use_q_norm = (graph_features & GRAPH_FEATURE_ATTENTION_Q_NORM) != 0;
    const bool use_k_norm = (graph_features & GRAPH_FEATURE_ATTENTION_K_NORM) != 0;
    const bool use_post_attention_norm = (graph_features & GRAPH_FEATURE_ATTENTION_POST_NORM) != 0;
    const bool use_post_ffn_norm = (graph_features & GRAPH_FEATURE_FFN_POST_NORM) != 0;
    const size_t context_bytes =
        ggml_tensor_overhead() * static_cast<size_t>(138 * layers + 16) + ggml_graph_overhead();
    ggml_init_params params{};
    params.mem_size = context_bytes;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }

    // llama.cpp decode starts from token ids, not a pre-materialized hidden
    // vector. The graph first performs `get_rows(token_embd.weight, token_id)`
    // and feeds that embedding row into layer 0. This is a small amount of
    // graph topology for large models, but it is a real source-visible decode
    // operation and the token embedding table can be comparable to the whole
    // transformer block for tiny models. Keep it inside the full-token probe so
    // model-fit does not need a backend or family correction for small GGUFs.
    ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor * token_embedding =
        ggml_new_tensor_2d(ctx, output_type, hidden, vocab);
    ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor * output_index = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_tensor * output = ggml_get_rows(ctx, token_embedding, input);
    std::vector<ggml_tensor *> keys;
    std::vector<ggml_tensor *> values;
    std::vector<ggml_tensor *> masks;
    std::vector<ggml_tensor *> key_indices;
    std::vector<ggml_tensor *> value_indices;
    std::vector<ggml_tensor *> f32_weights;
    keys.reserve(static_cast<size_t>(layers));
    values.reserve(static_cast<size_t>(layers));
    masks.reserve(static_cast<size_t>(layers));
    key_indices.reserve(static_cast<size_t>(layers));
    value_indices.reserve(static_cast<size_t>(layers));

    for (int64_t layer = 0; layer < layers; ++layer) {
        ggml_tensor * attn_norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden);
        ggml_tensor * ffn_norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden);
        f32_weights.push_back(attn_norm_weight);
        f32_weights.push_back(ffn_norm_weight);

        ggml_tensor * attn_input = output;
        if (!use_post_attention_norm) {
            // llama.cpp's `build_norm()` is not just `GGML_OP_RMS_NORM`: when
            // the model has a learned norm tensor, source graph construction
            // immediately multiplies the normalized activation by that weight.
            // The synthetic full-token probe used to skip these per-layer
            // norm-weight multiplies, leaving the matmul byte inventory correct
            // but the decode graph 2 nodes/layer thinner than llama.cpp. Tiny
            // models are exactly where those non-matmul nodes are visible, so
            // keep them in the GGML graph rather than hiding them behind a
            // scalar latency correction.
            attn_input = ggml_rms_norm(ctx, output, 1e-5f);
            attn_input = ggml_mul(ctx, attn_input, attn_norm_weight);
            ggml_set_name(attn_input, "attn_norm_weighted");
        }
        ggml_tensor * q = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, block_type, hidden, query_width), attn_input);
        ggml_tensor * k = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, block_type, hidden, kv_width), attn_input);
        ggml_tensor * v = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, block_type, hidden, kv_width), attn_input);
        if (use_q_norm) {
            q = rms_norm_attention_projection(ctx, q, query_width, norm_head_width);
        }
        if (use_k_norm) {
            k = rms_norm_attention_projection(ctx, k, kv_width, norm_head_width);
        }
        // Source llama.cpp reshapes Q/K/V projections into
        // [head_dim, heads, n_tokens] before RoPE and KV-cache handling:
        //
        //   Qcur = ggml_reshape_3d(...);
        //   Kcur = ggml_reshape_3d(...);
        //   Vcur = ggml_reshape_3d(...);
        //
        // Earlier synthetic probe versions jumped straight from 2-D projection
        // output to RoPE/cache writes. That preserved matmul bytes but missed
        // two source-visible reshape nodes per layer after accounting for the
        // synthetic-only q reshape below. Tiny models are sensitive to this
        // topology because these non-matmul nodes sit on the sampled-token
        // boundary. Keep the real GGML nodes here so the probe is source-shaped
        // rather than corrected with a backend or model-family constant.
        q = ggml_reshape_3d(ctx, q, head_dim, query_heads, 1);
        k = ggml_reshape_3d(ctx, k, head_dim, kv_heads, 1);
        v = ggml_reshape_3d(ctx, v, head_dim, kv_heads, 1);
        // Source llama.cpp applies RoPE to the current Q/K projections before
        // the token's K row is written into the layer KV cache. The synthetic
        // full-token probe used to skip this because RoPE is not a matmul and
        // contributes no model weight bytes. That made the probe's matmul
        // inventory look correct while its GGML graph topology was too thin,
        // especially for tiny models where non-matmul graph nodes dominate the
        // token latency. Keep this as a real GGML op rather than a scalar
        // penalty: the backend planner sees the same kind of source-visible
        // work llama.cpp submits for decode.
        q = ggml_rope(ctx, q, positions, static_cast<int>(head_dim), GGML_ROPE_TYPE_NEOX);
        k = ggml_rope(ctx, k, positions, static_cast<int>(head_dim), GGML_ROPE_TYPE_NEOX);
        ggml_build_forward_expand(graph, q);
        ggml_build_forward_expand(graph, v);
        ggml_build_forward_expand(graph, k);

        // llama.cpp decode does not attend over immutable synthetic K/V inputs.
        // Each token writes the current K and V rows into the layer KV cache with
        // GGML_OP_SET_ROWS, then attention reads from the updated cache view. For
        // small models, those cache-write/runtime nodes are large compared with
        // the matmuls, so the full-token probe must include them instead of
        // handing Flash Attention already-populated cache tensors.
        // llama.cpp separates KV allocation capacity from active attention
        // length. `llama_kv_cache::cpy_k/cpy_v()` writes into buffers sized by
        // `--ctx-size`, while `llama_kv_cache::get_n_kv()` pads the currently
        // used cells and `get_k/get_v()` expose only that active n_kv view to
        // Flash Attention. Tiny-model estimates were too optimistic when this
        // synthetic graph used one `context_tokens` value for both ideas: it
        // matched some total bytes but exercised a different backend attention
        // layout than the source graph. Keep the capacity-shaped SET_ROWS nodes
        // and use n_kv-shaped K/V/mask views for the attention path.
        ggml_tensor * key_cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv_width, kv_capacity_tokens);
        ggml_tensor * value_cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv_width, kv_capacity_tokens);
        ggml_tensor * key_index = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
        ggml_tensor * value_index = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
        // llama.cpp's KV cache write path (`cpy_k`/`cpy_v`) uses
        // `ggml_view_2d()` to merge the current head dimensions before
        // `ggml_set_rows()`. A reshape has the same logical dimensions for our
        // synthetic single-token tensor, but it does not produce the same
        // source-visible GGML op topology. Use a view here so graph inventory
        // lines up with the real decode graph without changing any bytes or
        // fitting to observed throughput.
        ggml_tensor * key_current = ggml_view_2d(ctx, k, kv_width, 1, k->nb[2], 0);
        ggml_tensor * value_current = ggml_view_2d(ctx, v, kv_width, 1, v->nb[2], 0);
        ggml_tensor * key_written = ggml_set_rows(ctx, key_cache, key_current, key_index);
        ggml_tensor * value_written = ggml_set_rows(ctx, value_cache, value_current, value_index);
        const size_t kv_element_size = ggml_element_size(key_written);
        ggml_tensor * key = ggml_view_4d(
            ctx,
            key_written,
            head_dim,
            kv_heads,
            n_kv,
            1,
            static_cast<size_t>(head_dim) * kv_element_size,
            key_written->nb[1],
            key_written->nb[1] * static_cast<size_t>(kv_capacity_tokens),
            0);
        ggml_tensor * value = ggml_view_4d(
            ctx,
            value_written,
            head_dim,
            kv_heads,
            n_kv,
            1,
            static_cast<size_t>(head_dim) * kv_element_size,
            value_written->nb[1],
            value_written->nb[1] * static_cast<size_t>(kv_capacity_tokens),
            0);
        ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, 1, 1, 1);
        // `build_attn_mha()` views the already-3D Q tensor as 4D to split
        // streams, then permutes it into Flash Attention layout. Using a
        // reshape here would add a synthetic-only node and hide the source
        // topology delta we are trying to measure.
        ggml_tensor * q4 = ggml_view_4d(
            ctx,
            q,
            q->ne[0],
            q->ne[1],
            q->ne[2],
            1,
            q->nb[1],
            q->nb[2],
            q->nb[3],
            0);
        q4 = ggml_permute(ctx, q4, 0, 2, 1, 3);
        key = ggml_permute(ctx, key, 0, 2, 1, 3);
        value = ggml_permute(ctx, value, 0, 2, 1, 3);
        ggml_tensor * attn = ggml_flash_attn_ext(
            ctx,
            q4,
            key,
            value,
            mask,
            1.0f / std::sqrt(static_cast<float>(head_dim)),
            0.0f,
            0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
        attn = ggml_reshape_2d(ctx, attn, query_width, 1);
        ggml_tensor * attn_out = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, block_type, query_width, hidden), attn);
        ggml_tensor * residual_source = output;
        if (layer == layers - 1) {
            // llama.cpp gathers only requested output rows on the final layer:
            //
            //   cur   = ggml_get_rows(cur,   inp_out_ids);
            //   inpSA = ggml_get_rows(inpSA, inp_out_ids);
            //
            // Decode uses one requested output token, but these two source
            // `GGML_OP_GET_ROWS` nodes are still present in the graph. They are
            // small, yet they are exactly the kind of non-matmul topology that
            // made tiny models look too fast when the synthetic probe only
            // matched weight bytes.
            attn_out = ggml_get_rows(ctx, attn_out, output_index);
            residual_source = ggml_get_rows(ctx, residual_source, output_index);
        }
        ggml_tensor * ffn_input;
        ggml_tensor * residual_after_attn;
        if (use_post_attention_norm) {
            ggml_tensor * attn_norm = ggml_rms_norm(ctx, attn_out, 1e-5f);
            attn_norm = ggml_mul(ctx, attn_norm, attn_norm_weight);
            ggml_set_name(attn_norm, "attn_post_norm_weighted");
            ffn_input = ggml_add(ctx, residual_source, attn_norm);
            residual_after_attn = ffn_input;
        } else {
            residual_after_attn = ggml_add(ctx, residual_source, attn_out);
            ffn_input = ggml_rms_norm(ctx, residual_after_attn, 1e-5f);
            ffn_input = ggml_mul(ctx, ffn_input, ffn_norm_weight);
            ggml_set_name(ffn_input, "ffn_norm_weighted");
        }
        ggml_tensor * gate = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, block_type, hidden, ffn), ffn_input);
        ggml_tensor * up = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, block_type, hidden, ffn), ffn_input);
        ggml_tensor * gated = ggml_swiglu_split(ctx, gate, up);
        ggml_tensor * down = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, block_type, ffn, hidden), gated);
        if (use_post_ffn_norm) {
            down = ggml_rms_norm(ctx, down, 1e-5f);
        }
        output = ggml_add(ctx, residual_after_attn, down);
        keys.push_back(key_cache);
        values.push_back(value_cache);
        masks.push_back(mask);
        key_indices.push_back(key_index);
        value_indices.push_back(value_index);
    }
    ggml_tensor * final_norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden);
    ggml_tensor * logits_input = ggml_rms_norm(ctx, output, 1e-5f);
    logits_input = ggml_mul(ctx, logits_input, final_norm_weight);
    ggml_tensor * logits = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, output_type, hidden, vocab), logits_input);
    f32_weights.push_back(final_norm_weight);
    ggml_set_name(input, "ggml_decode_full_token_input");
    ggml_set_name(final_norm_weight, "ggml_decode_full_token_output_norm");
    ggml_set_name(logits, "ggml_decode_full_token_logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(graph, logits);
    if (!graph_supported_by_backend(backend, graph)) {
        ggml_free(ctx);
        return false;
    }

    ScheduledGraph scheduled = alloc_sched_for_graph(backend, graph);
    if (scheduled.sched == nullptr) {
        ggml_free(ctx);
        return false;
    }

    double bytes = 0.0;
    double flops = 0.0;
    if (!set_i32_tensor(input, 1, 0, bytes)) {
        free_scheduled_graph(scheduled);
        ggml_free(ctx);
        return false;
    }
    if (!set_i32_tensor(
        positions,
        1,
        static_cast<int32_t>(std::max<int64_t>(0, n_kv - 1)),
        bytes)) {
        free_scheduled_graph(scheduled);
        ggml_free(ctx);
        return false;
    }
    if (!set_i32_tensor(output_index, 1, 0, bytes)) {
        free_scheduled_graph(scheduled);
        ggml_free(ctx);
        return false;
    }
    EncodedWeightCache weight_cache;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        if (t->op != GGML_OP_NONE) {
            continue;
        }
        const int64_t rows = t->ne[1];
        const int64_t cols = t->ne[0];
        if (t->type == block_type || t->type == output_type) {
            if (!set_encoded_weights(t, t->type, rows, cols, weight_cache, bytes, flops)) {
                free_scheduled_graph(scheduled);
                ggml_free(ctx);
                return false;
            }
        }
    }
    for (size_t i = 0; i < f32_weights.size(); ++i) {
        if (!set_f32_tensor(f32_weights[i], hidden, 2801 + static_cast<uint32_t>(i), bytes)) {
            free_scheduled_graph(scheduled);
            ggml_free(ctx);
            return false;
        }
    }
    for (size_t layer = 0; layer < keys.size(); ++layer) {
        const int64_t kv_elements = head_dim * kv_capacity_tokens * kv_heads;
        const int64_t mask_elements = n_kv;
        if (!set_f16_tensor(keys[layer], kv_elements, 1409 + static_cast<uint32_t>(layer), bytes)
            || !set_f16_tensor(values[layer], kv_elements, 1601 + static_cast<uint32_t>(layer), bytes)
            || !set_f16_tensor(masks[layer], mask_elements, 1801 + static_cast<uint32_t>(layer), bytes)
            || !set_i64_tensor(
                key_indices[layer],
                1,
                std::max<int64_t>(0, n_kv - 1),
                bytes)
            || !set_i64_tensor(
                value_indices[layer],
                1,
                std::max<int64_t>(0, n_kv - 1),
                bytes)) {
            free_scheduled_graph(scheduled);
            ggml_free(ctx);
            return false;
        }
        flops += 4.0 * static_cast<double>(query_heads)
            * static_cast<double>(n_kv)
            * static_cast<double>(head_dim);
    }

    const bool ok = measure_source_input
        ? compute_graph_source_input_timed(
            graph,
            input,
            positions,
            key_indices,
            value_indices,
            masks,
            scheduled,
            result,
            name,
            tensor_type,
            vocab,
            hidden,
            repeat_layers,
            n_kv,
            flops,
            graph_features,
            GRAPH_WARMUP_RUNS,
            SOURCE_BOUNDARY_TIMED_RUNS)
        : measure_source_sampled
        ? compute_graph_source_sampled_timed(
            backend,
            graph,
            input,
            positions,
            logits,
            key_indices,
            value_indices,
            masks,
            scheduled,
            result,
            name,
            tensor_type,
            vocab,
            hidden,
            repeat_layers,
            n_kv,
            bytes,
            flops,
            graph_features,
            GRAPH_WARMUP_RUNS,
            SOURCE_BOUNDARY_TIMED_RUNS)
        : measure_submission
        ? compute_graph_submission_timed(
            backend,
            graph,
            input,
            positions,
            logits,
            key_indices,
            value_indices,
            masks,
            scheduled,
            result,
            name,
            tensor_type,
            vocab,
            hidden,
            repeat_layers,
            n_kv,
            bytes,
            flops,
            graph_features,
            GRAPH_WARMUP_RUNS,
            SOURCE_BOUNDARY_TIMED_RUNS)
        : include_output_handoff
        ? compute_graph_output_handoff_timed(
            backend,
            graph,
            input,
            positions,
            logits,
            key_indices,
            value_indices,
            masks,
            scheduled,
            result,
            name,
            tensor_type,
            vocab,
            hidden,
            n_kv,
            bytes,
            flops,
            graph_features,
            GRAPH_WARMUP_RUNS,
            SOURCE_BOUNDARY_TIMED_RUNS)
        : compute_graph_timed(
            graph,
            scheduled,
            result,
            name,
            tensor_type,
            vocab,
            hidden,
            bytes,
            flops,
            graph_features,
            GRAPH_WARMUP_RUNS,
            SOURCE_BOUNDARY_TIMED_RUNS);
    ggml_free(ctx);
    return ok;
}

bool run_linear_attention_graph_probe(
    ggml_backend_t backend,
    enum ggml_type type,
    const char * name,
    const char * tensor_type,
    int64_t hidden,
    int64_t qkv_width,
    int64_t gate_width,
    int64_t state_width,
    int64_t output_input_width,
    int64_t ffn,
    int64_t recurrent_layers,
    int64_t full_attention_layers,
    int64_t kv_width,
    int graph_features,
    int64_t norm_head_width,
    ProbeResult & result) {
    const int64_t recurrent = std::max<int64_t>(1, recurrent_layers);
    const int64_t full_attention = std::max<int64_t>(0, full_attention_layers);
    const bool use_q_norm = (graph_features & GRAPH_FEATURE_ATTENTION_Q_NORM) != 0;
    const bool use_k_norm = (graph_features & GRAPH_FEATURE_ATTENTION_K_NORM) != 0;
    const bool use_post_attention_norm = (graph_features & GRAPH_FEATURE_ATTENTION_POST_NORM) != 0;
    const bool use_post_ffn_norm = (graph_features & GRAPH_FEATURE_FFN_POST_NORM) != 0;
    const int64_t safe_qkv = std::max<int64_t>(1, qkv_width);
    const int64_t safe_gate = std::max<int64_t>(1, gate_width);
    const int64_t safe_state = std::max<int64_t>(1, state_width);
    const int64_t safe_output_input = std::max<int64_t>(1, output_input_width);
    const int64_t safe_kv = std::max<int64_t>(1, std::min(kv_width, hidden));
    if (!matrix_shape_supported(type, safe_qkv, hidden)
        || !matrix_shape_supported(type, safe_gate, hidden)
        || !matrix_shape_supported(type, safe_state, hidden)
        || !matrix_shape_supported(type, hidden, safe_output_input)
        || !matrix_shape_supported(type, ffn, hidden)
        || !matrix_shape_supported(type, hidden, ffn)
        || !dense_llama_shape_supported(type, hidden, safe_kv, ffn)) {
        return false;
    }
    const size_t context_bytes =
        ggml_tensor_overhead() * static_cast<size_t>(96 * (recurrent + full_attention))
        + ggml_graph_overhead();
    ggml_init_params params{};
    params.mem_size = context_bytes;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, 1);
    ggml_tensor * output = input;
    ggml_cgraph * graph = ggml_new_graph(ctx);
    for (int64_t layer = 0; layer < recurrent; ++layer) {
        // Linear/recurrent attention blocks in llama.cpp are not simply dense
        // Q/K/V attention with a cheaper KV cache. In Qwen3.5-style graphs,
        // `build_layer_attn_linear()` submits independent source-visible
        // projections for qkv, z/gate, beta, alpha, and the final linear
        // attention output, with recurrent/SSM elementwise work between them.
        // The fit estimator needs a probe with that graph topology. This probe
        // intentionally stays structural: tensor-role widths come from GGUF
        // metadata and it does not embed a family/backend correction.
        ggml_tensor * attn_input = use_post_attention_norm ? output : ggml_rms_norm(ctx, output, 1e-5f);
        ggml_tensor * qkv = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, safe_qkv), attn_input);
        ggml_tensor * gate = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, safe_gate), attn_input);
        ggml_tensor * beta = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, safe_state), attn_input);
        ggml_tensor * alpha = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, safe_state), attn_input);
        ggml_build_forward_expand(graph, qkv);
        ggml_build_forward_expand(graph, gate);
        ggml_build_forward_expand(graph, beta);
        ggml_build_forward_expand(graph, alpha);

        ggml_tensor * alpha_softplus = ggml_softplus(ctx, alpha);
        ggml_tensor * recurrent_gate = ggml_mul(ctx, alpha_softplus, beta);
        ggml_tensor * qkv_activated = ggml_silu(ctx, qkv);
        ggml_tensor * state_proxy = ggml_mul(ctx, recurrent_gate, recurrent_gate);
        ggml_build_forward_expand(graph, state_proxy);

        // llama.cpp normalizes the recurrent attention output with the z/gate
        // projection before `ssm_out`. The full gated norm is family-specific;
        // this compact proxy keeps the source-visible dependency and
        // elementwise scheduling without manufacturing extra weight bytes.
        ggml_tensor * gate_reduced = ggml_mean(ctx, gate);
        ggml_tensor * gated_qkv = ggml_mul(ctx, qkv_activated, gate_reduced);
        ggml_tensor * output_input = gated_qkv;
        if (safe_output_input < safe_qkv) {
            output_input = ggml_view_2d(
                ctx,
                gated_qkv,
                safe_output_input,
                1,
                ggml_row_size(gated_qkv->type, safe_output_input),
                0);
        }
        ggml_tensor * projected = ggml_mul_mat(
            ctx,
            ggml_new_tensor_2d(ctx, type, safe_output_input, hidden),
            ggml_reshape_2d(ctx, output_input, safe_output_input, 1));
        ggml_tensor * ffn_input;
        ggml_tensor * residual_after_attn;
        if (use_post_attention_norm) {
            ggml_tensor * attn_norm = ggml_rms_norm(ctx, projected, 1e-5f);
            ffn_input = ggml_add(ctx, output, attn_norm);
            residual_after_attn = ffn_input;
        } else {
            residual_after_attn = ggml_add(ctx, output, projected);
            ffn_input = ggml_rms_norm(ctx, residual_after_attn, 1e-5f);
        }
        ggml_tensor * ffn_gate = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, ffn), ffn_input);
        ggml_tensor * ffn_up = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, ffn), ffn_input);
        ggml_tensor * gated = ggml_swiglu_split(ctx, ffn_gate, ffn_up);
        ggml_tensor * down = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, ffn, hidden), gated);
        if (use_post_ffn_norm) {
            down = ggml_rms_norm(ctx, down, 1e-5f);
        }
        output = ggml_add(ctx, residual_after_attn, down);
    }

    for (int64_t layer = 0; layer < full_attention; ++layer) {
        ggml_tensor * attn_input = use_post_attention_norm ? output : ggml_rms_norm(ctx, output, 1e-5f);
        ggml_tensor * q = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, hidden), attn_input);
        ggml_tensor * k = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, safe_kv), attn_input);
        ggml_tensor * v = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, safe_kv), attn_input);
        if (use_q_norm) {
            q = rms_norm_attention_projection(ctx, q, hidden, norm_head_width);
        }
        if (use_k_norm) {
            k = rms_norm_attention_projection(ctx, k, safe_kv, norm_head_width);
        }
        ggml_build_forward_expand(graph, q);
        ggml_build_forward_expand(graph, k);
        ggml_build_forward_expand(graph, v);
        ggml_tensor * attn = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, hidden), q);
        ggml_tensor * ffn_input;
        ggml_tensor * residual_after_attn;
        if (use_post_attention_norm) {
            ggml_tensor * attn_norm = ggml_rms_norm(ctx, attn, 1e-5f);
            ffn_input = ggml_add(ctx, output, attn_norm);
            residual_after_attn = ffn_input;
        } else {
            residual_after_attn = ggml_add(ctx, output, attn);
            ffn_input = ggml_rms_norm(ctx, residual_after_attn, 1e-5f);
        }
        ggml_tensor * gate = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, ffn), ffn_input);
        ggml_tensor * up = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, hidden, ffn), ffn_input);
        ggml_tensor * gated = ggml_swiglu_split(ctx, gate, up);
        ggml_tensor * down = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, type, ffn, hidden), gated);
        if (use_post_ffn_norm) {
            down = ggml_rms_norm(ctx, down, 1e-5f);
        }
        output = ggml_add(ctx, residual_after_attn, down);
    }

    ggml_set_name(input, "ggml_decode_linear_attn_graph_input");
    ggml_set_name(output, "ggml_decode_linear_attn_graph_output");
    ggml_set_output(output);
    ggml_build_forward_expand(graph, output);
    if (!graph_supported_by_backend(backend, graph)) {
        ggml_free(ctx);
        return false;
    }

    ScheduledGraph scheduled = alloc_sched_for_graph(backend, graph);
    if (scheduled.sched == nullptr) {
        ggml_free(ctx);
        return false;
    }

    double bytes = 0.0;
    double flops = 0.0;
    std::vector<float> input_f32 = deterministic_f32(hidden, 131);
    ggml_backend_tensor_set(input, input_f32.data(), 0, input_f32.size() * sizeof(float));
    bytes += static_cast<double>(input_f32.size() * sizeof(float));
    EncodedWeightCache weight_cache;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        if (t->type != type || t->op != GGML_OP_NONE) {
            continue;
        }
        const int64_t rows = t->ne[1];
        const int64_t cols = t->ne[0];
        if (!set_encoded_weights(t, type, rows, cols, weight_cache, bytes, flops)) {
            free_scheduled_graph(scheduled);
            ggml_free(ctx);
            return false;
        }
    }
    bytes += static_cast<double>((recurrent + full_attention) * (5 * hidden + ffn) * sizeof(float));
    ggml_backend_synchronize(backend);

    const bool ok = compute_graph_timed(
        graph,
        scheduled,
        result,
        name,
        tensor_type,
        ffn,
        hidden,
        bytes,
        flops,
        graph_features,
        GRAPH_WARMUP_RUNS,
        GRAPH_TIMED_RUNS);
    ggml_free(ctx);
    return ok;
}

bool run_moe_mul_mat_id_probe(
    ggml_backend_t backend,
    enum ggml_type type,
    const char * name,
    const char * tensor_type,
    ProbeResult & result) {
    constexpr int64_t expert_count = 128;
    constexpr int64_t experts_used = 8;
    constexpr int64_t expert_width = 768;
    constexpr int64_t hidden = 2048;
    constexpr int64_t tokens = 1;
    if (!moe_shape_supported(type, hidden, expert_width)) {
        return false;
    }
    const size_t context_bytes = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
    ggml_init_params params{};
    params.mem_size = context_bytes;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }

    ggml_tensor * experts = ggml_new_tensor_3d(ctx, type, hidden, expert_width, expert_count);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, experts_used, tokens);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden, experts_used, tokens);
    ggml_tensor * output = ggml_mul_mat_id(ctx, experts, input, ids);
    ggml_set_name(experts, "ggml_decode_moe_experts");
    ggml_set_name(ids, "ggml_decode_moe_ids");
    ggml_set_name(input, "ggml_decode_moe_input");
    ggml_set_name(output, "ggml_decode_moe_output");
    ggml_set_output(output);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    if (!graph_supported_by_backend(backend, graph)) {
        ggml_free(ctx);
        return false;
    }

    ScheduledGraph scheduled = alloc_sched_for_graph(backend, graph);
    if (scheduled.sched == nullptr) {
        ggml_free(ctx);
        return false;
    }

    double ignored_resident_bytes = 0.0;
    double flops = 0.0;
    EncodedWeightCache weight_cache;
    if (!set_encoded_weights(
        experts,
        type,
        expert_width * expert_count,
        hidden,
        weight_cache,
        ignored_resident_bytes,
        flops)) {
        free_scheduled_graph(scheduled);
        ggml_free(ctx);
        return false;
    }
    double bytes = static_cast<double>(ggml_row_size(type, hidden) * expert_width * experts_used);
    std::vector<float> input_f32 = deterministic_f32(hidden * experts_used * tokens, 223);
    ggml_backend_tensor_set(input, input_f32.data(), 0, input_f32.size() * sizeof(float));
    std::vector<int32_t> ids_i32(static_cast<size_t>(experts_used * tokens));
    for (int64_t i = 0; i < experts_used * tokens; ++i) {
        ids_i32[static_cast<size_t>(i)] = static_cast<int32_t>(i % experts_used);
    }
    ggml_backend_tensor_set(ids, ids_i32.data(), 0, ids_i32.size() * sizeof(int32_t));
    bytes += static_cast<double>(input_f32.size() * sizeof(float));
    bytes += static_cast<double>(ids_i32.size() * sizeof(int32_t));
    bytes += static_cast<double>(expert_width * experts_used * tokens * sizeof(float));
    flops = 2.0 * static_cast<double>(expert_width)
        * static_cast<double>(hidden)
        * static_cast<double>(experts_used)
        * static_cast<double>(tokens);
    ggml_backend_synchronize(backend);

    const bool ok = compute_graph_timed(
        graph,
        scheduled,
        result,
        name,
        tensor_type,
        expert_width,
        hidden,
        bytes,
        flops);
    ggml_free(ctx);
    return ok;
}

bool run_moe_graph_probe(
    ggml_backend_t backend,
    enum ggml_type type,
    const char * name,
    const char * tensor_type,
    int64_t expert_count,
    int64_t experts_used,
    int64_t expert_width,
    int64_t hidden,
    int64_t repeat_layers,
    ProbeResult & result) {
    const int64_t layers = std::max<int64_t>(1, repeat_layers);
    constexpr int64_t tokens = 1;
    if (expert_count <= 0 || experts_used <= 0 || expert_width <= 0 || hidden <= 0) {
        return false;
    }
    if (experts_used > expert_count || expert_count > MAX_MODEL_SHAPED_MOE_EXPERTS) {
        return false;
    }
    if (!moe_shape_supported(type, hidden, expert_width)) {
        return false;
    }
    const size_t context_bytes =
        ggml_tensor_overhead() * static_cast<size_t>(96 * layers) + ggml_graph_overhead();
    ggml_init_params params{};
    params.mem_size = context_bytes;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }

    std::vector<ggml_tensor *> routers;
    std::vector<ggml_tensor *> up_experts;
    std::vector<ggml_tensor *> gate_experts;
    std::vector<ggml_tensor *> down_experts;
    routers.reserve(static_cast<size_t>(layers));
    up_experts.reserve(static_cast<size_t>(layers));
    gate_experts.reserve(static_cast<size_t>(layers));
    down_experts.reserve(static_cast<size_t>(layers));
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden, 1, tokens);
    ggml_tensor * output = input;

    // Mirror the source-level shape of llama.cpp's `build_moe_ffn()` for the
    // common SILU routed-expert path used by OLMoE/Qwen-style GGUFs:
    //
    //   gate logits -> softmax -> argsort_top_k -> get_rows(weights)
    //   -> MUL_MAT_ID(up/gate) -> SWIGLU_SPLIT -> MUL_MAT_ID(down)
    //   -> multiply by routed weights -> view/add selected experts
    //
    // This is still a synthetic hardware probe: it does not run a GGUF model or
    // inspect model names. The dimensions come from GGUF metadata, and the
    // graph operations come from llama.cpp source. The cap on expert_count is a
    // resource guard so a validation probe cannot accidentally allocate a
    // production-scale expert pool just to measure one hardware row.
    //
    // Deep validation repeats this routed FFN subgraph l4/l8 in a single
    // scheduled graph. That is the MoE analogue of the dense stacked llama
    // graph probes: it measures source-visible graph depth and scheduler
    // amortization without fitting a multiplier to observed model throughput.
    for (int64_t layer = 0; layer < layers; ++layer) {
        ggml_tensor * router = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, expert_count);
        ggml_tensor * up_exps = ggml_new_tensor_3d(ctx, type, hidden, expert_width, expert_count);
        ggml_tensor * gate_exps = ggml_new_tensor_3d(ctx, type, hidden, expert_width, expert_count);
        ggml_tensor * down_exps = ggml_new_tensor_3d(ctx, type, expert_width, hidden, expert_count);
        routers.push_back(router);
        up_experts.push_back(up_exps);
        gate_experts.push_back(gate_exps);
        down_experts.push_back(down_exps);

        ggml_tensor * logits = ggml_mul_mat(ctx, router, output);
        ggml_tensor * probs = ggml_soft_max(ctx, logits);
        ggml_tensor * ids = ggml_argsort_top_k(ctx, probs, experts_used);
        ggml_tensor * weights = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, expert_count, tokens), ids);
        ggml_tensor * routed_input = ggml_reshape_3d(ctx, output, hidden, 1, tokens);
        ggml_tensor * up = ggml_mul_mat_id(ctx, up_exps, routed_input, ids);
        ggml_tensor * gate = ggml_mul_mat_id(ctx, gate_exps, routed_input, ids);
        ggml_tensor * activated = ggml_swiglu_split(ctx, gate, up);
        ggml_tensor * experts = ggml_mul_mat_id(ctx, down_exps, activated, ids);
        experts = ggml_mul(ctx, experts, weights);
        ggml_tensor * layer_output = nullptr;
        for (int64_t expert = 0; expert < experts_used; ++expert) {
            ggml_tensor * expert_view = ggml_view_2d(
                ctx,
                experts,
                hidden,
                tokens,
                experts->nb[2],
                expert * experts->nb[1]);
            layer_output = layer_output == nullptr ? expert_view : ggml_add(ctx, layer_output, expert_view);
        }
        output = layer_output;
    }
    ggml_set_name(input, "ggml_decode_moe_graph_input");
    ggml_set_name(output, "ggml_decode_moe_graph_output");
    ggml_set_output(output);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    if (!graph_supported_by_backend(backend, graph)) {
        ggml_free(ctx);
        return false;
    }

    ScheduledGraph scheduled = alloc_sched_for_graph(backend, graph);
    if (scheduled.sched == nullptr) {
        ggml_free(ctx);
        return false;
    }

    double ignored_resident_bytes = 0.0;
    double ignored_flops = 0.0;
    EncodedWeightCache weight_cache;
    for (int64_t layer = 0; layer < layers; ++layer) {
        set_f32_weights(
            routers[static_cast<size_t>(layer)],
            expert_count,
            hidden,
            307 + static_cast<uint32_t>(layer * 17),
            ignored_resident_bytes,
            ignored_flops);
        if (!set_active_encoded_weights(
            up_experts[static_cast<size_t>(layer)],
            type,
            expert_width * experts_used,
            hidden,
            weight_cache,
            ignored_resident_bytes,
            ignored_flops)) {
            free_scheduled_graph(scheduled);
            ggml_free(ctx);
            return false;
        }
        if (!set_active_encoded_weights(
            gate_experts[static_cast<size_t>(layer)],
            type,
            expert_width * experts_used,
            hidden,
            weight_cache,
            ignored_resident_bytes,
            ignored_flops)) {
            free_scheduled_graph(scheduled);
            ggml_free(ctx);
            return false;
        }
        if (!set_active_encoded_weights(
            down_experts[static_cast<size_t>(layer)],
            type,
            hidden * experts_used,
            expert_width,
            weight_cache,
            ignored_resident_bytes,
            ignored_flops)) {
            free_scheduled_graph(scheduled);
            ggml_free(ctx);
            return false;
        }
    }

    double bytes = static_cast<double>(layers)
        * (static_cast<double>(expert_count * hidden * sizeof(float))
        + static_cast<double>(expert_count * tokens * sizeof(float))
        + static_cast<double>(expert_count * tokens * sizeof(float))
        + 2.0 * static_cast<double>(ggml_row_size(type, hidden) * expert_width * experts_used)
        + static_cast<double>(ggml_row_size(type, expert_width) * hidden * experts_used));
    std::vector<float> input_f32 = deterministic_f32(hidden * tokens, 331);
    ggml_backend_tensor_set(input, input_f32.data(), 0, input_f32.size() * sizeof(float));
    bytes += static_cast<double>(input_f32.size() * sizeof(float));
    bytes += static_cast<double>(layers)
        * static_cast<double>(experts_used * tokens * sizeof(int32_t));
    bytes += static_cast<double>(layers)
        * static_cast<double>((2 * expert_width + hidden) * experts_used * tokens * sizeof(float));
    bytes += static_cast<double>(layers)
        * static_cast<double>(hidden * experts_used * tokens * sizeof(float));
    const double flops = static_cast<double>(layers)
        * (2.0 * static_cast<double>(expert_count)
        * static_cast<double>(hidden)
        * static_cast<double>(tokens)
        + 6.0 * static_cast<double>(expert_width)
        * static_cast<double>(hidden)
        * static_cast<double>(experts_used)
        * static_cast<double>(tokens));
    ggml_backend_synchronize(backend);

    const bool ok = compute_graph_timed(
        graph,
        scheduled,
        result,
        name,
        tensor_type,
        expert_width,
        hidden,
        bytes,
        flops);
    ggml_free(ctx);
    return ok;
}

bool run_moe_block_graph_probe(
    ggml_backend_t backend,
    enum ggml_type type,
    const char * name,
    const char * tensor_type,
    int64_t expert_count,
    int64_t experts_used,
    int64_t expert_width,
    int64_t hidden,
    int64_t kv_width,
    int64_t repeat_layers,
    bool submission_only,
    int64_t context_tokens,
    ProbeResult & result) {
    const int64_t layers = std::max<int64_t>(1, repeat_layers);
    constexpr int64_t tokens = 1;
    if (expert_count <= 0 || experts_used <= 0 || expert_width <= 0 || hidden <= 0 || kv_width <= 0) {
        return false;
    }
    if (experts_used > expert_count || expert_count > MAX_MODEL_SHAPED_MOE_EXPERTS) {
        return false;
    }
    kv_width = std::min(kv_width, hidden);
    if (!dense_llama_shape_supported(type, hidden, kv_width, expert_width)
        || !moe_shape_supported(type, hidden, expert_width)) {
        return false;
    }
    const size_t context_bytes =
        ggml_tensor_overhead() * static_cast<size_t>(160 * layers) + ggml_graph_overhead();
    ggml_init_params params{};
    params.mem_size = context_bytes;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }

    struct LayerTensors {
        ggml_tensor * wq;
        ggml_tensor * wk;
        ggml_tensor * wv;
        ggml_tensor * wo;
        ggml_tensor * router;
        ggml_tensor * up_experts;
        ggml_tensor * gate_experts;
        ggml_tensor * down_experts;
    };
    std::vector<LayerTensors> layer_tensors;
    layer_tensors.reserve(static_cast<size_t>(layers));

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, 1);
    ggml_tensor * output = input;
    ggml_cgraph * graph = ggml_new_graph(ctx);

    // This probe intentionally models a sparse transformer block, not only the
    // routed expert inner loop. In llama.cpp an OLMoE/Qwen-MoE style decode
    // layer still pays the attention projections, scheduler boundaries,
    // residual adds, RMSNorms, and then the `build_moe_ffn()` routed path. The
    // older `moe_graph` row below remains useful diagnostic evidence for
    // GGML_OP_MUL_MAT_ID itself, but using that row as the sole estimator input
    // made sparse models look too fast because attention and graph depth were
    // composed from unlike probes. This block row keeps the operations and
    // dimensions source-shaped while still avoiding a production KV-cache
    // allocation; KV-cache read traffic is workload dependent and is charged by
    // model-fit from GGUF metadata.
    for (int64_t layer = 0; layer < layers; ++layer) {
        ggml_tensor * wq = ggml_new_tensor_2d(ctx, type, hidden, hidden);
        ggml_tensor * wk = ggml_new_tensor_2d(ctx, type, hidden, kv_width);
        ggml_tensor * wv = ggml_new_tensor_2d(ctx, type, hidden, kv_width);
        ggml_tensor * wo = ggml_new_tensor_2d(ctx, type, hidden, hidden);
        ggml_tensor * router = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, expert_count);
        ggml_tensor * up_exps = ggml_new_tensor_3d(ctx, type, hidden, expert_width, expert_count);
        ggml_tensor * gate_exps = ggml_new_tensor_3d(ctx, type, hidden, expert_width, expert_count);
        ggml_tensor * down_exps = ggml_new_tensor_3d(ctx, type, expert_width, hidden, expert_count);
        layer_tensors.push_back(LayerTensors{
            wq,
            wk,
            wv,
            wo,
            router,
            up_exps,
            gate_exps,
            down_exps,
        });

        ggml_tensor * attn_input = ggml_rms_norm(ctx, output, 1e-5f);
        ggml_tensor * q = ggml_mul_mat(ctx, wq, attn_input);
        ggml_tensor * k = ggml_mul_mat(ctx, wk, attn_input);
        ggml_tensor * v = ggml_mul_mat(ctx, wv, attn_input);
        ggml_build_forward_expand(graph, q);
        ggml_build_forward_expand(graph, k);
        ggml_build_forward_expand(graph, v);
        ggml_tensor * attn = ggml_mul_mat(ctx, wo, q);
        ggml_tensor * attn_residual = ggml_add(ctx, output, attn);

        ggml_tensor * ffn_input = ggml_rms_norm(ctx, attn_residual, 1e-5f);
        ggml_tensor * logits = ggml_mul_mat(ctx, router, ffn_input);
        ggml_tensor * probs = ggml_soft_max(ctx, logits);
        ggml_tensor * ids = ggml_argsort_top_k(ctx, probs, experts_used);
        ggml_tensor * weights = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, expert_count, tokens), ids);
        ggml_build_forward_expand(graph, weights);
        ggml_tensor * routed_input = ggml_reshape_3d(ctx, ffn_input, hidden, 1, tokens);
        ggml_tensor * up = ggml_mul_mat_id(ctx, up_exps, routed_input, ids);
        ggml_tensor * gate = ggml_mul_mat_id(ctx, gate_exps, routed_input, ids);
        ggml_tensor * activated = ggml_swiglu_split(ctx, gate, up);
        ggml_tensor * experts = ggml_mul_mat_id(ctx, down_exps, activated, ids);
        experts = ggml_mul(ctx, experts, weights);
        ggml_build_forward_expand(graph, experts);
        ggml_tensor * moe_output = nullptr;
        for (int64_t expert = 0; expert < experts_used; ++expert) {
            ggml_tensor * expert_view = ggml_view_2d(
                ctx,
                experts,
                hidden,
                tokens,
                experts->nb[2],
                expert * experts->nb[1]);
            ggml_build_forward_expand(graph, expert_view);
            moe_output = moe_output == nullptr ? expert_view : ggml_add(ctx, moe_output, expert_view);
            if (moe_output != expert_view) {
                ggml_build_forward_expand(graph, moe_output);
            }
        }
        output = ggml_add(ctx, attn_residual, moe_output);
    }
    ggml_set_name(input, "ggml_decode_moe_block_graph_input");
    ggml_set_name(output, "ggml_decode_moe_block_graph_output");
    ggml_set_output(output);

    ggml_build_forward_expand(graph, output);
    if (!graph_supported_by_backend(backend, graph)) {
        ggml_free(ctx);
        return false;
    }

    ScheduledGraph scheduled = alloc_sched_for_graph(backend, graph);
    if (scheduled.sched == nullptr) {
        ggml_free(ctx);
        return false;
    }

    double bytes = 0.0;
    double flops = 0.0;
    EncodedWeightCache weight_cache;
    for (int64_t layer = 0; layer < layers; ++layer) {
        const LayerTensors & tensors = layer_tensors[static_cast<size_t>(layer)];
        if (!set_encoded_weights(tensors.wq, type, hidden, hidden, weight_cache, bytes, flops)
            || !set_encoded_weights(tensors.wk, type, kv_width, hidden, weight_cache, bytes, flops)
            || !set_encoded_weights(tensors.wv, type, kv_width, hidden, weight_cache, bytes, flops)
            || !set_encoded_weights(tensors.wo, type, hidden, hidden, weight_cache, bytes, flops)) {
            free_scheduled_graph(scheduled);
            ggml_free(ctx);
            return false;
        }
        set_f32_weights(
            tensors.router,
            expert_count,
            hidden,
            607 + static_cast<uint32_t>(layer * 19),
            bytes,
            flops);
        if (!set_active_encoded_weights(
            tensors.up_experts,
            type,
            expert_width * experts_used,
            hidden,
            weight_cache,
            bytes,
            flops)) {
            free_scheduled_graph(scheduled);
            ggml_free(ctx);
            return false;
        }
        if (!set_active_encoded_weights(
            tensors.gate_experts,
            type,
            expert_width * experts_used,
            hidden,
            weight_cache,
            bytes,
            flops)) {
            free_scheduled_graph(scheduled);
            ggml_free(ctx);
            return false;
        }
        if (!set_active_encoded_weights(
            tensors.down_experts,
            type,
            hidden * experts_used,
            expert_width,
            weight_cache,
            bytes,
            flops)) {
            free_scheduled_graph(scheduled);
            ggml_free(ctx);
            return false;
        }
    }

    std::vector<float> input_f32 = deterministic_f32(hidden, 631);
    ggml_backend_tensor_set(input, input_f32.data(), 0, input_f32.size() * sizeof(float));
    bytes += static_cast<double>(input_f32.size() * sizeof(float));
    bytes += static_cast<double>(layers)
        * static_cast<double>(
            (8 * hidden + 2 * kv_width + expert_count + (3 * experts_used)) * sizeof(float)
            + experts_used * sizeof(int32_t));
    ggml_backend_synchronize(backend);

    bool ok = false;
    if (submission_only) {
        const std::vector<ggml_tensor *> empty_indices;
        const std::vector<ggml_tensor *> empty_masks;
        ok = compute_graph_submission_timed(
            backend,
            graph,
            input,
            nullptr,
            output,
            empty_indices,
            empty_indices,
            empty_masks,
            scheduled,
            result,
            name,
            tensor_type,
            hidden,
            hidden,
            layers,
            std::max<int64_t>(1, context_tokens),
            bytes,
            flops,
            0,
            GRAPH_WARMUP_RUNS,
            GRAPH_TIMED_RUNS);
    } else {
        ok = compute_graph_timed(
            graph,
            scheduled,
            result,
            name,
            tensor_type,
            expert_width,
            hidden,
            bytes,
            flops,
            0,
            GRAPH_WARMUP_RUNS,
            GRAPH_TIMED_RUNS);
    }
    ggml_free(ctx);
    return ok;
}

std::string results_json(const std::vector<ProbeResult> & results) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < results.size(); ++i) {
        const ProbeResult & result = results[i];
        if (i > 0) {
            out << ",";
        }
        out << "{\"name\":\"" << result.name << "\","
            << "\"tensor_type\":\"" << result.tensor_type << "\","
            << "\"rows\":" << result.rows << ","
            << "\"cols\":" << result.cols << ","
            << "\"batch_tokens\":1,"
            << "\"graph_features\":" << result.graph_features << ","
            << "\"graph_node_count\":" << result.graph_node_count << ","
            << "\"effective_gbps\":" << result.effective_gbps << ","
            << "\"tflops\":" << result.tflops << ","
            << "\"elapsed_ms\":" << result.elapsed_ms << ","
            << "\"min_elapsed_ms\":" << result.min_elapsed_ms << ","
            << "\"max_elapsed_ms\":" << result.max_elapsed_ms << ","
            << "\"spread_pct\":" << result.spread_pct << ","
            << "\"graph_inventory\":[";
        for (size_t bucket_index = 0; bucket_index < result.graph_inventory.size(); ++bucket_index) {
            const GraphInventoryBucket & bucket = result.graph_inventory[bucket_index];
            if (bucket_index > 0) {
                out << ",";
            }
            out << "{\"family\":\"" << bucket.family << "\","
                << "\"ggml_op\":" << bucket.ggml_op << ","
                << "\"ggml_type\":" << bucket.ggml_type << ","
                << "\"node_count\":" << bucket.node_count << ","
                << "\"element_count\":" << bucket.element_count << ","
                << "\"output_bytes\":" << bucket.output_bytes << ","
                << "\"src0_bytes\":" << bucket.src0_bytes << ","
                << "\"src1_bytes\":" << bucket.src1_bytes << ","
                << "\"ne\":["
                << bucket.ne[0] << ","
                << bucket.ne[1] << ","
                << bucket.ne[2] << ","
                << bucket.ne[3] << "]}";
        }
        out << "],"
            << "\"runs\":" << result.runs << "}";
    }
    out << "]";
    return out.str();
}

std::string sampler_probe_json(const SamplerProbeResult & result) {
    std::ostringstream out;
    out << "{\"history_us_per_token\":" << result.history_us_per_token
        << ",\"vocab_us_per_token\":" << result.vocab_us_per_token
        << ",\"history_tokens\":" << result.history_tokens
        << ",\"vocab_tokens\":" << result.vocab_tokens
        << ",\"runs\":" << result.runs
        << "}";
    return out.str();
}

} // namespace

extern "C" char * mesh_llm_gpu_bench_ggml_sampler_probe_json(
    int64_t vocab_tokens,
    int64_t history_tokens,
    char ** error_out) {
    if (error_out != nullptr) {
        *error_out = nullptr;
    }
    if (vocab_tokens <= 0 || history_tokens <= 0) {
        set_error(error_out, "sampler probe dimensions must be positive");
        return nullptr;
    }

    SamplerProbeResult result{};
    if (!run_source_sampler_probe(vocab_tokens, history_tokens, result)) {
        set_error(error_out, "source-shaped sampler probe did not produce a supported result");
        return nullptr;
    }
    return copy_c_string(sampler_probe_json(result));
}

extern "C" char * mesh_llm_gpu_bench_ggml_output_projection_probe_json(
    int backend_kind,
    int tensor_type_kind,
    int64_t hidden,
    int64_t vocab,
    char ** error_out) {
    if (error_out != nullptr) {
        *error_out = nullptr;
    }
    enum ggml_type type = probe_tensor_type(tensor_type_kind);
    if (type == GGML_TYPE_COUNT) {
        set_error(error_out, "unsupported output projection probe tensor type");
        return nullptr;
    }
    if (hidden <= 0 || vocab <= 0) {
        set_error(error_out, "output projection probe dimensions must be positive");
        return nullptr;
    }

    ggml_backend_t backend = init_backend(backend_kind);
    if (backend == nullptr) {
        set_error(error_out, "GGML decode probe backend is not available");
        return nullptr;
    }

    std::ostringstream name;
    name << "ggml_decode_"
         << probe_tensor_type_name(tensor_type_kind)
         << "_matvec_output_"
         << vocab
         << "_"
         << hidden;

    ProbeResult result{};
    std::vector<ProbeResult> results;
    if (run_probe(
            backend,
            type,
            name.str().c_str(),
            probe_tensor_type_name(tensor_type_kind),
            ProbeShape{nullptr, vocab, hidden},
            result)) {
        results.push_back(result);
    }
    ggml_backend_free(backend);

    if (results.empty()) {
        set_error(error_out, "GGML output projection probe did not produce a supported result");
        return nullptr;
    }
    return copy_c_string(results_json(results));
}

extern "C" char * mesh_llm_gpu_bench_ggml_decode_probe_json(
    int backend_kind,
    int probe_depth,
    char ** error_out) {
    if (error_out != nullptr) {
        *error_out = nullptr;
    }

    ggml_backend_t backend = init_backend(backend_kind);
    if (backend == nullptr) {
        set_error(error_out, "GGML decode probe backend is not available");
        return nullptr;
    }
    const bool deep_probes = probe_depth == PROBE_DEPTH_DEEP;

    std::vector<ProbeResult> results;
    ProbeResult result{};
    for (const ProbeShape & shape : DECODE_SHAPES) {
        std::string f16_name = std::string("ggml_decode_f16_matvec_") + shape.suffix;
        if (run_probe(backend, GGML_TYPE_F16, f16_name.c_str(), "f16", shape, result)) {
            results.push_back(result);
        }
        std::string q8_name = std::string("ggml_decode_q8_0_matvec_") + shape.suffix;
        if (run_probe(backend, GGML_TYPE_Q8_0, q8_name.c_str(), "q8_0", shape, result)) {
            results.push_back(result);
        }
        std::string q4_name = std::string("ggml_decode_q4_k_matvec_") + shape.suffix;
        if (run_probe(backend, GGML_TYPE_Q4_K, q4_name.c_str(), "q4_k", shape, result)) {
            results.push_back(result);
        }
        std::string q6_name = std::string("ggml_decode_q6_k_matvec_") + shape.suffix;
        if (run_probe(backend, GGML_TYPE_Q6_K, q6_name.c_str(), "q6_k", shape, result)) {
            results.push_back(result);
        }
    }
    for (const ProbeShape & shape : LLAMA_GRAPH_SHAPES) {
        std::string q4_graph_name = std::string("ggml_decode_q4_k_llama_graph_") + shape.suffix;
        if (run_llama_graph_probe(
                backend,
                GGML_TYPE_Q4_K,
                q4_graph_name.c_str(),
                "q4_k",
                shape.rows,
                shape.rows,
                shape.cols,
                1,
                0,
                0,
                result)) {
            results.push_back(result);
        }
        // Deep validation can request a bounded graph-depth curve. Dense
        // llama.cpp decode submits many transformer blocks in one scheduled
        // graph, so source-shaped probes at l4/l8 let the estimator observe
        // how scheduler/allocator/kernel-launch behavior changes with graph
        // depth without constructing a full model-sized synthetic graph. The
        // full-depth experiment was too expensive even for a 28-layer 3B model,
        // and l16 was still too slow for a smoke-grade Metal deep benchmark on
        // an M1 Ultra. The curve therefore deliberately stops at small fixed
        // depths that can be gathered repeatedly.
        //
        // These rows are intentionally not part of the standard hardware
        // fingerprint: first-run Metal pipeline compilation plus deeper graphs
        // made `mesh-llm gpus detect` exceed its operator-facing timeout on an
        // M1 Ultra. The default benchmark should stay fast and broadly
        // portable; slow probes belong to validation.
        //
        // We also do not emit Q8 stack rows. Validation on the narrow SmolLM2 Q8
        // model falsified them as portable estimator inputs because they
        // over-amortized graph work relative to real llama.cpp decode.
        if (deep_probes && ((shape.rows == 2560 && shape.cols == 9728) ||
            (shape.rows == 4096 && shape.cols == 12288))) {
            for (int64_t layers : DEEP_LLAMA_GRAPH_LAYERS) {
                std::string q4_graph_l_name =
                    std::string("ggml_decode_q4_k_llama_graph_l")
                    + std::to_string(layers)
                    + "_"
                    + shape.suffix;
                if (run_llama_graph_probe(
                        backend,
                        GGML_TYPE_Q4_K,
                        q4_graph_l_name.c_str(),
                        "q4_k",
                        shape.rows,
                        shape.rows,
                        shape.cols,
                        layers,
                        0,
                        0,
                        result)) {
                    results.push_back(result);
                }
            }
        }
        if ((shape.rows != 768 || shape.cols != 2048) &&
            (shape.rows != 1024 || shape.cols != 4096) &&
            (shape.rows != 4096 || shape.cols != 12288)) {
            continue;
        }
        std::string q8_graph_name = std::string("ggml_decode_q8_0_llama_graph_") + shape.suffix;
        if (run_llama_graph_probe(
                backend,
                GGML_TYPE_Q8_0,
                q8_graph_name.c_str(),
                "q8_0",
                shape.rows,
                shape.rows,
                shape.cols,
                1,
                0,
                0,
                result)) {
            results.push_back(result);
        }
        std::string q6_graph_name = std::string("ggml_decode_q6_k_llama_graph_") + shape.suffix;
        if (run_llama_graph_probe(
                backend,
                GGML_TYPE_Q6_K,
                q6_graph_name.c_str(),
                "q6_k",
                shape.rows,
                shape.rows,
                shape.cols,
                1,
                0,
                0,
                result)) {
            results.push_back(result);
        }
    }
    for (const ProbeShape & shape : LLAMA_GQA_GRAPH_SHAPES) {
        constexpr int64_t kv_width = 1024;
        if (shape.rows <= kv_width) {
            continue;
        }
        std::string q4_graph_name = std::string("ggml_decode_q4_k_llama_graph_gqa_") + shape.suffix;
        if (run_llama_graph_probe(
                backend,
                GGML_TYPE_Q4_K,
                q4_graph_name.c_str(),
                "q4_k",
                shape.rows,
                kv_width,
                shape.cols,
                1,
                0,
                0,
                result)) {
            results.push_back(result);
        }
        if (deep_probes && shape.rows == 2560 && shape.cols == 9728) {
            for (int64_t layers : DEEP_LLAMA_GRAPH_LAYERS) {
                std::string q4_graph_l_name =
                    std::string("ggml_decode_q4_k_llama_graph_l")
                    + std::to_string(layers)
                    + "_gqa_"
                    + shape.suffix;
                if (run_llama_graph_probe(
                        backend,
                        GGML_TYPE_Q4_K,
                        q4_graph_l_name.c_str(),
                        "q4_k",
                        shape.rows,
                        kv_width,
                        shape.cols,
                        layers,
                        0,
                        0,
                        result)) {
                    results.push_back(result);
                }
            }
        }
    }
    if (run_moe_mul_mat_id_probe(
            backend,
            GGML_TYPE_Q4_K,
            "ggml_decode_moe_mul_mat_id_q4_k_128x8_768x2048",
            "q4_k",
            result)) {
        results.push_back(result);
    }
    if (run_moe_mul_mat_id_probe(
            backend,
            GGML_TYPE_Q6_K,
            "ggml_decode_moe_mul_mat_id_q6_k_128x8_768x2048",
            "q6_k",
            result)) {
        results.push_back(result);
    }
    if (run_moe_graph_probe(
            backend,
            GGML_TYPE_Q4_K,
            "ggml_decode_moe_graph_q4_k_128x8_768x2048",
            "q4_k",
            128,
            8,
            768,
            2048,
            1,
            result)) {
        results.push_back(result);
    }
    if (run_moe_graph_probe(
            backend,
            GGML_TYPE_Q6_K,
            "ggml_decode_moe_graph_q6_k_128x8_768x2048",
            "q6_k",
            128,
            8,
            768,
            2048,
            1,
            result)) {
        results.push_back(result);
    }
    if (run_logits_sync_probe(
            backend,
            "ggml_decode_logits_sync_vocab131072",
            131072,
            result)) {
        results.push_back(result);
    }

    ggml_backend_free(backend);

    if (results.empty()) {
        set_error(error_out, "GGML decode probe did not produce supported matvec results");
        return nullptr;
    }
    return copy_c_string(results_json(results));
}

extern "C" char * mesh_llm_gpu_bench_ggml_moe_graph_probe_json(
    int backend_kind,
    int tensor_type_kind,
    int64_t expert_count,
    int64_t experts_used,
    int64_t expert_width,
    int64_t hidden,
    int64_t repeat_layers,
    char ** error_out) {
    if (error_out != nullptr) {
        *error_out = nullptr;
    }
    enum ggml_type type = probe_tensor_type(tensor_type_kind);
    if (type == GGML_TYPE_COUNT) {
        set_error(error_out, "unsupported MoE probe tensor type");
        return nullptr;
    }

    ggml_backend_t backend = init_backend(backend_kind);
    if (backend == nullptr) {
        set_error(error_out, "GGML decode probe backend is not available");
        return nullptr;
    }

    std::ostringstream name;
    name << "ggml_decode_moe_graph_"
         << "l"
         << std::max<int64_t>(1, repeat_layers)
         << "_"
         << probe_tensor_type_name(tensor_type_kind)
         << "_"
         << expert_count
         << "x"
         << experts_used
         << "_"
         << expert_width
         << "x"
         << hidden;

    ProbeResult result{};
    std::vector<ProbeResult> results;
    if (run_moe_graph_probe(
            backend,
            type,
            name.str().c_str(),
            probe_tensor_type_name(tensor_type_kind),
            expert_count,
            experts_used,
            expert_width,
            hidden,
            repeat_layers,
            result)) {
        results.push_back(result);
    }
    ggml_backend_free(backend);

    if (results.empty()) {
        set_error(error_out, "GGML MoE graph probe did not produce a supported result");
        return nullptr;
    }
    return copy_c_string(results_json(results));
}

extern "C" char * mesh_llm_gpu_bench_ggml_moe_block_graph_probe_json(
    int backend_kind,
    int tensor_type_kind,
    int64_t expert_count,
    int64_t experts_used,
    int64_t expert_width,
    int64_t hidden,
    int64_t kv_width,
    int64_t repeat_layers,
    char ** error_out) {
    if (error_out != nullptr) {
        *error_out = nullptr;
    }
    enum ggml_type type = probe_tensor_type(tensor_type_kind);
    if (type == GGML_TYPE_COUNT) {
        set_error(error_out, "unsupported MoE block probe tensor type");
        return nullptr;
    }

    ggml_backend_t backend = init_backend(backend_kind);
    if (backend == nullptr) {
        set_error(error_out, "GGML decode probe backend is not available");
        return nullptr;
    }

    std::ostringstream name;
    name << "ggml_decode_moe_block_graph_"
         << "l"
         << std::max<int64_t>(1, repeat_layers)
         << "_"
         << probe_tensor_type_name(tensor_type_kind)
         << "_"
         << expert_count
         << "x"
         << experts_used
         << "_"
         << expert_width
         << "x"
         << hidden;
    if (kv_width > 0 && kv_width < hidden) {
        name << "_kv" << kv_width;
    }

    ProbeResult result{};
    std::vector<ProbeResult> results;
    if (run_moe_block_graph_probe(
            backend,
            type,
            name.str().c_str(),
            probe_tensor_type_name(tensor_type_kind),
            expert_count,
            experts_used,
            expert_width,
            hidden,
            kv_width,
            repeat_layers,
            false,
            1,
            result)) {
        results.push_back(result);
    }
    ggml_backend_free(backend);

    if (results.empty()) {
        set_error(error_out, "GGML MoE block graph probe did not produce a supported result");
        return nullptr;
    }
    return copy_c_string(results_json(results));
}

extern "C" char * mesh_llm_gpu_bench_ggml_moe_block_decode_submission_probe_json(
    int backend_kind,
    int tensor_type_kind,
    int64_t expert_count,
    int64_t experts_used,
    int64_t expert_width,
    int64_t hidden,
    int64_t kv_width,
    int64_t repeat_layers,
    int64_t context_tokens,
    char ** error_out) {
    if (error_out != nullptr) {
        *error_out = nullptr;
    }
    enum ggml_type type = probe_tensor_type(tensor_type_kind);
    if (type == GGML_TYPE_COUNT) {
        set_error(error_out, "unsupported MoE block submission probe tensor type");
        return nullptr;
    }

    ggml_backend_t backend = init_backend(backend_kind);
    if (backend == nullptr) {
        set_error(error_out, "GGML decode probe backend is not available");
        return nullptr;
    }

    std::ostringstream name;
    name << "ggml_decode_moe_block_submission_"
         << "l"
         << std::max<int64_t>(1, repeat_layers)
         << "_"
         << probe_tensor_type_name(tensor_type_kind)
         << "_"
         << expert_count
         << "x"
         << experts_used
         << "_"
         << expert_width
         << "x"
         << hidden
         << "_ctx"
         << std::max<int64_t>(1, context_tokens);
    if (kv_width > 0 && kv_width < hidden) {
        name << "_kv" << kv_width;
    }

    ProbeResult result{};
    std::vector<ProbeResult> results;
    if (run_moe_block_graph_probe(
            backend,
            type,
            name.str().c_str(),
            probe_tensor_type_name(tensor_type_kind),
            expert_count,
            experts_used,
            expert_width,
            hidden,
            kv_width,
            repeat_layers,
            true,
            context_tokens,
            result)) {
        results.push_back(result);
    }
    ggml_backend_free(backend);

    if (results.empty()) {
        set_error(error_out, "GGML MoE block submission probe did not produce a supported result");
        return nullptr;
    }
    return copy_c_string(results_json(results));
}

extern "C" char * mesh_llm_gpu_bench_ggml_dense_graph_probe_json(
    int backend_kind,
    int tensor_type_kind,
    int64_t hidden,
    int64_t kv_width,
    int64_t ffn,
    int64_t repeat_layers,
    int graph_features,
    int64_t norm_head_width,
    char ** error_out) {
    if (error_out != nullptr) {
        *error_out = nullptr;
    }
    enum ggml_type type = probe_tensor_type(tensor_type_kind);
    if (type == GGML_TYPE_COUNT) {
        set_error(error_out, "unsupported dense graph probe tensor type");
        return nullptr;
    }
    if (hidden <= 0 || kv_width <= 0 || ffn <= 0) {
        set_error(error_out, "dense graph probe dimensions must be positive");
        return nullptr;
    }

    ggml_backend_t backend = init_backend(backend_kind);
    if (backend == nullptr) {
        set_error(error_out, "GGML decode probe backend is not available");
        return nullptr;
    }

    std::ostringstream name;
    name << "ggml_decode_"
         << probe_tensor_type_name(tensor_type_kind)
         << "_llama_graph";
    if (repeat_layers > 1) {
        name << "_l" << repeat_layers;
    }
    if ((graph_features & GRAPH_FEATURE_ATTENTION_Q_NORM) != 0 &&
        (graph_features & GRAPH_FEATURE_ATTENTION_K_NORM) != 0) {
        name << "_qknorm";
    } else if ((graph_features & GRAPH_FEATURE_ATTENTION_Q_NORM) != 0) {
        name << "_qnorm";
    } else if ((graph_features & GRAPH_FEATURE_ATTENTION_K_NORM) != 0) {
        name << "_knorm";
    }
    if ((graph_features & GRAPH_FEATURE_ATTENTION_POST_NORM) != 0 ||
        (graph_features & GRAPH_FEATURE_FFN_POST_NORM) != 0) {
        name << "_postnorm";
    }
    if (kv_width < hidden) {
        name << "_gqa_" << hidden << "_kv" << kv_width << "_" << ffn;
    } else {
        name << "_" << hidden << "_" << ffn;
    }
    ProbeResult result{};
    std::vector<ProbeResult> results;
    if (run_llama_graph_probe(
            backend,
            type,
            name.str().c_str(),
            probe_tensor_type_name(tensor_type_kind),
            hidden,
            std::min(kv_width, hidden),
            ffn,
            repeat_layers,
            graph_features,
            norm_head_width,
            result)) {
        results.push_back(result);
    }
    ggml_backend_free(backend);

    if (results.empty()) {
        set_error(error_out, "GGML dense graph probe did not produce a supported result");
        return nullptr;
    }
    return copy_c_string(results_json(results));
}

extern "C" char * mesh_llm_gpu_bench_ggml_attention_runtime_probe_json(
    int backend_kind,
    int64_t head_dim,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t context_tokens,
    int64_t repeat_layers,
    char ** error_out) {
    if (error_out != nullptr) {
        *error_out = nullptr;
    }
    if (head_dim <= 0 || query_heads <= 0 || kv_heads <= 0 || context_tokens <= 0) {
        set_error(error_out, "attention runtime probe dimensions must be positive");
        return nullptr;
    }

    ggml_backend_t backend = init_backend(backend_kind);
    if (backend == nullptr) {
        set_error(error_out, "GGML attention runtime probe backend is not available");
        return nullptr;
    }

    std::ostringstream name;
    name << "ggml_decode_flash_attn_ext";
    if (repeat_layers > 1) {
        name << "_l" << repeat_layers;
    }
    name << "_h" << head_dim
         << "_qh" << query_heads
         << "_kvh" << kv_heads
         << "_ctx" << context_tokens;

    ProbeResult result{};
    std::vector<ProbeResult> results;
    if (run_attention_runtime_probe(
            backend,
            name.str().c_str(),
            head_dim,
            query_heads,
            kv_heads,
            context_tokens,
            repeat_layers,
            result)) {
        results.push_back(result);
    }
    ggml_backend_free(backend);

    if (results.empty()) {
        set_error(error_out, "GGML attention runtime probe did not produce a supported result");
        return nullptr;
    }
    return copy_c_string(results_json(results));
}

extern "C" char * mesh_llm_gpu_bench_ggml_logits_readback_probe_json(
    int backend_kind,
    int64_t vocab,
    char ** error_out) {
    if (error_out != nullptr) {
        *error_out = nullptr;
    }
    if (vocab <= 0) {
        set_error(error_out, "logits readback probe vocabulary size must be positive");
        return nullptr;
    }

    ggml_backend_t backend = init_backend(backend_kind);
    if (backend == nullptr) {
        set_error(error_out, "GGML logits readback probe backend is not available");
        return nullptr;
    }

    std::ostringstream name;
    name << "ggml_decode_logits_readback_vocab" << vocab;

    ProbeResult result{};
    std::vector<ProbeResult> results;
    if (run_logits_readback_probe(backend, name.str().c_str(), vocab, result)) {
        results.push_back(result);
    }
    ggml_backend_free(backend);

    if (results.empty()) {
        set_error(error_out, "GGML logits readback probe did not produce a supported result");
        return nullptr;
    }
    return copy_c_string(results_json(results));
}

extern "C" char * mesh_llm_gpu_bench_ggml_logits_sync_probe_json(
    int backend_kind,
    int64_t vocab,
    char ** error_out) {
    if (error_out != nullptr) {
        *error_out = nullptr;
    }
    if (vocab <= 0) {
        set_error(error_out, "logits sync probe vocabulary size must be positive");
        return nullptr;
    }

    ggml_backend_t backend = init_backend(backend_kind);
    if (backend == nullptr) {
        set_error(error_out, "GGML logits sync probe backend is not available");
        return nullptr;
    }

    std::ostringstream name;
    name << "ggml_decode_logits_sync_vocab" << vocab;

    ProbeResult result{};
    std::vector<ProbeResult> results;
    if (run_logits_sync_probe(backend, name.str().c_str(), vocab, result)) {
        results.push_back(result);
    }
    ggml_backend_free(backend);

    if (results.empty()) {
        set_error(error_out, "GGML logits sync probe did not produce a supported result");
        return nullptr;
    }
    return copy_c_string(results_json(results));
}

extern "C" char * mesh_llm_gpu_bench_ggml_logits_output_handoff_probe_json(
    int backend_kind,
    int64_t vocab,
    char ** error_out) {
    if (error_out != nullptr) {
        *error_out = nullptr;
    }
    if (vocab <= 0) {
        set_error(error_out, "logits output handoff probe vocabulary size must be positive");
        return nullptr;
    }

    ggml_backend_t backend = init_backend(backend_kind);
    if (backend == nullptr) {
        set_error(error_out, "GGML logits output handoff probe backend is not available");
        return nullptr;
    }

    std::ostringstream name;
    name << "ggml_decode_logits_output_handoff_vocab" << vocab;

    ProbeResult result{};
    std::vector<ProbeResult> results;
    if (run_logits_output_handoff_probe(backend, name.str().c_str(), vocab, result)) {
        results.push_back(result);
    }
    ggml_backend_free(backend);

    if (results.empty()) {
        set_error(error_out, "GGML logits output handoff probe did not produce a supported result");
        return nullptr;
    }
    return copy_c_string(results_json(results));
}

extern "C" char * mesh_llm_gpu_bench_ggml_dense_sampled_token_probe_json(
    int backend_kind,
    int tensor_type_kind,
    int64_t hidden,
    int64_t kv_width,
    int64_t ffn,
    int64_t vocab,
    int64_t repeat_layers,
    int graph_features,
    int64_t norm_head_width,
    char ** error_out) {
    if (error_out != nullptr) {
        *error_out = nullptr;
    }
    enum ggml_type type = probe_tensor_type(tensor_type_kind);
    if (type == GGML_TYPE_COUNT) {
        set_error(error_out, "unsupported tensor type for dense sampled-token probe");
        return nullptr;
    }
    if (hidden <= 0 || kv_width <= 0 || ffn <= 0 || vocab <= 0) {
        set_error(error_out, "dense sampled-token probe dimensions must be positive");
        return nullptr;
    }

    ggml_backend_t backend = init_backend(backend_kind);
    if (backend == nullptr) {
        set_error(error_out, "GGML dense sampled-token probe backend is not available");
        return nullptr;
    }

    std::ostringstream name;
    name << "ggml_decode_"
         << probe_tensor_type_name(tensor_type_kind)
         << "_sampled_token";
    if (repeat_layers > 1) {
        name << "_l" << repeat_layers;
    }
    if ((graph_features & GRAPH_FEATURE_ATTENTION_Q_NORM) != 0 &&
        (graph_features & GRAPH_FEATURE_ATTENTION_K_NORM) != 0) {
        name << "_qknorm";
    } else if ((graph_features & GRAPH_FEATURE_ATTENTION_Q_NORM) != 0) {
        name << "_qnorm";
    } else if ((graph_features & GRAPH_FEATURE_ATTENTION_K_NORM) != 0) {
        name << "_knorm";
    }
    if ((graph_features & GRAPH_FEATURE_ATTENTION_POST_NORM) != 0 ||
        (graph_features & GRAPH_FEATURE_FFN_POST_NORM) != 0) {
        name << "_postnorm";
    }
    if (kv_width < hidden) {
        name << "_gqa_" << hidden << "_kv" << kv_width << "_" << ffn;
    } else {
        name << "_" << hidden << "_" << ffn;
    }
    name << "_vocab" << vocab;

    ProbeResult result{};
    std::vector<ProbeResult> results;
    if (run_dense_sampled_token_probe(
            backend,
            type,
            name.str().c_str(),
            probe_tensor_type_name(tensor_type_kind),
            hidden,
            std::min(kv_width, hidden),
            ffn,
            vocab,
            repeat_layers,
            graph_features,
            norm_head_width,
            result)) {
        results.push_back(result);
    }
    ggml_backend_free(backend);

    if (results.empty()) {
        set_error(error_out, "GGML dense sampled-token probe did not produce a supported result");
        return nullptr;
    }
    return copy_c_string(results_json(results));
}

extern "C" char * mesh_llm_gpu_bench_ggml_dense_full_token_probe_json(
    int backend_kind,
    int block_tensor_type_kind,
    int output_tensor_type_kind,
    int64_t hidden,
    int64_t kv_width,
    int64_t ffn,
    int64_t vocab,
    int64_t repeat_layers,
    int graph_features,
    int64_t norm_head_width,
    int64_t head_dim,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t context_tokens,
    int64_t active_context_tokens,
    char ** error_out) {
    if (error_out != nullptr) {
        *error_out = nullptr;
    }
    enum ggml_type block_type = probe_tensor_type(block_tensor_type_kind);
    enum ggml_type output_type = probe_tensor_type(output_tensor_type_kind);
    if (block_type == GGML_TYPE_COUNT || output_type == GGML_TYPE_COUNT) {
        set_error(error_out, "unsupported tensor type for dense full-token probe");
        return nullptr;
    }
    if (hidden <= 0 || kv_width <= 0 || ffn <= 0 || vocab <= 0 || head_dim <= 0
        || query_heads <= 0 || kv_heads <= 0 || context_tokens <= 0 || active_context_tokens <= 0) {
        set_error(error_out, "dense full-token probe dimensions must be positive");
        return nullptr;
    }
    active_context_tokens = std::min(active_context_tokens, context_tokens);

    ggml_backend_t backend = init_backend(backend_kind);
    if (backend == nullptr) {
        set_error(error_out, "GGML dense full-token probe backend is not available");
        return nullptr;
    }

    std::ostringstream name;
    name << "ggml_decode_"
         << probe_tensor_type_name(block_tensor_type_kind)
         << "_full_token";
    if (output_tensor_type_kind != block_tensor_type_kind) {
        name << "_out" << probe_tensor_type_name(output_tensor_type_kind);
    }
    if (repeat_layers > 1) {
        name << "_l" << repeat_layers;
    }
    if ((graph_features & GRAPH_FEATURE_ATTENTION_Q_NORM) != 0 &&
        (graph_features & GRAPH_FEATURE_ATTENTION_K_NORM) != 0) {
        name << "_qknorm";
    } else if ((graph_features & GRAPH_FEATURE_ATTENTION_Q_NORM) != 0) {
        name << "_qnorm";
    } else if ((graph_features & GRAPH_FEATURE_ATTENTION_K_NORM) != 0) {
        name << "_knorm";
    }
    if ((graph_features & GRAPH_FEATURE_ATTENTION_POST_NORM) != 0 ||
        (graph_features & GRAPH_FEATURE_FFN_POST_NORM) != 0) {
        name << "_postnorm";
    }
    if (kv_width < hidden) {
        name << "_gqa_" << hidden << "_kv" << kv_width << "_" << ffn;
    } else {
        name << "_" << hidden << "_" << ffn;
    }
    name << "_vocab" << vocab
         << "_ctx" << context_tokens
         << "_nkv" << active_context_tokens
         << "_h" << head_dim
         << "_qh" << query_heads
         << "_kvh" << kv_heads;

    ProbeResult result{};
    std::vector<ProbeResult> results;
    if (run_dense_full_token_probe(
            backend,
            block_type,
            output_type,
            name.str().c_str(),
            probe_tensor_type_name(block_tensor_type_kind),
            hidden,
            std::min(kv_width, hidden),
            ffn,
            vocab,
            repeat_layers,
            graph_features,
            norm_head_width,
            head_dim,
            query_heads,
            kv_heads,
            context_tokens,
            active_context_tokens,
            false,
            false,
            false,
            false,
            result)) {
        results.push_back(result);
    }
    ggml_backend_free(backend);

    if (results.empty()) {
        set_error(error_out, "GGML dense full-token probe did not produce a supported result");
        return nullptr;
    }
    return copy_c_string(results_json(results));
}

extern "C" char * mesh_llm_gpu_bench_ggml_dense_full_token_handoff_probe_json(
    int backend_kind,
    int block_tensor_type_kind,
    int output_tensor_type_kind,
    int64_t hidden,
    int64_t kv_width,
    int64_t ffn,
    int64_t vocab,
    int64_t repeat_layers,
    int graph_features,
    int64_t norm_head_width,
    int64_t head_dim,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t context_tokens,
    int64_t active_context_tokens,
    char ** error_out) {
    if (error_out != nullptr) {
        *error_out = nullptr;
    }
    enum ggml_type block_type = probe_tensor_type(block_tensor_type_kind);
    enum ggml_type output_type = probe_tensor_type(output_tensor_type_kind);
    if (block_type == GGML_TYPE_COUNT || output_type == GGML_TYPE_COUNT) {
        set_error(error_out, "unsupported tensor type for dense full-token handoff probe");
        return nullptr;
    }
    if (hidden <= 0 || kv_width <= 0 || ffn <= 0 || vocab <= 0 || head_dim <= 0
        || query_heads <= 0 || kv_heads <= 0 || context_tokens <= 0 || active_context_tokens <= 0) {
        set_error(error_out, "dense full-token handoff probe dimensions must be positive");
        return nullptr;
    }
    active_context_tokens = std::min(active_context_tokens, context_tokens);

    ggml_backend_t backend = init_backend(backend_kind);
    if (backend == nullptr) {
        set_error(error_out, "GGML dense full-token handoff probe backend is not available");
        return nullptr;
    }

    std::ostringstream name;
    name << "ggml_decode_"
         << probe_tensor_type_name(block_tensor_type_kind)
         << "_full_token_handoff";
    if (output_tensor_type_kind != block_tensor_type_kind) {
        name << "_out" << probe_tensor_type_name(output_tensor_type_kind);
    }
    if (repeat_layers > 1) {
        name << "_l" << repeat_layers;
    }
    if ((graph_features & GRAPH_FEATURE_ATTENTION_Q_NORM) != 0 &&
        (graph_features & GRAPH_FEATURE_ATTENTION_K_NORM) != 0) {
        name << "_qknorm";
    } else if ((graph_features & GRAPH_FEATURE_ATTENTION_Q_NORM) != 0) {
        name << "_qnorm";
    } else if ((graph_features & GRAPH_FEATURE_ATTENTION_K_NORM) != 0) {
        name << "_knorm";
    }
    if ((graph_features & GRAPH_FEATURE_ATTENTION_POST_NORM) != 0 ||
        (graph_features & GRAPH_FEATURE_FFN_POST_NORM) != 0) {
        name << "_postnorm";
    }
    if (kv_width < hidden) {
        name << "_gqa_" << hidden << "_kv" << kv_width << "_" << ffn;
    } else {
        name << "_" << hidden << "_" << ffn;
    }
    name << "_vocab" << vocab
         << "_ctx" << context_tokens
         << "_nkv" << active_context_tokens
         << "_h" << head_dim
         << "_qh" << query_heads
         << "_kvh" << kv_heads;

    ProbeResult result{};
    std::vector<ProbeResult> results;
    if (run_dense_full_token_probe(
            backend,
            block_type,
            output_type,
            name.str().c_str(),
            probe_tensor_type_name(block_tensor_type_kind),
            hidden,
            std::min(kv_width, hidden),
            ffn,
            vocab,
            repeat_layers,
            graph_features,
            norm_head_width,
            head_dim,
            query_heads,
            kv_heads,
            context_tokens,
            active_context_tokens,
            true,
            false,
            false,
            false,
            result)) {
        results.push_back(result);
    }
    ggml_backend_free(backend);

    if (results.empty()) {
        set_error(error_out, "GGML dense full-token handoff probe did not produce a supported result");
        return nullptr;
    }
    return copy_c_string(results_json(results));
}

extern "C" char * mesh_llm_gpu_bench_ggml_dense_decode_submission_probe_json(
    int backend_kind,
    int block_tensor_type_kind,
    int output_tensor_type_kind,
    int64_t hidden,
    int64_t kv_width,
    int64_t ffn,
    int64_t vocab,
    int64_t repeat_layers,
    int graph_features,
    int64_t norm_head_width,
    int64_t head_dim,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t context_tokens,
    int64_t active_context_tokens,
    char ** error_out) {
    if (error_out != nullptr) {
        *error_out = nullptr;
    }
    enum ggml_type block_type = probe_tensor_type(block_tensor_type_kind);
    enum ggml_type output_type = probe_tensor_type(output_tensor_type_kind);
    if (block_type == GGML_TYPE_COUNT || output_type == GGML_TYPE_COUNT) {
        set_error(error_out, "unsupported tensor type for dense decode submission probe");
        return nullptr;
    }
    if (hidden <= 0 || kv_width <= 0 || ffn <= 0 || vocab <= 0 || head_dim <= 0
        || query_heads <= 0 || kv_heads <= 0 || context_tokens <= 0 || active_context_tokens <= 0) {
        set_error(error_out, "dense decode submission probe dimensions must be positive");
        return nullptr;
    }
    active_context_tokens = std::min(active_context_tokens, context_tokens);

    ggml_backend_t backend = init_backend(backend_kind);
    if (backend == nullptr) {
        set_error(error_out, "GGML dense decode submission probe backend is not available");
        return nullptr;
    }

    std::ostringstream name;
    name << "ggml_decode_"
         << probe_tensor_type_name(block_tensor_type_kind)
         << "_submission";
    if (output_tensor_type_kind != block_tensor_type_kind) {
        name << "_out" << probe_tensor_type_name(output_tensor_type_kind);
    }
    if (repeat_layers > 1) {
        name << "_l" << repeat_layers;
    }
    if ((graph_features & GRAPH_FEATURE_ATTENTION_Q_NORM) != 0 &&
        (graph_features & GRAPH_FEATURE_ATTENTION_K_NORM) != 0) {
        name << "_qknorm";
    } else if ((graph_features & GRAPH_FEATURE_ATTENTION_Q_NORM) != 0) {
        name << "_qnorm";
    } else if ((graph_features & GRAPH_FEATURE_ATTENTION_K_NORM) != 0) {
        name << "_knorm";
    }
    if ((graph_features & GRAPH_FEATURE_ATTENTION_POST_NORM) != 0 ||
        (graph_features & GRAPH_FEATURE_FFN_POST_NORM) != 0) {
        name << "_postnorm";
    }
    if (kv_width < hidden) {
        name << "_gqa_" << hidden << "_kv" << kv_width << "_" << ffn;
    } else {
        name << "_" << hidden << "_" << ffn;
    }
    name << "_vocab" << vocab
         << "_ctx" << context_tokens
         << "_nkv" << active_context_tokens
         << "_h" << head_dim
         << "_qh" << query_heads
         << "_kvh" << kv_heads;

    ProbeResult result{};
    std::vector<ProbeResult> results;
    if (run_dense_full_token_probe(
            backend,
            block_type,
            output_type,
            name.str().c_str(),
            probe_tensor_type_name(block_tensor_type_kind),
            hidden,
            std::min(kv_width, hidden),
            ffn,
            vocab,
            repeat_layers,
            graph_features,
            norm_head_width,
            head_dim,
            query_heads,
            kv_heads,
            context_tokens,
            active_context_tokens,
            false,
            true,
            false,
            false,
            result)) {
        results.push_back(result);
    }
    ggml_backend_free(backend);

    if (results.empty()) {
        set_error(error_out, "GGML dense decode submission probe did not produce a supported result");
        return nullptr;
    }
    return copy_c_string(results_json(results));
}

extern "C" char * mesh_llm_gpu_bench_ggml_dense_source_sampled_token_probe_json(
    int backend_kind,
    int block_tensor_type_kind,
    int output_tensor_type_kind,
    int64_t hidden,
    int64_t kv_width,
    int64_t ffn,
    int64_t vocab,
    int64_t repeat_layers,
    int graph_features,
    int64_t norm_head_width,
    int64_t head_dim,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t context_tokens,
    int64_t active_context_tokens,
    char ** error_out) {
    if (error_out != nullptr) {
        *error_out = nullptr;
    }
    enum ggml_type block_type = probe_tensor_type(block_tensor_type_kind);
    enum ggml_type output_type = probe_tensor_type(output_tensor_type_kind);
    if (block_type == GGML_TYPE_COUNT || output_type == GGML_TYPE_COUNT) {
        set_error(error_out, "unsupported tensor type for dense source sampled-token probe");
        return nullptr;
    }
    if (hidden <= 0 || kv_width <= 0 || ffn <= 0 || vocab <= 0 || head_dim <= 0
        || query_heads <= 0 || kv_heads <= 0 || context_tokens <= 0 || active_context_tokens <= 0) {
        set_error(error_out, "dense source sampled-token probe dimensions must be positive");
        return nullptr;
    }
    active_context_tokens = std::min(active_context_tokens, context_tokens);

    ggml_backend_t backend = init_backend(backend_kind);
    if (backend == nullptr) {
        set_error(error_out, "GGML dense source sampled-token probe backend is not available");
        return nullptr;
    }

    std::ostringstream name;
    name << "ggml_decode_"
         << probe_tensor_type_name(block_tensor_type_kind)
         << "_full_token_source_sampled";
    if (output_tensor_type_kind != block_tensor_type_kind) {
        name << "_out" << probe_tensor_type_name(output_tensor_type_kind);
    }
    if (repeat_layers > 1) {
        name << "_l" << repeat_layers;
    }
    if ((graph_features & GRAPH_FEATURE_ATTENTION_Q_NORM) != 0 &&
        (graph_features & GRAPH_FEATURE_ATTENTION_K_NORM) != 0) {
        name << "_qknorm";
    } else if ((graph_features & GRAPH_FEATURE_ATTENTION_Q_NORM) != 0) {
        name << "_qnorm";
    } else if ((graph_features & GRAPH_FEATURE_ATTENTION_K_NORM) != 0) {
        name << "_knorm";
    }
    if ((graph_features & GRAPH_FEATURE_ATTENTION_POST_NORM) != 0 ||
        (graph_features & GRAPH_FEATURE_FFN_POST_NORM) != 0) {
        name << "_postnorm";
    }
    if (kv_width < hidden) {
        name << "_gqa_" << hidden << "_kv" << kv_width << "_" << ffn;
    } else {
        name << "_" << hidden << "_" << ffn;
    }
    name << "_vocab" << vocab
         << "_ctx" << context_tokens
         << "_nkv" << active_context_tokens
         << "_h" << head_dim
         << "_qh" << query_heads
         << "_kvh" << kv_heads;

    ProbeResult result{};
    std::vector<ProbeResult> results;
    if (run_dense_full_token_probe(
            backend,
            block_type,
            output_type,
            name.str().c_str(),
            probe_tensor_type_name(block_tensor_type_kind),
            hidden,
            std::min(kv_width, hidden),
            ffn,
            vocab,
            repeat_layers,
            graph_features,
            norm_head_width,
            head_dim,
            query_heads,
            kv_heads,
            context_tokens,
            active_context_tokens,
            false,
            false,
            true,
            false,
            result)) {
        results.push_back(result);
    }
    ProbeResult source_input_result{};
    std::string source_input_name = name.str();
    const size_t source_sampled_marker = source_input_name.find("_full_token_source_sampled");
    if (source_sampled_marker != std::string::npos) {
        source_input_name.replace(source_sampled_marker, 26, "_source_input");
    }
    if (run_dense_full_token_probe(
            backend,
            block_type,
            output_type,
            source_input_name.c_str(),
            probe_tensor_type_name(block_tensor_type_kind),
            hidden,
            std::min(kv_width, hidden),
            ffn,
            vocab,
            repeat_layers,
            graph_features,
            norm_head_width,
            head_dim,
            query_heads,
            kv_heads,
            context_tokens,
            active_context_tokens,
            false,
            false,
            false,
            true,
            source_input_result)) {
        results.push_back(source_input_result);
    }
    ggml_backend_free(backend);

    if (results.empty()) {
        set_error(error_out, "GGML dense source sampled-token probe did not produce a supported result");
        return nullptr;
    }
    return copy_c_string(results_json(results));
}

extern "C" char * mesh_llm_gpu_bench_ggml_linear_attention_graph_probe_json(
    int backend_kind,
    int tensor_type_kind,
    int64_t hidden,
    int64_t qkv_width,
    int64_t gate_width,
    int64_t state_width,
    int64_t output_input_width,
    int64_t ffn,
    int64_t recurrent_layers,
    int64_t full_attention_layers,
    int64_t kv_width,
    int graph_features,
    int64_t norm_head_width,
    char ** error_out) {
    if (error_out != nullptr) {
        *error_out = nullptr;
    }
    enum ggml_type type = probe_tensor_type(tensor_type_kind);
    if (type == GGML_TYPE_COUNT) {
        set_error(error_out, "unsupported linear attention graph probe tensor type");
        return nullptr;
    }
    if (hidden <= 0 || qkv_width <= 0 || gate_width <= 0 || state_width <= 0 ||
        output_input_width <= 0 || ffn <= 0 || recurrent_layers <= 0 || kv_width <= 0) {
        set_error(error_out, "linear attention graph probe dimensions must be positive");
        return nullptr;
    }
    if (output_input_width > qkv_width) {
        set_error(error_out, "linear attention output input width cannot exceed qkv width");
        return nullptr;
    }

    ggml_backend_t backend = init_backend(backend_kind);
    if (backend == nullptr) {
        set_error(error_out, "GGML decode probe backend is not available");
        return nullptr;
    }

    std::ostringstream name;
    name << "ggml_decode_"
         << probe_tensor_type_name(tensor_type_kind)
         << "_linear_attn_graph"
         << "_r" << recurrent_layers
         << "_f" << std::max<int64_t>(0, full_attention_layers);
    if ((graph_features & GRAPH_FEATURE_ATTENTION_Q_NORM) != 0 &&
        (graph_features & GRAPH_FEATURE_ATTENTION_K_NORM) != 0) {
        name << "_qknorm";
    } else if ((graph_features & GRAPH_FEATURE_ATTENTION_Q_NORM) != 0) {
        name << "_qnorm";
    } else if ((graph_features & GRAPH_FEATURE_ATTENTION_K_NORM) != 0) {
        name << "_knorm";
    }
    if ((graph_features & GRAPH_FEATURE_ATTENTION_POST_NORM) != 0 ||
        (graph_features & GRAPH_FEATURE_FFN_POST_NORM) != 0) {
        name << "_postnorm";
    }
    name << "_h" << hidden
         << "_qkv" << qkv_width
         << "_gate" << gate_width
         << "_state" << state_width
         << "_out" << output_input_width
         << "_kv" << kv_width
         << "_ffn" << ffn;

    ProbeResult result{};
    std::vector<ProbeResult> results;
    if (run_linear_attention_graph_probe(
            backend,
            type,
            name.str().c_str(),
            probe_tensor_type_name(tensor_type_kind),
            hidden,
            qkv_width,
            gate_width,
            state_width,
            output_input_width,
            ffn,
            recurrent_layers,
            full_attention_layers,
            kv_width,
            graph_features,
            norm_head_width,
            result)) {
        results.push_back(result);
    }
    ggml_backend_free(backend);

    if (results.empty()) {
        set_error(error_out, "GGML linear attention graph probe did not produce a supported result");
        return nullptr;
    }
    return copy_c_string(results_json(results));
}

extern "C" void mesh_llm_gpu_bench_ggml_decode_probe_free(void * ptr) {
    std::free(ptr);
}
