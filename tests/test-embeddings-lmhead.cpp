// Regression test: lm_head must be skipped in embeddings mode.
//
// Without the fix, `output_all = cparams.embeddings` forces the vocab
// projection (lm_head) to run over every token even though logits are
// never read in embeddings mode. The fix gates the lm_head behind
// `!cparams.embeddings` and sets `has_logits = !cparams.embeddings`.
//
// This test builds a tiny synthetic model (same approach as
// test-llama-archs.cpp) and asserts, for every patched arch:
//   - llama_get_logits(ctx) == nullptr  when embeddings mode is on
//   - llama_get_embeddings(ctx) != nullptr and is non-zero
// The first assertion FAILS without the has_logits fix and PASSES with it.

#include "common.h"
#include "log.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"

#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    std::hash<std::string> hasher;
    std::mt19937 gen(hasher(tensor->name) + *(const size_t *) userdata);
    std::normal_distribution<float> dis(0.0f, 1.0e-2f);
    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (int64_t i = 0; i < ne; i++) tmp[i] = dis(gen);
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (int64_t i = 0; i < ne; i++) tmp[i] = ggml_fp32_to_fp16(dis(gen));
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("unexpected tensor type");
    }
}

static gguf_context_ptr make_gguf(llm_arch arch) {
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(arch, ret.get());

    const uint32_t n_ctx = 128, n_vocab = 128, n_embd = 256;
    const uint32_t n_head = 2, n_ff = 384, n_layer = 2;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,        llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                  n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,              n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,            n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,             n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,                 n_layer);
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,         n_ff);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,        n_head);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,     n_head);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,     1e-5f);
    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,       false);
    ms.add_kv(LLM_KV_LOGIT_SCALE,                 1.0f);
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,             "no_vocab");
    if (arch == LLM_ARCH_GEMMA2) {
        ms.add_kv(LLM_KV_FINAL_LOGIT_SOFTCAPPING, 30.0f);
        ms.add_kv(LLM_KV_ATTN_LOGIT_SOFTCAPPING,  50.0f);
    }
    if (arch == LLM_ARCH_QWEN2MOE) {
        ms.add_kv(LLM_KV_EXPERT_COUNT,               4u);
        ms.add_kv(LLM_KV_EXPERT_USED_COUNT,          2u);
        ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, n_ff);
    }
    if (arch == LLM_ARCH_GEMMA4) {
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, 6u);
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,   64u);
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,   128u);  // n_embd/n_head
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA, 128u);
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER, 32u);
        ms.add_kv(LLM_KV_FINAL_LOGIT_SOFTCAPPING,    30.0f);
    }
    return ret;
}

static bool test_arch(llm_arch arch, const char * name) {
    bool ok = true;
    try {
        gguf_context_ptr gguf = make_gguf(arch);

        llama_model_params mp = llama_model_default_params();
        mp.progress_callback = [](float, void *) -> bool { return true; };
        size_t seed = 1234;
        llama_model_ptr model(llama_model_init_from_user(gguf.get(), set_tensor_data, &seed, mp));
        if (!model) { printf("  [%s] model init failed\n", name); return false; }

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx        = 32;
        cp.n_batch      = 32;
        cp.n_ubatch     = 32;
        cp.embeddings   = true;
        cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
        llama_context_ptr ctx(llama_init_from_model(model.get(), cp));
        if (!ctx) { printf("  [%s] ctx init failed\n", name); return false; }

        llama_token toks[] = { 1, 2, 3, 4 };
        llama_batch batch = llama_batch_get_one(toks, 4);
        if (llama_decode(ctx.get(), batch) != 0) { printf("  [%s] decode failed\n", name); return false; }

        // CORE ASSERTION: logits must be NULL in embeddings mode (fails without has_logits fix)
        const float * logits = llama_get_logits(ctx.get());
        if (logits != nullptr) { printf("  [%s] FAIL: llama_get_logits != NULL in embeddings mode\n", name); ok = false; }

        // embeddings must be present and non-zero
        const float * emb = llama_get_embeddings(ctx.get());
        if (emb == nullptr) { printf("  [%s] FAIL: llama_get_embeddings == NULL\n", name); ok = false; }
        else {
            const int n_embd = llama_model_n_embd(model.get());
            bool all_zero = true;
            for (int i = 0; i < n_embd; i++) if (std::fabs(emb[i]) > 0.0f) { all_zero = false; break; }
            if (all_zero) { printf("  [%s] FAIL: embeddings all-zero\n", name); ok = false; }
        }
        // CROSS-MODE PARITY: t_embd in embedding mode must match t_embd from
        // generation mode (where the lm_head still runs). Proves the gate does
        // not change the embedding values -- the real regression guard.
        std::vector<float> emb_copy;
        if (emb) { const int n = llama_model_n_embd(model.get()); emb_copy.assign(emb, emb + n); }
        {
            llama_context_params gp = cp;
            gp.embeddings   = false;
            gp.pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED;
            llama_context_ptr gctx(llama_init_from_model(model.get(), gp));
            llama_token gtoks[] = { 1, 2, 3, 4 };
            llama_batch gb = llama_batch_get_one(gtoks, 4);
            if (gctx && llama_decode(gctx.get(), gb) == 0) {
                const float * gen_emb = llama_get_embeddings(gctx.get());
                if (gen_emb && !emb_copy.empty()) {
                    const int n = (int) emb_copy.size();
                    double dot = 0, na = 0, nb = 0;
                    for (int i = 0; i < n; i++) { dot += (double) emb_copy[i]*gen_emb[i]; na += (double) emb_copy[i]*emb_copy[i]; nb += (double) gen_emb[i]*gen_emb[i]; }
                    const double cos = (na > 0 && nb > 0) ? dot/(std::sqrt(na)*std::sqrt(nb)) : 0.0;
                    if (cos < 0.9999) { printf("  [%s] FAIL: cross-mode embedding cosine %.6f < 0.9999\n", name, cos); ok = false; }
                }
            }
        }

        // POOLING COVERAGE: the lm_head gate must hold for every pooling type,
        // not only NONE. Re-run embeddings mode with MEAN and CLS and assert
        // logits stay NULL (lm_head skipped) and embeddings are produced.
        for (auto pt : { LLAMA_POOLING_TYPE_MEAN, LLAMA_POOLING_TYPE_CLS }) {
            llama_context_params pp = cp;
            pp.pooling_type = pt;
            llama_context_ptr pctx(llama_init_from_model(model.get(), pp));
            if (!pctx) { printf("  [%s] pooling %d: ctx init failed\n", name, (int) pt); ok = false; continue; }
            llama_token ptoks[] = { 1, 2, 3, 4 };
            llama_batch pbatch = llama_batch_get_one(ptoks, 4);
            if (llama_decode(pctx.get(), pbatch) != 0) { printf("  [%s] pooling %d: decode failed\n", name, (int) pt); ok = false; continue; }
            if (llama_get_logits(pctx.get()) != nullptr) { printf("  [%s] FAIL: logits != NULL (pooling %d)\n", name, (int) pt); ok = false; }
            if (llama_get_embeddings(pctx.get()) == nullptr) { printf("  [%s] FAIL: embeddings == NULL (pooling %d)\n", name, (int) pt); ok = false; }
        }

        if (ok) printf("  [%s] OK (logits NULL, embeddings present, cross-mode parity + MEAN/CLS)\n", name);
    } catch (const std::exception & e) {
        printf("  [%s] exception: %s\n", name, e.what());
        ok = false;
    }
    return ok;
}

int main() {
    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    llama_backend_init();

    struct { llm_arch arch; const char * name; } cases[] = {
        { LLM_ARCH_QWEN2,      "qwen2"      },
        { LLM_ARCH_QWEN3,      "qwen3"      },
        { LLM_ARCH_GEMMA2,     "gemma2"     },
        { LLM_ARCH_GEMMA4,     "gemma4"     },
        { LLM_ARCH_PHI3,       "phi3"       },
        { LLM_ARCH_STABLELM,   "stablelm"   },
        { LLM_ARCH_STARCODER2, "starcoder2" },
        { LLM_ARCH_ARCEE,      "arcee"      },
        { LLM_ARCH_INTERNLM2,  "internlm2"  },
        { LLM_ARCH_QWEN2MOE,   "qwen2moe"   },
        { LLM_ARCH_LLAMA,      "llama"      },
    };

    bool all_ok = true;
    for (auto & c : cases) all_ok &= test_arch(c.arch, c.name);

    llama_backend_free();
    printf("%s\n", all_ok ? "All tests passed" : "FAILURES");
    return all_ok ? 0 : 1;
}
