#ifndef COMMON_H
#define COMMON_H

#include <QString>
#include <QThread>
#include "llama/llama.h"

struct gpt_params{
    int32_t seed            = -1; // RNG seed
    int32_t n_threads       = QThread::idealThreadCount()/2;
    int32_t n_predict       = 128; // new tokens to predict
    int32_t repeat_last_n   = 64;  // last n tokens to penalize

    int32_t n_ctx           = 512;

    // sampling parameters
    int32_t top_k           = 40;
    float   top_p           = 0.95f;
    float   temp            = 0.80f;
    float   repeat_penalty  = 1.30f;

    int32_t n_batch         = 8; // batch size for prompt processing
    int32_t n_keep          = 0;

    QString model           = "models/lamma-7B/ggml-model.bin"; // model path

    bool memory_f16        = true;  // use f16 instead of f32 for memory kv
    bool interactive       = false; // interactive mode

    bool interactive_start = false; // wait for user input immediately
    bool embedding         = false; // get only sentence embedding
    bool instruct          = false; // instruction mode (used for Alpaca models)
    bool ignore_eos        = false; // do not stop generating after eos
    bool perplexity        = false; // compute perplexity over the prompt
    bool use_mlock         = false; // use mlock to keep model in memory
    bool mem_test          = false; // compute maximum memory usage
    bool verbose_prompt    = false; // print prompt tokens before generation
};
Q_DECLARE_METATYPE(gpt_params);


std::vector<llama_token> llama_tokenize(struct llama_context * ctx, const QString & text, bool add_bos);
#endif // COMMON_H
