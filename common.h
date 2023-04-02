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


typedef struct _keep_prompt_token{
    bool init(llama_context *ctx, const QString prompt);
    int32_t get_n_keep();
    std::vector<llama_token>& get_initial_token();
private:
    std::vector<llama_token> initial_token;
    int32_t n_keep        = 0;// number of tokens to keep when resetting context
private:
    llama_context *m_ctx = nullptr;
}keep_prompt_token_t;

typedef struct _instruction_info{
    void init(llama_context *ctx);
    int inject(const QString buffer, std::vector<llama_token> &output);
private:
    std::vector<llama_token> input_prefix; // instruction prefix
    std::vector<llama_token> input_suffix; // response prefix
private:
    llama_context *m_ctx = nullptr;
}instruction_info_t;

typedef struct _embedding_queue{
    void init(llama_context *ctx);
    void input_copy(const std::vector<llama_token> input);
    bool input_is_empty();
    int input_produce(instruction_info_t &instruction_info, const QString buffer);
    void input_consume();
    std::vector<llama_token>& get_embd_output();
    bool output_is_empty();
private:
    std::vector<llama_token> embd_input; // sentence embedding storage
    std::vector<llama_token> embd_output; // sentence embedding to process
    int32_t n_consumed = 0;
private:
    llama_context *m_ctx = nullptr;
}embedding_queue_t;

typedef struct _last_n_tokens{
    void init(llama_context *ctx);
    void push(llama_token id);
    std::vector<llama_token>& get_last_n_tokens();
private:
    std::vector<llama_token> last_n_tokens;
}last_n_tokens_t;

typedef struct _env_configs{
    void init(const gpt_params &params);
    int32_t n_predict       = 128; // new tokens to predict
    int32_t repeat_last_n   = 64;  // last n tokens to penalize

    // sampling parameters
    int32_t top_k           = 40;
    float   top_p           = 0.95f;
    float   temp            = 0.80f;
    float   repeat_penalty  = 1.30f;

    int32_t n_batch         = 8; // batch size for prompt processing
    int32_t n_keep          = 0;
    int32_t n_threads       = 4;
}env_configs_t;

typedef struct _env_state{
    bool can_reamain();
    int32_t n_past = 0;
    int32_t n_remain = 0;
}env_state_t;

typedef struct _session_env{
    llama_context *ctx = nullptr; // context instance
    env_configs_t configs;  // params for model load and eval
    last_n_tokens_t last_n_tokens;
    keep_prompt_token_t keep_token;
    instruction_info_t instruction_info; // inject info
    embedding_queue_t embedding_queue;
    env_state_t state;
}session_env_t;

bool load_model(session_env_t *env, const gpt_params &params, llama_progress_callback progress_callback,void *progress_callback_user_data);
void unload_model(session_env_t *env);

bool init_chat_env(session_env_t *env);
void init_user_input(session_env_t *env, const QString &msg);
QString generate_token(session_env_t *env);
bool should_generate(session_env_t *env);
#endif // COMMON_H
