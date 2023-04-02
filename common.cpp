#include "common.h"

inline std::vector<llama_token> llama_tokenize(llama_context *ctx, const QString &text, bool add_bos)
{
    std::string text_ = text.toStdString();
    // initialize to prompt numer of chars, since n_tokens <= n_prompt_chars
    std::vector<llama_token> res(text_.size() + (int)add_bos);
    int n = llama_tokenize(ctx, text_.c_str(), res.data(), res.size(), add_bos);
    assert(n >= 0);
    res.resize(n);

    return res;
}

bool load_model(session_env_t *env, const gpt_params &params, llama_progress_callback progress_callback,void *progress_callback_user_data)
{
    env->configs.init(params);
    std::string model = params.model.toStdString();
    auto lparams = llama_context_default_params();
    lparams.seed = params.seed;
    lparams.n_ctx = params.n_ctx;
    lparams.f16_kv = params.memory_f16;
    lparams.use_mlock = params.use_mlock;
    lparams.progress_callback=progress_callback;
    lparams.progress_callback_user_data=progress_callback_user_data;
    env->ctx = llama_init_from_file(model.c_str(),lparams);
    if(env->ctx)
        return true;
    return false;
}

void unload_model(session_env_t *env)
{
    llama_free(env->ctx);
}

inline void consume_tokens(session_env_t *env, int32_t n_batch)
{
    for(int i=0;i< n_batch; i++)
    {
        env->embedding_queue.input_consume();
    }
}

inline void process_output_embd(session_env_t *env)
{
    if(!env->embedding_queue.output_is_empty())
    {
        const int n_ctx = llama_n_ctx(env->ctx);

        std::vector<llama_token>& embd = env->embedding_queue.get_embd_output();
        if (env->state.n_past + (int) embd.size() > n_ctx) {
            const int n_left = env->state.n_past - env->configs.n_keep;

            env->state.n_past = env->keep_token.get_n_keep();
            // insert n_left/2 tokens at the start of embd from last_n_tokens
            embd.insert(embd.begin(), env->last_n_tokens.get_last_n_tokens().begin() + n_ctx - n_left/2 - embd.size(), env->last_n_tokens.get_last_n_tokens().end() - embd.size());
        }
        if (llama_eval(env->ctx, embd.data(), embd.size(), env->state.n_past, env->configs.n_threads)) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return;
        }
        env->state.n_past += embd.size();
        embd.clear();
    }
}
#include <QDebug>
void init_user_input(session_env_t *env, const QString &msg)
{
    env->state.n_remain = env->configs.n_predict;
    env->state.n_remain -= env->embedding_queue.input_produce(env->instruction_info, msg);
}

QString generate_token(session_env_t *env)
{
    QString result = "";
    if(env->state.can_reamain())
    {
        if(!env->embedding_queue.input_is_empty())
        {
            consume_tokens(env,env->configs.n_batch);
        }
        process_output_embd(env);
        const int32_t top_k          = env->configs.top_k;
        const float   top_p          = env->configs.top_p;
        const float   temp           = env->configs.temp;
        const float   repeat_penalty = env->configs.repeat_penalty;
        const int32_t repeat_last_n  = env->configs.repeat_last_n;
        const int n_ctx = llama_n_ctx(env->ctx);
        llama_token id = 0;
        {
            id = llama_sample_top_p_top_k(env->ctx,
                    env->last_n_tokens.get_last_n_tokens().data() + n_ctx - repeat_last_n,
                    repeat_last_n, top_k, top_p, temp, repeat_penalty);

            env->last_n_tokens.push(id);
        }
        result = QString(llama_token_to_str(env->ctx, id));
        std::vector<llama_token>& embd = env->embedding_queue.get_embd_output();
        embd.push_back(id);
        if(embd.back() == llama_token_eos())
        {
            env->state.n_remain = 0;
        }
        else
        {
            env->state.n_remain--;
        }
    }
    return result;
}

bool init_chat_env(session_env_t *env)
{
    env->instruction_info.init(env->ctx);
    bool success = env->keep_token.init(env->ctx,"Below is an instruction that describes a task. Write a response that appropriately completes the request.");
    if(success)
    {
        env->last_n_tokens.init(env->ctx);
        env->embedding_queue.init(env->ctx);

        env->embedding_queue.input_copy(env->keep_token.get_initial_token());
        while(!env->embedding_queue.input_is_empty())
        {
            consume_tokens(env,env->configs.n_batch);
            process_output_embd(env);
        }
    }
    else
    {
        return false;
    }
    return true;
}

bool should_generate(session_env_t *env)
{
    return env->state.can_reamain();
}

bool _keep_prompt_token::init(llama_context *ctx, const QString prompt)
{
    m_ctx = ctx;
    QString prompt_ = QString(" %1").arg(prompt);
    auto embd_inp = ::llama_tokenize(ctx, prompt_, true);
    const int n_ctx = llama_n_ctx(ctx);
    if ((int32_t) embd_inp.size() > n_ctx - 4) {
        fprintf(stderr, "%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
        return false;
    }
    initial_token = embd_inp;
    n_keep = (int32_t)embd_inp.size();
    return true;
}

int32_t _keep_prompt_token::get_n_keep()
{
    return n_keep;
}

std::vector<llama_token> &_keep_prompt_token::get_initial_token()
{
    return initial_token;
}

void _last_n_tokens::init(llama_context *ctx)
{
    const int n_ctx = llama_n_ctx(ctx);
    last_n_tokens.resize(n_ctx);
}

void _last_n_tokens::push(llama_token id)
{
    last_n_tokens.erase(last_n_tokens.begin());
    last_n_tokens.push_back(id);
}

std::vector<llama_token> &_last_n_tokens::get_last_n_tokens()
{
    return last_n_tokens;
}

void _embedding_queue::init(llama_context *ctx)
{
    m_ctx = ctx;
}

void _embedding_queue::input_copy(const std::vector<llama_token> input)
{
    embd_input.assign(input.begin(),input.end());
}

bool _embedding_queue::input_is_empty()
{
    return embd_input.size() <= n_consumed;
}

int _embedding_queue::input_produce(instruction_info_t &instruction_info, const QString buffer)
{
    // Clear input if its consume all token
    if(input_is_empty())
    {
        embd_input.clear();
        n_consumed = 0;
    }
    return instruction_info.inject(buffer, embd_input);
}

void _embedding_queue::input_consume()
{
    if(!input_is_empty())
    {
        embd_output.push_back(embd_input[n_consumed]);
        ++n_consumed;
    }
}

std::vector<llama_token> &_embedding_queue::get_embd_output()
{
    return embd_output;
}

bool _embedding_queue::output_is_empty()
{
    return embd_output.empty();
}

void _instruction_info::init(llama_context *ctx)
{
    m_ctx = ctx;
    input_prefix = ::llama_tokenize(ctx, "\n\n### Instruction:\n\n", true);
    input_suffix = ::llama_tokenize(ctx, "\n\n### Response:\n\n", false);
}

int _instruction_info::inject(const QString buffer, std::vector<llama_token> &output)
{
    //output.insert(output.end(), input_prefix.begin(), input_prefix.end());
    auto line_inp = ::llama_tokenize(m_ctx, buffer, false);
    output.insert(output.end(), line_inp.begin(), line_inp.end());
    output.insert(output.end(), input_suffix.begin(), input_suffix.end());
    return line_inp.size();
}

void _env_configs::init(const gpt_params &params)
{
    n_batch = params.n_batch;
    n_keep = params.n_keep;
    n_predict = params.n_predict;
    n_threads = params.n_threads;
    repeat_last_n = params.repeat_last_n;
    repeat_penalty = params.repeat_penalty;
    temp = params.temp;
    top_k = params.top_k;
    top_p = params.top_p;
}

bool _env_state::can_reamain()
{
    return n_remain > 0;
}
