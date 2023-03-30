#include <QFile>
#include "processor.h"

class Processor::InternalData
{
public:
    llama_context *ctx = nullptr;
    gpt_params params;
};

Processor::Processor(QObject *parent) : QObject(parent)
{
    Q_UNUSED(parent)
    qRegisterMetaType<gpt_params>();
    m_data = QSharedPointer<InternalData>(new InternalData());
}

void Processor::handleLoadModel(const gpt_params &params)
{
    std::string model = params.model.toStdString();
    auto lparams = llama_context_default_params();
    m_data->params = params;
    lparams.seed = params.seed;
    lparams.n_ctx = params.n_ctx;
    lparams.f16_kv = params.memory_f16;
    lparams.use_mlock = params.use_mlock;
    lparams.progress_callback=updateLoadProgress;
    lparams.progress_callback_user_data=this;
    m_data->ctx = llama_init_from_file(model.c_str(),lparams);
    if(m_data->ctx)
    {
       emit modelLoadSuccessed();
    }
    else
    {
        emit modelLoadFailed("Failed to OpenFile");
    }
}

void Processor::handleUnloadModel()
{
    llama_free(m_data->ctx);
    emit modelUnloaded();
}

void Processor::handleEvalToken(const QString &prompt)
{
    emit tokenRemaining();
    // Add a space in front of the first character to match OG llama tokenizer behavior
    QString prompt_ = QString(" %1").arg(prompt);
    auto embd_inp = ::llama_tokenize(m_data->ctx, prompt_, true);
    const int n_ctx = llama_n_ctx(m_data->ctx);

    if ((int) embd_inp.size() > n_ctx - 4) {
        fprintf(stderr, "%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
        return;
    }

    // number of tokens to keep when resetting context
    if (m_data->params.n_keep < 0 || m_data->params.n_keep > (int)embd_inp.size() || m_data->params.instruct) {
        m_data->params.n_keep = (int)embd_inp.size();
    }
    // determine newline token
    auto llama_token_newline = ::llama_tokenize(m_data->ctx, "\n", false);

    // TODO: replace with ring-buffer
    std::vector<llama_token> last_n_tokens(n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    int n_past     = 0;
    int n_remain   = m_data->params.n_predict;
    int n_consumed = 0;

    std::vector<llama_token> embd;
    while (n_remain != 0 || m_data->params.interactive) {
        // predict
        if (embd.size() > 0) {
            // infinite text generation via context swapping
            // if we run out of context:
            // - take the n_keep first tokens from the original prompt (via n_past)
            // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in a batch
            if (n_past + (int) embd.size() > n_ctx) {
                const int n_left = n_past - m_data->params.n_keep;

                n_past = m_data->params.n_keep;

                // insert n_left/2 tokens at the start of embd from last_n_tokens
                embd.insert(embd.begin(), last_n_tokens.begin() + n_ctx - n_left/2 - embd.size(), last_n_tokens.end() - embd.size());
            }

            if (llama_eval(m_data->ctx, embd.data(), embd.size(), n_past, m_data->params.n_threads)) {
                fprintf(stderr, "%s : failed to eval\n", __func__);
                return;
            }
        }

        n_past += embd.size();
        embd.clear();

        if ((int) embd_inp.size() <= n_consumed) {
            // out of user input, sample next token
            const int32_t top_k          = m_data->params.top_k;
            const float   top_p          = m_data->params.top_p;
            const float   temp           = m_data->params.temp;
            const float   repeat_penalty = m_data->params.repeat_penalty;

            llama_token id = 0;

            {
                auto logits = llama_get_logits(m_data->ctx);

                if (m_data->params.ignore_eos) {
                    logits[llama_token_eos()] = 0;
                }

                id = llama_sample_top_p_top_k(m_data->ctx,
                        last_n_tokens.data() + n_ctx - m_data->params.repeat_last_n,
                        m_data->params.repeat_last_n, top_k, top_p, temp, repeat_penalty);

                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(id);
            }

            // replace end of text token with newline token when in interactive mode
            if (id == llama_token_eos() && m_data->params.interactive && !m_data->params.instruct) {
                id = llama_token_newline.front();
            }

            // add it to the context
            embd.push_back(id);

            // decrement remaining sampling budget
            --n_remain;
        } else {
            // some user input remains from prompt or interaction, forward it to processing
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);
                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(embd_inp[n_consumed]);
                ++n_consumed;
                if ((int) embd.size() >= m_data->params.n_batch) {
                    break;
                }
            }
        }

        // display text
        for (auto id : embd) {
            emit tokenSampled(llama_token_to_str(m_data->ctx, id));
        }

        // end of text token
        if (embd.back() == llama_token_eos()) {
            emit tokenSampled(" [end of text]\n");
            break;
        }

        // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
        if (m_data->params.interactive && n_remain <= 0 && m_data->params.n_predict != -1) {
            n_remain = m_data->params.n_predict;
        }
    }
    emit tokenConsumed();
}

void Processor::updateLoadProgress(float progress, void *ctx)
{
    Processor* ctx_ = (Processor*)ctx;
    emit ctx_->modelLoading(100*progress);
}
