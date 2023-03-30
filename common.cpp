#include "common.h"

std::vector<llama_token> llama_tokenize(llama_context *ctx, const QString &text, bool add_bos)
{
    std::string text_ = text.toStdString();
    // initialize to prompt numer of chars, since n_tokens <= n_prompt_chars
    std::vector<llama_token> res(text_.size() + (int)add_bos);
    int n = llama_tokenize(ctx, text_.c_str(), res.data(), res.size(), add_bos);
    assert(n >= 0);
    res.resize(n);

    return res;
}
