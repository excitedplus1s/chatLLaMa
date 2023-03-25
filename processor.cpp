#include <QFile>
#include "processor.h"


Processor::Processor(QObject *parent) : QObject(parent)
{
    Q_UNUSED(parent)
    qRegisterMetaType<gpt_params>();
}

void Processor::handleLoadModel(const gpt_params &params)
{
    m_params = params;
    QFile fin(m_params.model);
    if(fin.open(QIODevice::ReadOnly))
    {
        uint32_t magic = 0;
        fin.read((char *) &magic, sizeof(magic));
        if(magic != 0x67676d6c)
        {
            emit modelLoadFailed("Invaild model,bad magic");
            fin.close();
            return;
        }
        int n_ff = 0;
        int n_parts = 0;

        // load hparams
        {
            auto & hparams = m_model.hparams;

            fin.read((char *) &hparams.n_vocab, sizeof(hparams.n_vocab));
            fin.read((char *) &hparams.n_embd,  sizeof(hparams.n_embd));
            fin.read((char *) &hparams.n_mult,  sizeof(hparams.n_mult));
            fin.read((char *) &hparams.n_head,  sizeof(hparams.n_head));
            fin.read((char *) &hparams.n_layer, sizeof(hparams.n_layer));
            fin.read((char *) &hparams.n_rot,   sizeof(hparams.n_rot));
            fin.read((char *) &hparams.f16,     sizeof(hparams.f16));

            hparams.n_ctx = 512;

            n_ff = ((2*(4*hparams.n_embd)/3 + hparams.n_mult - 1)/hparams.n_mult)*hparams.n_mult;
            n_parts = LLAMA_N_PARTS.value(hparams.n_embd);
        }
        // load vocab
        {
            const int32_t n_vocab = m_model.hparams.n_vocab;

            if (n_vocab != m_model.hparams.n_vocab) {
                emit modelLoadFailed(QString("invalid model file '%1' (bad vocab size %2 != %3)")
                                     .arg(m_params.model)
                                     .arg(n_vocab)
                                     .arg(m_model.hparams.n_vocab)
                                     );
                fin.close();
                return;
            }

            QString word;
            for (int i = 0; i < n_vocab; i++) {
                uint32_t len;
                fin.read((char *) &len, sizeof(len));

                word.resize(len);
                word = QString::fromUtf8(fin.read(len));

                m_vocab.token_to_id[word] = i;
                m_vocab.id_to_token[i] = word;
            }
        }
        // for the big tensors, we have the option to store the data in 16-bit floats or quantized
        // in order to save memory and also to speed up the computation
        ggml_type wtype = GGML_TYPE_COUNT;
        switch (m_model.hparams.f16) {
            case 0: wtype = GGML_TYPE_F32;  break;
            case 1: wtype = GGML_TYPE_F16;  break;
            case 2: wtype = GGML_TYPE_Q4_0; break;
            case 3: wtype = GGML_TYPE_Q4_1; break;
            default:
                    emit modelLoadFailed(QString("invalid model file '%1' (bad f16 value %2)")
                                 .arg(m_params.model)
                                 .arg(m_model.hparams.f16)
                                 );
                    fin.close();
                    return;
        }

        auto & ctx = m_model.ctx;

        size_t ctx_size = 0;

        {
            const auto & hparams = m_model.hparams;

            const int n_embd  = hparams.n_embd;
            const int n_layer = hparams.n_layer;
            const int n_ctx   = hparams.n_ctx;
            const int n_vocab = hparams.n_vocab;

            ctx_size += n_embd*n_vocab*ggml_type_sizef(wtype); // tok_embeddings

            ctx_size += n_embd*ggml_type_sizef(GGML_TYPE_F32); // norm

            ctx_size += n_embd*n_vocab*ggml_type_sizef(wtype); // output

            ctx_size += n_layer*(n_embd*ggml_type_sizef(GGML_TYPE_F32)); // attention_norm

            ctx_size += n_layer*(n_embd*n_embd*ggml_type_sizef(wtype)); // wq
            ctx_size += n_layer*(n_embd*n_embd*ggml_type_sizef(wtype)); // wk
            ctx_size += n_layer*(n_embd*n_embd*ggml_type_sizef(wtype)); // wv
            ctx_size += n_layer*(n_embd*n_embd*ggml_type_sizef(wtype)); // wo

            ctx_size += n_layer*(n_embd*ggml_type_sizef(GGML_TYPE_F32)); // ffn_norm

            ctx_size += n_layer*(n_ff*n_embd*ggml_type_sizef(wtype)); // w1
            ctx_size += n_layer*(n_ff*n_embd*ggml_type_sizef(wtype)); // w2
            ctx_size += n_layer*(n_ff*n_embd*ggml_type_sizef(wtype)); // w3

            ctx_size += n_ctx*n_layer*n_embd*ggml_type_sizef(GGML_TYPE_F32); // memory_k
            ctx_size += n_ctx*n_layer*n_embd*ggml_type_sizef(GGML_TYPE_F32); // memory_v

            ctx_size += (5 + 10*n_layer)*256; // object overhead
        }
        // create the ggml context
        {
            struct ggml_init_params params = {
                /*.mem_size   =*/ ctx_size,
                /*.mem_buffer =*/ NULL,
            };

            m_model.ctx = ggml_init(params);
            if (!m_model.ctx) {
                emit modelLoadFailed(QString("%1: ggml_init() failed")
                             .arg(__func__)
                             );
                fin.close();
                return;
            }
        }
        // prepare memory for the weights
        {
           const auto & hparams = m_model.hparams;

           const int n_embd  = hparams.n_embd;
           const int n_layer = hparams.n_layer;
           const int n_vocab = hparams.n_vocab;

           m_model.layers.resize(n_layer);

           m_model.tok_embeddings = ggml_new_tensor_2d(ctx, wtype, n_embd, n_vocab);

           m_model.norm   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
           m_model.output = ggml_new_tensor_2d(ctx, wtype, n_embd, n_vocab);

           // map by name
           m_model.tensors["tok_embeddings.weight"] = m_model.tok_embeddings;

           m_model.tensors["norm.weight"]   = m_model.norm;
           m_model.tensors["output.weight"] = m_model.output;

           for (int i = 0; i < n_layer; ++i) {
               auto & layer = m_model.layers[i];

               layer.attention_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

               layer.wq = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
               layer.wk = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
               layer.wv = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
               layer.wo = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);

               layer.ffn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

               layer.w1 = ggml_new_tensor_2d(ctx, wtype, n_embd,   n_ff);
               layer.w2 = ggml_new_tensor_2d(ctx, wtype,   n_ff, n_embd);
               layer.w3 = ggml_new_tensor_2d(ctx, wtype, n_embd,   n_ff);

               // map by name
               m_model.tensors["layers." + QString::number(i) + ".attention_norm.weight"] = layer.attention_norm;

               m_model.tensors["layers." + QString::number(i) + ".attention.wq.weight"] = layer.wq;
               m_model.tensors["layers." + QString::number(i) + ".attention.wk.weight"] = layer.wk;
               m_model.tensors["layers." + QString::number(i) + ".attention.wv.weight"] = layer.wv;
               m_model.tensors["layers." + QString::number(i) + ".attention.wo.weight"] = layer.wo;

               m_model.tensors["layers." + QString::number(i) + ".ffn_norm.weight"] = layer.ffn_norm;

               m_model.tensors["layers." + QString::number(i) + ".feed_forward.w1.weight"] = layer.w1;
               m_model.tensors["layers." + QString::number(i) + ".feed_forward.w2.weight"] = layer.w2;
               m_model.tensors["layers." + QString::number(i) + ".feed_forward.w3.weight"] = layer.w3;
           }
        }
        // key + value memory
        {
            const auto & hparams = m_model.hparams;

            const int n_embd  = hparams.n_embd;
            const int n_layer = hparams.n_layer;
            const int n_ctx   = hparams.n_ctx;

            const int n_mem      = n_layer*n_ctx;
            const int n_elements = n_embd*n_mem;

            m_model.memory_k = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
            m_model.memory_v = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
        }
        const qint64 file_offset = fin.pos();
        fin.close();

        //now load model
        for (int i = 0; i < n_parts; ++i) {
            const int part_id = i;

            QString fname_part = m_params.model;
            if (i > 0) {
                fname_part = QString("%1.%2")
                        .arg(m_params.model)
                        .arg(i);
            }
            QFile fin(fname_part);
            if(fin.open(QIODevice::ReadOnly)){
                const size_t file_size = fin.size();
                fin.seek(file_offset);
                // load weights
                {
                    int n_tensors = 0;
                    size_t total_size = 0;

                    while (true) {
                        int32_t n_dims;
                        int32_t length;
                        int32_t ftype;

                        fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
                        fin.read(reinterpret_cast<char *>(&length), sizeof(length));
                        fin.read(reinterpret_cast<char *>(&ftype),  sizeof(ftype));

                        if (fin.atEnd()) {
                            break;
                        }

                        int32_t nelements = 1;
                        int32_t ne[2] = { 1, 1 };
                        for (int i = 0; i < n_dims; ++i) {
                            fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
                            nelements *= ne[i];
                        }

                        std::string name(length, 0);
                        fin.read(&name[0], length);

                        if (m_model.tensors.find(name.data()) == m_model.tensors.end()) {
                            emit modelLoadFailed(QString("unknown tensor '%1' in model file")
                                                 .arg(name.data())
                                         );
                            fin.close();
                            return;
                        }

                        int split_type = 0;
                        if (name.find("tok_embeddings") != std::string::npos) {
                            split_type = 0;
                        } else if (name.find("layers") != std::string::npos) {
                            if (name.find("attention.wo.weight") != std::string::npos) {
                                split_type = 0;
                            } else if (name.find("feed_forward.w2.weight") != std::string::npos) {
                                split_type = 0;
                            } else {
                                split_type = 1;
                            }
                        } else if (name.find("output") != std::string::npos) {
                            split_type = 1;
                        }

                        auto tensor = m_model.tensors[name.data()];

                        if (n_dims == 1) {
                            if (ggml_nelements(tensor) != nelements) {
                                emit modelLoadFailed(QString("tensor '%1' has wrong size in model file")
                                                     .arg(name.data())
                                             );
                                fin.close();
                                return;
                            }
                        } else {
                            if (ggml_nelements(tensor)/n_parts != nelements) {
                                emit modelLoadFailed(QString("tensor '%1' has wrong size in model file")
                                                     .arg(name.data())
                                             );
                                fin.close();
                                return;
                            }
                        }

                        if (n_dims == 1) {
                            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
                                emit modelLoadFailed(QString("tensor '%1' has wrong shape in model file: got [%2, %3], expected [%4, %5]")
                                                     .arg(name.data())
                                                     .arg(tensor->ne[0])
                                                     .arg(tensor->ne[1])
                                                     .arg(ne[0])
                                                     .arg(ne[1])
                                             );
                                fin.close();
                                return;
                            }
                        } else {
                            if (split_type == 0) {
                                if (tensor->ne[0]/n_parts != ne[0] || tensor->ne[1] != ne[1]) {
                                    emit modelLoadFailed(QString("tensor '%1' has wrong shape in model file: got [%2, %3], expected [%4, %5]")
                                                         .arg(name.data())
                                                         .arg(tensor->ne[0]/n_parts)
                                                         .arg(tensor->ne[1])
                                                         .arg(ne[0])
                                                         .arg(ne[1])
                                                 );
                                    fin.close();
                                    return;
                                }
                            } else {
                                if (tensor->ne[0] != ne[0] || tensor->ne[1]/n_parts != ne[1]) {
                                    emit modelLoadFailed(QString("tensor '%1' has wrong shape in model file: got [%2, %3], expected [%4, %5]")
                                                         .arg(name.data())
                                                         .arg(tensor->ne[0])
                                                         .arg(tensor->ne[1]/n_parts)
                                                         .arg(ne[0])
                                                         .arg(ne[1])
                                                 );
                                    fin.close();
                                    return;
                                }
                            }
                        }

                        size_t bpe = 0;

                        switch (ftype) {
                            case 0: bpe = ggml_type_size(GGML_TYPE_F32);  break;
                            case 1: bpe = ggml_type_size(GGML_TYPE_F16);  break;
                            case 2: bpe = ggml_type_size(GGML_TYPE_Q4_0); assert(ne[0] % 64 == 0); break;
                            case 3: bpe = ggml_type_size(GGML_TYPE_Q4_1); assert(ne[0] % 64 == 0); break;
                            default:
                                    {
                                        emit modelLoadFailed(QString("unknown ftype %1 in model file")
                                                             .arg(ftype)
                                                     );
                                        fin.close();
                                        return;
                                    }
                        };

                        if (n_dims == 1 || n_parts == 1) {
                            if ((nelements*bpe)/ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
                                emit modelLoadFailed(QString("tensor '%1' has wrong size in model file: got %2, expected %3")
                                                     .arg(name.data())
                                                     .arg(ggml_nbytes(tensor))
                                                     .arg(nelements*bpe)
                                             );
                                fin.close();
                                return;
                            }

                            if (part_id == 0) {
                                fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));
                            } else {
                                fin.skip(ggml_nbytes(tensor));
                            }

                            total_size += ggml_nbytes(tensor);
                        } else {
                            if ((nelements*bpe)/ggml_blck_size(tensor->type) != ggml_nbytes(tensor)/n_parts) {
                                emit modelLoadFailed(QString("tensor '%1' has wrong size in model file: got %2, expected %3")
                                                     .arg(name.data())
                                                     .arg(ggml_nbytes(tensor)/n_parts)
                                                     .arg(nelements*bpe)
                                             );
                                fin.close();
                                return;
                            }

                            if (split_type == 0) {
                                const int np0 = ne[0];

                                const size_t row_size = (tensor->ne[0]/ggml_blck_size(tensor->type))*ggml_type_size(tensor->type);
                                assert(row_size == tensor->nb[1]);

                                for (int i1 = 0; i1 < ne[1]; ++i1) {
                                    const size_t offset_row = i1*row_size;
                                    const size_t offset = offset_row + ((part_id*np0)/ggml_blck_size(tensor->type))*ggml_type_size(tensor->type);
                                    fin.read(reinterpret_cast<char *>(tensor->data) + offset, row_size/n_parts);
                                }
                            } else {
                                const int np1 = ne[1];

                                const size_t row_size = (tensor->ne[0]/ggml_blck_size(tensor->type))*ggml_type_size(tensor->type);

                                for (int i1 = 0; i1 < ne[1]; ++i1) {
                                    const size_t offset_row = (i1 + part_id*np1)*row_size;
                                    fin.read(reinterpret_cast<char *>(tensor->data) + offset_row, row_size);
                                }
                            }

                            total_size += ggml_nbytes(tensor)/n_parts;
                        }

                        double current_file_progress = double(size_t(fin.pos()) - file_offset) / double(file_size - file_offset);
                        double current_progress = (double(i) + current_file_progress) / double(n_parts);
                        emit modelLoading(current_progress*100);
                    }
                }

                fin.close();
            }
            else
            {
                emit modelLoadFailed(QString("Failed to OpenFile '%1'")
                                     .arg(fname_part));
                return;
            }
        }
    }
    else
    {
        emit modelLoadFailed("Failed to OpenFile");
        return;
    }
    mem_per_token = 0;
    llama_eval(m_model, params.n_threads, 0, { 0, 1, 2, 3 }, m_logits, mem_per_token);
    emit modelLoadSuccessed();
}

void Processor::handleUnloadModel()
{
    ggml_free(m_model.ctx);
    emit modelUnloaded();
}

void Processor::handleEvalToken(const QString &prompt)
{
    emit tokenRemaining();
    QVector<gpt_vocab::id> embd_inp = llama_tokenize(m_vocab, prompt, true);
    for (int i = 0; i < (int) embd_inp.size(); i++) {
            fprintf(stderr, "%6d -> '%s'\n", embd_inp[i], m_vocab.id_to_token.value(embd_inp[i]).data());
        }
    m_params.n_predict = std::min(m_params.n_predict, m_model.hparams.n_ctx - (int) embd_inp.size());
    int last_n_size = m_params.repeat_last_n;
    QVector<gpt_vocab::id> last_n_tokens(last_n_size);
    last_n_tokens.fill(0);
    int remaining_tokens = m_params.n_predict;
    int input_consumed = 0;
    int n_past = 0;
    QVector<gpt_vocab::id> embd;
    while (remaining_tokens > 0)
    {
        // predict
        if (embd.size() > 0)
        {
            if (!llama_eval(m_model, m_params.n_threads, n_past, embd, m_logits, mem_per_token)) {
                emit tokenConsumed();
                return;
            }
        }

        n_past += embd.size();
        embd.clear();

        if (embd_inp.size() <= input_consumed)
        {
            // out of user input, sample next token
            const float top_k = m_params.top_k;
            const float top_p = m_params.top_p;
            const float temp  = m_params.temp;
            const float repeat_penalty = m_params.repeat_penalty;

            const int n_vocab = m_model.hparams.n_vocab;

            gpt_vocab::id id = 0;

            {
                std::mt19937 rng(m_params.seed);

                id = llama_sample_top_p_top_k(m_vocab, m_logits.data() + (m_logits.size() - n_vocab), last_n_tokens, repeat_penalty, top_k, top_p, temp, rng);

                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(id);
            }

            // add it to the context
            embd.push_back(id);

            // decrement remaining sampling budget
            --remaining_tokens;
        } else {
            // some user input remains from prompt or interaction, forward it to processing
            while (embd_inp.size() > input_consumed) {
                embd.push_back(embd_inp[input_consumed]);
                last_n_tokens.erase(last_n_tokens.begin());
                last_n_tokens.push_back(embd_inp[input_consumed]);
                ++input_consumed;
                if (embd.size() > m_params.n_batch) {
                    break;
                }
            }
        }
        // send text
        int id;
        foreach (id, embd) {
            emit tokenSampled(m_vocab.id_to_token[id]);
        }
        if (embd.back() == 2) {
                 break;
        }
    }

    emit tokenConsumed();
}

QVector<gpt_vocab::id> Processor::llama_tokenize(const gpt_vocab & vocab, const QString & text, bool bos)
{

    QVector<gpt_vocab::id> res;

    if (bos) {
        res.push_back(1); // TODO: replace with vocab.bos
    }

     //find the longest token that matches the text
    int pos = 0;
    while (true) {
        int l = 0;
        int t = 0;
        for (auto it = vocab.id_to_token.cbegin();it!=vocab.id_to_token.cend();++it) {
            if (it.value().size() < l) continue;
            if (it.value().size() > text.size() - pos) continue;
            if (text.mid(pos, it.value().size()) == it.value()) {
                l = it.value().size();
                t = it.key();
            }
        }

        if (l == 0) {
            break;
        }

        res.push_back(t);
        pos += l;
    }

    return res;
}

bool Processor::llama_eval(
        const llama_model & model,
        const int n_threads,
        const int n_past,
        const QVector<gpt_vocab::id> & embd_inp,
              QVector<float>         & embd_w,
              size_t                     & mem_per_token)
{
    const int N = embd_inp.size();

    const auto & hparams = model.hparams;

    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx   = hparams.n_ctx;
    const int n_head  = hparams.n_head;
    const int n_vocab = hparams.n_vocab;
    const int n_rot   = hparams.n_embd/hparams.n_head;


    static size_t buf_size = 512u*1024*1024;
    static void * buf = malloc(buf_size);

    if (mem_per_token > 0 && mem_per_token*N > buf_size) {
        const size_t buf_size_new = 1.1*(mem_per_token*N); // add 10% to account for ggml object overhead
        //fprintf(stderr, "\n%s: reallocating buffer from %zu to %zu bytes\n", __func__, buf_size, buf_size_new);

        // reallocate
        buf_size = buf_size_new;
        buf = realloc(buf, buf_size);
        if (buf == nullptr) {
            fprintf(stderr, "%s: failed to allocate %zu bytes\n", __func__, buf_size);
            return false;
        }
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    ggml_cgraph gf = {};
    gf.n_threads = n_threads;

    struct ggml_tensor * embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    memcpy(embd->data, embd_inp.data(), N*ggml_element_size(embd));

    struct ggml_tensor * inpL = ggml_get_rows(ctx0, model.tok_embeddings, embd);

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * inpSA = inpL;

        struct ggml_tensor * cur;

        // norm
        {
            cur = ggml_norm(ctx0, inpL);

            // cur = attention_norm*cur
            cur = ggml_mul(ctx0,
                        ggml_repeat(ctx0, model.layers[il].attention_norm, cur),
                        cur);
        }

        // self-attention
        {
            struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
            struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, model.layers[il].wk, cur);
            struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, model.layers[il].wv, cur);

            // store key and value to memory
            if (N >= 1) {
                struct ggml_tensor * k = ggml_view_1d(ctx0, model.memory_k, N*n_embd, (ggml_element_size(model.memory_k)*n_embd)*(il*n_ctx + n_past));
                struct ggml_tensor * v = ggml_view_1d(ctx0, model.memory_v, N*n_embd, (ggml_element_size(model.memory_v)*n_embd)*(il*n_ctx + n_past));

                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Kcur, k));
                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Vcur, v));
            }

            // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
            struct ggml_tensor * Q =
                ggml_permute(ctx0,
                        ggml_rope(ctx0,
                            ggml_cpy(ctx0,
                                Qcur,
                                ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd/n_head, n_head, N)),
                            n_past, n_rot, 0),
                        0, 2, 1, 3);

            // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
            struct ggml_tensor * K =
                ggml_permute(ctx0,
                        ggml_rope(ctx0,
                            ggml_reshape_3d(ctx0,
                                ggml_view_1d(ctx0, model.memory_k, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(model.memory_k)*n_embd),
                                n_embd/n_head, n_head, n_past + N),
                            n_past, n_rot, 1),
                        0, 2, 1, 3);

            // K * Q
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            struct ggml_tensor * KQ_scaled =
                ggml_scale(ctx0,
                        KQ,
                        ggml_new_f32(ctx0, 1.0f/sqrt(float(n_embd)/n_head))
                        );

            // KQ_masked = mask_past(KQ_scaled)
            struct ggml_tensor * KQ_masked = ggml_diag_mask_inf(ctx0, KQ_scaled, n_past);

            // KQ = soft_max(KQ_masked)
            struct ggml_tensor * KQ_soft_max = ggml_soft_max(ctx0, KQ_masked);

            // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
            struct ggml_tensor * V_trans =
                ggml_permute(ctx0,
                        ggml_reshape_3d(ctx0,
                            ggml_view_1d(ctx0, model.memory_v, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(model.memory_v)*n_embd),
                            n_embd/n_head, n_head, n_past + N),
                        1, 2, 0, 3);

            // KQV = transpose(V) * KQ_soft_max
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V_trans, KQ_soft_max);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

            // cur = KQV_merged.contiguous().view(n_embd, N)
            cur = ggml_cpy(ctx0,
                    KQV_merged,
                    ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));

            // projection (no bias)
            cur = ggml_mul_mat(ctx0,
                    model.layers[il].wo,
                    cur);
        }

        struct ggml_tensor * inpFF = ggml_add(ctx0, cur, inpSA);

        // feed-forward network
        {
            // norm
            {
                cur = ggml_norm(ctx0, inpFF);

                // cur = ffn_norm*cur
                cur = ggml_mul(ctx0,
                        ggml_repeat(ctx0, model.layers[il].ffn_norm, cur),
                        cur);
            }

            struct ggml_tensor * tmp = ggml_mul_mat(ctx0,
                    model.layers[il].w3,
                    cur);


            cur = ggml_mul_mat(ctx0,
                    model.layers[il].w1,
                    cur);

            // SILU activation
            cur = ggml_silu(ctx0, cur);

            cur = ggml_mul(ctx0, cur, tmp);

            cur = ggml_mul_mat(ctx0,
                    model.layers[il].w2,
                    cur);
        }

        cur  = ggml_add(ctx0, cur, inpFF);

        // input for next layer
        inpL = cur;
    }

    // norm
    {
        inpL = ggml_norm(ctx0, inpL);

        // inpL = norm*inpL
        inpL = ggml_mul(ctx0,
                    ggml_repeat(ctx0, model.norm, inpL),
                    inpL);
    }

    // lm_head
    {
        inpL = ggml_mul_mat(ctx0, model.output, inpL);
    }

    // logits -> probs
    //inpL = ggml_soft_max(ctx0, inpL);

    // run the computation
    ggml_build_forward_expand(&gf, inpL);
    ggml_graph_compute       (ctx0, &gf);

    //if (n_past%100 == 0) {
    //    ggml_graph_print   (&gf);
    //    ggml_graph_dump_dot(&gf, NULL, "gpt-2.dot");
    //}

    //embd_w.resize(n_vocab*N);
    //memcpy(embd_w.data(), ggml_get_data(inpL), sizeof(float)*n_vocab*N);

    // return result for just the last token
    embd_w.resize(n_vocab);
    memcpy(embd_w.data(), (float *) ggml_get_data(inpL) + (n_vocab*(N-1)), sizeof(float)*n_vocab);

    if (mem_per_token == 0) {
        mem_per_token = ggml_used_mem(ctx0)/N;
    }
    //fprintf(stderr, "used_mem = %zu\n", ggml_used_mem(ctx0));

    ggml_free(ctx0);

    return true;
}

void Processor::sample_top_k(QVector<QPair<double, gpt_vocab::id>> & logits_id, int top_k) {
    // find the top K tokens
    std::partial_sort(
            logits_id.begin(),
            logits_id.begin() + top_k, logits_id.end(),
            [](const QPair<double, gpt_vocab::id> & a, const QPair<double, gpt_vocab::id> & b) {
        return a.first > b.first;
    });

    logits_id.resize(top_k);
}

gpt_vocab::id Processor::llama_sample_top_p_top_k(
        const gpt_vocab & vocab,
        const float * logits,
        QVector<gpt_vocab::id> & last_n_tokens,
        double repeat_penalty,
        int top_k,
        double top_p,
        double temp,
        std::mt19937 & rng) {
    int n_logits = vocab.id_to_token.size();

    QVector<QPair<double, gpt_vocab::id>> logits_id;
    logits_id.reserve(n_logits);

    {

        const double scale = 1.0/temp;
        for (int i = 0; i < n_logits; ++i) {
            // repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
            // credit https://github.com/facebookresearch/llama/compare/main...shawwn:llama:main
            if (std::find(last_n_tokens.begin(), last_n_tokens.end(), i) != last_n_tokens.end()) {
                // if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                if (logits[i] < 0.0) {
                    logits_id.push_back(qMakePair(logits[i]*scale*repeat_penalty, i));
                } else {
                    logits_id.push_back(qMakePair(logits[i]*scale/repeat_penalty, i));
                }
            } else {
                logits_id.push_back(qMakePair(logits[i]*scale, i));
            }
        }
    }

    sample_top_k(logits_id, top_k);

    double maxl = -INFINITY;
    QPair<double, gpt_vocab::id> kv;
    foreach (kv, logits_id) {
        maxl = std::max(maxl, kv.first);
    }

    // compute probs for the top K tokens
    QVector<double> probs;
    probs.reserve(logits_id.size());

    double sum = 0.0;
    foreach (kv, logits_id) {
        double p = exp(kv.first - maxl);
        probs.push_back(p);
        sum += p;
    }

    // normalize the probs
    for (auto & p : probs) {
        p /= sum;
    }

    if (top_p < 1.0f) {
        double cumsum = 0.0f;
        for (int i = 0; i < (int) probs.size(); i++) {
            cumsum += probs[i];
            if (cumsum >= top_p) {
                probs.resize(i + 1);
                logits_id.resize(i + 1);
                break;
            }
        }

        cumsum = 1.0/cumsum;
        for (int i = 0; i < (int) probs.size(); i++) {
            probs[i] *= cumsum;
        }
    }

    std::discrete_distribution<> dist(probs.begin(), probs.end());
    int idx = dist(rng);

    return logits_id[idx].second;
}
