#ifndef PROCESSOR_H
#define PROCESSOR_H

#include <QObject>
#include <QString>
#include <random>
#include "gptutils.h"

class Processor : public QObject
{
    Q_OBJECT
public:
    explicit Processor(QObject *parent = nullptr);

signals:
    void modelLoading(int percent);
    void modelLoadSuccessed();
    void modelLoadFailed(const QString &reason);
    void modelUnloaded();
    void tokenRemaining();
    void tokenSampled(const QString &token);
    void tokenConsumed();

public slots:
    void handleLoadModel(const gpt_params &params);
    void handleUnloadModel();
    void handleEvalToken(const QString &prompt);

private:
    QVector<gpt_vocab::id> llama_tokenize(const gpt_vocab & vocab, const QString & text, bool bos);
    bool llama_eval(
            const llama_model & model,
            const int n_threads,
            const int n_past,
            const QVector<gpt_vocab::id> & embd_inp,
                  QVector<float>         & embd_w,
                  size_t                 & mem_per_token);
    void sample_top_k(QVector<QPair<double, gpt_vocab::id>> & logits_id, int top_k);
    gpt_vocab::id llama_sample_top_p_top_k(
            const gpt_vocab & vocab,
            const float * logits,
            QVector<gpt_vocab::id> & last_n_tokens,
            double repeat_penalty,
            int top_k,
            double top_p,
            double temp,
            std::mt19937 & rng);
private:
    gpt_params m_params;
    gpt_vocab m_vocab;
    llama_model m_model;
    size_t mem_per_token;
    QVector<float> m_logits;
};

#endif // PROCESSOR_H
