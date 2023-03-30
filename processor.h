#ifndef PROCESSOR_H
#define PROCESSOR_H

#include <QObject>
#include <QString>
#include <random>
#include <QSharedPointer>
#include "common.h"

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
    static void updateLoadProgress(float progress, void *ctx);
private:
    class InternalData;
    QSharedPointer<InternalData> m_data;
};

#endif // PROCESSOR_H
