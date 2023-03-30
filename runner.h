#ifndef RUNNER_H
#define RUNNER_H

#include <QObject>
#include <QThread>
#include <QString>
#include "common.h"

class Runner : public QObject
{
    Q_OBJECT
public:
    explicit Runner(QObject *parent = nullptr);
    ~Runner();

signals: //recv from gui
    void loadModel(const gpt_params &params);
    void unloadModel();
    void sendMessage(const QString &prompt);

signals: //send to gui
    void loadModelPercent(int percent);
    void loadModelStatus(bool successed,const QString &reason);
    void resetModelStatus();
    void botWaitting();
    void botTalk(const QString &token);
    void botEnd();

private slots:
    void handleModelLoading(int percent);
    void handleModelLoadSuccessed();
    void handleModelLoadFailed(const QString &reason);
    void handleModelUnloaded();
    void handleTokenRemaining();
    void handleTokenSampled(const QString &token);
    void handleTokenConsumed();
private:
   Runner(const Runner&) = delete;
   Runner& operator=(const Runner&) = delete;

private:
    QThread m_thread;
};

#endif // RUNNER_H
