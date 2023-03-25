#include "runner.h"
#include "processor.h"

Runner::Runner(QObject *parent) : QObject(parent)
{
    Q_UNUSED(parent)
    Processor *processor = new Processor();
    processor->moveToThread(&m_thread);
    connect(&m_thread, &QThread::finished, processor, &QObject::deleteLater);
    connect(this, &Runner::loadModel, processor, &Processor::handleLoadModel);
    connect(this, &Runner::unloadModel, processor, &Processor::handleUnloadModel);
    connect(this, &Runner::sendMessage, processor, &Processor::handleEvalToken);

    connect(processor, &Processor::modelLoading, this, &Runner::handleModelLoading);
    connect(processor, &Processor::modelLoadFailed, this, &Runner::handleModelLoadFailed);
    connect(processor, &Processor::modelLoadSuccessed, this, &Runner::handleModelLoadSuccessed);
    connect(processor, &Processor::modelUnloaded, this, &Runner::handleModelUnloaded);
    connect(processor, &Processor::tokenRemaining, this, &Runner::handleTokenRemaining);
    connect(processor, &Processor::tokenSampled, this, &Runner::handleTokenSampled);
    connect(processor, &Processor::tokenConsumed, this, &Runner::handleTokenConsumed);

    m_thread.start();
}

Runner::~Runner()
{
    m_thread.terminate();
    //m_thread.quit();
    //m_thread.wait();
}

void Runner::handleModelLoading(int percent)
{
    emit loadModelPercent(percent);
}

void Runner::handleModelLoadSuccessed()
{
    emit loadModelStatus(true, "Success!");
}

void Runner::handleModelLoadFailed(const QString &reason)
{
    emit loadModelStatus(false, reason);
}

void Runner::handleModelUnloaded()
{
    emit resetModelStatus();
}

void Runner::handleTokenRemaining()
{
    emit botWaitting();
}

void Runner::handleTokenSampled(const QString &token)
{
    emit botTalk(token);
}

void Runner::handleTokenConsumed()
{
    emit botEnd();
}
