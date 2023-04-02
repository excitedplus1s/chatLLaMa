#include <QFile>
#include "processor.h"

class Processor::InternalData
{
public:
    bool chat_init_status = false;
    session_env_t env;
};

Processor::Processor(QObject *parent) : QObject(parent)
{
    Q_UNUSED(parent)
    qRegisterMetaType<gpt_params>();
    m_data = QSharedPointer<InternalData>(new InternalData());
}

void Processor::handleLoadModel(const gpt_params &params)
{
    bool success = ::load_model(&m_data->env,params,updateLoadProgress,this);
    if(success)
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
    ::unload_model(&m_data->env);
    emit modelUnloaded();
}

void Processor::handleEvalToken(const QString &prompt)
{
    emit tokenRemaining();
    if(!m_data->chat_init_status)
    {
        ::init_chat_env(&m_data->env);
        m_data->chat_init_status=true;
    }
    ::init_user_input(&m_data->env, prompt);
    while(::should_generate(&m_data->env))
    {
        QString result = ::generate_token(&m_data->env);
        emit tokenSampled(result);
    }
    emit tokenConsumed();
}

void Processor::updateLoadProgress(float progress, void *ctx)
{
    Processor* ctx_ = (Processor*)ctx;
    emit ctx_->modelLoading(100*progress);
}
