#include "modelsetting.h"
#include "./ui_modelsetting.h"
#include <QTime>
#include <QMessageBox>

modelsetting::modelsetting(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::modelsetting)
{
    ui->setupUi(this);
}

modelsetting::~modelsetting()
{
    delete ui;
}

void modelsetting::showEvent(QShowEvent *e)
{
    ui->seed->setValidator(new QIntValidator(-1,INT_MAX,ui->seed));
    ui->batch_size->setValidator(new QIntValidator(1,INT_MAX,ui->batch_size));
    ui->top_k->setValidator(new QIntValidator(1,INT_MAX,ui->top_k));
    ui->top_p->setValidator(new QDoubleValidator(0.0f,1.0f,3,ui->top_p));
    QDialog::showEvent(e);
}

void modelsetting::updateModelPercent(int percent)
{
    ui->progressBar_load->setValue(percent);
}

void modelsetting::updateloadStatus(bool successed, const QString &reason)
{
    if(successed)
    {
        QMessageBox::information(this, "",reason);
        ui->btn_load->setEnabled(false);
    }
    else
    {
        QMessageBox::critical(this, "",reason);
    }
}

void modelsetting::on_btn_unload_clicked()
{
    emit unloadModel();
    ui->progressBar_load->setValue(0);
    ui->btn_load->setEnabled(true);
}


void modelsetting::on_btn_load_clicked()
{
    gpt_params params;
    packModelParams(params);
    emit loadModel(params);
}

void modelsetting::packModelParams(gpt_params &params)
{
    params.model = QString("models/%1/ggml-model.bin").arg(ui->modelSize->currentText());
    params.seed = ui->seed->text().toInt();
    if(params.seed < 0)
    {
        params.seed = QTime(0,0,0).secsTo(QTime::currentTime());
    }
    params.n_predict = ui->n_predict->text().toInt();
    params.top_k = ui->top_k->text().toInt();
    params.top_p = ui->top_p->text().toFloat();
    params.repeat_last_n = ui->repeat_last_n->text().toInt();
    params.repeat_penalty = ui->repeat_penalty->text().toFloat();
    params.temp = ui->temperature->text().toFloat();
    params.n_batch = ui->batch_size->text().toInt();
}

