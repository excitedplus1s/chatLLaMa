#include "mainwindow.h"
#include "./ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->sendMessageButton->setEnabled(false);
    runner = new Runner(this);
    connect(runner, &Runner::botTalk, this, &MainWindow::set_label);
    connect(runner, &Runner::botWaitting, this, &MainWindow::disableSendMessageButton);
    connect(runner, &Runner::botEnd, this, &MainWindow::enableSendMessageButton);
    dia = new modelsetting(this);
    connect(dia, &modelsetting::loadModel, runner, &Runner::loadModel);
    connect(dia, &modelsetting::unloadModel, runner, &Runner::unloadModel);
    connect(dia, &modelsetting::unloadModel, this, &MainWindow::disableSendMessageButton);
    connect(runner, &Runner::loadModelPercent, dia, &modelsetting::updateModelPercent);
    connect(runner, &Runner::loadModelStatus, dia, &modelsetting::updateloadStatus);
    connect(runner, &Runner::loadModelStatus, [this](bool successed, const QString){if(successed) this->enableSendMessageButton();});
}

MainWindow::~MainWindow()
{
    delete ui;
    delete runner;
    delete dia;
}


void MainWindow::on_action_LoadModel_triggered()
{
    dia->show();
}


void MainWindow::on_sendMessageButton_clicked()
{
    auto sendbuf = ui->sendMessageTextEdit->toPlainText();
    emit runner->sendMessage(sendbuf);
    ui->sendMessageTextEdit->clear();
    set_label(QString("\nUser: %1\nChatLLaMa:").arg(sendbuf));
}

void MainWindow::set_label(const QString &token)
{
    QString txt = ui->chatMeaasge->toPlainText() + token;
    ui->chatMeaasge->setPlainText(txt);
    ui->chatMeaasge->moveCursor(QTextCursor::End);
}

void MainWindow::disableSendMessageButton()
{
    ui->sendMessageButton->setEnabled(false);
}

void MainWindow::enableSendMessageButton()
{
    ui->sendMessageButton->setEnabled(true);
}

