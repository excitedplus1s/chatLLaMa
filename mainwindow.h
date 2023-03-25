#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "runner.h"
#include "modelsetting.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_action_LoadModel_triggered();

    void on_sendMessageButton_clicked();

    void set_label(const QString &token);

    void disableSendMessageButton();

    void enableSendMessageButton();

private:
    Ui::MainWindow *ui;
    Runner *runner;
    modelsetting *dia;
};
#endif // MAINWINDOW_H
