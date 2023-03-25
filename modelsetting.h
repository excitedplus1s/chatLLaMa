#ifndef MODELSETTING_H
#define MODELSETTING_H

#include <QDialog>
#include "gptutils.h"

namespace Ui {
class modelsetting;
}

class modelsetting : public QDialog
{
    Q_OBJECT

public:
    explicit modelsetting(QWidget *parent = nullptr);
    ~modelsetting();
protected:
    void showEvent(QShowEvent *) final;

signals:
    void loadModel(const gpt_params &params);
    void unloadModel();

public slots:
    void updateModelPercent(int percent);
    void updateloadStatus(bool successed,const QString &reason);

private slots:
    void on_btn_unload_clicked();

    void on_btn_load_clicked();

private:
    void packModelParams(gpt_params &params);

private:
    Ui::modelsetting *ui;
};

#endif // MODELSETTING_H
