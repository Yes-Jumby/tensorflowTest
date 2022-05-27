#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <QtDebug>
#include <QDialog>
#include <QMessageBox>
#include <QFileDialog>

using namespace cv;
using namespace std;
using namespace cv::dnn;

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
    void on_pushButton_clicked();

private:
    Ui::MainWindow *ui;
    std::vector<string> m_strclass_name;
    cv::dnn::Net m_net;
};
#endif // MAINWINDOW_H
