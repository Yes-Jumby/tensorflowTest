#include "mainwindow.h"
#include "ui_mainwindow.h"
//D:\Qt\5.14.0\msvc2017_64\bin > .\windeployqt.exe D : \Demo\MyDemoWindows\tensorflowTest\release\tensorflowTest.exe
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    m_strclass_name = {"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};


    string path = "D://Demo//MyDemoWindows//tensorflowTest//data//alex_net//frozen_graph_alex_net.pb";
    m_net = readNetFromTensorflow(path);
    //m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    //m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);//调用GPU
    if(m_net.empty())
    {
        qDebug()<<"load model failed!";
    }
    else
    {
        qDebug()<<"load model success!";
    }
}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_pushButton_clicked()
{
    ui->plainTextEdit->clear();
    ui->plainTextEdit_2->clear();
    QString filename;
    QFileDialog QFileDialog_;
    QFileDialog_.setDirectory("D://Demo//MyDemoWindows//tensorflowTest//data//alex_net");
    filename = QFileDialog_.getOpenFileName(this,
                                         tr("选择图像"),
                                         "",
                                         tr("Images (*.png *.bmp *.jpg *.tif *.GIF )"));
    if(filename.isEmpty())
    {
        ui->plainTextEdit->appendPlainText(QString("null pic!"));
        return;
    }
    else
    {

       //forward
       cv::Mat frame = imread(filename.toStdString(),IMREAD_UNCHANGED);
       cv::cvtColor(frame, frame, COLOR_BGRA2RGB);
       cv::Mat frame_32F;
       cout<<(frame.channels())<<endl;
       cout<<(frame.size())<<endl;
       frame.convertTo(frame_32F,CV_32FC3);
       cv::Mat blob = blobFromImage(frame_32F/255.0,
                                1.0,
                                Size(227,227),
                                Scalar(0,0,0));
       cout<<(blob.channels())<<endl;
       //cout<<(blob.size())<<endl;
       m_net.setInput(blob);
       cv::Mat out = m_net.forward();
       cout<<out.cols<<endl;
       cout<<out.rows<<endl;
       cv::Point maxclass;
       cout <<"result:"<<out<<endl;
       minMaxLoc(out, nullptr, nullptr, nullptr, &maxclass);
       float* pout = reinterpret_cast<float*>(out.data);
	   float fsum = 0.0f;
	   for (int i = 0; i < 10; i++)
	   {
		   fsum += pout[i];
	   }
	   ui->plainTextEdit->appendPlainText(QString("this pic is :%1").
                                          arg(QString::fromStdString(m_strclass_name[maxclass.x])));
       ui->plainTextEdit_2->appendPlainText(QString("0:%1   1:%2   2:%3   3:%4   4:%5   5:%6   6:%7   7:%8   8:%9   9:%10		sum:%11").
                                            arg(pout[0]).arg(pout[1]).arg(pout[2]).arg(pout[3]).arg(pout[4]).
                                            arg(pout[5]).arg(pout[6]).arg(pout[7]).arg(pout[8]).arg(pout[9]).arg(fsum));
       //display
       QImage qimg(frame.data, frame.cols, frame.rows, static_cast<int>(frame.step), QImage::Format_RGB888);
       if (qimg.isNull()) //加载图像
       {
           ui->plainTextEdit->appendPlainText(QString("load pic failed!"));
           return;
       }
       cout << qimg.width() << endl;
       cout << qimg.height() << endl;
       qimg = qimg.scaled(ui->label->width(), ui->label->height(), Qt::IgnoreAspectRatio);
       cout << qimg.width() << endl;
       cout << qimg.height() << endl;
       ui->label->setPixmap(QPixmap::fromImage(qimg));
    }
}
