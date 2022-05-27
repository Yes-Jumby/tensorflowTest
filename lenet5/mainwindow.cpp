#include "mainwindow.h"
#include "ui_mainwindow.h"
//D:\Qt\5.14.0\msvc2017_64\bin > .\windeployqt.exe D : \Demo\MyDemoWindows\tensorflowTest\release\tensorflowTest.exe
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    m_strclass_name = {"0", "1", "2","3", "4","5", "6", "7", "8", "9"};


    string path = "D://Demo//MyDemoWindows//tensorflowTest//data//lenet5//frozen_graph_lenet5.pb";
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
    QFileDialog_.setDirectory("D://Demo//MyDemoWindows//tensorflowTest//data//lenet5");
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
       cv::Mat frame = imread(filename.toStdString(),0);
       //设定结构元素的大小与滑动块系统相联系
       int s = 6;
       //创建结构元素
       Mat structureElenent = getStructuringElement(MORPH_RECT,Size(s,s));
       //膨胀操作
	   cv::Mat frame_dilate;
	   erode(frame, frame_dilate,structureElenent);
       cv::Mat frame_32F;
	   frame_dilate.convertTo(frame_32F,CV_32FC1);
       cv::Mat blob = blobFromImage(1-frame_32F/255.0,
                                1.0,
                                Size(28,28),
                                Scalar(0,0,0));
       cout<<(blob.channels())<<endl;
       cout<<(blob.size())<<endl;
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
                                          arg(std::stoi(m_strclass_name[maxclass.x])));
       ui->plainTextEdit_2->appendPlainText(QString("0:%1   1:%2   2:%3   3:%4   4:%5   5:%6   6:%7   7:%8   8:%9   9:%10		sum:%11").
                                            arg(pout[0]).arg(pout[1]).arg(pout[2]).arg(pout[3]).arg(pout[4]).
                                            arg(pout[5]).arg(pout[6]).arg(pout[7]).arg(pout[8]).arg(pout[9]).arg(fsum));
	   //display
	   QImage qimg(frame_dilate.data, frame_dilate.cols, frame_dilate.rows, static_cast<int>(frame_dilate.step), QImage::Format_Grayscale8);
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
