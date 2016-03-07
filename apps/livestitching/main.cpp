#include "mainwindow.h"
#include <QApplication>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#ifdef _WIN32
#include <windows.h>
int WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, char*, int nShowCmd) {
    QApplication a(0, 0);
#else
int main(int argc, char *argv[]) {
    QApplication a(argc, argv);
#endif
    MainWindow w;
    w.show();

    return a.exec();
}
