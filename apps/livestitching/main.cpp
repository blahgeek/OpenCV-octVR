#include "mainwindow.h"
#include <QApplication>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#if defined(_WIN32) && defined(OWLLIVE_DISABLE_CONSOLE)
#include <windows.h>
int WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, char*, int nShowCmd) {
    int argc = 0;
    QApplication a(argc, 0);
#else
int main(int argc, char *argv[]) {
    QApplication a(argc, argv);
#endif
    MainWindow w;
    w.show();

    return a.exec();
}
