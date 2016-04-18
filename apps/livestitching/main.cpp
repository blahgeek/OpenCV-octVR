#include "mainwindow.h"
#include <QApplication>
#include <QSplashScreen>
#include <QPixmap>
#include <QPainter>
#include <QTime>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define OWLLIVE_VERSION "0.4.2"

class OwlSpashScreen: public QSplashScreen {
public:
    OwlSpashScreen(): QSplashScreen(QPixmap(":/icons/splash.png")) {}
    void drawContents(QPainter * painter) override {
        painter->setPen(QPen(QColor(255, 255, 255)));
        painter->drawText(288, 150, "Version: " OWLLIVE_VERSION " Starting...");
    }
};

#if defined(_WIN32) && defined(OWLLIVE_DISABLE_CONSOLE)
#include <windows.h>
int WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, char*, int nShowCmd) {
    int argc = 0;
    QApplication a(argc, 0);
#else
int main(int argc, char *argv[]) {
    QApplication a(argc, argv);
#endif

    OwlSpashScreen splash;
    splash.show();
    a.processEvents();

    QTime t;
    t.start();

    MainWindow w;

    while(t.elapsed() < 3000) {
        a.processEvents();
        QThread::msleep(5);
    }

    w.show();
    splash.finish(&w);

    return a.exec();
}
