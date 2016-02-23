/* 
* @Author: BlahGeek
* @Date:   2016-02-23
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-02-23
*/

#ifndef LIVESTITCHING_RUNNER_H__
#define LIVESTITCHING_RUNNER_H__ value

#include <QProcess>
#include <QJsonDocument>

class Runner : public QObject {
    Q_OBJECT

private:
    QProcess dumper_proc, ffmpeg_proc;
    QStringList dumper_args, ffmpeg_args;

public:
    enum RunningStatus { NOT_RUNNING, DUMPER_RUNNING, FFMPEG_RUNNING };

    enum RunningStatus status() const;
    void start(QStringList dumper_args, QStringList ffmpeg_args);

    Runner();

public slots:
    void onDumperProcessFinished(int exitCode, QProcess::ExitStatus status);
    void onFfmpegProcessFinished(int exitCode, QProcess::ExitStatus status);
    void stop();

signals:
    void statusChanged();

};

#endif
