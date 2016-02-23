#if defined(_WIN32)
#define Q_COMPILER_INITIALIZER_LISTS value
#endif
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QMessageBox>
#include "qfile.h"
#include <QTextStream>
#include <QProcess>
#include <QDebug>
#include <QRegExp>
#include <QLineEdit>
#include <QCameraInfo>
#include <QProcess>
#include <QDebug>

#include <cassert>

void MainWindow::run() {
    auto json_doc = this->pto_template->getJsonDocument();
    QJsonObject doc_obj = json_doc.object();
    QJsonArray inputs = doc_obj["inputs"].toArray();

    auto selected_cams = this->inputs_selector->getSelected();

    if(inputs.size() != selected_cams.size()) {
        QMessageBox::warning(this, "Bad Template", "Input count does not match");
        return;
    }
    if(selected_cams.size() == 0) {
        QMessageBox::warning(this, "", "No input selected");
        return;
    }
    if(this->ui->hls_enable->checkState() != Qt::Checked &&
       this->ui->file_enable->checkState() != Qt::Checked &&
       this->ui->decklink_enable->checkState() != Qt::Checked) {
        QMessageBox::warning(this, "", "No output selected");
        return;
    }

    QString output_json_path = temp_dir.path() + "/vr.json";
    QFile output_json(output_json_path);
    output_json.open(QIODevice::WriteOnly);
    output_json.write(json_doc.toJson());
    output_json.close();

    QStringList dumper_args;
    dumper_args << "-w" << QString::number(ui->paranoma_width->value())
                << "-o" << (temp_dir.path() + "/vr.dat")
                << output_json_path;
    qDebug() << "Running dumper: " << dumper_args;

    this->onRunningStatusChanged(DUMPER_RUNNING);

    QProcess dumper_proc;
    dumper_proc.start("/home/blahgeek/dumper", dumper_args); // FIXME
    bool finished = dumper_proc.waitForFinished();
    if(!(finished && dumper_proc.exitStatus() == QProcess::NormalExit && dumper_proc.exitCode() == 0)) {
        QMessageBox::warning(nullptr, "", "Unable to create dat file");
        this->onRunningStatusChanged(NOT_RUNNING);
        return;
    }

    QStringList args;
    for(auto & input: selected_cams)
        args << "-f" << "v4l2" << "-input_format" << "uyvy422"
             << "-framerate" << "30" << "-i" << input.deviceName();
    args << "-filter_complex"
         << QString("vr_map=inputs=%1:outputs=%2:crop_x=%3:crop_w=%4:blend=%5")
            .arg(selected_cams.size())
            .arg(temp_dir.path() + "/vr.dat")
            .arg(ui->inputs_crop_x->value())
            .arg(ui->inputs_crop_w->value())
            .arg(ui->paranoma_algorithm->currentIndex() == 0 ? 128 : -5);
    args << "-c:v" << ui->encoding_codec->currentText()
         << "-b:v" << QString("%1M").arg(ui->encoding_bitrate->value())
         << "-g" << QString::number(ui->encoding_gopsize->value());

    QString tee_output;
    if(this->ui->hls_enable->checkState() == Qt::Checked) {
        if(tee_output.size() == 0)
            tee_output.append("|");
        tee_output.append(QString("[f=hls:hls_time=%1:hls_list_size=%2:hls_flags=delete_segments:hls_allow_cache=0]%3")
                          .arg(ui->hls_segment_time->value())
                          .arg(ui->hls_list_size->value())
                          .arg(ui->hls_path->text()));
    }
    if(this->ui->file_enable->checkState() == Qt::Checked) {
        if(tee_output.size() == 0)
            tee_output.append("|");
        tee_output.append(ui->file_path->text());
    }
    if(this->ui->decklink_enable->checkState() == Qt::Checked) {
        if(tee_output.size() == 0)
            tee_output.append("|");
        tee_output.append(QString("[f=decklink]%1")
                          .arg(ui->decklink_device->currentText()));
    }

    if(tee_output.size() > 0)
        tee_output.remove(0, 1);

    args << "-f" << "tee" << "-y" << tee_output;
    qDebug() << "Running ffmpeg: " << args;

    this->onRunningStatusChanged(FFMPEG_RUNNING);
    this->ffmpeg_proc.start("/home/blahgeek/ffmpeg", args); // FIXME
}

void MainWindow::onRunningStatusChanged(enum RunningStatus status) {
    switch(status){
        case NOT_RUNNING:
            this->ui->pushButton_run->setEnabled(true);
            this->ui->pushButton_stop->setEnabled(false);
            this->ui->running_status->setText("Not running");
            break;
        case DUMPER_RUNNING:
            this->ui->pushButton_run->setEnabled(false);
            this->ui->pushButton_stop->setEnabled(false);
            this->ui->running_status->setText("Preparing...");
            break;
        case FFMPEG_RUNNING:
            this->ui->pushButton_run->setEnabled(false);
            this->ui->pushButton_stop->setEnabled(true);
            this->ui->running_status->setText("Running...");
            break;
        default:
            break;
    }
}

void MainWindow::onFfmpegStateChanged(QProcess::ProcessState state) {
    if(state == QProcess::NotRunning)
        this->onRunningStatusChanged(NOT_RUNNING);
    else
        this->onRunningStatusChanged(FFMPEG_RUNNING);
}

void MainWindow::onTabChanged(int index) {
    qDebug() << "onTabChanged: " << index;
    if(index == 0)
        this->inputs_selector->start();
    else {
        this->inputs_selector->stop();
        auto selected = this->inputs_selector->getSelected();
        qDebug() << "selected count: " << selected.size();
    }
}

void MainWindow::onInputsSelectChanged() {
    auto selected = this->inputs_selector->getSelected();
    auto all = this->inputs_selector->getAll();
    this->ui->inputs_info->setText(QString("%1 found, %2 selected")
                                   .arg(all.size())
                                   .arg(selected.size()));
    this->ui->inputs_action_save->setEnabled(selected.size() > 0);
}

void MainWindow::onInputSaveButtonClicked() {
    this->inputs_selector->saveImages(this->ui->inputs_crop_x->value(),
                                      this->ui->inputs_crop_w->value());
}

void MainWindow::onTemplateChanged() {
    auto json_doc = this->pto_template->getJsonDocument();
    QJsonObject doc_obj = json_doc.object();
    QJsonArray inputs = doc_obj["inputs"].toArray();
    this->ui->template_info->setText(QString("%1 inputs").arg(inputs.size()));
}

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    inputs_selector.reset(new InputsSelector(ui->inputs_grid));
    pto_template.reset(new PTOTemplate(ui->template_tree_view));
    preview_video.reset(new PreviewVideoWidget(this));
    this->ui->preview_layout->addWidget(preview_video.get());

    preview_timer.setInterval(100);
    preview_timer.start();
    connect(&preview_timer, &QTimer::timeout, preview_video.get(), &PreviewVideoWidget::updatePreview);

    {
        QFile f(":qdarkstyle/style.qss");
        Q_ASSERT(f.exists());
        f.open(QFile::ReadOnly | QFile::Text);
        QTextStream ts(&f);
        qApp->setStyleSheet(ts.readAll());
    }

    connect(ui->pushButton_run, &QPushButton::clicked, this, &MainWindow::run);
    connect(ui->pushButton_stop, &QPushButton::clicked, &this->ffmpeg_proc, &QProcess::terminate);
    connect(&this->ffmpeg_proc, &QProcess::stateChanged, this, &MainWindow::onFfmpegStateChanged);

    connect(ui->tabWidget, &QTabWidget::currentChanged, this, &MainWindow::onTabChanged);
    connect(ui->inputs_action_save, &QPushButton::clicked, this, &MainWindow::onInputSaveButtonClicked);
    connect(this->inputs_selector.get(), &InputsSelector::selectedChanged, this, &MainWindow::onInputsSelectChanged);

    connect(ui->template_load, &QPushButton::clicked, this->pto_template.get(), &PTOTemplate::loadPTO);
    connect(this->pto_template.get(), &PTOTemplate::dataChanged, this, &MainWindow::onTemplateChanged);

    this->onInputsSelectChanged();
    this->onTemplateChanged();
    this->onRunningStatusChanged(NOT_RUNNING);

    this->ui->tabWidget->setCurrentIndex(0);

    assert(temp_dir.isValid());
    qDebug() << "Temp dir: " << temp_dir.path();
}

MainWindow::~MainWindow()
{
    delete ui;
}
