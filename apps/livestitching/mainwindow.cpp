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

    QStringList args;
    for(auto & input: selected_cams)
        args << "-f" << "v4l2" << "-input_format" << "uyvy422"
             << "-framerate" << "30" << "-i" << input.deviceName();

    QString filter_complex = QString("vr_map=");
    filter_complex.append(QString("inputs=%1:outputs=%2:crop_x=%3:crop_w=%4")
                          .arg(selected_cams.size())
                          .arg(temp_dir.path() + "/vr.dat")
                          .arg(ui->inputs_crop_x->value())
                          .arg(ui->inputs_crop_w->value()) );
    filter_complex.append(QString(":blend=%1:preview_ow=%2:preview_oh=%3")
                          .arg(ui->paranoma_algorithm->currentIndex() == 0 ? 128 : -5)
                          .arg(PREVIEW_WIDTH)
                          .arg(PREVIEW_HEIGHT));
    filter_complex.append(QString(":scale_ow=%1:scale_oh=%2")
                          .arg(ui->paranoma_width->value())
                          .arg(ui->paranoma_height->value()));

    if(this->ui->decklink_enable->checkState() == Qt::Checked)
        filter_complex.append(QString(",split=2[o0][o1]"));

    args << "-filter_complex" << filter_complex;

    if(this->ui->decklink_enable->checkState() == Qt::Checked)
        args << "-map" << "[o0]";
    args << "-c:v" << ui->encoding_codec->currentText()
         << "-pix_fmt" << "yuv420p"
         << "-b:v" << QString("%1M").arg(ui->encoding_bitrate->value())
         << "-g" << QString::number(ui->encoding_gopsize->value());

    QString tee_output("[f=null]-");
    if(this->ui->hls_enable->checkState() == Qt::Checked) {
        tee_output.append(QString("|[f=hls:hls_time=%1:hls_list_size=%2:hls_flags=delete_segments:hls_allow_cache=0]%3")
                          .arg(ui->hls_segment_time->value())
                          .arg(ui->hls_list_size->value())
                          .arg(ui->hls_path->text()));
    }
    if(this->ui->file_enable->checkState() == Qt::Checked) {
        tee_output.append("|");
        tee_output.append(ui->file_path->text());
    }

    args << "-f" << "tee" << "-y" << tee_output;

    if(this->ui->decklink_enable->checkState() == Qt::Checked)
        args << "-map" << "[o1]" 
             << "-pix_fmt" << "uyvy422"
             << "-preroll" << "0.5" << "-vsync" << "drop"
             << "-f" << "decklink" << this->ui->decklink_device->text();

    this->runner->start(dumper_args, args);
}

void MainWindow::onRunningStatusChanged() {
    auto status = this->runner->status();
    switch(status){
        case Runner::NOT_RUNNING:
            this->ui->pushButton_run->setEnabled(true);
            this->ui->pushButton_stop->setEnabled(false);
            this->ui->running_status->setText("Not running");
            break;
        case Runner::DUMPER_RUNNING:
            this->ui->pushButton_run->setEnabled(false);
            this->ui->pushButton_stop->setEnabled(false);
            this->ui->running_status->setText("Preparing...");
            break;
        case Runner::FFMPEG_RUNNING:
            this->ui->pushButton_run->setEnabled(false);
            this->ui->pushButton_stop->setEnabled(true);
            this->ui->running_status->setText("Running...");
            break;
        default:
            break;
    }
}

void MainWindow::onTabChanged(int index) {
    qDebug() << "onTabChanged: " << index;
    if(index == 0 && this->runner->status() == Runner::NOT_RUNNING)
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
    preview_video.reset(new PreviewVideoWidget(this, PREVIEW_WIDTH, PREVIEW_HEIGHT));
    runner.reset(new Runner());
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
    connect(ui->pushButton_stop, &QPushButton::clicked, this->runner.get(), &Runner::stop);
    connect(this->runner.get(), &Runner::statusChanged, this, &MainWindow::onRunningStatusChanged);

    connect(ui->tabWidget, &QTabWidget::currentChanged, this, &MainWindow::onTabChanged);
    connect(ui->inputs_action_save, &QPushButton::clicked, this, &MainWindow::onInputSaveButtonClicked);
    connect(this->inputs_selector.get(), &InputsSelector::selectedChanged, this, &MainWindow::onInputsSelectChanged);

    connect(ui->template_load, &QPushButton::clicked, this->pto_template.get(), &PTOTemplate::loadPTO);
    connect(this->pto_template.get(), &PTOTemplate::dataChanged, this, &MainWindow::onTemplateChanged);

    this->onInputsSelectChanged();
    this->onTemplateChanged();
    this->onRunningStatusChanged();

    this->ui->tabWidget->setCurrentIndex(0);

    assert(temp_dir.isValid());
    qDebug() << "Temp dir: " << temp_dir.path();
}

MainWindow::~MainWindow()
{
    delete ui;
}
