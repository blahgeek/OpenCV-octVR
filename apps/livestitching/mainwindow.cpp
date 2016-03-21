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
#include <QKeyEvent>

#include <cassert>

void MainWindow::onGenerateCMD() {
    auto json_doc = this->pto_template->getJsonDocument();
    QJsonObject doc_obj = json_doc.object();
    QJsonArray inputs = doc_obj["inputs"].toArray();

    auto selected_cams = this->inputs_selector->getSelected();

    if(ui->check_cheat->checkState() != Qt::Checked) {
        if(inputs.size() != selected_cams.size()) {
            QMessageBox::warning(this, "Bad Template", "Input count does not match");
            this->ui->line_cmd->setText("");
            return;
        }
        if(selected_cams.size() == 0) {
            QMessageBox::warning(this, "", "No input selected");
            this->ui->line_cmd->setText("");
            return;
        }
    }

    // BEGIN input args
    QStringList args = this->inputs_selector->getInputArgs();

    // BEGIN filter args
    QString filter_complex = QString("vr_map=");
    filter_complex.append(QString("inputs=%1:outputs=%2:crop_x=%3:crop_w=%4:blend=%5")
                          .arg(selected_cams.size())
                          .arg("vr.dat")
                          .arg(ui->inputs_crop_x->value())
                          .arg(ui->inputs_crop_w->value())
                          .arg(ui->paranoma_algorithm->currentIndex()==0?128:-5) );
    if(this->preview_video->isValid())
        filter_complex.append(QString(":preview_ow=%1:preview_oh=%2")
                              .arg(PREVIEW_WIDTH)
                              .arg(PREVIEW_HEIGHT));
    filter_complex.append(QString(":scale_ow=%1:scale_oh=%2")
                          .arg(ui->paranoma_width->value())
                          .arg(ui->paranoma_height->value()));

    // BEGIN output args
    QStringList output_args;
    int output_count = 0;

    if(this->ui->hls_enable->checkState() == Qt::Checked) {
        output_args << "-map" << QString("[o%1]").arg(output_count);
        output_args << "-c:v"  << ui->hls_codec->currentText()
                    << "-pix_fmt" << "yuv420p"
                    << "-b:v" << QString("%1M").arg(ui->hls_bitrate->value())
                    << "-g" << QString::number(ui->hls_gopsize->value())
                    << "-f" << "hls"
                    << "-hls_time" << QString::number(ui->hls_segment_time->value())
                    << "-hls_list_size" << QString::number(ui->hls_list_size->value())
                    << "-hls_flags" << "delete_segments"
                    << "-hls_allow_cache" << "0"
                    << "-y" << ui->hls_path->text();
        output_count += 1;
    }
    if(this->ui->file_enable->checkState() == Qt::Checked) {
        output_args << "-map" << QString("[o%1]").arg(output_count);
        output_args << "-c:v"  << ui->file_codec->currentText()
                    << "-pix_fmt" << "yuv420p"
                    << "-b:v" << QString("%1M").arg(ui->file_bitrate->value())
                    << "-g" << QString::number(ui->file_gopsize->value())
                    << "-y" << ui->file_path->text();
        output_count += 1;
    }
    if(this->ui->decklink_enable->checkState() == Qt::Checked) {
        output_args << "-map" << QString("[o%1]").arg(output_count);
        output_args << "-pix_fmt" << "uyvy422"
                    << "-f" << "decklink"
                    << "-vsync" << "drop"
                    << "-preroll" << "0.5"
                    << this->ui->decklink_device->text();
        output_count += 1;
    }
    if(this->ui->rtmp_enable->checkState() == Qt::Checked && 
       !this->ui->rtmp_url->text().isEmpty()) {
        output_args << "-map" << QString("[o%1]").arg(output_count);
        output_args << "-c:v"  << ui->rtmp_codec->currentText()
                    << "-pix_fmt" << "yuv420p"
                    << "-b:v" << QString("%1M").arg(ui->rtmp_bitrate->value())
                    << "-g" << QString::number(ui->rtmp_gopsize->value())
                    << "-f" << "flv"
                    << "-y" << ui->rtmp_url->text();
        output_count += 1;
    }

    if(output_count == 0) {
        output_args << "-map" << QString("[o%1]").arg(output_count);
        output_args << "-c:v" << "rawvideo" << "-f" << "null" << "-";
        output_count += 1;
    }

    filter_complex.append(QString(",split=%1").arg(output_count));
    for(int i = 0 ; i < output_count ; i += 1)
        filter_complex.append(QString("[o%1]").arg(i));

    args << "-filter_complex" << filter_complex;
    args += output_args;

    QString args_line;
    for(int i = 0 ; i < args.size() ; i += 1) {
        if(i != 0)
            args_line += " ";
        args_line += "\"" + args[i].replace("\"", "\"\"\"") + "\"";
    }
    this->ui->line_cmd->setText(args_line);
}

void MainWindow::onEncryptCMD() {
    this->ui->line_cmd_encrypted->setText(
        Encryptor::encryptArgString(this->ui->line_cmd->text())
        );
}

void MainWindow::run() {
    auto json_doc = this->pto_template->getJsonDocument();
    if(ui->check_cheat->checkState() != Qt::Checked)
        this->onGenerateCMD();
    QString args = this->ui->line_cmd->text();
    if(args.isEmpty())
        return;

    this->runner->start(json_doc, ui->paranoma_width->value(), args);
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

void MainWindow::onHLSPathSelect() {
    QString filename = QFileDialog::getSaveFileName(this);
    if(filename.size() > 0)
        this->ui->hls_path->setText(filename);
}

void MainWindow::onFilePathSelect() {
    QString filename = QFileDialog::getSaveFileName(this);
    if(filename.size() > 0)
        this->ui->file_path->setText(filename);
}

const int MainWindow::magic_key_seqs[] = {
    Qt::Key_Up, Qt::Key_Up, Qt::Key_Down, Qt::Key_Down,
    Qt::Key_Left, Qt::Key_Right, Qt::Key_Left, Qt::Key_Right,
    Qt::Key_B, Qt::Key_A,
};
const int MainWindow::magic_key_seqs_len = 10;

void MainWindow::keyPressEvent(QKeyEvent * event) {
    if(event->key() == magic_key_seqs[magic_key_current_state]) {
        magic_key_current_state += 1;
        if (magic_key_current_state >= magic_key_seqs_len) {
            qDebug() << "Magic!";
            this->ui->groupBox_magic->show();
            magic_key_current_state = 0;
        }
    }
    else {
        magic_key_current_state = 0;
    }
    QMainWindow::keyPressEvent(event);
}

void MainWindow::onCheatStateChanged(int state) {
    this->ui->line_cmd->setReadOnly(state != Qt::Checked);
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

    connect(ui->hls_path_select, &QPushButton::clicked, this, &MainWindow::onHLSPathSelect);
    connect(ui->file_path_select, &QPushButton::clicked, this, &MainWindow::onFilePathSelect);

    connect(ui->template_load, &QPushButton::clicked, this->pto_template.get(), &PTOTemplate::loadPTO);
    connect(this->pto_template.get(), &PTOTemplate::dataChanged, this, &MainWindow::onTemplateChanged);

    connect(ui->pushButton_gen_cmd, &QPushButton::clicked, this, &MainWindow::onGenerateCMD);
    connect(ui->check_cheat, &QCheckBox::stateChanged, this, &MainWindow::onCheatStateChanged);
    connect(ui->pushButton_encrypt_cmd, &QPushButton::clicked, this, &MainWindow::onEncryptCMD);

    this->onInputsSelectChanged();
    this->onTemplateChanged();
    this->onRunningStatusChanged();

    this->ui->hls_path->setText(QDir::toNativeSeparators(QDir::homePath() + "/vr.m3u8"));
    this->ui->file_path->setText(QDir::toNativeSeparators(QDir::homePath() + "/vr.ts"));

    this->ui->groupBox_magic->hide();
    this->ui->tabWidget->setCurrentIndex(0);
}

MainWindow::~MainWindow()
{
    delete ui;
}
