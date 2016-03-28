/*
* @Author: StrayWarrior
* @Date:   2016-03-21
* @Last Modified by:   StrayWarrior
* @Last Modified time: 2016-03-21
*/

#ifndef LIVESTITCHING_ENCRYPTOR_H__
#define LIVESTITCHING_ENCRYPTOR_H__ value_

#include <QObject>
#include <QByteArray>
#include <QString>

class Encryptor : public QObject {
    Q_OBJECT

public:
    static QString encryptArgString(const QString & _str);

};

#endif /* LIVESTITCHING_ENCRYPTOR_H__ */