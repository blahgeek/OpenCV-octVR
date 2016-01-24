#ifndef JSONITEM_H
#define JSONITEM_H
#include <QtCore>
#include <QJsonValue>
#include <QJsonArray>
#include <QJsonObject>
class QJsonTreeItem
{
public:
    QJsonTreeItem(QJsonTreeItem * parent = 0);
    ~QJsonTreeItem();
    void appendChild(QJsonTreeItem * item);
    QJsonTreeItem *child(int row);
    QJsonTreeItem *parent();
    int childCount() const;
    int row() const;
    void setKey(const QString& key);
    void setValue(const QJsonValue& value);
    // void setValue(const QString& value);
    void setType(const QJsonValue::Type& type);
    QString key() const;
    QString stringValue() const;
    // QJsonValue::Type type() const;
    QJsonValue jsonValue() const;


    static QJsonTreeItem* load(const QJsonValue& value, QJsonTreeItem * parent = 0);

protected:


private:
    QString mKey;
    QJsonValue mJValue;
    // QString mValue;
    QJsonValue::Type mType; // only for array and object

    QList<QJsonTreeItem*> mChilds;
    QJsonTreeItem * mParent;


};

#endif // JSONITEM_H
