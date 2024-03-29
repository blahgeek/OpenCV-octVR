/***********************************************
    Copyright (C) 2014  Schutz Sacha
    This file is part of QJsonModel (https://github.com/dridk/QJsonmodel).

    QJsonModel is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    QJsonModel is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with QJsonModel.  If not, see <http://www.gnu.org/licenses/>.

**********************************************/

#include "qjsonitem.h"

QJsonTreeItem::QJsonTreeItem(QJsonTreeItem *parent)
{

    mParent = parent;


}

QJsonTreeItem::~QJsonTreeItem()
{
    qDeleteAll(mChilds);

}

void QJsonTreeItem::appendChild(QJsonTreeItem *item)
{
    mChilds.append(item);
}

QJsonTreeItem *QJsonTreeItem::child(int row)
{
    return mChilds.value(row);
}

QJsonTreeItem *QJsonTreeItem::parent()
{
    return mParent;
}

int QJsonTreeItem::childCount() const
{
    return mChilds.count();
}

int QJsonTreeItem::row() const
{
    if (mParent)
        return mParent->mChilds.indexOf(const_cast<QJsonTreeItem*>(this));

    return 0;
}

void QJsonTreeItem::setKey(const QString &key)
{
    mKey = key;
}

void QJsonTreeItem::setValue(const QJsonValue &value) {
    mJValue = value;
}

// void QJsonTreeItem::setValue(const QString &value)
// {
//     mValue = value;
// }

void QJsonTreeItem::setType(const QJsonValue::Type &type)
{
    mType = type;
}

QString QJsonTreeItem::key() const
{
    return mKey;
}

QString QJsonTreeItem::stringValue() const {
    return mJValue.toVariant().toString();
}

// QString QJsonTreeItem::value() const
// {
//     return mValue;
// }

// QJsonValue::Type QJsonTreeItem::type() const
// {
//     return mJValue.type();
// }

QJsonTreeItem* QJsonTreeItem::load(const QJsonValue& value, QJsonTreeItem* parent)
{


    QJsonTreeItem * rootItem = new QJsonTreeItem(parent);
    rootItem->setKey("root");

    if ( value.isObject())
    {

        //Get all QJsonValue childs
        foreach (QString key , value.toObject().keys()){
            QJsonValue v = value.toObject().value(key);
            QJsonTreeItem * child = load(v,rootItem);
            child->setKey(key);
            child->setType(v.type());
            rootItem->appendChild(child);

        }

    }

    else if ( value.isArray())
    {
        //Get all QJsonValue childs
        int index = 0;
        foreach (QJsonValue v , value.toArray()){

            QJsonTreeItem * child = load(v,rootItem);
            child->setKey(QString::number(index));
            child->setType(v.type());
            rootItem->appendChild(child);
            ++index;
        }
    }
    else
    {
        rootItem->setValue(value);
        // rootItem->setValue(value.toVariant().toString());
        // rootItem->setType(value.type());
    }

    return rootItem;
}

QJsonValue QJsonTreeItem::jsonValue() const {
    if(this->mType == QJsonValue::Object) {
        QJsonObject ret;
        foreach(QJsonTreeItem * item, this->mChilds) {
            ret.insert(item->key(), item->jsonValue());
        }
        return ret;
    }
    if(this->mType == QJsonValue::Array) {
        QJsonArray ret;
        foreach(QJsonTreeItem * item, this->mChilds) {
            ret.append(item->jsonValue());
        }
        return ret;
    }
    return this->mJValue;
}

