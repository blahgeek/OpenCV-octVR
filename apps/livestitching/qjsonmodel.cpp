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

#include "qjsonmodel.h"
#include <QFile>
#include <QDebug>
#include <QJsonDocument>
#include <QJsonObject>
#include <QIcon>
#include <QFont>
#include <QDebug>

QJsonModel::QJsonModel(QObject *parent) :
    QAbstractItemModel(parent)
{
    mRootItem = new QJsonTreeItem;
    mHeaders.append("key");
    mHeaders.append("value");

    this->setIcon(QJsonValue::Bool, QIcon(":/icons/bullet_black.png"));
    this->setIcon(QJsonValue::Double, QIcon(":/icons/bullet_blue.png"));
    this->setIcon(QJsonValue::String, QIcon(":/icons/bullet_blue.png"));
    this->setIcon(QJsonValue::Array, QIcon(":/icons/table.png"));
    this->setIcon(QJsonValue::Object, QIcon(":/icons/brick.png"));
    this->mNonEditableIcon = QIcon(":/icons/bullet_red.png");
}

void QJsonModel::setEditableFields(const QStringList & s) {
    mEditableFields = s;
}

Qt::ItemFlags QJsonModel::flags(const QModelIndex &index) const {
    if (!index.isValid())
        return 0;

    QJsonTreeItem *item = static_cast<QJsonTreeItem*>(index.internalPointer());

    Qt::ItemFlags ret = QAbstractItemModel::flags(index);
    if(index.column() == 1 && mEditableFields.indexOf(item->key()) != -1)
        ret |= Qt::ItemIsEditable;

    return ret;
}

QJsonDocument QJsonModel::document() const {
    return mDocument;
}

bool QJsonModel::load(const QString &fileName)
{
    QFile file(fileName);
    bool success = false;
    if (file.open(QIODevice::ReadOnly)) {
        success = load(&file);
        file.close();
    }
    else success = false;

    return success;
}

bool QJsonModel::load(QIODevice *device)
{
    return loadJson(device->readAll());
}

bool QJsonModel::loadJson(const QByteArray &json) {
    return this->loadDocument(QJsonDocument::fromJson(json));
}

bool QJsonModel::loadDocument(const QJsonDocument & d) {
    mDocument = d;

    if (!mDocument.isNull())
    {
        beginResetModel();
        if (mDocument.isArray()) {
            mRootItem = QJsonTreeItem::load(QJsonValue(mDocument.array()));
            mRootItem->setType(QJsonValue::Array);
        } else {
            mRootItem = QJsonTreeItem::load(QJsonValue(mDocument.object()));
            mRootItem->setType(QJsonValue::Object);
        }
        endResetModel();
        emit dataChanged(QModelIndex(), QModelIndex());
        return true;
    }
    return false;
}


QVariant QJsonModel::data(const QModelIndex &index, int role) const
{

    if (!index.isValid())
        return QVariant();


    QJsonTreeItem *item = static_cast<QJsonTreeItem*>(index.internalPointer());


    if ((role == Qt::DecorationRole) && (index.column() == 0)){
        if(item->childCount() > 0 || this->mEditableFields.indexOf(item->key()) != -1)
            return mTypeIcons.value(item->jsonValue().type());
        return mNonEditableIcon;
    }


    if (role == Qt::DisplayRole) {

        if (index.column() == 0)
            return QString("%1").arg(item->key());

        if (index.column() == 1) {
            if(item->childCount() > 0)
                return QString("[%1 item(s)]").arg(item->childCount());
            else
                return QString("%1").arg(item->stringValue());
        }
    }



    return QVariant();

}

bool QJsonModel::setData(const QModelIndex &index, const QVariant & value, int role) {
    if(role != Qt::EditRole || !index.isValid())
        return false;

    double _val;
    bool ok = false;

    QJsonTreeItem *item = static_cast<QJsonTreeItem *>(index.internalPointer());
    switch(item->jsonValue().type()) {
        case QJsonValue::Bool:
            item->setValue(value.toBool()); break;
        case QJsonValue::Double:
            _val = value.toDouble(&ok);
            if(ok) {
                item->setValue(_val);
                break;
            } else {
                return false;
            }
        case QJsonValue::String:
            item->setValue(value.toString()); break;
        default:
            return false;
    }

    // update document
    QJsonValue root = mRootItem->jsonValue();
    if(root.type() == QJsonValue::Array)
        mDocument = QJsonDocument(root.toArray());
    else
        mDocument = QJsonDocument(root.toObject());

    emit dataChanged(index, index);

    return true;
}

QVariant QJsonModel::headerData(int section, Qt::Orientation orientation, int role) const
{
    if (role != Qt::DisplayRole)
        return QVariant();

    if (orientation == Qt::Horizontal) {

        return mHeaders.value(section);
    }
    else
        return QVariant();
}

QModelIndex QJsonModel::index(int row, int column, const QModelIndex &parent) const
{
    if (!hasIndex(row, column, parent))
        return QModelIndex();

    QJsonTreeItem *parentItem;

    if (!parent.isValid())
        parentItem = mRootItem;
    else
        parentItem = static_cast<QJsonTreeItem*>(parent.internalPointer());

    QJsonTreeItem *childItem = parentItem->child(row);
    if (childItem)
        return createIndex(row, column, childItem);
    else
        return QModelIndex();
}

QModelIndex QJsonModel::parent(const QModelIndex &index) const
{
    if (!index.isValid())
        return QModelIndex();

    QJsonTreeItem *childItem = static_cast<QJsonTreeItem*>(index.internalPointer());
    QJsonTreeItem *parentItem = childItem->parent();

    if (parentItem == mRootItem)
        return QModelIndex();

    return createIndex(parentItem->row(), 0, parentItem);
}

int QJsonModel::rowCount(const QModelIndex &parent) const
{
    QJsonTreeItem *parentItem;
    if (parent.column() > 0)
        return 0;

    if (!parent.isValid())
        parentItem = mRootItem;
    else
        parentItem = static_cast<QJsonTreeItem*>(parent.internalPointer());

    return parentItem->childCount();
}

int QJsonModel::columnCount(const QModelIndex &parent) const
{
    Q_UNUSED(parent)
    return 2;
}

void QJsonModel::setIcon(const QJsonValue::Type &type, const QIcon &icon)
{
    mTypeIcons.insert(type,icon);
}
