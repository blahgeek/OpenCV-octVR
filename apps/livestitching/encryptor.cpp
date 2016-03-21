/*
* @Author: StrayWarrior
* @Date:   2016-03-21
* @Last Modified by:   StrayWarrior
* @Last Modified time: 2016-03-21
*/

#include "./encryptor.hpp"

#ifdef OWLLIVE_ENCRYPT_ARG
#include <sodium.h>
#endif

QString Encryptor::encryptArgString(const QString & _str) {
#ifndef OWLLIVE_ENCRYPT_ARG
    return QString(_str);
#else
    // Modify the encrypt method here.
    const unsigned char secret[] = {103, 246, 81, 250, 242, 200, 201, 94, 240,
                                    238, 74, 26, 34, 3, 148, 59, 107, 95, 189,
                                    173, 111, 120, 101, 65, 74, 154, 28, 96,
                                    200, 247, 247, 52};
    if (sodium_init() == -1) {
        return QString();
    }
    QByteArray _str_bytes;
    _str_bytes.append(_str);
    const unsigned char * _str_cptr = reinterpret_cast<const unsigned char *>(_str_bytes.data());
    unsigned int _str_len = _str.length();

    unsigned char nonce[crypto_secretbox_NONCEBYTES];
    randombytes_buf(nonce, sizeof nonce);

    unsigned int cipher_len = crypto_secretbox_MACBYTES + _str_len;
    unsigned char * cipher_str = new unsigned char[cipher_len];

    if (crypto_secretbox_easy(cipher_str, _str_cptr, _str_len, nonce, secret) != 0){
        return QString();
    }

    QByteArray ret_bytes = QByteArray(reinterpret_cast<const char *>(nonce), crypto_secretbox_NONCEBYTES);
    ret_bytes.append(reinterpret_cast<const char *>(cipher_str), cipher_len);

    QString ret = ret_bytes.toBase64();
    delete cipher_str;

    return ret;
#endif
}