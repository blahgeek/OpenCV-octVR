/*
* @Author: BlahGeek
* @Date:   2016-01-21
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-01-25
*/

#include "./monkey.hpp"

extern "C"
JNIEXPORT void JNICALL Java_loft_monkeyVR_NativeBridge_onStart(JNIEnv * env,
                                                               jclass cls,
                                                               jint index,
                                                               jint width,
                                                               jint height) {
    MonkeyVR::getInstance()->onStart(index, width, height);
}

extern "C"
JNIEXPORT void JNICALL Java_loft_monkeyVR_NativeBridge_onStop(JNIEnv * env,
                                                              jclass cls,
                                                              jint index) {
    MonkeyVR::getInstance()->onStop(index);
}

extern "C"
JNIEXPORT jint JNICALL Java_loft_monkeyVR_NativeBridge_onFrame(JNIEnv * env,
                                                               jclass cls,
                                                               jint index,
                                                               jlong pIn,
                                                               jlong pOut) {
    cv::UMat * in = (cv::UMat *)pIn;
    cv::Mat * out = (cv::Mat *)pOut;
    return MonkeyVR::getInstance()->onFrame(index, in, out);
}

extern "C"
JNIEXPORT jstring JNICALL Java_loft_monkeyVR_NativeBridge_setParams(JNIEnv * env,
                                                                 jclass cls,
                                                                 jint _bitrate,
                                                                 jstring _outfile_path,
                                                                 jstring _remote_addr,
                                                                 jint _remote_port,
                                                                 jboolean _ifStitch,
                                                                 jboolean _ifSocket) {
    const char * outfile_path;
    const char * remote_addr;
    outfile_path = (env)->GetStringUTFChars(_outfile_path, nullptr);
    remote_addr = (env)->GetStringUTFChars(_remote_addr, nullptr);
    MonkeyVR::getInstance()->setParams(_bitrate, outfile_path, remote_addr, _remote_port, _ifStitch, _ifSocket);
    return (env)->NewStringUTF(MonkeyVR::getInstance()->printParams().c_str());
}
