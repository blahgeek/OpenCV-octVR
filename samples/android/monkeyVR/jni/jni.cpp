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
                                                                 jint bitrate,
                                                                 jstring outfile_path,
                                                                 jstring remote_addr,
                                                                 jint remote_port,
                                                                 jboolean ifStitch,
                                                                 jboolean ifSocket) {
    const char * out;
    //sprintf(out, "Hello from JNI!");
    out = (env)->GetStringUTFChars(outfile_path, nullptr);
    return (env)->NewStringUTF(out);
}
