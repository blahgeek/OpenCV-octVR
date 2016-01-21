#include <jni.h>

int initCL();
int processFrontFrame(long in, long out);
int processBackFrame(long in, long out);

JNIEXPORT jint JNICALL Java_org_opencv_samples_tutorial4_NativePart_processFrontFrame(JNIEnv * env,
                                                                                      jclass cls,
                                                                                      jlong pIn, jlong pOut) {
    return processFrontFrame(pIn, pOut);
}

JNIEXPORT jint JNICALL Java_org_opencv_samples_tutorial4_NativePart_processBackFrame(JNIEnv * env,
                                                                                     jclass cls,
                                                                                     jlong pIn, jlong pOut) {
    return processBackFrame(pIn, pOut);
}

JNIEXPORT jint JNICALL Java_org_opencv_samples_tutorial4_NativePart_initCL(JNIEnv * env,
                                                                           jclass cls) {
    return initCL();
}
