#include <jni.h>

int initCL();
int processFrontFrame(int texIn, int texOut, int width, int height);
int processBackFrame(int texIn, int texOut, int width, int height);

JNIEXPORT jint JNICALL Java_org_opencv_samples_tutorial4_NativePart_initCL(JNIEnv * env, jclass cls)
{
    return initCL();
}

JNIEXPORT jint JNICALL Java_org_opencv_samples_tutorial4_NativePart_processFrontFrame(JNIEnv * env, jclass cls,
                                                                                       jint texIn, jint texOut,
                                                                                       jint width, jint height) {
    return processFrontFrame(texIn, texOut, width, height);
}

JNIEXPORT jint JNICALL Java_org_opencv_samples_tutorial4_NativePart_processBackFrame(JNIEnv * env, jclass cls,
                                                                                      jint texIn, jint texOut,
                                                                                      jint width, jint height) {
    return processBackFrame(texIn, texOut, width, height);
}
