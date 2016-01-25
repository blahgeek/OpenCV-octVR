/* 
* @Author: BlahGeek
* @Date:   2016-01-25
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-01-25
*/

#include "common.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

extern "C"
JNIEXPORT void JNICALL Java_org_opencv_android_UMatCameraViewFrame_releaseUMat(JNIEnv * env,
                                                                               jclass * cls,
                                                                               jlong p) {
    cv::UMat * m = (cv::UMat *)p;
    if(m != NULL)
        delete m;
}

extern "C"
JNIEXPORT jlong JNICALL Java_org_opencv_android_UMatCameraViewFrame_copyToUMat(JNIEnv * env,
                                                                               jclass * cls,
                                                                               jlong p,
                                                                               jbyteArray data,
                                                                               jint len,
                                                                               jint w,
                                                                               jint h,
                                                                               jint t) {
    cv::UMat * m = (cv::UMat *)p;
    if(m == NULL)
        m = new cv::UMat(h, w, t /*, cv::USAGE_ALLOCATE_SHARED_MEMORY */);
    if(data != NULL) {
        char * native_data = (char *)env->GetPrimitiveArrayCritical(data, 0);
        cv::Mat raw_data_mat(1, len, CV_8U, native_data);
        raw_data_mat.reshape(m->channels(), m->rows).copyTo(*m);
        env->ReleasePrimitiveArrayCritical(data, native_data, 0);
    }
    return (jlong)m;
}

extern "C"
JNIEXPORT void JNICALL Java_org_opencv_android_UMatCameraViewFrame_convertToRgba(JNIEnv * env,
                                                                                 jclass * cls,
                                                                                 jlong src,
                                                                                 jlong dst) {
    cv::UMat * u = (cv::UMat *) src;
    cv::Mat * m = (cv::Mat *) dst;
    if(u == NULL || m == NULL)
        return;
    cv::cvtColor(*u, *m, cv::COLOR_YUV2RGBA_NV21, 4);
}

extern "C"
JNIEXPORT void JNICALL Java_org_opencv_android_UMatCameraViewFrame_convertToGray(JNIEnv * env,
                                                                                 jclass * cls,
                                                                                 jlong src,
                                                                                 jlong dst) {
    // TODO
}
