#define __CL_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS /*let's give a chance for OpenCL 1.1 devices*/

#include <GLES2/gl2.h>
#include <EGL/egl.h>

#include <CL/cl.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/ocl.hpp>

#include <mutex>

#include "common.hpp"

void dumpCLinfo()
{
    LOGD("*** OpenCL info ***");
    try
    {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        LOGD("OpenCL info: Found %d OpenCL platforms", platforms.size());
        for (int i = 0; i < platforms.size(); ++i)
        {
            std::string name = platforms[i].getInfo<CL_PLATFORM_NAME>();
            std::string version = platforms[i].getInfo<CL_PLATFORM_VERSION>();
            std::string profile = platforms[i].getInfo<CL_PLATFORM_PROFILE>();
            std::string extensions = platforms[i].getInfo<CL_PLATFORM_EXTENSIONS>();
            LOGD( "OpenCL info: Platform[%d] = %s, ver = %s, prof = %s, ext = %s",
                  i, name.c_str(), version.c_str(), profile.c_str(), extensions.c_str() );
        }

        std::vector<cl::Device> devices;
        platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);

        for (int i = 0; i < devices.size(); ++i)
        {
            std::string name = devices[i].getInfo<CL_DEVICE_NAME>();
            std::string extensions = devices[i].getInfo<CL_DEVICE_EXTENSIONS>();
            cl_ulong type = devices[i].getInfo<CL_DEVICE_TYPE>();
            LOGD( "OpenCL info: Device[%d] = %s (%s), ext = %s",
                  i, name.c_str(), (type==CL_DEVICE_TYPE_GPU ? "GPU" : "CPU"), extensions.c_str() );
        }
    }
    catch(cl::Error& e)
    {
        LOGE( "OpenCL info: error while gathering OpenCL info: %s (%d)", e.what(), e.err() );
    }
    catch(std::exception& e)
    {
        LOGE( "OpenCL info: error while gathering OpenCL info: %s", e.what() );
    }
    catch(...)
    {
        LOGE( "OpenCL info: unknown error while gathering OpenCL info" );
    }
    LOGD("*******************");
}

cl::Context theContext;
cl::CommandQueue theQueue;
bool haveOpenCL = false;
std::mutex mutex;

extern "C" void initCL() {
    std::lock_guard<std::mutex> lock(mutex);
    static bool inited = false;
    if (inited) {
        LOGD("OpenCL already inited, return");
        return;
    }
    inited = true;

    dumpCLinfo();

    EGLDisplay mEglDisplay = eglGetCurrentDisplay();
    if (mEglDisplay == EGL_NO_DISPLAY)
        LOGE("initCL: eglGetCurrentDisplay() returned 'EGL_NO_DISPLAY', error = %x", eglGetError());

    EGLContext mEglContext = eglGetCurrentContext();
    if (mEglContext == EGL_NO_CONTEXT)
        LOGE("initCL: eglGetCurrentContext() returned 'EGL_NO_CONTEXT', error = %x", eglGetError());

    cl_context_properties props[] =
    {   CL_GL_CONTEXT_KHR,   (cl_context_properties) mEglContext,
        CL_EGL_DISPLAY_KHR,  (cl_context_properties) mEglDisplay,
        CL_CONTEXT_PLATFORM, 0,
        0 };

    try
    {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        cl::Platform p = platforms[0];

        //cl::Platform p = cl::Platform::getDefault();
        std::string ext = p.getInfo<CL_PLATFORM_EXTENSIONS>();
        if(ext.find("cl_khr_gl_sharing") == std::string::npos)
            LOGE("Warning: CL-GL sharing isn't supported by PLATFORM");
        props[5] = (cl_context_properties) p();

        theContext = cl::Context(CL_DEVICE_TYPE_GPU, props);
        std::vector<cl::Device> devs = theContext.getInfo<CL_CONTEXT_DEVICES>();
        LOGD("Context returned %d devices, taking the 1st one", devs.size());
        ext = devs[0].getInfo<CL_DEVICE_EXTENSIONS>();
        if(ext.find("cl_khr_gl_sharing") == std::string::npos)
            LOGE("Warning: CL-GL sharing isn't supported by DEVICE");

        theQueue = cl::CommandQueue(theContext, devs[0]);

        cv::ocl::attachContext(p.getInfo<CL_PLATFORM_NAME>(), p(), theContext(), devs[0]());
        if( cv::ocl::useOpenCL() )
            LOGD("OpenCV+OpenCL works OK!");
        else
            LOGE("Can't init OpenCV with OpenCL TAPI");

        haveOpenCL = true;
    }
    catch(cl::Error& e)
    {
        LOGE("cl::Error: %s (%d)", e.what(), e.err());
    }
    catch(std::exception& e)
    {
        LOGE("std::exception: %s", e.what());
    }
    catch(...)
    {
        LOGE( "OpenCL info: unknown error while initializing OpenCL stuff" );
    }
    LOGD("initCL completed");
}

cv::UMat umat, umat_rgb;

extern "C"
int processFrontFrame(long pIn, long pOut) {
    cv::Mat * in = (cv::Mat *)pIn;
    cv::Mat * out = (cv::Mat *)pOut;
    CV_Assert(in != NULL && out != NULL);

    LOGD("processFrontFrame, %dx%d, type=%d (pIn=%d, pOut=%d)", 
         in->cols, in->rows, in->type(), pIn, pOut);

    // cv::UMat umat, umat_rgb;
    in->copyTo(umat);
    cv::cvtColor(umat, umat_rgb, cv::COLOR_YUV2RGBA_NV21, 4);
    umat_rgb.copyTo(*out);

    return 0;
}

extern "C"
int processBackFrame(long pIn, long pOut) {
    cv::Mat * in = (cv::Mat *)pIn;
    cv::Mat * out = (cv::Mat *)pOut;
    CV_Assert(in != NULL && out != NULL);

    LOGD("processBackFrame, %dx%d, type=%d (pIn=%d, pOut=%d)", 
         in->cols, in->rows, in->type(), pIn, pOut);

    return 0;
}
