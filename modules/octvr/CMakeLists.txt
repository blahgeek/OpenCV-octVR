set(the_description "octVR, you know it")
ocv_warnings_disable(CMAKE_CXX_FLAGS -Wnon-virtual-dtor)
ocv_define_module(octvr 
                  opencv_imgproc opencv_features2d opencv_calib3d opencv_objdetect opencv_stitching
                  OPTIONAL opencv_cudaarithm opencv_cudafilters opencv_cudafeatures2d opencv_cudalegacy)
