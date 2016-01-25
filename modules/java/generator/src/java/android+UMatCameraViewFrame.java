/* 
* @Author: BlahGeek
* @Date:   2016-01-25
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-01-25
*/

package org.opencv.android;

import java.util.List;

import android.content.Context;
import android.graphics.ImageFormat;
import android.graphics.SurfaceTexture;
import android.hardware.Camera;
import android.hardware.Camera.PreviewCallback;
import android.os.Build;
import android.util.AttributeSet;
import android.util.Log;
import android.view.ViewGroup.LayoutParams;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class UMatCameraViewFrame implements CameraBridgeViewBase.CvCameraViewFrame {
    static {
        System.loadLibrary("opencv_java3");
    }
    
    private long pUmat;
    private Mat mRgba;
    private Mat mGray;
    private int mWidth, mHeight;

    public static native void releaseUMat(long p);
    public static native long copyToUMat(long p, byte[] data, int len, int w, int h, int t);
    public static native void convertToRgba(long src, long dst);
    public static native void convertToGray(long src, long dst);

    public UMatCameraViewFrame(int width, int height) {
        pUmat = copyToUMat(pUmat, null, 0, width, height + height / 2, CvType.CV_8U);
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mGray = new Mat(height, width, CvType.CV_8U);
        mWidth = width;
        mHeight = height;
    }

    public void update(byte[] data) {
        copyToUMat(pUmat, data, data.length, mWidth, mHeight + mHeight / 2, CvType.CV_8U);
    }

    public void release() {
        releaseUMat(pUmat);
        mRgba.release();
        mGray.release();
    }

    @Override
    public long raw() {
        return pUmat;
    }

    @Override
    public Mat gray() {
        convertToGray(pUmat, mGray.getNativeObjAddr());
        return mGray;
    }

    @Override
    public Mat rgba() {
        convertToRgba(pUmat, mRgba.getNativeObjAddr());
        return mRgba;
    }
}
