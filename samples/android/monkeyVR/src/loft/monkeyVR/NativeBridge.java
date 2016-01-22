/* 
* @Author: BlahGeek
* @Date:   2016-01-21
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-01-21
*/

package loft.monkeyVR;

import org.opencv.core.Mat;
import org.opencv.android.JavaCameraView;

import android.util.Log;

public class NativeBridge implements JavaCameraView.CvCameraViewListener2 {
    static {
        System.loadLibrary("opencv_java3");
        System.loadLibrary("MonkeyVRJNI");
    }

    private Mat in;
    private Mat out = new Mat();
    private int camIndex = 0;
    private static final String LOGTAG = "NativeBridge";

    // Index: 0 for back camera and 1 for front
    public static native void onStart(int index, int width, int height);
    public static native void onStop(int index);
    // return: 1 for modified
    public static native int onFrame(int index, long pIn, long pOut);

    public NativeBridge(boolean _isFrontCam) {
        camIndex = _isFrontCam ? 1 : 0;
    }

    public void onCameraViewStarted(int width, int height) {
        onStart(camIndex, width, height);
    }

    public void onCameraViewStopped() {
        onStop(camIndex);
    }

    public Mat onCameraFrame(JavaCameraView.CvCameraViewFrame frame) {
        in = frame.raw();
        int modified = onFrame(camIndex, in.getNativeObjAddr(), out.getNativeObjAddr());
        return modified > 0 ? out : null;
    }

}
