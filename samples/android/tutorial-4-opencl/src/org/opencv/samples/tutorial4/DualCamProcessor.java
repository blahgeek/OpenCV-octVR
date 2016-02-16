/* 
* @Author: BlahGeek
* @Date:   2016-01-19
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-01-20
*/

package org.opencv.samples.tutorial4;

import android.util.Log;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.locks.Condition;

public class DualCamProcessor {

    static final String LOGTAG = "DualCamProcessor";

    // final Lock lock = new ReentrantLock();
    // final Condition cond = lock.newCondition();

    int frontTexIn, frontTexOut;
    int frontWidth, frontHeight;
    boolean frontCameraAvailable = false;

    public boolean onFrontCameraTexture(int texIn, int texOut, int width, int height) {
        Log.i(LOGTAG, String.format("onFrontCameraTexture(%d, %d, %d, %d)", texIn, texOut, width, height));

        frontTexIn = texIn;
        frontTexOut = texOut;
        frontWidth = width;
        frontHeight = height;
        frontCameraAvailable = true;

        return true;
    }

    public boolean onBackCameraTexture(int texIn, int texOut, int width, int height) {
        Log.i(LOGTAG, String.format("onBackCameraTexture(%d, %d, %d, %d)", texIn, texOut, width, height));
        if(!frontCameraAvailable) {
            Log.w(LOGTAG, "Front camera not available, return");
            return false;
        }

        NativePart.processFrame(frontTexIn, frontTexOut, frontWidth, frontHeight,
                                texIn, texOut, width, height);
        return true;
    }

    public void onCameraViewStarted(int width, int height) {
        Log.i(LOGTAG, String.format("onCameraViewStarted(%d, %d)", width, height));
    }
    public void onCameraViewStopped() {
        Log.i(LOGTAG, "onCameraViewStopped");
    }

}
