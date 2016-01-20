package org.opencv.samples.tutorial4;

public class NativePart {
    static
    {
        System.loadLibrary("opencv_java3");
        System.loadLibrary("JNIpart");
    }

    public static native int initCL();
    public static native int processFrontFrame(int texIn, int texOut,
                                               int width, int height);
    public static native int processBackFrame(int texIn, int texOut,
                                              int width, int height);
}
