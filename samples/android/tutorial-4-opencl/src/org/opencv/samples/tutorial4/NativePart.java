package org.opencv.samples.tutorial4;

public class NativePart {
    static
    {
        System.loadLibrary("opencv_java3");
        System.loadLibrary("JNIpart");
    }

    /**
     * Process frame, pFrame is native pointer of Mat
     * @return        1: modified, 0: unmodified
     */
    public static native int processFrontFrame(long pIn, long pOut);
    public static native int processBackFrame(long pIn, long pOut);

    public static native int initCL();
}
