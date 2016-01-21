package loft.monkeyVR;

public class NativePart {
    static
    {
        System.loadLibrary("opencv_java3");
        System.loadLibrary("MonkeyVRJNI");
    }

    /**
     * Process frame, pFrame is native pointer of Mat
     * @return        1: modified, 0: unmodified
     */
    public static native int processFrontFrame(long pIn, long pOut);
    public static native int processBackFrame(long pIn, long pOut);

    public static native int initCL();
}
