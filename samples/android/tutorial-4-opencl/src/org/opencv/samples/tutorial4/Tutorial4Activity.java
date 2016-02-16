package org.opencv.samples.tutorial4;

import org.opencv.android.JavaCameraView;
import org.opencv.core.Mat;
import org.opencv.core.Core;
import org.opencv.imgproc.Imgproc;

import android.util.Log;
import android.app.Activity;
import android.content.pm.ActivityInfo;
import android.os.Bundle;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.Window;
import android.view.WindowManager;
import android.widget.TextView;

public class Tutorial4Activity extends Activity {

    static
    {
        System.loadLibrary("opencv_java3");
    }

    private JavaCameraView frontCamView, backCamView;
    private TextView mProcMode;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON,
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);

        setContentView(R.layout.activity);

        // processor = new DualCamProcessor();

        frontCamView = (JavaCameraView) findViewById(R.id.front_cam_gl_surface_view);
        frontCamView.setMaxFrameSize(1280, 720);
        frontCamView.setCvCameraViewListener(new JavaCameraView.CvCameraViewListener2() {
            private Mat in = new Mat();
            private Mat out = new Mat();
            public void onCameraViewStarted(int width, int height) {
                // NativePart.initCL();
            }
            public void onCameraViewStopped() {
            }
            public Mat onCameraFrame(JavaCameraView.CvCameraViewFrame frame) {
                in = frame.raw();
                int modified = NativePart.processFrontFrame(in.getNativeObjAddr(), out.getNativeObjAddr());
                if(modified > 0)
                    return out;
                else
                    return null;
            }
        });

        backCamView = (JavaCameraView) findViewById(R.id.back_cam_gl_surface_view);
        backCamView.setMaxFrameSize(1280, 720);
        backCamView.setCvCameraViewListener(new JavaCameraView.CvCameraViewListener2() {
            private Mat in = new Mat();
            private Mat out = new Mat();
            public void onCameraViewStarted(int width, int height) {
                // NativePart.initCL();
            }
            public void onCameraViewStopped() {
            }
            public Mat onCameraFrame(JavaCameraView.CvCameraViewFrame frame) {
                in = frame.raw();
                int modified = NativePart.processBackFrame(in.getNativeObjAddr(), out.getNativeObjAddr());
                if(modified > 0)
                    return out;
                else
                    return null;
            }
        });

        // TextView tv = (TextView)findViewById(R.id.fps_text_view);
        // mProcMode = (TextView)findViewById(R.id.proc_mode_text_view);
        // runOnUiThread(new Runnable() {
        //     public void run() {
        //         mProcMode.setText("Processing mode: No processing");
        //     }
        // });

    }

    @Override
    protected void onPause() {
        frontCamView.disableView();
        backCamView.disableView();
        super.onPause();
    }

    @Override
    protected void onResume() {
        super.onResume();
        backCamView.enableView();
        frontCamView.enableView();
    }

    // @Override
    // public boolean onCreateOptionsMenu(Menu menu) {
    //     MenuInflater inflater = getMenuInflater();
    //     inflater.inflate(R.menu.menu, menu);
    //     return super.onCreateOptionsMenu(menu);
    // }

    // @Override
    // public boolean onOptionsItemSelected(MenuItem item) {
    //     switch (item.getItemId()) {
    //     case R.id.no_proc:
    //         runOnUiThread(new Runnable() {
    //             public void run() {
    //                 mProcMode.setText("Processing mode: No Processing");
    //             }
    //         });
    //         mLeftView.setProcessingMode(NativePart.PROCESSING_MODE_NO_PROCESSING);
    //         mRightView.setProcessingMode(NativePart.PROCESSING_MODE_NO_PROCESSING);
    //         return true;
    //     case R.id.cpu:
    //         runOnUiThread(new Runnable() {
    //             public void run() {
    //                 mProcMode.setText("Processing mode: CPU");
    //             }
    //         });
    //         mLeftView.setProcessingMode(NativePart.PROCESSING_MODE_CPU);
    //         mRightView.setProcessingMode(NativePart.PROCESSING_MODE_CPU);
    //         return true;
    //     case R.id.ocl_ocv:
    //         runOnUiThread(new Runnable() {
    //             public void run() {
    //                 mProcMode.setText("Processing mode: OpenCL via OpenCV (TAPI)");
    //             }
    //         });
    //         mLeftView.setProcessingMode(NativePart.PROCESSING_MODE_OCL_OCV);
    //         mRightView.setProcessingMode(NativePart.PROCESSING_MODE_OCL_OCV);
    //         return true;
    //     default:
    //         return false;
    //     }
    // }
}
