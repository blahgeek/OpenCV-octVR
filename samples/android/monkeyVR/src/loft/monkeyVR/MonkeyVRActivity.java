/* 
* @Author: BlahGeek
* @Date:   2016-01-21
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-01-21
*/

package loft.monkeyVR;

import org.opencv.android.JavaCameraView;

import android.util.Log;
import android.app.Activity;
import android.os.Bundle;
import android.view.Window;
import android.view.WindowManager;
import android.content.pm.ActivityInfo;

public class MonkeyVRActivity extends Activity {

    private JavaCameraView frontCamView, backCamView;

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

        frontCamView = (JavaCameraView) findViewById(R.id.front_cam_gl_surface_view);
        frontCamView.setMaxFrameSize(1280, 720);
        frontCamView.setCvCameraViewListener(new NativeBridge(true));

        backCamView = (JavaCameraView) findViewById(R.id.back_cam_gl_surface_view);
        backCamView.setMaxFrameSize(1280, 720);
        backCamView.setCvCameraViewListener(new NativeBridge(false));
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

}
