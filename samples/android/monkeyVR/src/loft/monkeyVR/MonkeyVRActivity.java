/*
* @Author: BlahGeek
* @Date:   2016-01-21
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-01-26
*/

package loft.monkeyVR;

import org.opencv.android.JavaCameraView;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;

import android.util.Log;
import android.app.Activity;
import android.os.Bundle;
import android.view.Window;
import android.view.WindowManager;
import android.content.pm.ActivityInfo;

import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaCodecList;
import android.media.MediaFormat;

import android.app.AlertDialog;
import android.app.Dialog;
import android.content.Context;
import android.content.DialogInterface;
import android.view.LayoutInflater;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.CheckBox;
import android.widget.Toast;
import android.widget.TextView;

public class MonkeyVRActivity extends Activity implements DialogInterface.OnClickListener {

    private JavaCameraView frontCamView, backCamView;

    private String outfile_path = "/sdcard/octvr.mp4";
    private String remote_addr = "192.168.1.103";
    private int remote_port = 23456;
    private int bitrate = 10000000;
    private boolean ifStitch = true;
    private boolean ifSocket = true;

    private NativeBridge frontBridge;
    private NativeBridge backBridge;

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

        Button startButton = (Button) findViewById(R.id.startButton);
        Button stopButton = (Button) findViewById(R.id.stopButton);
        startButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {


                String out = frontBridge.setParams(bitrate, outfile_path,
                                      remote_addr, remote_port,
                                      ifStitch, ifSocket);
                //backCamView.enableView();
                //frontCamView.enableView();

                Context ctx = getApplicationContext();
                //CharSequence text = "Started!";

                Toast toast = Toast.makeText(ctx, out, Toast.LENGTH_SHORT);
                toast.show();
            }
        });
        stopButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {

                //frontCamView.disableView();
                //backCamView.disableView();

                CharSequence text = "Stopped!";
                Context ctx = getApplicationContext();

                Toast toast = Toast.makeText(ctx, text, Toast.LENGTH_SHORT);
                toast.show();
            }
        });

        int numCodecs = MediaCodecList.getCodecCount();
        for (int i = 0; i < numCodecs; i++) {
            MediaCodecInfo codecInfo = MediaCodecList.getCodecInfoAt(i);
            if (!codecInfo.isEncoder()) {
                continue;
            }
            String[] types = codecInfo.getSupportedTypes();
            for (int j = 0; j < types.length; j++) {
                Log.d("MonkeyVR", types[j]);
            }
        }

        frontBridge = new NativeBridge(true);
        backBridge = new NativeBridge(false);

        frontCamView = (JavaCameraView) findViewById(R.id.front_cam_gl_surface_view);
        frontCamView.setMaxFrameSize(1280, 720);
        frontCamView.setCvCameraViewListener(frontBridge);

        backCamView = (JavaCameraView) findViewById(R.id.back_cam_gl_surface_view);
        backCamView.setMaxFrameSize(1280, 720);
        backCamView.setCvCameraViewListener(backBridge);
    }

    public void showSettings(View v) {
        LayoutInflater li = LayoutInflater.from(this);
        View view = li.inflate(R.layout.keyboard_prompt, null);

        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Settings");

        builder.setView(view);

        builder.setPositiveButton("Confirm", this);
        builder.setNegativeButton("Cancel", this);

        EditText bitrateEditText = (EditText) view.findViewById(R.id.bitrateEditText);
        bitrateEditText.setText(Integer.toString(bitrate), TextView.BufferType.EDITABLE);
        EditText outputEditText = (EditText) view.findViewById(R.id.outputEditText);
        outputEditText.setText(outfile_path, TextView.BufferType.EDITABLE);
        EditText dstAddrEditText = (EditText) view.findViewById(R.id.dstAddrEditText);
        dstAddrEditText.setText(remote_addr, TextView.BufferType.EDITABLE);
        EditText dstPortEditText = (EditText) view.findViewById(R.id.dstPortEditText);
        dstPortEditText.setText(Integer.toString(remote_port), TextView.BufferType.EDITABLE);
        CheckBox stitchCheckBox = (CheckBox) view.findViewById(R.id.stitchCheckBox);
        stitchCheckBox.setChecked(ifStitch);
        CheckBox socketCheckBox = (CheckBox) view.findViewById(R.id.socketCheckBox);
        socketCheckBox.setChecked(ifSocket);

        AlertDialog ad = builder.create();
        ad.show();
    }

    @Override
    public void onClick(DialogInterface dialog, int which) {

        if(which == Dialog.BUTTON_POSITIVE){

            AlertDialog ad = (AlertDialog) dialog;
            EditText t = (EditText) ad.findViewById(R.id.bitrateEditText);
            bitrate = Integer.parseInt(t.getText().toString());
            t = (EditText) ad.findViewById(R.id.outputEditText);
            outfile_path = t.getText().toString();
            t = (EditText) ad.findViewById(R.id.dstAddrEditText);
            remote_addr = t.getText().toString();
            t = (EditText) ad.findViewById(R.id.dstPortEditText);
            remote_port = Integer.parseInt(t.getText().toString());

            CheckBox c = (CheckBox) ad.findViewById(R.id.stitchCheckBox);
            ifStitch = c.isChecked();
            c = (CheckBox) ad.findViewById(R.id.socketCheckBox);
            ifSocket = c.isChecked();

            CharSequence text = "Settings set.";
            Toast.makeText(this, text, Toast.LENGTH_SHORT).show();
        }

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
        //backCamView.enableView();
        //frontCamView.enableView();
    }

}
