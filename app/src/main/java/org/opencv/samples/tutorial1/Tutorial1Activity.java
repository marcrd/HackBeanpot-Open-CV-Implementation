package org.opencv.samples.tutorial1;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;

import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.Toast;
import org.opencv.imgproc.Imgproc;
import org.opencv.features2d.*;
import org.opencv.core.*;
import org.opencv.video.Video;

public class Tutorial1Activity extends Activity implements CvCameraViewListener2, SensorEventListener {
    private static final String TAG = "OCVSample::Activity";

    private CameraBridgeViewBase mOpenCvCameraView;
    private boolean              mIsJavaCamera = true;
    private MenuItem             mItemSwitchCamera = null;

    FeatureDetector detector;
    DescriptorExtractor descriptor;
    DescriptorMatcher matcher;

    private SensorManager mSensorManager;
    private Sensor mSensor;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    private float sensorData[];
    private float rotData[];

    @Override
    public final void onSensorChanged(SensorEvent event)
    {
        rotData[0] = event.values[0];
        rotData[1] = event.values[1];
        rotData[2] = event.values[2];
        //rotData[3] = event.values[3];

        /*Log.v(TAG, "SENSOR INFO HERE");
        Log.v(TAG, "SensorValue1 = " + sensorData[0]);
        Log.v(TAG, "SensorValue2 = " + sensorData[1]);
        Log.v(TAG, "SensorValue3 = " + sensorData[2]);*/

    };

    @Override
    public final void onAccuracyChanged(Sensor sensor, int accuracy) {
        // Do something here if sensor accuracy changes.
    }

    public Tutorial1Activity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.tutorial1_surface_view);

        if (mIsJavaCamera)
            mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.tutorial1_activity_java_surface_view);
        else
            mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.tutorial1_activity_native_surface_view);

        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);

        mOpenCvCameraView.setCvCameraViewListener(this);

        mSensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        mSensor = mSensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR);

        sensorData = new float[3];
        rotData = new float[4];
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        mSensorManager.registerListener(this, mSensor, SensorManager.SENSOR_DELAY_NORMAL);
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_9, this, mLoaderCallback);
    }

    public void onDestroy() {
        super.onDestroy();
        /*if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();*/
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemSwitchCamera = menu.add("Toggle Native/Java camera");
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        String toastMesage = new String();
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);

        if (item == mItemSwitchCamera) {
            mOpenCvCameraView.setVisibility(SurfaceView.GONE);
            mIsJavaCamera = !mIsJavaCamera;

            if (mIsJavaCamera) {
                mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.tutorial1_activity_java_surface_view);
                toastMesage = "Java Camera";
            } else {
                mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.tutorial1_activity_native_surface_view);
                toastMesage = "Native Camera";
            }

            mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
            mOpenCvCameraView.setCvCameraViewListener(this);
            mOpenCvCameraView.enableView();
            Toast toast = Toast.makeText(this, toastMesage, Toast.LENGTH_LONG);
            toast.show();
        }

        return true;
    }

    private Mat mRgba;
    private Mat mRgbaOLD;
    private Mat descriptorsOLD;
    private Mat descriptors1;
    private Mat keypoints1;
    private MatOfKeyPoint keypointsOLD;
    private Mat featuredImg;
    private MatOfDMatch matches;
    private boolean firstFrame;

    public Mat saveImage1;
    public boolean issaved = false;
    MatOfPoint2f prevPts;
    MatOfPoint2f nextPts;
    MatOfPoint initial;

    double posX;
    double posY;

    public void onCameraViewStarted(int width, int height)
    {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mRgbaOLD = new Mat(height, width, CvType.CV_8UC4);

        detector = FeatureDetector.create(FeatureDetector.ORB);
        descriptor = DescriptorExtractor.create(DescriptorExtractor.ORB);
        matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);

        descriptorsOLD = new Mat();
        keypointsOLD = new MatOfKeyPoint();

        descriptors1 = new Mat();
        keypoints1 = new MatOfKeyPoint();

        featuredImg = new Mat();
        matches = new MatOfDMatch();

        firstFrame = true;

        initial = new MatOfPoint();
        nextPts = new MatOfPoint2f();
        prevPts = new MatOfPoint2f();

        posX = posY = 0;
    }

    public void onCameraViewStopped() {
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame)
    {
        mRgba = inputFrame.rgba();
        Mat mRgbaGray = inputFrame.gray();
        Imgproc.resize(mRgbaGray, mRgbaGray, new Size(mRgbaGray.cols()*0.25, mRgbaGray.rows()*0.25));
        Imgproc.resize(mRgba, mRgba, new Size(mRgba.cols()*0.25, mRgba.rows()*0.25));

        MatOfByte status = new MatOfByte();
        MatOfFloat err = new MatOfFloat();

        double avgX = 0;
        double avgY = 0;
        double avgMag = 0;
        double avgExp = 0;

        if(!issaved)
        {
            issaved = true;
        }
        else
        {
            //mRgbaOLD = mRgbaGray;
            Imgproc.goodFeaturesToTrack(mRgbaGray, initial, 100, 0.01, 0.01);
            initial.convertTo(prevPts, CvType.CV_32FC2);
            nextPts = new MatOfPoint2f();
            Video.calcOpticalFlowPyrLK(mRgbaGray, mRgbaOLD, prevPts, nextPts, status, err);

            Point[] pArray = prevPts.toArray();
            Point[] pnArray = nextPts.toArray();

            double xSum = 0;
            double xCounter = 0;
            double ySum = 0;
            double yCounter = 0;
            double magSum = 0;
            double magCounter = 0;
            double expSum = 0;
            double expCounter = 0;

            for(int i = 0; i < pArray.length; i++)
            {
                Core.circle(mRgba, pArray[i], 2, new Scalar(255, 0, 0));
                Core.circle(mRgba, pnArray[i], 5, new Scalar(0, 255, 0));
                Core.line(mRgba, pArray[i], pnArray[i], new Scalar(0, 0, 255), 1);

                xSum += pArray[i].x - pnArray[i].x;
                ySum += pArray[i].y - pnArray[i].y;
                magSum += Math.pow(pArray[i].y - pnArray[i].y, 2) + Math.pow(pArray[i].x - pnArray[i].x, 2);

                double mid = mRgba.cols() / 2;
                double tmpValX = pArray[i].x - pnArray[i].x;

                if(i+1 < pArray.length)
                    expSum += Math.pow((pArray[i].x-pArray[i+1].x), 2) + Math.pow((pArray[i].y-pArray[i+1].y), 2) - (Math.pow((pnArray[i].x-pnArray[i+1].x), 2)) - (Math.pow((pnArray[i].y-pnArray[i+1].y), 2));

                xCounter++;
                yCounter++;
                magCounter++;
                expCounter++;
            }

            avgX = xSum / xCounter;
            if(Math.abs(avgX) < 0.001)
                avgX = 0;

            avgY = ySum / yCounter;
            if(Math.abs(avgY) < 0.001)
                avgY = 0;

            avgMag = magSum / magCounter;
            if(Math.abs(avgMag) < 0.001)
                avgMag = 0;

            avgExp = expSum / expCounter;
            if(Math.abs(avgExp) < 0.001)
                avgExp = 0;

            // Dump Logic.. assume noise and that we aren't turning.. meh
            if(avgExp > 500 && Math.abs(avgX) < 17 && Math.abs(avgY) < 17) {
                posY += 0.1*Math.cos(rotData[2]);
                posX += 0.1*Math.sin(rotData[2]);
            }
            else if(avgExp < -500 && Math.abs(avgX) < 17 && Math.abs(avgY)<17) {
                posY -= 0.1*Math.cos(rotData[2]);
                posX -= 0.1*Math.sin(rotData[2]);
            }
            prevPts = nextPts;
        }

        mRgbaOLD = mRgbaGray.clone();

        Imgproc.resize(mRgba, mRgba, new Size(mRgba.cols()*4, mRgba.rows()*4));

        Core.putText(mRgba, "X = " + avgX, new Point(5, 150), 0, 2, new Scalar(255, 0, 255), 3, 8, false);
        Core.putText(mRgba, "Y = " + avgY, new Point(5, 200), 0, 2, new Scalar(255, 0, 255), 3, 8, false);
        Core.putText(mRgba, "M = " + avgMag, new Point(5, 250), 0, 2, new Scalar(255, 0, 255), 3, 8, false);
        Core.putText(mRgba, "E = " + avgExp, new Point(5, 300), 0, 2, new Scalar(255, 0, 255), 3, 8, false);

        Core.putText(mRgba, "POSX = " + posX, new Point(5, 350), 0, 2, new Scalar(255, 0, 255), 3, 8, false);
        Core.putText(mRgba, "POSY = " + posY, new Point(5, 400), 0, 2, new Scalar(255, 0, 255), 3, 8, false);

        Core.putText(mRgba, "GRAVX = " + rotData[0], new Point(5, 550), 0, 2, new Scalar(255, 0, 255), 3, 8, false);
        Core.putText(mRgba, "GRAVY = " + rotData[1], new Point(5, 600), 0, 2, new Scalar(255, 0, 255), 3, 8, false);
        Core.putText(mRgba, "GRAVZ = " + rotData[2], new Point(5, 650), 0, 2, new Scalar(255, 0, 255), 3, 8, false);
        //Core.putText(mRgba, "RMAG = " + rotData[3], new Point(5, 700), 0, 2, new Scalar(255, 0, 255), 3, 8, false);
        return mRgba;
        /*MatOfKeyPoint keypoints1 = new MatOfKeyPoint();
        detector.detect(mRgba, keypoints1);
        descriptor.compute(mRgba, keypoints1, descriptors1);

        Features2d.drawKeypoints(mRgba, keypoints1, featuredImg, new Scalar(255, 159, 10), 0);
*/
        /*if(!firstFrame)
        {
            MatOfDMatch matches = new MatOfDMatch();
            matcher.match(descriptors1, descriptorsOLD, matches);

            Scalar RED = new Scalar(255, 0, 0);
            Scalar GREEN = new Scalar(0, 255, 0);
            Mat outputImg = new Mat();
            MatOfByte drawnMatches = new MatOfByte();

            Features2d.drawMatches(mRgba, keypoints1, mRgbaOLD, keypointsOLD, matches, outputImg, GREEN, RED, drawnMatches, Features2d.NOT_DRAW_SINGLE_POINTS);

            featuredImg = outputImg;
        }*/

        //Imgproc.resize(featuredImg, featuredImg, new Size(featuredImg.cols()*4, featuredImg.rows()*4));

        /*descriptorsOLD = descriptors1;
        keypointsOLD = keypoints1;
        firstFrame = true;
        mRgbaOLD = mRgba;

        return featuredImg;*/
    }
}
