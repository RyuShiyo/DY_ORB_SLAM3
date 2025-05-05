#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <sophus/so3.hpp>

#include "yolo.h"
#include "System.h"

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames,
				vector<double> &vTimestamps);



void opticalFlow(const Mat& frame1, Mat& frame2, const Mat& prvs, const Mat& next, Mat& bgr)
{
    // Create some random colors
    vector<Scalar> colors;
    RNG rng;
    for(int i = 0; i < 100; i++)
    {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        colors.push_back(Scalar(r,g,b));
    }

    vector<Point2f> p0, p1;

    // Take first frame and find corners in it
    goodFeaturesToTrack(prvs, p0, 100, 0.3, 7, Mat(), 7, false, 0.04);

    // Create a mask image for drawing purposes
    Mat mask = Mat::zeros(frame1.size(), frame1.type());

    // calculate optical flow
    vector<uchar> status;
    vector<float> err;
    TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
    calcOpticalFlowPyrLK(prvs, next, p0, p1, status, err, Size(15,15), 2, criteria);

    vector<Point2f> good_new;
    for(uint i = 0; i < p0.size(); i++)
    {
        // Select good points
        if(status[i] == 1) {
            good_new.push_back(p1[i]);
            // draw the tracks
            line(mask,p1[i], p0[i], colors[i], 2);
            circle(frame2, p1[i], 5, colors[i], -1);
        }
    }
    add(frame2, mask, bgr);

}

void opticalFlowDense(const Mat& prvs, const Mat& next, Mat& bgr)
{
	Mat flow(prvs.size(), CV_32FC2);
	calcOpticalFlowFarneback(prvs, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

	// visualization
	Mat flow_parts[2];
	split(flow, flow_parts);
	Mat magnitude, angle, magn_norm;
	cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
	normalize(magnitude, magn_norm, 0.0f, 1.0f, NORM_MINMAX);
	angle *= ((1.f / 360.f) * (180.f / 255.f));

	//build hsv image
	Mat _hsv[3], hsv, hsv8;
	_hsv[0] = angle;
	_hsv[1] = Mat::ones(angle.size(), CV_32F);
	_hsv[2] = magn_norm;
	merge(_hsv, 3, hsv);
	hsv.convertTo(hsv8, CV_8U, 255.0);
	cvtColor(hsv8, bgr, COLOR_HSV2BGR);
}

int main(int argc, char **argv)
{
	Net_config yolo_nets[4] = {
		{0.5, 0.4, 416, 416,"coco.names", "yolov3/yolov3.cfg", "yolov3/yolov3.weights", "yolov3"},
		{0.5, 0.4, 608, 608,"coco.names", "yolov4/yolov4.cfg", "yolov4/yolov4.weights", "yolov4"},
		{0.5, 0.4, 320, 320,"coco.names", "yolo-fastest/yolo-fastest-xl.cfg", "yolo-fastest/yolo-fastest-xl.weights", "yolo-fastest"},
		{0.5, 0.4, 320, 320,"coco.names", "yolobile/csdarknet53s-panet-spp.cfg", "yolobile/yolobile.weights", "yolobile"}
	};

	if(argc != 4)
    {
        cerr << endl << "Usage: ./mono_tum path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    string strFile = string(argv[3])+"/rgb.txt";
    LoadImages(strFile, vstrImageFilenames, vTimestamps);

    int nImages = vstrImageFilenames.size();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::MONOCULAR,true);
    float imageScale = SLAM.GetImageScale();

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    double t_resize = 0.f;
    double t_track = 0.f;

    // Main loop
    cv::Mat im;
    for(int ni=0; ni<nImages; ni++)
    {
        // Read image from file
        im = cv::imread(string(argv[3])+"/"+vstrImageFilenames[ni],cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);
        double tframe = vTimestamps[ni];
    	Mat mask = Mat::zeros(im.size(), CV_8UC1); // mask对应单通道图像所以是8UC1 (8 uchar channel 1)
        if(im.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenames[ni] << endl;
            return 1;
        }

        if(imageScale != 1.f)
        {
#ifdef REGISTER_TIMES
    #ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t_Start_Resize = std::chrono::steady_clock::now();
    #else
            std::chrono::monotonic_clock::time_point t_Start_Resize = std::chrono::monotonic_clock::now();
    #endif
#endif
            int width = im.cols * imageScale;
            int height = im.rows * imageScale;
            cv::resize(im, im, cv::Size(width, height));
#ifdef REGISTER_TIMES
    #ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t_End_Resize = std::chrono::steady_clock::now();
    #else
            std::chrono::monotonic_clock::time_point t_End_Resize = std::chrono::monotonic_clock::now();
    #endif
            t_resize = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t_End_Resize - t_Start_Resize).count();
            SLAM.InsertResizeTime(t_resize);
#endif
        }

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        // Pass the image to the SLAM system
        SLAM.TrackMonocular(im,tframe,mask);

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    return 0;


	// YOLO yolo_model(yolo_nets[0]);
	// string imgpath = "frame1.jpg";
	// Mat srcimg = imread(imgpath);
	// yolo_model.detect(srcimg);
	// static const string kWinName = "Deep learning object detection in OpenCV";
	// namedWindow(kWinName, WINDOW_NORMAL);
	// imshow(kWinName, srcimg);
	// waitKey(0);
	// destroyAllWindows();

}

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
	ifstream f;
	f.open(strFile.c_str());

	// skip first three lines
	string s0;
	getline(f,s0);
	getline(f,s0);
	getline(f,s0);

	while(!f.eof())
	{
		string s;
		getline(f,s);
		if(!s.empty())
		{
			stringstream ss;
			ss << s;
			double t;
			string sRGB;
			ss >> t;
			vTimestamps.push_back(t);
			ss >> sRGB;
			vstrImageFilenames.push_back(sRGB);
		}
	}
}
