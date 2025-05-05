#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <sophus/so3.hpp>

#include "yolo.h"
#include "System.h"

void LoadImagesEuStereo(const string &strPathLeft, const string &strPathRight, const string &strPathTimes,
				vector<string> &vstrImageLeft, vector<string> &vstrImageRight, vector<double> &vTimeStamps);

void removeKeypointsInsideBoxes(const std::vector<KeyPoint>& keypoints,
								const std::vector<Rect>& boxes,
								std::vector<KeyPoint>& filteredKeypoints,
								const Size& imageSize);

void opticalFlow(const Mat& frame1, Mat& frame2, const Mat& prvs, const Mat& next, Mat& bgr)
{
	// cvtColor(frame1, frame1, COLOR_GRAY2BGR);
	// cvtColor(frame2, frame2, COLOR_GRAY2BGR);
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

void opticalFlowDense(const Mat& prvs, const Mat& next, Mat& segmented,
					  const Mat& original, float lowerBound, float upperBound)
{
	// 计算光流
	Mat flow(prvs.size(), CV_32FC2);
	calcOpticalFlowFarneback(prvs, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

	// 分割光流的水平和垂直分量
	Mat flow_parts[2];
	split(flow, flow_parts);

	// 计算光流模值和角度
	Mat magnitude, angle;
	cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);

	// 创建掩码，筛选模值在区间 [lowerBound, upperBound] 的像素点
	Mat mask = (magnitude >= lowerBound) & (magnitude <= upperBound);

	// 创建分割图像，将符合条件的原图像素点保留，其他设为黑色
	segmented = Mat::zeros(original.size(), original.type());
	original.copyTo(segmented, mask);
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

void computeAndSaveOpticalFlow(const Mat& prvs, const Mat& next) {
	string outputFilename = "optical_flow_result.jpg";
	// 计算稠密光流
	Mat flow(prvs.size(), CV_32FC2);
	calcOpticalFlowFarneback(prvs, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

	// 分割光流的水平和垂直分量
	Mat flow_parts[2];
	split(flow, flow_parts);

	// 计算光流模值和方向角度
	Mat magnitude, angle;
	cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);

	// 归一化模值
	Mat magn_norm;
	normalize(magnitude, magn_norm, 0.0f, 1.0f, NORM_MINMAX);

	// 构建 HSV 图像
	Mat hsv[3], hsv_image, bgr_image;
	hsv[0] = angle * (1.f / 360.f) * (180.f / 255.f); // 角度映射到 [0, 1]
	hsv[1] = Mat::ones(angle.size(), CV_32F);          // 饱和度为 1
	hsv[2] = magn_norm;                                // 使用归一化后的模值作为亮度
	merge(hsv, 3, hsv_image);

	// 转换到 8 位 HSV 图像
	Mat hsv_8u;
	hsv_image.convertTo(hsv_8u, CV_8U, 255.0);

	// HSV 转换到 BGR 颜色空间
	cvtColor(hsv_8u, bgr_image, COLOR_HSV2BGR);

	// 保存结果图像
	imwrite(outputFilename, bgr_image);
}

int main(int argc, char **argv)
{
	int k = 0;
	Net_config yolo_nets[4] = {
		{0.5, 0.4, 416, 416,"coco.names", "YoloModel/yolov3/yolov3.cfg", "YoloModel/yolov3/yolov3.weights", "yolov3"},
		{0.5, 0.4, 608, 608,"coco.names", "YoloModel/yolov4/yolov4.cfg", "YoloModel/yolov4/yolov4.weights", "yolov4"},
		{0.5, 0.4, 320, 320,"coco.names", "YoloModel/yolo-fastest/yolo-fastest-xl.cfg", "YoloModel/yolo-fastest/yolo-fastest-xl.weights", "yolo-fastest"},
		{0.5, 0.4, 320, 320,"coco.names", "YoloModel/yolobile/csdarknet53s-panet-spp.cfg", "YoloModel/yolobile/yolobile.weights", "yolobile"}
	};

	if (argc < 5) {
		cerr << endl <<
				"Usage: ./stereo_euroc path_to_vocabulary path_to_settings path_to_sequence_folder_1 path_to_times_file_1 (path_to_image_folder_2 path_to_times_file_2 ... path_to_image_folder_N path_to_times_file_N) (trajectory_file_name)"
				<< endl;

		return 1;
	}

	const int num_seq = (argc - 3) / 2;
	cout << "num_seq = " << num_seq << endl;
	bool bFileName = (((argc - 3) % 2) == 1);
	string file_name;
	if (bFileName) {
		file_name = string(argv[argc - 1]);
		cout << "file name: " << file_name << endl;
	}

	// Load all sequences:
	int seq;
	vector<vector<string> > vstrImageLeft;
	vector<vector<string> > vstrImageRight;
	vector<vector<double> > vTimestampsCam;
	vector<int> nImages;

	vstrImageLeft.resize(num_seq);
	vstrImageRight.resize(num_seq);
	vTimestampsCam.resize(num_seq);
	nImages.resize(num_seq);

	int tot_images = 0;
	for (seq = 0; seq < num_seq; seq++) {
		cout << "Loading images for sequence " << seq << "...";

		string pathSeq(argv[(2 * seq) + 3]);
		string pathTimeStamps(argv[(2 * seq) + 4]);

		string pathCam0 = pathSeq + "/mav0/cam0/data";
		string pathCam1 = pathSeq + "/mav0/cam1/data";

		LoadImagesEuStereo(pathCam0, pathCam1, pathTimeStamps, vstrImageLeft[seq], vstrImageRight[seq], vTimestampsCam[seq]);
		cout << "LOADED!" << endl;

		nImages[seq] = vstrImageLeft[seq].size();
		tot_images += nImages[seq];
	}

	// Vector for tracking time statistics
	vector<float> vTimesTrack;
	vTimesTrack.resize(tot_images);

	cout << endl << "-------" << endl;
	cout.precision(17);


	// Create SLAM system. It initializes all system threads and gets ready to process frames.// Create SLAM system. It initializes all system threads and gets ready to process frames.
	ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::STEREO, true);

	cv::Mat imLeft, imRight;
	for (seq = 0; seq < num_seq; seq++) {
		// Seq loop
		double t_resize = 0;
		double t_rect = 0;
		double t_track = 0;
		int num_rect = 0;
		int proccIm = 0;

		// Main loop
		Mat pre_im;
		// nImages[seq] = 10;
		for (int ni = 0; ni < nImages[seq]; ni++, proccIm++) {
			// Read left and right images from file
			imLeft = cv::imread(vstrImageLeft[seq][ni], cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);
			imRight = cv::imread(vstrImageRight[seq][ni], cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);

			if (imLeft.empty()) {
				cerr << endl << "Failed to load image at: "
						<< string(vstrImageLeft[seq][ni]) << endl;
				return 1;
			}

			if (imRight.empty()) {
				cerr << endl << "Failed to load image at: "
						<< string(vstrImageRight[seq][ni]) << endl;
				return 1;
			}

			// Yolo
			vector<CandidateBox> boxes;
			Mat grayImage, colorImage;
			Mat outputImage = Mat::zeros(imLeft.size(), CV_8UC1);
			Mat mask = Mat::zeros(imLeft.size(), CV_8UC1); // mask对应单通道图像所以是8UC1 (8 uchar channel 1)
			if (imLeft.channels() == 1) {
				grayImage = imLeft.clone();
				cv::cvtColor(imLeft, colorImage, cv::COLOR_GRAY2BGR);
			}else {
				cvtColor(imLeft, grayImage, COLOR_BGR2GRAY);
			}

			Mat output = colorImage.clone();
			YOLO yolo_model(yolo_nets[2]);
			boxes = yolo_model.detect(colorImage, output);

			// 获得提取到的目标的灰度图，后续光流检测用
			// for (const auto& box : boxes) {
			// 	Rect validBox = box & Rect(0, 0, im.cols, im.rows);
			// 	grayImage(validBox).copyTo(outputImage(validBox));
			// }

			// 获得掩码mask，筛除特征点用
			for (const auto& box : boxes) {
				if (box.label == 0)
					rectangle(mask, box.box, cv::Scalar(255,255,255), cv::FILLED);
				// else
				// 	rectangle(mask, box.box, cv::Scalar(0,0,0), cv::FILLED);
			}
			// imwrite("./Output/Tum/mask_"+to_string(k)+".png", mask);
			// k++;

			// outputImage = imLeft.clone();
			// 提取光流
			// if (ni != 0) {
			// 	Mat bgr, prvs, next;
			// 	cvtColor(pre_im, prvs, COLOR_BGR2GRAY);
			// 	cvtColor(imLeft, next, COLOR_BGR2GRAY);
			// 	opticalFlowDense(prvs, next, bgr);
			// 	// imwrite("./Output/Tum/img_opticalFlowDense_"+to_string(k)+".png", bgr);
			// 	// k++;
			// 	pre_im = outputImage.clone();
			// }else {
			// 	pre_im = outputImage.clone();
			// }

			double tframe = vTimestampsCam[seq][ni];

			std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

			// Pass the image to the SLAM system
			SLAM.TrackStereo(imLeft, imRight, tframe, mask, vector<ORB_SLAM3::IMU::Point>(), vstrImageLeft[seq][ni]);

			std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

#ifdef REGISTER_TIMES
			t_track = t_resize + t_rect + std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t2 - t1).count();
			SLAM.InsertTrackTime(t_track);
#endif

			double ttrack = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
			vTimesTrack[ni] = ttrack;

			// Wait to load the next frame
			double T = 0;
			if (ni < nImages[seq] - 1)
				T = vTimestampsCam[seq][ni + 1] - tframe;
			else if (ni > 0)
				T = tframe - vTimestampsCam[seq][ni - 1];

			if (ttrack < T)
				usleep((T - ttrack) * 1e6); // 1e6
		}

		if (seq < num_seq - 1) {
			cout << "Changing the dataset" << endl;

			SLAM.ChangeDataset();
		}
	}
    // Stop all threads
    SLAM.Shutdown();

	// Save camera trajectory
	if (bFileName) {
		// const string kf_file = "kf_" + string(argv[argc - 1]) + ".txt";
		// const string f_file = "f_" + string(argv[argc - 1]) + ".txt";
		const string kf_file = string(argv[argc-1]) + "_kf" + ".txt";
		const string f_file = string(argv[argc-1]) + "_f" + ".txt";
		SLAM.SaveTrajectoryEuRoC(f_file);
		SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file);
	} else {
		SLAM.SaveTrajectoryEuRoC("CameraTrajectory.txt");
		SLAM.SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory.txt");
	}

	// YOLO yolo_model(yolo_nets[3]);
	// string imgpath = "bus.jpg";
	// Mat srcimg = imread(imgpath);
	// Mat output = srcimg.clone();
	// yolo_model.detect(srcimg,output);
	// imwrite("yolov3.jpg", output);

	return 0;
	// string imgpath1 = "frame1.jpg";
	// string imgpath2 = "frame2.jpg";
	// Mat srcimg1 = imread(imgpath1), prvs;
	// Mat srcimg2 = imread(imgpath2), next;
	// Mat bgr;
	// cvtColor(srcimg1, prvs, COLOR_BGR2GRAY);
	// cvtColor(srcimg2, next, COLOR_BGR2GRAY);
	// opticalFlowDense(prvs, next, bgr);
	// opticalFlow(srcimg1, srcimg2, prvs, next, bgr);
	// computeAndSaveOpticalFlow(prvs, next);
	// imshow("opt", bgr);
	// waitKey(0);
	// destroyAllWindows();
	// imwrite("optical.png", bgr);

}


void LoadImagesEuStereo(const string &strPathLeft, const string &strPathRight, const string &strPathTimes,
				vector<string> &vstrImageLeft, vector<string> &vstrImageRight, vector<double> &vTimeStamps) {
	ifstream fTimes;
	fTimes.open(strPathTimes.c_str());
	vTimeStamps.reserve(5000);
	vstrImageLeft.reserve(5000);
	vstrImageRight.reserve(5000);
	while (!fTimes.eof()) {
		string s;
		getline(fTimes, s);
		if (!s.empty()) {
			stringstream ss;
			ss << s;
			vstrImageLeft.push_back(strPathLeft + "/" + ss.str() + ".png");
			vstrImageRight.push_back(strPathRight + "/" + ss.str() + ".png");
			double t;
			ss >> t;
			vTimeStamps.push_back(t / 1e9);
		}
	}
}


