/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include "yolo.h"
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>

#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;
using namespace cv;
int mk=0;
// 任务队列
std::queue<vector<Mat>> taskQueue; // 存储目标检测后的结果
std::mutex queueMutex;
std::condition_variable queueCV;
bool processingDone = false; // 标志目标检测是否完成

void LoadImagesRgbdTum(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
				vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);

void detectionThread(int& nImages, const vector<string> &vstrImageFilenamesRGB, const float& imageScale, const string &argv) {
	Mat mask, imRGB;
	vector<Mat> fullMask;
	for(int ni = 20; ni < nImages; ni += 1) {
		// Read image and depthmap from file
		imRGB = cv::imread(argv + "/" + vstrImageFilenamesRGB[ni], cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);
		if(imRGB.empty())
		{
			cerr << endl << "Failed to load image at: "
				 << argv << "/" << vstrImageFilenamesRGB[ni] << endl;
			return;
		}

		if(imageScale != 1.f)
		{
			int width = imRGB.cols * imageScale;
			int height = imRGB.rows * imageScale;
			cv::resize(imRGB, imRGB, cv::Size(width, height));
		}

		mask = Mat::zeros(imRGB.size(), CV_8UC1); // mask对应单通道图像所以是8UC1 (8 uchar channel 1)
		fullMask = YOLO::DetectYolo(imRGB, mask);
		{
			std::lock_guard<std::mutex> lock(queueMutex); // **只对 taskQueue 加锁**
			taskQueue.push(fullMask);
		}
		queueCV.notify_one(); // 通知主线程有新数据
	}
	// 目标检测完成，设置标志
	{
		std::lock_guard<std::mutex> lock(queueMutex);
		processingDone = true;
	}
	queueCV.notify_one(); // 通知主线程任务完成
}

// 计算特征点的欧几里得距离
double computeDistance(const Point2f& p1, const Point2f& p2) {
	return norm(p1 - p2);
}

// 过滤动态特征点
vector<Point2f> filterDynamicPoints(const vector<Point2f>& prevPts,const vector<Point2f>& currPts,const vector<uchar>& status,double threshold){
	vector<Point2f> staticPoints;
	for (size_t i = 0; i < prevPts.size(); i++) {
		if (status[i] && computeDistance(prevPts[i], currPts[i]) < threshold) {
			staticPoints.push_back(currPts[i]); // 只保留静态点
		}
	}
	return staticPoints;
}

int main(int argc, char **argv)
{
    if(argc != 5)
    {
        cerr << endl << "Usage: ./rgbd_tum path_to_vocabulary path_to_settings path_to_sequence path_to_association" << endl;
        return 1;
    }
    // Retrieve paths to images
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;
    string strAssociationFilename = string(argv[4]);
    LoadImagesRgbdTum(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);

    // Check consistency in the number of images and depthmaps
    int nImages = vstrImageFilenamesRGB.size();
    if(vstrImageFilenamesRGB.empty())
    {
        cerr << endl << "No images found in provided path." << endl;
        return 1;
    }
    else if(vstrImageFilenamesD.size()!=vstrImageFilenamesRGB.size())
    {
        cerr << endl << "Different number of images for rgb and depth." << endl;
        return 1;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::RGBD,true);
    float imageScale = SLAM.GetImageScale();

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat imRGB, imD;
	double total_elapsed_time = 0.0; // 总计时间（秒）
	std::chrono::high_resolution_clock::time_point start;
	std::chrono::high_resolution_clock::time_point end;

	Mat ImRGB = cv::imread(string(argv[3])+"/"+vstrImageFilenamesRGB[0],cv::IMREAD_UNCHANGED);
	Mat _mask = Mat::zeros(ImRGB.size(), CV_8UC1);
	vector<Mat> _fullMask;
	_fullMask.push_back(_mask);
	for (int i = 0; i < 20; i++) {
		taskQueue.push(_fullMask);
	}

	std::thread t(detectionThread, std::ref(nImages), std::ref(vstrImageFilenamesRGB), std::ref(imageScale), string(argv[3]));

	for(int ni=0; ni<nImages; ni++)
    {
        // Read image and depthmap from file
        imRGB = cv::imread(string(argv[3])+"/"+vstrImageFilenamesRGB[ni],cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);
        imD = cv::imread(string(argv[3])+"/"+vstrImageFilenamesD[ni],cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(imRGB.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenamesRGB[ni] << endl;
            return 1;
        }

        if(imageScale != 1.f)
        {
            int width = imRGB.cols * imageScale;
            int height = imRGB.rows * imageScale;
            cv::resize(imRGB, imRGB, cv::Size(width, height));
            cv::resize(imD, imD, cv::Size(width, height));
        }

    	if (ni>=60 && ni<=156)
    		start = std::chrono::high_resolution_clock::now(); // 记录开始时间

    	// Mat mask = Mat::zeros(imRGB.size(), CV_8UC1); // mask对应单通道图像所以是8UC1 (8 uchar channel 1)
		vector<Mat> fullMask;
		std::unique_lock<std::mutex> lock(queueMutex);
		queueCV.wait(lock, [] { return !taskQueue.empty() || processingDone; });

		if (!taskQueue.empty()) {
			fullMask = taskQueue.front();
			taskQueue.pop();
		}

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        // Pass the image to the SLAM system
        SLAM.TrackRGBD(imRGB,imD,tframe,fullMask);

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

    	if (ni>=60 && ni<=156) {
    		end = std::chrono::high_resolution_clock::now(); // 记录结束时间
    		std::chrono::duration<double> elapsed = end - start; // 计算时间（单位：秒）
    		total_elapsed_time += elapsed.count(); // 累加时间
    	}

    	if (ni==156)
    		std::cout << "96フレームの実行時間：" << total_elapsed_time << " 秒" << std::endl;
        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }

	t.join();
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
    SLAM.SaveTrajectoryTUM("yolo_rgbd_tum_f.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("yolo_rgbd_tum_kf.txt");

	// // 读取前后两帧
	// Mat prevImage  = imread("frame1.png", IMREAD_GRAYSCALE);
	// Mat nextImage  = imread("frame2.png", IMREAD_GRAYSCALE);
	//
	// Mat img = imread("frame1.png", cv::IMREAD_UNCHANGED);
	// // Yolo
	// Mat grayImage,  output;
	// if (img.channels() == 1) {
	// 	cvtColor(img, output, cv::COLOR_GRAY2BGR);
	// }else {
	// 	cvtColor(img, grayImage, COLOR_BGR2GRAY);
	// }
	// Mat mask = Mat::zeros(img.size(), CV_8UC1);
	// Mat colorImage = img.clone();
	// YOLO yolo_model(yolo_nets[2]);
	// // **1. 候选框（假设检测到的目标）**
	//
	// imshow("img", mask);
	// waitKey(0);
	// vector<Point2f> prevPts, nextPts;
	// goodFeaturesToTrack(prevImg, prevPts, 100, 0.01, 10);
	//
	// vector<cv::KeyPoint> mvKeys;
	// cv::Mat mDescriptors;
	// vector<int> vLapping;
	// ORB_SLAM3::ORBextractor::operator()(prevImage,mask,mvKeys,mDescriptors,vLapping);
	// // 计算光流（使用Lucas-Kanade方法）
	// vector<uchar> status;
	// vector<float> err;
	// calcOpticalFlowPyrLK(prevImage, nextImage, prevPts, nextPts, status, err);
	//
	// // 用来保存去除动态特征点后的结果
	// vector<Point2f> filteredPts;
	//
	// // 设定光流位移阈值，用于判断动态特征点
	// float displacementThreshold = 2.0;  // 可以调整这个阈值
	//
	// // 遍历每个特征点，判断其是否在掩码中的白色区域，且位移是否过大
	// for (size_t i = 0; i < prevPts.size(); i++) {
	// 	if (status[i] == 1) {  // 光流计算成功
	// 		// 判断当前特征点是否在掩码中的白色区域
	// 		if (mask.at<uchar>(prevPts[i]) == 255) {
	// 			// 计算光流的位移
	// 			float dx = nextPts[i].x - prevPts[i].x;
	// 			float dy = nextPts[i].y - prevPts[i].y;
	// 			float displacement = sqrt(dx * dx + dy * dy);
	//
	// 			// 如果位移小于阈值，认为该点是静态的，保留它
	// 			if (displacement < displacementThreshold) {
	// 				filteredPts.push_back(prevPts[i]);
	// 			}
	// 		}
	// 	}
	// }
	//
	// // 显示去除动态特征点后的结果
	// Mat outputImg = nextImage.clone();
	// for (size_t i = 0; i < filteredPts.size(); i++) {
	// 	circle(outputImg, filteredPts[i], 3, Scalar(0, 255, 0), -1);  // 绿色圆点表示静态特征点
	// }
	//
	// imshow("Filtered Feature Points", outputImg);
	// waitKey(0);

    return 0;
}

void LoadImagesRgbdTum(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)
{
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while(!fAssociation.eof())
    {
        string s;
        getline(fAssociation,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);

        }
    }
}
