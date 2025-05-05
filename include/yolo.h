#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

struct Net_config
{
	float confThreshold; // Confidence threshold
	float nmsThreshold;  // Non-maximum suppression threshold
	int inpWidth;  // Width of network's input image
	int inpHeight; // Height of network's input image
	string classesFile;
	string modelConfiguration;
	string modelWeights;
	string netname;
};

static const Net_config yolo_nets[4] = {
	{0.5, 0.4, 416, 416,"coco.names", "YoloModel/yolov3/yolov3.cfg", "YoloModel/yolov3/yolov3.weights", "yolov3"},
	{0.5, 0.4, 608, 608,"coco.names", "YoloModel/yolov4/yolov4.cfg", "YoloModel/yolov4/yolov4.weights", "yolov4"},
	{0.5, 0.4, 320, 320,"coco.names", "YoloModel/yolo-fastest/yolo-fastest-xl.cfg", "YoloModel/yolo-fastest/yolo-fastest-xl.weights", "yolo-fastest"},
	{0.5, 0.4, 320, 320,"coco.names", "YoloModel/yolobile/csdarknet53s-panet-spp.cfg", "YoloModel/yolobile/yolobile.weights", "yolobile"}
};

struct CandidateBox {
	cv::Rect box;       // 候选框
	int label;  // 对应的类别
};

class YOLO
{
public:
	YOLO(Net_config config);
	vector<CandidateBox> detect(Mat& frame, Mat& output);
	static vector<Mat> DetectYolo(Mat& imRGB, Mat& mask);
private:
	float confThreshold;
	float nmsThreshold;
	int inpWidth;
	int inpHeight;
	char netname[20];
	vector<string> classes;
	Net net;
	void maskOutsideBoxes(const cv::Mat &inputImage, const Rect &box, cv::Mat &outputImage);
	vector<CandidateBox> postprocess(Mat& frame, const vector<Mat>& outs, cv::Mat& output);
	void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);
};
