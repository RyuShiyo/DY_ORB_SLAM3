#include "yolo.h"

int m = 0;

YOLO::YOLO(Net_config config) {
    // cout << "Net use " << config.netname << endl;
    this->confThreshold = config.confThreshold;
    this->nmsThreshold = config.nmsThreshold;
    this->inpWidth = config.inpWidth;
    this->inpHeight = config.inpHeight;
    strcpy(this->netname, config.netname.c_str());

    ifstream ifs(config.classesFile.c_str());
    string line;
    while (getline(ifs, line)) this->classes.push_back(line);

    this->net = readNetFromDarknet(config.modelConfiguration, config.modelWeights);
    this->net.setPreferableBackend(DNN_BACKEND_OPENCV);
    this->net.setPreferableTarget(DNN_TARGET_CPU);
}

vector<CandidateBox> YOLO::postprocess(Mat &frame, const vector<Mat> &outs, Mat &output)
// Remove the bounding boxes with low confidence using non-maxima suppression
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    vector<CandidateBox> boxes_mask;

    for (size_t i = 0; i < outs.size(); ++i) {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float *data = (float *) outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > this->confThreshold) {
                int centerX = (int) (data[0] * frame.cols);
                int centerY = (int) (data[1] * frame.rows);
                int width = (int) (data[2] * frame.cols);
                int height = (int) (data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float) confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);
    int flag = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        Rect box = boxes[idx];
        if (classIds[idx] == 0) flag++; // if people
        boxes_mask.push_back({box,classIds[idx]});
        // this->drawPred(classIds[idx], confidences[idx], box.x, box.y,
        //                box.x + box.width, box.y + box.height, output);
    }
    if (flag == 0) boxes_mask.clear();
    return boxes_mask;
}

void YOLO::drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat &frame)
// Draw the predicted bounding box
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 3);

    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!this->classes.empty()) {
        CV_Assert(classId < (int)this->classes.size());
        label = this->classes[classId] + ":" + label;
        // printf("%s", label.c_str());
    }

    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    // rectangle(frame, Point(left, top - int(1.5 * labelSize.height)),
    //           Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.85, Scalar(0, 255, 0), 2);
}

vector<CandidateBox> YOLO::detect(Mat &frame, Mat &output) {
    Mat blob;
    blobFromImage(frame, blob, 1 / 255.0, Size(this->inpWidth, this->inpHeight), Scalar(0, 0, 0), true, false);
    this->net.setInput(blob);
    vector<Mat> outs;
    this->net.forward(outs, this->net.getUnconnectedOutLayersNames());
    vector<CandidateBox> boxes;
    boxes = this->postprocess(frame, outs, output);
    // return boxes;

    // vector<double> layersTimes;
    // double freq = getTickFrequency() / 1000;
    // double t = net.getPerfProfile(layersTimes) / freq;
    // string label = format("%s Inference time : %.2f ms", this->netname, t);
    //
    // int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    // double fontScale = 1.0;
    // int thickness = 2;
    // int baseline = 0;
    // cv::Size textSize = cv::getTextSize(label, FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
    // baseline += thickness; // 让背景稍微大一点
    // cv::Point textOrg(0, 30);
    //
    // // 计算背景矩形区域
    // cv::Point rectTopLeft(textOrg.x, textOrg.y - textSize.height - baseline);
    // cv::Point rectBottomRight(textOrg.x + textSize.width, textOrg.y + baseline);
    // cv::rectangle(output, rectTopLeft, rectBottomRight, cv::Scalar(255, 255, 255), cv::FILLED);

    // putText(output, label, textOrg, fontFace, fontScale, Scalar(0, 0, 255), thickness);
    // imwrite(format("%s_out.jpg", this->netname), frame);
    // imwrite("./Output/Tum/yolo3_detect_img_"+to_string(m)+".png",output);
    // m++;

    return boxes;
}

vector<Mat> YOLO::DetectYolo(Mat& imRGB, Mat& mask) {
    // Yolo
    Mat output = imRGB.clone();
    if (imRGB.channels() == 1) {
        cvtColor(imRGB, output, cv::COLOR_GRAY2BGR);
    }

    vector<Mat> fullMask;
    Mat _mask = mask.clone();
    Mat colorImage = imRGB.clone();
    YOLO yolo_model(yolo_nets[2]);
    vector<CandidateBox> boxes = yolo_model.detect(colorImage, output);

    for (const auto& box : boxes) {
        if (box.label == 0) {
            rectangle(_mask, box.box, cv::Scalar(255,255,255), cv::FILLED);
            // for (const auto& _box : boxes) {
            //     if (_box.label != 0) {
            //         rectangle(_mask, _box.box, cv::Scalar(0,0,0), cv::FILLED);
            //     }
            // }
            fullMask.push_back(_mask.clone());
            rectangle(_mask, box.box, cv::Scalar(0,0,0), cv::FILLED);

            rectangle(mask, box.box, cv::Scalar(255,255,255), cv::FILLED);
        }

        else
            rectangle(mask, box.box, cv::Scalar(0,0,0), cv::FILLED);
    }

    fullMask.push_back(mask.clone());

    return fullMask;
}