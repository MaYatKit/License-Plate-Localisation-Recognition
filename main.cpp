// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* \brief The entry point for the Inference Engine interactive_Vehicle_detection demo application
* \file security_barrier_camera_demo/main.cpp
* \example security_barrier_camera_demo/main.cpp
*/
#include <iostream>
#include <fstream>
#include <memory>
#include <chrono>
#include <algorithm>
#include <iterator>
#include <string>
#include <vector>
#include <queue>

#include <inference_engine.hpp>

#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>
#include <ie_iextension.h>
#include <ext_list.hpp>
#include <opencv2/videoio/videoio_c.h>


using namespace InferenceEngine;

static const int LICENSE_PLATE_LABEL = 1;
static const int BACKGROUND_LABEL = 0;

typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
static const std::string LPR_MODEL_PATH = "LPR.xml";
static const std::string LPL_MODEL_PATH = "LPL.xml";
static const std::string HARDWARE = "CPU";


struct BaseInferRequest {
    InferRequest::Ptr request;
    std::string inputName;
    size_t requestId;

    BaseInferRequest(ExecutableNetwork &net, std::string &inputName) :
            request(net.CreateInferRequestPtr()), inputName(inputName), requestId(0) {
    }

    virtual ~BaseInferRequest() {}

    virtual void startAsync() {
        request->StartAsync();
    }

    virtual void Infer() {
        request->Infer();
    }

    virtual void wait() {
        request->Wait(IInferRequest::WaitMode::RESULT_READY);
    }

    virtual void setBlob(const Blob::Ptr &frameBlob) {
        request->SetBlob(inputName, frameBlob);
    }

    virtual void setImage(const cv::Mat &frame) {
        Blob::Ptr inputBlob;

        inputBlob = request->GetBlob(inputName);
        matU8ToBlob<uint8_t>(frame, inputBlob);
    }

    virtual Blob::Ptr getBlob(const std::string &name) {
        return request->GetBlob(name);
    }


    void setId(size_t id) {
        requestId = id;
    }

    size_t getId() {
        return requestId;
    }

    using Ptr = std::shared_ptr<BaseInferRequest>;
};

struct BaseDetection {
    ExecutableNetwork net;
    InferencePlugin plugin;
    std::string topoName;
    std::string inputName;
    std::string outputName;

    BaseDetection(std::string topoName)
            : topoName(topoName) {}

    ExecutableNetwork *operator->() {
        return &net;
    }

    virtual CNNNetwork read() = 0;

    mutable bool enablingChecked = false;
};

struct LPLInferRequest : BaseInferRequest {
    cv::Size srcImageSize;

    LPLInferRequest(ExecutableNetwork &net, std::string &inputName) :
            BaseInferRequest(net, inputName), srcImageSize(0, 0) {}

    void setImage(const cv::Mat &frame) override {
        BaseInferRequest::setImage(frame);

        srcImageSize.height = frame.rows;
        srcImageSize.width = frame.cols;
    }

    void setSourceImageSize(int width, int height) {
        srcImageSize.height = height;
        srcImageSize.width = width;
    }

    cv::Size getSourceImageSize() {
        return srcImageSize;
    }

    using Ptr = std::shared_ptr<LPLInferRequest>;
};

struct LPRInferRequest : BaseInferRequest {
    std::string inputSeqName;

    LPRInferRequest(ExecutableNetwork &net, std::string &inputName, std::string &inputSeqName) :
            BaseInferRequest(net, inputName), inputSeqName(inputSeqName) {}

    void fillSeqBlob() {
        Blob::Ptr seqBlob = request->GetBlob(inputSeqName);
        int maxSequenceSizePerPlate = seqBlob->getTensorDesc().getDims()[0];
        // second input is sequence, which is some relic from the training
        // it should have the leading 0.0f and rest 1.0f
        float *blob_data = seqBlob->buffer().as<float *>();
        blob_data[0] = 0.0f;
        std::fill(blob_data + 1, blob_data + maxSequenceSizePerPlate, 1.0f);
    }

    void setBlob(const Blob::Ptr &frameBlob) override {
        BaseInferRequest::setBlob(frameBlob);
        if (!inputSeqName.empty()) {
            fillSeqBlob();
        }
    }

    void setImage(const cv::Mat &frame) override {
        BaseInferRequest::setImage(frame);
        if (!inputSeqName.empty()) {
            fillSeqBlob();
        }
    }

    using Ptr = std::shared_ptr<LPRInferRequest>;
};

struct LPLDetection : BaseDetection {
    int maxProposalCount;
    int objectSize;

    struct Result {
        int label;
        float confidence;
        cv::Rect location;
    };

    LPLDetection() : BaseDetection("Vehicle"), maxProposalCount(0), objectSize(0) {}

    LPLInferRequest::Ptr createInferRequest() {

        return std::make_shared<LPLInferRequest>(net, inputName);
    }

    CNNNetwork read() override {
        CNNNetReader netReader;
        netReader.ReadNetwork(LPL_MODEL_PATH);
        netReader.getNetwork().setBatchSize(1);
        std::string binFileName = fileNameNoExt(LPL_MODEL_PATH) + ".bin";
        netReader.ReadWeights(binFileName);

        InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
        if (inputInfo.size() != 1) {
            throw std::logic_error("Vehicle Detection network should have only one input");
        }
        InputInfo::Ptr &inputInfoFirst = inputInfo.begin()->second;
        inputInfoFirst->setInputPrecision(Precision::U8);

        inputInfoFirst->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
        inputInfoFirst->getInputData()->setLayout(Layout::NHWC);
        inputName = inputInfo.begin()->first;

        OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
        if (outputInfo.size() != 1) {
            throw std::logic_error("License Plate Detection network should have only one output");
        }
        DataPtr &_output = outputInfo.begin()->second;
        const SizeVector outputDims = _output->getTensorDesc().getDims();
        outputName = outputInfo.begin()->first;
        maxProposalCount = outputDims[2];
        objectSize = outputDims[3];
        if (objectSize != 7) {
            throw std::logic_error("Output should have 7 as a last dimension");
        }
        if (outputDims.size() != 4) {
            throw std::logic_error("Incorrect output dimensions for SSD");
        }
        _output->setPrecision(Precision::FP32);
        _output->setLayout(Layout::NCHW);

        return netReader.getNetwork();
    }

    void fetchResults(LPLInferRequest::Ptr request, std::vector<Result> &results) {
        cv::Size srcImageSize = request->getSourceImageSize();
        const float *detections = request->getBlob(
                outputName)->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
        // pretty much regular SSD post-processing
        for (int i = 0; i < maxProposalCount; i++) {
            float image_id = detections[i * objectSize + 0];  // in case of batch
            Result r;
            r.label = static_cast<int>(detections[i * objectSize + 1]);
            r.confidence = detections[i * objectSize + 2];
            if (r.confidence <= 0.5) {
                continue;
            }

            r.location.x = static_cast<int>(detections[i * objectSize + 3] * srcImageSize.width);
            r.location.y = static_cast<int>(detections[i * objectSize + 4] * srcImageSize.height);
            r.location.width = static_cast<int>(detections[i * objectSize + 5] * srcImageSize.width - r.location.x);
            r.location.height = static_cast<int>(detections[i * objectSize + 6] * srcImageSize.height - r.location.y);

            if (image_id < 0) {  // indicates end of detections
                break;
            }
            if (r.label == LICENSE_PLATE_LABEL) {
                results.push_back(r);
            }
        }
    }
};


struct LPRDetection : BaseDetection {
    std::string inputSeqName;
    const size_t maxSequenceSizePerPlate = 88;

    LPRDetection() : BaseDetection("License Plate Recognition") {}

    LPRInferRequest::Ptr createInferRequest() {
        return std::make_shared<LPRInferRequest>(net, inputName, inputSeqName);
    }

    std::string GetLicencePlateText(BaseInferRequest::Ptr request) {
        static std::vector<std::string> items = {
                "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                "<Anhui>", "<Beijing>", "<Chongqing>", "<Fujian>",
                "<Gansu>", "<Guangdong>", "<Guangxi>", "<Guizhou>",
                "<Hainan>", "<Hebei>", "<Heilongjiang>", "<Henan>",
                "<HongKong>", "<Hubei>", "<Hunan>", "<InnerMongolia>",
                "<Jiangsu>", "<Jiangxi>", "<Jilin>", "<Liaoning>",
                "<Macau>", "<Ningxia>", "<Qinghai>", "<Shaanxi>",
                "<Shandong>", "<Shanghai>", "<Shanxi>", "<Sichuan>",
                "<Tianjin>", "<Tibet>", "<Xinjiang>", "<Yunnan>",
                "<Zhejiang>", "-WJ",
                "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
                "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
                "U", "V", "W", "X", "Y", "Z"
        };
        // up to 88 items per license plate, ended with "-1"
        const auto data = request->getBlob(outputName)->buffer().as<float *>();
        std::string result;
        for (size_t i = 0; i < maxSequenceSizePerPlate; i++) {
            if (data[i] == -1)
                break;
            result += items[static_cast<size_t>(data[i])];
        }

        return result;
    }

    CNNNetwork read() override {
        slog::info << "Loading network files for Licence Plate Recognition (LPR)" << slog::endl;
        CNNNetReader netReader;
        netReader.ReadNetwork(LPR_MODEL_PATH);
        slog::info << "Batch size is forced to  1 for LPR Network" << slog::endl;
        netReader.getNetwork().setBatchSize(1);
        std::string binFileName = fileNameNoExt(LPR_MODEL_PATH) + ".bin";
        netReader.ReadWeights(binFileName);

        slog::info << "Checking LPR Network inputs" << slog::endl;
        InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());
        if (inputInfo.size() == 2) {
            auto sequenceInput = (++inputInfo.begin());
            inputSeqName = sequenceInput->first;
            if (sequenceInput->second->getTensorDesc().getDims()[0] != maxSequenceSizePerPlate) {
                throw std::logic_error("LPR post-processing assumes certain maximum sequences");
            }
        } else if (inputInfo.size() == 1) {
            inputSeqName = "";
        } else {
            throw std::logic_error("LPR should have 1 or 2 inputs");
        }

        InputInfo::Ptr &inputInfoFirst = inputInfo.begin()->second;
        inputInfoFirst->setInputPrecision(Precision::U8);
        inputInfoFirst->getInputData()->setLayout(Layout::NCHW);
        inputName = inputInfo.begin()->first;
        slog::info << "Checking LPR Network outputs" << slog::endl;
        OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());
        if (outputInfo.size() != 1) {
            throw std::logic_error("LPR should have 1 output");
        }
        outputName = outputInfo.begin()->first;

        return netReader.getNetwork();
    }
};


struct LicensePlateObject {
    std::string conf;
    cv::Rect location;
    std::string text;
};


void fillROIColor(cv::Mat &frame, cv::Rect roi, cv::Scalar color, double opacity) {
    if (opacity > 0) {
        roi = roi & cv::Rect(0, 0, frame.rows, frame.cols);
        cv::Mat textROI = frame(roi);
        cv::addWeighted(color, opacity, textROI, 1.0 - opacity, 0.0, textROI);
    }
}

void putTextOnImage(cv::Mat &frame, std::string str, cv::Point p,
                    cv::HersheyFonts font, double fontScale, cv::Scalar color,
                    int thickness, cv::Scalar bgcolor = cv::Scalar(),
                    double opacity = 0) {
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(str, font, 0.5, 1, &baseline);
    fillROIColor(frame, cv::Rect(cv::Point(p.x, p.y + baseline),
                                 cv::Point(p.x + textSize.width, p.y - textSize.height)),
                 bgcolor, opacity);
    cv::putText(frame, str, p, font, fontScale, color, thickness);
}


void setUpPlugin(InferencePlugin plugin) {
    /** Load default extensions lib for the CPU plugin (e.g. SSD's DetectionOutput)**/
    plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
}


int main(int argc, char *argv[]) {
    try {

        cv::VideoCapture cap;
        std::vector<cv::Mat> images;
        std::vector<std::string> videoFiles;


        cv::VideoCapture camera;
        camera.open(0);
        camera.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        camera.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        camera.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        camera.set(cv::CAP_PROP_FPS, 30);
        cap = camera;

        LPLDetection LPL;
        LPRDetection LPR;


        slog::info << "Loading plugin " << HARDWARE << slog::endl;
        InferencePlugin plugin_LPL = PluginDispatcher().getPluginByDevice(HARDWARE);
        InferencePlugin plugin_LRP = PluginDispatcher().getPluginByDevice(HARDWARE);

        setUpPlugin(plugin_LPL);
        setUpPlugin(plugin_LRP);


        LPR.net = plugin_LRP.LoadNetwork(LPR.read(), {});
        LPR.plugin = plugin_LRP;
        LPL.net = plugin_LPL.LoadNetwork(LPL.read(), {});
        LPL.plugin = plugin_LPL;
        std::shared_ptr<LPLInferRequest> availableLPLDetectionRequests = LPL.createInferRequest();
        std::shared_ptr<LPLInferRequest> nextLPLDetectionRequests = LPL.createInferRequest();


        std::shared_ptr<LPRInferRequest> LPRDetectionRequest = LPR.createInferRequest();
        cv::Mat frame;
        cv::Mat nextFrame;
        do {

        } while (!cap.read(frame));
        Blob::Ptr frameBlob;

        slog::info << "Start inference " << slog::endl;

        auto lastFrameTime = std::chrono::high_resolution_clock::now();
        std::deque<double> frameTimeCache;
        std::deque<std::pair<cv::Mat, double>> frameCache;


        int pause(1);

        availableLPLDetectionRequests->setBlob(wrapMat2Blob(frame));
        availableLPLDetectionRequests->setSourceImageSize(frame.cols, frame.rows);
        availableLPLDetectionRequests->startAsync();
        auto startTime = std::chrono::high_resolution_clock::now();
        double avgFps = 0;
        do {
            if (!cap.read(nextFrame)) {
                if (nextFrame.empty()) {
                    break;
                }
                continue;
            }
            std::vector<LicensePlateObject> licensePlates;
            std::vector<LPLDetection::Result> results;

            nextLPLDetectionRequests->setBlob(wrapMat2Blob(nextFrame));
            nextLPLDetectionRequests->setSourceImageSize(nextFrame.cols, nextFrame.rows);


            nextLPLDetectionRequests->startAsync();

            availableLPLDetectionRequests->wait();

            LPL.fetchResults(availableLPLDetectionRequests, results);


            size_t index = 0;
            for (; index < results.size(); index++) {
                LicensePlateObject lp;
                lp.location = results[index].location;
                std::ostringstream ss;
                ss << (results[index].confidence * 100);
                lp.conf = ss.str();

                auto clippedRect = lp.location & cv::Rect(0, 0, frame.cols, frame.rows);
                cv::Mat licensePlate = frame(clippedRect);
                LPRDetectionRequest->setImage(licensePlate);
                LPRDetectionRequest->Infer();
                std::string text = LPR.GetLicencePlateText(LPRDetectionRequest);
                lp.text = text;

                licensePlates.push_back(lp);
            }

            int licensePlateRectThickness = 1;
            double licensePlateFontScale = 1;
            int baseline = 0;
            auto v = licensePlates.begin();
            for (; v != licensePlates.end(); v++) {
                auto lp = v;

                if (lp == licensePlates.end()) {
                    continue;
                }
                cv::rectangle(frame, lp->location, cv::Scalar(0, 255, 0), licensePlateRectThickness);
                cv::putText(frame,
                            lp->conf + "%",
                            cv::Point(lp->location.x + lp->location.width, lp->location.y + lp->location.height),
                            cv::FONT_HERSHEY_COMPLEX_SMALL, licensePlateFontScale,
                            cv::Scalar(102, 102, 255));


                cv::Size licensePlateFontSize = cv::getTextSize(lp->text, cv::FONT_HERSHEY_COMPLEX_SMALL,
                                                                licensePlateFontScale,
                                                                licensePlateRectThickness,
                                                                &baseline);
                cv::putText(frame,
                            lp->text,
                            cv::Point(lp->location.x,
                                      lp->location.y - licensePlateFontSize.height / 2),
                            cv::FONT_HERSHEY_COMPLEX_SMALL, licensePlateFontScale,
                            cv::Scalar(102, 255, 153), licensePlateRectThickness);
            }


            fillROIColor(frame, cv::Rect(0, 0, 300, 80), cv::Scalar(255, 0, 0), 0.6);

            std::ostringstream out;


            auto interval = std::chrono::duration_cast<ms>(std::chrono::high_resolution_clock::now() - lastFrameTime);
            lastFrameTime = std::chrono::high_resolution_clock::now();
            if (frameTimeCache.size() == 5) {
                frameTimeCache.pop_front();
            }
            frameTimeCache.push_back(interval.count());
            double average = 0;
            for (double time : frameTimeCache) {
                average += time;
            }
            average = average / frameTimeCache.size();


            avgFps += (1000. / average);

            out << std::fixed << std::setprecision(2)
                << 1000. / average << " fps";
            putTextOnImage(frame, out.str(), cv::Point(5, 35), cv::FONT_HERSHEY_TRIPLEX, 1.1,
                           cv::Scalar(255, 255, 255), 2);

            out.str("");
            out << "License plate amount: " << licensePlates.size();
            putTextOnImage(frame, out.str(), cv::Point(5, 60), cv::FONT_HERSHEY_TRIPLEX, 0.6,
                           cv::Scalar(255, 255, 255), 1);


            if (frame.empty())
                break;
            cv::imshow("Detection results", frame);
            const int key = cv::waitKey(pause);
            if (key == 27) {
                break;
            } else if (key == 32) {
                pause = (pause + 1) & 1;
            }

            frame = nextFrame;
            nextFrame = cv::Mat();
            availableLPLDetectionRequests.swap(nextLPLDetectionRequests);
        } while (true);
        cap.release();
        cv::destroyAllWindows();
    }
    catch (const std::exception &error) {
        std::cerr << "[ ERROR ] " << error.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    return 0;
}