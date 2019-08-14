#pragma once
#include <vector>
#include <opencv2/imgproc.hpp>
#include "utilities.h"

class FocusReconstructor {
private:
	// algorithm parameters
	unsigned int m_blur_kernel_size;
	unsigned int m_focus_kernel_size;
	float m_ml_threshold;
	float m_focus_threshold;
	std::vector<cv::Mat> m_inContainer, m_focusContainer;

	void preProcessImage(cv::Mat &inImg, cv::Mat &outImg);
	void computeFocusScore(cv::Mat &inImg, cv::Mat &outImg);

public:
	FocusReconstructor(unsigned int bks, unsigned int fks, float mlt, float ft) :
		m_blur_kernel_size(bks), m_focus_kernel_size(fks),
		m_ml_threshold(mlt), m_focus_threshold(ft) {};

	void addImage(const cv::Mat &inImg);
	void processInputs();
	void reconstructCoarse(cv::Mat &outImg, cv::Mat &depthMap);
	void reconstructFine(cv::Mat &outImg, cv::Mat &depthMap);
};