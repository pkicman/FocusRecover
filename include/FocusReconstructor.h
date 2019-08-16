#pragma once
#include <vector>
#include <opencv2/core.hpp>

#include "myMat.h"
#include "utilities.h"

class FocusReconstructor {
private:
	// algorithm parameters
	unsigned int m_blur_kernel_size;
	unsigned int m_focus_kernel_size;
	float m_ml_threshold;
	float m_focus_threshold;
	std::vector< myMat<uchar> > m_inContainerNew;
	std::vector< myMat<float> > m_focusContainerNew;

	void preProcessImage(myMat<uchar> &inImg, myMat<uchar> &outImg);
	void computeFocusScore(myMat<float> &inImg, myMat<float> &outImg);

public:
	FocusReconstructor(unsigned int bks, unsigned int fks, float mlt, float ft, unsigned int n_frames) :
		m_blur_kernel_size(bks), m_focus_kernel_size(fks),
		m_ml_threshold(mlt), m_focus_threshold(ft) 
	{
		m_inContainerNew.reserve(n_frames);
	};

	void addImage(const cv::Mat &inImg);
	void processInputs();
	void reconstructCoarse(myMat<uchar> &outImg, myMat<float> &depthMap);
	void reconstructFine(myMat<uchar> &outImg, myMat<float> &depthMap);
};