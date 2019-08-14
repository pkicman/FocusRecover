#include "FocusReconstructor.h"

void FocusReconstructor::preProcessImage(cv::Mat & inImg, cv::Mat & outImg)
{
	cv::Mat gray;
	convertToGray(inImg, gray);
	applyGaussianBlur(gray, outImg, m_blur_kernel_size);
}

void FocusReconstructor::computeFocusScore(cv::Mat & inImg, cv::Mat & outImg)
{
	threshToZero(inImg, m_ml_threshold);
	sumOverKernel(inImg, outImg, m_focus_kernel_size);
}

void FocusReconstructor::addImage(const cv::Mat & inImg)
{
	m_inContainer.push_back(inImg.clone());
}

void FocusReconstructor::processInputs()
{
	if (m_inContainer.empty()) return;
	cv::Mat gray, modifiedLap, focusScore;
	
	for (auto &frame : m_inContainer)
	{
		preProcessImage(frame, gray);
		computeModifiedLaplace(gray, modifiedLap);
		computeFocusScore(modifiedLap, focusScore);
		m_focusContainer.push_back(focusScore.clone());
	}
}

void FocusReconstructor::reconstructCoarse(cv::Mat & outImg, cv::Mat & depthMap)
{
	if (m_inContainer.empty()) return;
	// Copy the furthest image as a default
	m_inContainer.back().copyTo(outImg);
	depthMap = cv::Mat::zeros(m_inContainer.back().size(), CV_32FC1);
	std::vector<float> focus_score;
	focus_score.reserve(m_focusContainer.size());
	int nimg = m_focusContainer.size();
	int nrows = m_focusContainer.back().rows;
	int ncols = m_focusContainer.back().cols;

	for (int i = 0; i < nrows; ++i)
		for (int j = 0; j < ncols; ++j)
		{
			focus_score.clear();
			for (int k = 0; k < nimg; ++k)
			{				
				focus_score.push_back(m_focusContainer[k].at<float>(i, j));
			}
			// find position of most focused image
			int idx = std::distance(focus_score.begin(),
				std::max_element(focus_score.begin(), focus_score.end()));
			if (focus_score[idx] > m_focus_threshold)
			{
				outImg.at<cv::Vec3b>(i, j)[0] = m_inContainer[idx].at<cv::Vec3b>(i, j)[0];
				outImg.at<cv::Vec3b>(i, j)[1] = m_inContainer[idx].at<cv::Vec3b>(i, j)[1];
				outImg.at<cv::Vec3b>(i, j)[2] = m_inContainer[idx].at<cv::Vec3b>(i, j)[2];
				depthMap.at<float>(i, j) = 1 - (float)idx / (float)nimg;
			}
		}
}

void FocusReconstructor::reconstructFine(cv::Mat & outImg, cv::Mat & depthMap)
{
	if (m_inContainer.empty()) return;
	// Copy the furthest image as a default
	m_inContainer.back().copyTo(outImg);
	depthMap = cv::Mat::zeros(m_inContainer.back().size(), CV_32FC1);
	std::vector<float> focus_score;
	focus_score.reserve(m_focusContainer.size());
	int nimg = m_focusContainer.size();
	int nrows = m_focusContainer.back().rows;
	int ncols = m_focusContainer.back().cols;

	int counter = 0;
	for (int i = 0; i < nrows; ++i)
		for (int j = 0; j < ncols; ++j)
		{
			focus_score.clear();
			for (int k = 0; k < nimg; ++k)
				focus_score.push_back(m_focusContainer[k].at<float>(i, j));

			// find position of most focused image
			int idx = std::distance(focus_score.begin(),
				std::max_element(focus_score.begin(), focus_score.end()));
			if (focus_score[idx] > m_focus_threshold)
			{
				bool refined_result = false;
				float fine_idx;
				if (idx > 1 && idx < nimg - 1)
				{
					if (focus_score[idx - 1] > 0 && focus_score[idx + 1] > 0)
					{
						float a, b, c;
						fitQuadratic((float)idx - 1, focus_score[idx - 1],
							(float)idx, focus_score[idx],
							(float)idx + 1, focus_score[idx + 1],
							a, b, c);
						if (abs(a) > 0.0000000001) {
							refined_result = true;
							fine_idx = -b / (2 * a);
							//std::cout << idx << " " << fine_idx << std::endl;
							counter++;
						}
					}
				}
				outImg.at<cv::Vec3b>(i, j)[0] = m_inContainer[idx].at<cv::Vec3b>(i, j)[0];
				outImg.at<cv::Vec3b>(i, j)[1] = m_inContainer[idx].at<cv::Vec3b>(i, j)[1];
				outImg.at<cv::Vec3b>(i, j)[2] = m_inContainer[idx].at<cv::Vec3b>(i, j)[2];
				if (refined_result)
					depthMap.at<float>(i, j) = 1 - fine_idx / (float)nimg;
				else
					depthMap.at<float>(i, j) = 1 - (float)idx / (float)nimg;
			}
		}
}
