#include "FocusReconstructor.h"

void FocusReconstructor::preProcessImage(myMat<uchar>& inImg, myMat<uchar>& outImg)
{
	myMat<uchar> gray;
	convertToGray(inImg, gray);
	applyGaussianBlur(gray, outImg, m_blur_kernel_size);
}

void FocusReconstructor::computeFocusScore(myMat<float>& inImg, myMat<float>& outImg)
{
	threshToZero(inImg, m_ml_threshold);
	sumOverKernel(inImg, outImg, m_focus_kernel_size);
}

void FocusReconstructor::addImage(const cv::Mat & inImg)
{
	m_inContainerNew.emplace_back(inImg);
}

void FocusReconstructor::processInputs()
{	
	if (m_inContainerNew.empty()) return;
	m_focusContainerNew.reserve(m_inContainerNew.size());
	myMat<uchar> mygray;
	myMat<float> mymodifiedLap, myfocusScore;
	
	for (auto &frame : m_inContainerNew)
	{
		preProcessImage(frame, mygray);
		computeModifiedLaplace(mygray, mymodifiedLap);
		computeFocusScore(mymodifiedLap, myfocusScore);
		m_focusContainerNew.push_back(myfocusScore);
	}
}

void FocusReconstructor::reconstructCoarse(myMat<uchar>& outImg, myMat<float>& depthMap)
{
	if (m_inContainerNew.empty()) return;
	int nimg = m_focusContainerNew.size();
	int nrows = m_focusContainerNew.back().nrows();
	int ncols = m_focusContainerNew.back().ncols();

	// Copy the furthest image as a default
	outImg = m_inContainerNew.back();
	depthMap = myMat<float>(nrows, ncols, 1);
	depthMap.fill(0.0f);
	std::vector<float> focus_score;
	focus_score.reserve(nimg);

	for (int i = 0; i < nrows; ++i)
		for (int j = 0; j < ncols; ++j)
		{
			focus_score.clear();
			for (int k = 0; k < nimg; ++k)
				focus_score.push_back(m_focusContainerNew[k].at(i, j)[0]);

			// find position of most focused image
			int idx = std::distance(focus_score.begin(),
				std::max_element(focus_score.begin(), focus_score.end()));
			if (focus_score[idx] > m_focus_threshold)
			{
				outImg.at(i, 3 * j)[0] = m_inContainerNew[idx].at(i, 3 * j)[0];
				outImg.at(i, 3 * j)[1] = m_inContainerNew[idx].at(i, 3 * j)[1];
				outImg.at(i, 3 * j)[2] = m_inContainerNew[idx].at(i, 3 * j)[2];
				depthMap.at(i, j)[0] = 1 - (float)idx / (float)nimg;
			}
		}
}

void FocusReconstructor::reconstructFine(myMat<uchar>& outImg, myMat<float>& depthMap)
{
	if (m_inContainerNew.empty()) return;
	int nimg = m_focusContainerNew.size();
	int nrows = m_focusContainerNew.back().nrows();
	int ncols = m_focusContainerNew.back().ncols();

	// Copy the furthest image as a default
	outImg = m_inContainerNew.back();
	depthMap = myMat<float>(nrows, ncols, 1);
	depthMap.fill(0.0f);
	std::vector<float> focus_score;
	focus_score.reserve(nimg);
	
	for (int i = 0; i < nrows; ++i)
		for (int j = 0; j < ncols; ++j)
		{
			focus_score.clear();
			for (int k = 0; k < nimg; ++k)
				focus_score.push_back(m_focusContainerNew[k].at(i, j)[0]);

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
						}
					}
				}
				outImg.at(i, 3*j)[0] = m_inContainerNew[idx].at(i, 3*j)[0];
				outImg.at(i, 3*j)[1] = m_inContainerNew[idx].at(i, 3*j)[1];
				outImg.at(i, 3*j)[2] = m_inContainerNew[idx].at(i, 3*j)[2];
				if (refined_result)
					depthMap.at(i, j)[0] = 1 - fine_idx / (float)nimg;
				else
					depthMap.at(i, j)[0] = 1 - (float)idx / (float)nimg;
			}
		}
}
