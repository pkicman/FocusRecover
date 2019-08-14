#include "utilities.h"


void fitQuadratic(float x1, float y1, float x2, float y2, float x3, float y3, float &a, float &b, float &c)
{
	// To avoid using linear algebra libraries an anlytical solution is used
	float denom = (x1 - x2)*(x1 - x3)*(x2 - x3);
	a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom;
	b = (x3*x3 * (y1 - y2) + x2*x2 * (y3 - y1) + x1*x1 *(y2 - y3)) / denom;
	c = (x2 * x3 * (x2 - x3) * y1 + x3*x1*(x3 - x1)*y2 + x1 * x2 * (x1 - x2) * y3) / denom;
}

void threshToZero(cv::Mat &inImg, const float threshold)
{
	// set (in-place) to zero every pixel whose value is below threshold
	CV_Assert(inImg.depth() == CV_32FC1);

	float *p;
	for (int i = 0; i < inImg.rows; ++i)
	{
		p = inImg.ptr<float>(i);
		for (int j = 0; j < inImg.cols; ++j)
			if (p[j] < threshold) p[j] = 0.0f;
	}
}

void convertToGray(const cv::Mat &inImg, cv::Mat &outImg)
{
	// Assuming standard opencv convention of BGR color coding
	if (inImg.empty()) return;
	if (inImg.type() == CV_8UC3)
		outImg = cv::Mat(inImg.size(), CV_8U);
	else
		return;

	uchar *out;
	const cv::Vec3b *in;
	for (int i = 0; i < inImg.rows; ++i)
	{
		out = outImg.ptr<uchar>(i);
		in = inImg.ptr<cv::Vec3b>(i);
		for (int j = 0; j < inImg.cols; ++j)
		{
			out[j] = (uchar)(0.114*(float)in[j][0] + 0.587*(float)in[j][1] + 0.299*(float)in[j][2]);
		}
	}
}

void applyGaussianBlur(const cv::Mat & inImg, cv::Mat & outImg, const unsigned int kernel_size)
{
	std::vector<float> kernel;

	// use precomputed kernel coefficients for kernel 3x3 and 5x5
	// other kernel sizes not supported
	if (kernel_size == 1)
	{
		outImg = inImg.clone();
		return;
	}
	else if (kernel_size == 3)
	{
		std::array<float, 2> a = { 0.44198f,	0.27901f };
		kernel.insert(kernel.end(), a.begin(), a.end());
	}
	else if (kernel_size == 5)
	{
		std::array<float, 3> a = { 0.38774f, 0.24477f, 0.06136f };
		kernel.insert(kernel.end(), a.begin(), a.end());
	}
	else
	{
		std::cout << "Unsupported kernel size" << std::endl;
		outImg = inImg.clone();
		return;
	}

	cv::Mat convHor = cv::Mat::zeros(inImg.size(), CV_8UC1);

	// convolve horizontally
	int d = (kernel_size - 1) / 2;
	const uchar *r_in;
	uchar *r_out;
	for (int i = 0; i < inImg.rows; ++i)
	{
		r_in = inImg.ptr<uchar>(i);
		r_out = convHor.ptr<uchar>(i);
		for (int j = d; j < inImg.cols - d; ++j)
		{
			r_out[j] = (uchar)(kernel[0] * (float) r_in[j]);
			for (int k = 1; k <= d; ++k)
				r_out[j] += (uchar)(kernel[k]*((float)r_in[j - k] + (float)r_in[j + k]));
		}
		// first columns with mirror of edge pixels
		for (int j = 0; j < d; ++j)
		{
			r_out[j] = (uchar)(kernel[0] * (float)r_in[j]);
			for (int k = 1; k <= d; ++k)
			{
				r_out[j] += (uchar)(kernel[k] * (float)r_in[j + k]);
				// this line just reflects the border pixels
				// should probably be changed to something more readable
				r_out[j] += (uchar)(kernel[k] * (float) ((j - k < 0) ? r_in[k - j - 1] : r_in[j - k]));
			}
		}
		// last columns with mirror of edge pixels
		for (int j = inImg.cols - 1; j > inImg.cols - d - 1; --j)
		{
			r_out[j] = (uchar)(kernel[0] * (float)r_in[j]);
			for (int k = 1; k <= d; ++k)
			{
				r_out[j] += (uchar)(kernel[k]*(float)r_in[j - k]);
				r_out[j] += (uchar)(kernel[k]*(float)((j + k >= inImg.cols) ? r_in[2 * inImg.cols - (j + k) - 1] : r_in[j + k]));
			}
		}
	}

	// convolve vertically
	outImg = cv::Mat::zeros(inImg.size(), CV_8UC1);
	
	for (int i = d; i < outImg.rows - d; ++i)
	{
		r_out = outImg.ptr<uchar>(i);
		r_in = convHor.ptr<uchar>(i);

		for (int j = 0; j < outImg.cols; ++j)
		{
			r_out[j] = (uchar)(kernel[0] * (float)r_in[j]);
			for (int k = 1; k <= d; ++k)
			{
				r_out[j] += (uchar)(kernel[k] * (float)((r_in - outImg.cols*k)[j]));
				r_out[j] += (uchar)(kernel[k] * (float)((r_in + outImg.cols*k)[j]));
			}
		}
	}

	// first rows
	for (int i = 0; i < d; ++i)
	{
		r_out = outImg.ptr<uchar>(i);
		r_in = convHor.ptr<uchar>(i);
		for (int j = 0; j < outImg.cols; ++j)
		{
			r_out[j] = (uchar)(kernel[0] * (float)r_in[j]);
			for (int k = 1; k <= d; ++k)
			{
				r_out[j] += (uchar)(kernel[k] * (float) ((i - k < 0) ? convHor.at<uchar>(-i + k - 1, j) : (r_in - outImg.cols*k)[j]));
				r_out[j] += (uchar)(kernel[k] * (float)((r_in + outImg.cols*k)[j]));
			}
		}
	}

	// last rows
	for (int i = outImg.rows - 1; i > outImg.rows - d - 1; --i)
	{
		r_out = outImg.ptr<uchar>(i);
		r_in = convHor.ptr<uchar>(i);
		for (int j = 0; j < outImg.cols; ++j)
		{
			r_out[j] = (uchar)(kernel[0] * (float)r_in[j]);
			for (int k = 1; k <= d; ++k)
			{
				r_out[j] += (uchar)(kernel[k]*(float)((r_in - outImg.cols*k)[j]));
				r_out[j] += (uchar)(kernel[k]*(float)((i + k >= outImg.rows) ? convHor.at<uchar>(2 * outImg.rows - (i + k) - 1, j) : (r_in + outImg.cols*k)[j]));
			}
		}
	}
}


void sumOverKernel(cv::Mat &inImg, cv::Mat &outImg, const unsigned int kernel_size)
{
	cv::Mat temp = cv::Mat::zeros(inImg.size(), CV_32FC1);

	// convolve horizontally
	int d = (kernel_size - 1) / 2;
	float *r_in, *r_out;
	for (int i = 0; i < inImg.rows; ++i)
	{
		r_in = inImg.ptr<float>(i);
		r_out = temp.ptr<float>(i);
		for (int j = d; j < inImg.cols - d; ++j)
		{
			r_out[j] = r_in[j];
			for (int k = 1; k <= d; ++k)
				r_out[j] += r_in[j - k] + r_in[j + k];
		}
		// first columns with mirror of edge pixels
		for (int j = 0; j < d; ++j)
		{
			r_out[j] = r_in[j];
			for (int k = 1; k <= d; ++k)
			{
				r_out[j] += r_in[j + k];
				r_out[j] += (j - k < 0) ? r_in[k - j - 1] : r_in[j - k];
			}
		}
		// last columns with mirror of edge pixels
		for (int j = inImg.cols - 1; j > inImg.cols - d - 1; --j)
		{
			r_out[j] = r_in[j];
			for (int k = 1; k <= d; ++k)
			{
				r_out[j] += r_in[j - k];
				r_out[j] += (j + k >= inImg.cols) ? r_in[2 * inImg.cols - (j + k) - 1] : r_in[j + k];
			}
		}
	}

	// convolve vertically
	temp.copyTo(outImg);
	for (int i = d; i < temp.rows - d; ++i)
	{
		r_out = outImg.ptr<float>(i);
		r_in = temp.ptr<float>(i);

		for (int j = 0; j < temp.cols; ++j)
		{
			for (int k = 1; k <= d; ++k)
			{
				r_out[j] += (r_in - temp.cols*k)[j];
				r_out[j] += (r_in + temp.cols*k)[j];
			}
		}
	}

	// first rows
	for (int i = 0; i < d; ++i)
	{
		r_out = outImg.ptr<float>(i);
		r_in = temp.ptr<float>(i);
		for (int j = 0; j < temp.cols; ++j)
		{
			for (int k = 1; k <= d; ++k)
			{
				r_out[j] += (i - k < 0) ? temp.at<float>(-i + k - 1, j) : (r_in - temp.cols*k)[j];
				r_out[j] += (r_in + temp.cols*k)[j];
			}
		}
	}

	// last rows
	for (int i = outImg.rows - 1; i > outImg.rows - d - 1; --i)
	{
		r_out = outImg.ptr<float>(i);
		r_in = temp.ptr<float>(i);
		for (int j = 0; j < temp.cols; ++j)
		{
			for (int k = 1; k <= d; ++k)
			{
				r_out[j] += (r_in - temp.cols*k)[j];
				r_out[j] += (i + k >= temp.rows) ? temp.at<float>(2 * temp.rows - (i + k) - 1, j) : (r_in + temp.cols*k)[j];

			}
		}
	}
}


void computeModifiedLaplace(const cv::Mat &inImg, cv::Mat &lap)
{
	lap = cv::Mat::zeros(inImg.size(), CV_32FC1);
	float *l;
	const uchar *r, *r_minus, *r_plus;

	// Process first and last rows separately, mirror the pixels
	// first row
	r = inImg.ptr<uchar>(0);
	r_plus = inImg.ptr<uchar>(1);
	l = lap.ptr<float>(0);
	for (int y = 1; y < inImg.cols - 1; ++y)
		l[y] = abs(2 * (float)r[y] - (float)r[y - 1] - (float)r[y + 1]) +
		abs(2 * (float)r[y] - 2 * (float)r_plus[y]);

	// last row
	r = inImg.ptr<uchar>(inImg.rows - 1);
	r_minus = inImg.ptr<uchar>(inImg.rows - 2);
	l = lap.ptr<float>(lap.rows - 1);
	for (int y = 1; y < inImg.cols - 1; ++y)
		l[y] = abs(2 * (float)r[y] - (float)r[y - 1] - (float)r[y + 1]) +
		abs(2 * (float)r[y] - 2 * (float)r_minus[y]);

	// Loop over the main body
	for (int x = 1; x < inImg.rows - 1; ++x)
	{
		l = lap.ptr<float>(x);
		r = inImg.ptr<uchar>(x);
		r_minus = inImg.ptr<uchar>(x - 1);
		r_plus = inImg.ptr<uchar>(x + 1);
		for (int y = 1; y < inImg.cols - 1; ++y)
		{
			l[y] = abs(2 * (float)r[y] - (float)r_minus[y] - (float)r_plus[y]) +
				abs(2 * (float)r[y] - (float)r[y - 1] - (float)r[y + 1]);
		}
		// first column
		l[0] = abs(2 * (float)r[0] - 2 * (float)r[1]) +
			abs(2 * (float)r[0] - (float)r_minus[0] - (float)r_plus[0]);
		// last column
		unsigned int c = inImg.cols - 1;
		l[c] = abs(2 * (float)r[c] - 2 * (float)r[c - 1]) +
			abs(2 * (float)r[c] - (float)r_minus[c] - (float)r_plus[c]);
	}
}