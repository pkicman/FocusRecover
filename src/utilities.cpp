#include "utilities.h"


void fitQuadratic(float x1, float y1, float x2, float y2, float x3, float y3, float &a, float &b, float &c)
{
	// To avoid using linear algebra libraries an anlytical solution is used
	float denom = (x1 - x2)*(x1 - x3)*(x2 - x3);
	a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom;
	b = (x3*x3 * (y1 - y2) + x2*x2 * (y3 - y1) + x1*x1 *(y2 - y3)) / denom;
	c = (x2 * x3 * (x2 - x3) * y1 + x3*x1*(x3 - x1)*y2 + x1 * x2 * (x1 - x2) * y3) / denom;
}

void threshToZero(myMat<float>& inImg, const float threshold)
{
	// set (in-place) to zero every pixel whose value is below threshold

	float *p;
	for (int i = 0; i < inImg.nrows(); ++i)
	{
		p = inImg.rowPtr(i);
		for (int j = 0; j < inImg.ncols(); ++j)
			if (p[j] < threshold) p[j] = 0.0f;
	}
}


void convertToGray(const myMat<uchar> &inImg, myMat<uchar> &outImg)
{
	// Assuming standard opencv convention of BGR color coding
	if (inImg.nrows() == 0 || inImg.ncols() == 0 || inImg.nchannels() == 0) return;
	if (inImg.nchannels() == 3)
		outImg = myMat<uchar>(inImg.nrows(), inImg.ncols(), 1);
	else
		return;

	uchar *out;
	const uchar *in;
	for (int i = 0; i < inImg.nrows(); ++i)
	{
		out = outImg.rowPtr(i);
		in = inImg.rowPtr(i);
		for (int j = 0; j < inImg.ncols(); ++j)
		{
			out[j] = (uchar)(0.114*(float)in[j*3] + 0.587*(float)in[j*3+1] + 0.299*(float)in[j*3+2]);
		}
	}
}


void applyGaussianBlur(const myMat<uchar>& inImg, myMat<uchar>& outImg, const unsigned int kernel_size)
{
	std::vector<float> kernel;

	// use precomputed kernel coefficients for kernel 3x3 and 5x5
	// other kernel sizes not supported
	if (kernel_size == 1)
	{
		outImg = inImg;
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
		outImg = inImg;
		return;
	}

	myMat<uchar> convHor(inImg.nrows(), inImg.ncols(), 1);

	// convolve horizontally
	int d = (kernel_size - 1) / 2;
	const uchar *r_in;
	uchar *r_out;
	for (int i = 0; i < inImg.nrows(); ++i)
	{
		r_in = inImg.rowPtr(i);
		r_out = convHor.rowPtr(i);
		for (int j = d; j < inImg.ncols() - d; ++j)
		{
			r_out[j] = (uchar)(kernel[0] * (float)r_in[j]);
			for (int k = 1; k <= d; ++k)
				r_out[j] += (uchar)(kernel[k] * ((float)r_in[j - k] + (float)r_in[j + k]));
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
				r_out[j] += (uchar)(kernel[k] * (float)((j - k < 0) ? r_in[k - j - 1] : r_in[j - k]));
			}
		}
		// last columns with mirror of edge pixels
		for (int j = inImg.ncols() - 1; j > inImg.ncols() - d - 1; --j)
		{
			r_out[j] = (uchar)(kernel[0] * (float)r_in[j]);
			for (int k = 1; k <= d; ++k)
			{
				r_out[j] += (uchar)(kernel[k] * (float)r_in[j - k]);
				r_out[j] += (uchar)(kernel[k] * (float)((j + k >= inImg.ncols()) ? r_in[2 * inImg.ncols() - (j + k) - 1] : r_in[j + k]));
			}
		}
	}

	// convolve vertically
	outImg = myMat<uchar>(inImg.nrows(), inImg.ncols(), 1);

	for (int i = d; i < outImg.nrows() - d; ++i)
	{
		r_out = outImg.rowPtr(i);
		r_in = convHor.rowPtr(i);

		for (int j = 0; j < outImg.ncols(); ++j)
		{
			r_out[j] = (uchar)(kernel[0] * (float)r_in[j]);
			for (int k = 1; k <= d; ++k)
			{
				r_out[j] += (uchar)(kernel[k] * (float)((r_in - outImg.ncols()*k)[j]));
				r_out[j] += (uchar)(kernel[k] * (float)((r_in + outImg.ncols()*k)[j]));
			}
		}
	}

	// first rows
	for (int i = 0; i < d; ++i)
	{
		r_out = outImg.rowPtr(i);
		r_in = convHor.rowPtr(i);
		for (int j = 0; j < outImg.ncols(); ++j)
		{
			r_out[j] = (uchar)(kernel[0] * (float)r_in[j]);
			for (int k = 1; k <= d; ++k)
			{
				r_out[j] += (uchar)(kernel[k] * (float)((i - k < 0) ? convHor.at(-i + k - 1, j)[0] : (r_in - outImg.ncols()*k)[j]));
				r_out[j] += (uchar)(kernel[k] * (float)((r_in + outImg.ncols()*k)[j]));
			}
		}
	}

	// last rows
	for (int i = outImg.nrows() - 1; i > outImg.nrows() - d - 1; --i)
	{
		r_out = outImg.rowPtr(i);
		r_in = convHor.rowPtr(i);
		for (int j = 0; j < outImg.ncols(); ++j)
		{
			r_out[j] = (uchar)(kernel[0] * (float)r_in[j]);
			for (int k = 1; k <= d; ++k)
			{
				r_out[j] += (uchar)(kernel[k] * (float)((r_in - outImg.ncols()*k)[j]));
				r_out[j] += (uchar)(kernel[k] * (float)((i + k >= outImg.nrows()) ? convHor.at(2 * outImg.nrows() - (i + k) - 1, j)[0] : (r_in + outImg.ncols()*k)[j]));
			}
		}
	}

}


void sumOverKernel(myMat<float>& inImg, myMat<float>& outImg, const unsigned int kernel_size)
{
	myMat<float> temp(inImg.nrows(), inImg.ncols(), 1);

	// convolve horizontally
	int d = (kernel_size - 1) / 2;
	float *r_in, *r_out;
	for (int i = 0; i < inImg.nrows(); ++i)
	{
		r_in = inImg.rowPtr(i);
		r_out = temp.rowPtr(i);
		for (int j = d; j < inImg.ncols() - d; ++j)
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
		for (int j = inImg.ncols() - 1; j > inImg.ncols() - d - 1; --j)
		{
			r_out[j] = r_in[j];
			for (int k = 1; k <= d; ++k)
			{
				r_out[j] += r_in[j - k];
				r_out[j] += (j + k >= inImg.ncols()) ? r_in[2 * inImg.ncols() - (j + k) - 1] : r_in[j + k];
			}
		}
	}

	// convolve vertically
	outImg = temp;
	for (int i = d; i < temp.nrows() - d; ++i)
	{
		r_out = outImg.rowPtr(i);
		r_in = temp.rowPtr(i);

		for (int j = 0; j < temp.ncols(); ++j)
		{
			for (int k = 1; k <= d; ++k)
			{
				r_out[j] += (r_in - temp.ncols()*k)[j];
				r_out[j] += (r_in + temp.ncols()*k)[j];
			}
		}
	}

	// first rows
	for (int i = 0; i < d; ++i)
	{
		r_out = outImg.rowPtr(i);
		r_in = temp.rowPtr(i);
		for (int j = 0; j < temp.ncols(); ++j)
		{
			for (int k = 1; k <= d; ++k)
			{
				r_out[j] += (i - k < 0) ? temp.at(-i + k - 1, j)[0] : (r_in - temp.ncols()*k)[j];
				r_out[j] += (r_in + temp.ncols()*k)[j];
			}
		}
	}

	// last rows
	for (int i = outImg.nrows() - 1; i > outImg.nrows() - d - 1; --i)
	{
		r_out = outImg.rowPtr(i);
		r_in = temp.rowPtr(i);
		for (int j = 0; j < temp.ncols(); ++j)
		{
			for (int k = 1; k <= d; ++k)
			{
				r_out[j] += (r_in - temp.ncols()*k)[j];
				r_out[j] += (i + k >= temp.nrows()) ? temp.at(2 * temp.nrows() - (i + k) - 1, j)[0] : (r_in + temp.ncols()*k)[j];
			}
		}
	}
}


void computeModifiedLaplace(const myMat<uchar>& inImg, myMat<float>& lap)
{
	lap = myMat<float>(inImg.nrows(), inImg.ncols(), 1);

	float *l;
	const uchar *r, *r_minus, *r_plus;

	// Process first and last rows separately, mirror the pixels
	// first row
	r = inImg.rowPtr(0);
	r_plus = inImg.rowPtr(1);
	l = lap.rowPtr(0);
	for (int y = 1; y < inImg.ncols() - 1; ++y)
		l[y] = abs(2 * (float)r[y] - (float)r[y - 1] - (float)r[y + 1]) +
		abs(2 * (float)r[y] - 2 * (float)r_plus[y]);

	// last row
	r = inImg.rowPtr(inImg.nrows() - 1);
	r_minus = inImg.rowPtr(inImg.nrows() - 2);
	l = lap.rowPtr(lap.nrows() - 1);
	for (int y = 1; y < inImg.ncols() - 1; ++y)
		l[y] = abs(2 * (float)r[y] - (float)r[y - 1] - (float)r[y + 1]) +
		abs(2 * (float)r[y] - 2 * (float)r_minus[y]);

	// Loop over the main body
	for (int x = 1; x < inImg.nrows() - 1; ++x)
	{
		l = lap.rowPtr(x);
		r = inImg.rowPtr(x);
		r_minus = inImg.rowPtr(x - 1);
		r_plus = inImg.rowPtr(x + 1);
		for (int y = 1; y < inImg.ncols() - 1; ++y)
		{
			l[y] = abs(2 * (float)r[y] - (float)r_minus[y] - (float)r_plus[y]) +
				abs(2 * (float)r[y] - (float)r[y - 1] - (float)r[y + 1]);
		}
		// first column
		l[0] = abs(2 * (float)r[0] - 2 * (float)r[1]) +
			abs(2 * (float)r[0] - (float)r_minus[0] - (float)r_plus[0]);
		// last column
		unsigned int c = inImg.ncols() - 1;
		l[c] = abs(2 * (float)r[c] - 2 * (float)r[c - 1]) +
			abs(2 * (float)r[c] - (float)r_minus[c] - (float)r_plus[c]);
	}
}
