Development process:
Carefully analyze requirements and the task at hand
Literature review to look for papers on the subject
Rapid prototyping in Python and experimenting on the approach, getting better understanding of the problem
Coding up a final solution in C++
Short profiling and optimization
Cleaning and packaging the code for delivery

===
The code
In itself the process of reconstructing the depth from a set of narrowly focused images consists from the two major steps:
Finding the focused areas in every image
Recovering the depth of each pixel, based on the score.

A great overview of the available approaches is provided in [1]. In the solution a method based on [2] is used, with the modification of 
using quadratic function to estimate the fine depth of pixel (this deviation was mostly due to the simplicity). According to the [1] the 
selected method is one of the most accurate ones, while at the same time fairly fast. The major drawback of the method is sensitivity to 
the noise in the image.

===
Results
Two reconstructions are computed - coarse (the depth interpolated, simply the integer value of the image index is used)
fine - (the depth of the image is computed fitting a quadratic function into three highest focus_scores

The compensate for the sensitivity of the selected method to the noise the image is initially blurred with a small kernel. 
This obviousy reduces the ability of reconstructing fine details, on the other hand the when the image is not blured the background
is recunstructed with a very high level of noise.

===
Useful resources:
[1] Analysis of focus measure operators for shape-from-focus
[2] Nayar, S.K., Shape from Focus, 1989
[1] https://stackoverflow.com/questions/7765810/is-there-a-way-to-detect-if-an-image-is-blurry