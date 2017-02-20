**Vehicle Detection**
======================


<div align="center">
<a href="http://www.youtube.com/watch?feature=player_embedded&v=YYMYOsoYpwo
" target="_blank"><img src="http://img.youtube.com/vi/YYMYOsoYpwo/0.jpg"
alt="youtube link" width="480" height="360" border="10" /></a>
</div>

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2a]: ./output_images/hog_car.png
[image2b]: ./output_images/hog_notcar.png
[image3]: ./output_images/sliding_windows.png
[image4]: ./output_images/sliding_window.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./output.mp4

---

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 103 through 153 of the file called `utils.py`). The function `extract_features`
is referenced from `get_classifier` of the file `project.py`. This is the location where classifier is created and trained. 

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `HLS` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` for all channels:

Example of car image:
![alt text][image2a]

Example of not car image
![alt text][image2b]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. Larger values of pixels per cell, cells per block 
resulted in a greater number of false positives. `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`
seem to be good fit. 

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using only HOG features. The code for this step is contained in lines 95 through 126 of the file called `project.py`)
Features have been scaled to zero mean and unit variance before training the classifier. Using histograms and spatial binary features
resulted in unacceptable number of false positives.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search window positions along potential trajectories of vehicles.
Algorithm uses two window sizes - at the begining of a path (relatively close to the camera) and smaller at the far end
where cars sizes shrink. All sub-windows are determined by the simple interpolation between those two ends.
Moreover, at each position of sub-window additional space is added to the left and right and classic sliding window
algorithm is applied to this boundary. The search is performed in lines 136 through 193 of the file called `project.py`
 
![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Performance is acceptable because I used only HOG and various window sizes along the potential trajectories.
Full processing of `project_video.mp4` takes around 10-13 minutes. 

![alt text][image4]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output.mp4)

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I decided to use only HOG. Spatial binary features and color histograms resulted in poor performance. 
Moreover, I used smaller number of windows. Windows are generated based on list of trajectories. Bigger windows closer to the camera
and smaller windows closer to the horizon. This optimization significantly helped to improve performance.
For this project I used only simple lines as trajectories but in more advanced approach those trajectories
may correspond to detected lanes lines and may be in the form of a second order polynomial curve.
The solution may fail when the cars move too fast - they may be thresholded, when cars are partially occluded, 
during worse environmental conditions - common problems with all vision techniques.