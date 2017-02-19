import time
import glob
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
from datetime import timedelta
import matplotlib.pyplot as plt
from moviepy.editor import *

from utils import *

# debug settings
DEBUG_TEST_PICTURE = 0
DEBUG_TEST_VIDEO = 1
DEBUG_DISABLED = 2
debug = DEBUG_DISABLED
debug_show_paths = False   # shows paths with base interpolated windows
debug_show_windows = False  # show paths wita all generated windows - slided base windows

# videos
video = "project_video.mp4"
test_video = "project_video_test.mp4"

# images
test_image = "test_images/test4.jpg"
car_imgs = glob.glob('dataset/vehicles/GTI**/*.png')
notcar_imgs = glob.glob('dataset/non-vehicles/GTI**/*.png')
image_width = 1280
image_height = 720

# training parameters
color_space = 'HLS'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
spatial_size = (8, 8)  # Spatial binning dimensions
hist_bins = 64  # Number of histogram bins
spatial_feat = False  # Spatial features on or off
hist_feat = False  # Histogram features on or off
hog_feat = True  # HOG features on or off

# detection params
x_start_stop = [[1079, 750], [1148, 794], [1201, 830], [1241, 858]]
y_start_stop = [[623, 473], [583, 458], [532, 438], [465, 419]]
window_start = [150, 128]
window_stop = [100, 76]
heat_depth = 8 # smoothing heatmap by heatmaps stack
heat_threshold = 4

# global params
cls, bboxes, X_scaler = None, None, None
heatmap = []


# draws bounding boxes in the given image
def process_image(image):

    draw_image = np.copy(image)

    if debug_show_paths or debug_show_windows:
        draw_boxes(draw_image, bboxes)

    else:
        hot_windows = search_windows(image, bboxes, cls, X_scaler, color_space=color_space,
                                     spatial_size=spatial_size, hist_bins=hist_bins,
                                     orient=orient, pix_per_cell=pix_per_cell,
                                     cell_per_block=cell_per_block,
                                     hog_channel=hog_channel, spatial_feat=spatial_feat,
                                     hist_feat=hist_feat, hog_feat=hog_feat)

        hm = np.zeros_like(image[:, :, 0]).astype(np.float)
        heatmap_stack = np.zeros_like(image[:, :, 0]).astype(np.float)

        hm = add_heat(hm, hot_windows)
        heatmap.append(hm)

        while (len(heatmap) > heat_depth):
            del heatmap[0]
        for h in heatmap:
            heatmap_stack += h

        heatmap_stack = apply_threshold(heatmap_stack, heat_depth * heat_threshold)

        labels = label(heatmap_stack)
        labeled_windows = labeled_bboxes(labels)

        draw_boxes(draw_image, labeled_windows)

    return draw_image


# returns a classifier cars/not cars
def get_classifier(cars, not_cars):
    car_features = extract_features(cars, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(not_cars, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)
    # fit the data
    svc = LinearSVC()
    svc.fit(X_train, y_train)
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    return svc, X_scaler


# evaluates a polynomial
def ev(coef1, coef2, coef3, x):
    return coef1 * x ** 2 + coef2 * x + coef3


# returns windows along a given path. x_start_stop, y_start_stop determine the start and stop of
# the path.
def slide_window_path(x_start_stop, y_start_stop, xy_window_start, xy_window_stop,
                      xy_overlap=(0.5, 0.5), steps = 10, margin = 40):

    window_list = []

    for x, y in zip(x_start_stop, y_start_stop):

        # fit path
        x = x[::-1]
        y = y[::-1]
        # TODO: in this version only simple line is used
        center_path = np.polyfit(y, x, 2)

        # sample path - array of x,y coords
        y_path = np.linspace(np.min(y), np.max(y), steps)
        x_path = ev(center_path[0], center_path[1], center_path[2], y_path)

        # deltas for window at each step
        dx = np.linspace(xy_window_stop[0] / 2, xy_window_start[0] / 2, steps)
        dy = np.linspace(xy_window_stop[1] / 2, xy_window_start[1] / 2, steps)

        for p_x, p_y, s_x, s_y in zip(x_path, y_path, dx, dy):
            startx = int(p_x - s_x)
            endx = int(p_x + s_x)
            starty = int(p_y - s_y)
            endy = int(p_y + s_y)

            if debug_show_paths:
                window_list.append(((startx, starty), (endx, endy)))
            else:
                # validate coordinates
                startx = startx - margin
                if startx < 0:
                    startx = 0

                endx = endx + margin
                if endx > image_width:
                    endx = image_width - 1

                if endy > image_height:
                    endy = image_height - 1

                # window size
                win_w = int(s_x * 2)
                win_h = int(s_y * 2)

                # slide window within a base window + margin
                sub_windows = slide_window(
                    x_start_stop=[startx, endx],
                    y_start_stop=[starty, endy],
                    xy_window=(win_w, win_h),
                    xy_overlap=xy_overlap)

                for sw in sub_windows:
                    window_list.append(sw)

    # Return the list of windows
    return window_list


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):

    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(img, bbox[0], bbox[1], color, thick)

    if debug_show_paths or debug_show_paths:
        for sta, sto in zip(x_start_stop, y_start_stop):
            cv2.line(img, (sta[0], sto[0]), (sta[1], sto[1]), (255, 0, 0), 6)


if __name__ == "__main__":

    t = time.time()

    if not (debug_show_windows or debug_show_paths):
        print("Training...")
        cls, X_scaler = get_classifier(car_imgs, notcar_imgs)
        print("Processing...")

    bboxes = slide_window_path(
        x_start_stop=x_start_stop,
        y_start_stop=y_start_stop,
        xy_window_start = window_start,
        xy_window_stop = window_stop,
        xy_overlap=(0.6, 0.6),
        steps=15,
        margin=40
    )

    if debug == DEBUG_TEST_PICTURE:
        image = cv2.imread(test_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        output = process_image(image)
        plt.imshow(output)
        plt.waitforbuttonpress()

    elif debug == DEBUG_TEST_VIDEO:
        white_output = 'output_test.mp4'
        clip1 = VideoFileClip(test_video, audio=False)
        white_clip = clip1.fl_image(process_image)
        white_clip.write_videofile(white_output, audio=False)

    else:
        white_output = 'output.mp4'
        clip1 = VideoFileClip(video, audio=False)
        white_clip = clip1.fl_image(process_image)
        white_clip.write_videofile(white_output, audio=False)

    elapsed = time.time() - t
    print("Done ! Finished in: ", str(timedelta(seconds=elapsed)))