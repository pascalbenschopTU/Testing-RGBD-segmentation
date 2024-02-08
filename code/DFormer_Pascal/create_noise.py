import numpy as np 
import cv2 
from PIL import Image

from scipy.interpolate import griddata
from scipy import ndimage

def add_gaussian_shifts(depth, std=1/2.0):

    rows, cols = depth.shape 
    gaussian_shifts = np.random.normal(0, std, size=(rows, cols, 2))
    gaussian_shifts = gaussian_shifts.astype(np.float32)

    # creating evenly spaced coordinates  
    xx = np.linspace(0, cols-1, cols)
    yy = np.linspace(0, rows-1, rows)

    # get xpixels and ypixels 
    xp, yp = np.meshgrid(xx, yy)

    xp = xp.astype(np.float32)
    yp = yp.astype(np.float32)

    xp_interp = np.minimum(np.maximum(xp + gaussian_shifts[:, :, 0], 0.0), cols)
    yp_interp = np.minimum(np.maximum(yp + gaussian_shifts[:, :, 1], 0.0), rows)

    depth_interp = cv2.remap(depth, xp_interp, yp_interp, cv2.INTER_LINEAR)

    return depth_interp
    

def filterDisp(disp, dot_pattern_, invalid_disp_):

    size_filt_ = 9

    xx = np.linspace(0, size_filt_-1, size_filt_)
    yy = np.linspace(0, size_filt_-1, size_filt_)

    xf, yf = np.meshgrid(xx, yy)

    xf = xf - int(size_filt_ / 2.0)
    yf = yf - int(size_filt_ / 2.0)

    sqr_radius = (xf**2 + yf**2)
    vals = sqr_radius * 1.2**2 

    vals[vals==0] = 1 
    weights_ = 1 /vals  

    fill_weights = 1 / ( 1 + sqr_radius)
    fill_weights[sqr_radius > 9] = -1.0 

    disp_rows, disp_cols = disp.shape 
    dot_pattern_rows, dot_pattern_cols = dot_pattern_.shape

    lim_rows = np.minimum(disp_rows - size_filt_, dot_pattern_rows - size_filt_)
    lim_cols = np.minimum(disp_cols - size_filt_, dot_pattern_cols - size_filt_)

    center = int(size_filt_ / 2.0)

    window_inlier_distance_ = 0.1

    out_disp = np.ones_like(disp) * invalid_disp_

    interpolation_map = np.zeros_like(disp)

    for r in range(0, lim_rows):

        for c in range(0, lim_cols):

            if dot_pattern_[r+center, c+center] > 0:
                                
                # c and r are the top left corner 
                window  = disp[r:r+size_filt_, c:c+size_filt_] 
                dot_win = dot_pattern_[r:r+size_filt_, c:c+size_filt_] 
  
                valid_dots = dot_win[window < invalid_disp_]

                n_valids = np.sum(valid_dots) / 255.0 
                n_thresh = np.sum(dot_win) / 255.0 

                if n_valids > n_thresh / 1.2: 

                    mean = np.mean(window[window < invalid_disp_])

                    diffs = np.abs(window - mean)
                    diffs = np.multiply(diffs, weights_)

                    cur_valid_dots = np.multiply(np.where(window<invalid_disp_, dot_win, 0), 
                                                 np.where(diffs < window_inlier_distance_, 1, 0))

                    n_valids = np.sum(cur_valid_dots) / 255.0

                    if n_valids > n_thresh / 1.2: 
                    
                        accu = window[center, center] 

                        assert(accu < invalid_disp_)

                        out_disp[r+center, c + center] = round((accu)*8.0) / 8.0

                        interpolation_window = interpolation_map[r:r+size_filt_, c:c+size_filt_]
                        disp_data_window     = out_disp[r:r+size_filt_, c:c+size_filt_]

                        substitutes = np.where(interpolation_window < fill_weights, 1, 0)
                        interpolation_window[substitutes==1] = fill_weights[substitutes ==1 ]

                        disp_data_window[substitutes==1] = out_disp[r+center, c+center]

    return out_disp


def create_noisy_depth(file):
    # reading the image directly in gray with 0 as input 
    dot_pattern_ = Image.open("./data/kinect-pattern_3x3.png")
    dot_pattern_ = dot_pattern_.convert('L')
    dot_pattern_ = np.array(dot_pattern_)

    # various variables to handle the noise modelling
    scale_factor  = 100     # converting depth from m to cm 
    focal_length  = 480.0   # focal length of the camera used 
    baseline_m    = 0.075   # baseline in m 
    invalid_disp_ = 99999999.9

    depth_uint16 = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    h, w = depth_uint16.shape 

    # Our depth images were scaled by 5000 to store in png format so dividing to get 
    # depth in meters 
    depth = depth_uint16.astype('float')
    # Normalize depth to 0, 10
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth)) * 10

    depth_interp = add_gaussian_shifts(depth)

    disp_= focal_length * baseline_m / (depth_interp + 1e-10)
    depth_f = np.round(disp_ * 8.0)/8.0

    out_disp = filterDisp(depth_f, dot_pattern_, invalid_disp_)

    depth = focal_length * baseline_m / out_disp
    depth[out_disp == invalid_disp_] = 0 
    
    # The depth here needs to converted to cms so scale factor is introduced 
    # though often this can be tuned from [100, 200] to get the desired banding / quantisation effects 
    noisy_depth = (35130/(np.round((35130/(np.round(depth*scale_factor) + 1e-16)) + np.random.normal(size=(h, w))*(1.0/6.0) + 0.5)+1e-16))/ scale_factor 

    # Normalize depth back to 0, 255
    noisy_depth = (noisy_depth - np.min(noisy_depth)) / (np.max(noisy_depth) - np.min(noisy_depth)) * 255
    noisy_depth = noisy_depth.astype('uint16')

    return noisy_depth


def create_depth_noise_2(depth_map, rgb_image, noise_level=0.2):
    """
    Simulate noise in the depth map based on object properties, scene layout, motion blur, and scene illumination.

    Parameters:
        depth_map (numpy.ndarray): Perfect depth map.
        rgb_image (numpy.ndarray): RGB image.
        noise_level (float): Noise level.

    Returns:
        numpy.ndarray: Noisy depth map.
    """
    # Initialize noisy depth map
    noisy_depth_map = depth_map.copy().astype(np.float64)

    # Simulate edges where the depth is not well defined, i.e. black
    # Compute gradients using Sobel operator
    grad_x = cv2.Sobel(depth_map.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_map.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
    # Compute gradient magnitude and threshold for edge detection
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    edge_mask = (grad_mag < 50).astype(np.uint8)  # Adjust threshold as needed
    # Invert the edge mask
    inverted_edges = 1.0 - edge_mask
    random_edges = np.zeros_like(inverted_edges)
    # Give the edges a chance to disappear
    random_edges[inverted_edges > 0] = np.random.choice([0, 1], size=inverted_edges.shape, p=[1.0-noise_level, noise_level])[inverted_edges > 0]
    # Make the edges wider
    kernel_size = (5, 5)
    kernel = np.ones(kernel_size, np.uint8)
    dilated_edges = cv2.dilate(random_edges.astype(np.uint8), kernel, iterations=1)
    # Convert depth_map to float64
    depth_map = depth_map.astype(np.float64)
    # Subtract the edge mask from the depth map
    noisy_depth_map[dilated_edges > 0] = dilated_edges[dilated_edges > 0]

    # Simulate noise in bright regions of the RGB image
    # Calculate brightness of the RGB image
    brightness = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    bright_regions = brightness > 200  # Define threshold for brightness
    bright_noise = noise_level * np.random.normal(128, 50, depth_map.shape)
    noisy_depth_map += bright_noise * bright_regions

    # Simulate freckles
    # Where noise looks like random blobs of black
    noisy_region_mask = (depth_map > 50) & (brightness < 100)
    # Create a random noise mask with the same shape as the depth map
    noise_mask = np.random.rand(*depth_map.shape) < 0.005
    # Combine the noisy region mask and the noise mask
    noisy_regions = noisy_region_mask & noise_mask
    dilated_regions = cv2.dilate(noisy_regions.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
    noisy_depth_map[dilated_regions == 1] = dilated_regions[dilated_regions == 1]

    # Clip the noisy depth map to the range [0, 255]
    noisy_depth_map = np.clip(noisy_depth_map, 0, 255)

    # Blur the noisy depth map neurest neighbors
    noisy_depth_map = cv2.medianBlur(noisy_depth_map.astype(np.uint8), 3)

    return noisy_depth_map