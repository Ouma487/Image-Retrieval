from skimage.color import rgb2gray, rgb2hsv
import cv2 
from transformers import pipeline
import numpy as np
from skimage import filters
import matplotlib.pyplot as plt

def convert_to_hsv(img):
    """Convert RGB image to HSV.

    Args:
        img (ndarray): Input RGB image.

    Returns:
        ndarray: HSV formatted image.
    """
    return (255 * rgb2hsv(img)).astype(int)

def load_depth_estimator():
    """Load the DPT model.

    Returns:
        object: Loaded depth estimator.
    """
    estimator = pipeline(task="depth-estimation", model="Intel/dpt-large")
    return estimator

def estimate_depth(img_path, estimator, shape):
    """Estimate the depth of the image.

    Args:
        img_path (str): Path to the input image.
        estimator (object): Depth estimator.
        shape (tuple): Desired shape of the output image.

    Returns:
        ndarray: Image-sized array of depth of each pixel.
    """
    d_img = np.array(estimator(img_path)['depth'])
    d_img = cv2.resize(d_img, shape)
    return d_img

def savola_transformation(img):
    """Binarize the image to detect edges.

    Args:
        img (ndarray): Input RGB image.

    Returns:
        ndarray: Binarized image.
    """
    gray_image = rgb2gray(img)
    threshold = filters.threshold_sauvola(gray_image)
    binarized_image = (gray_image > threshold) * 1
    return (255 * binarized_image).astype(int)

def compose_features(img_path, estimator, shape=(256, 256), weights=[1, 1, 1, 1]):
    """Stack the different layers of image transformations (features).

    Args:
        img_path (str): Path to the input image.
        estimator : Depth estimator.
        shape (tuple): Desired shape of the output image.
        weights (list): List of weights for each feature.

    Returns:
        ndarray: Composed features.
        tuple: Tuple containing the original image and individual features (depth, HSV, and binarized).
    """
    img = cv2.imread(img_path)
    img = cv2.resize(img, shape)
    
    h_img = convert_to_hsv(img)
    d_img = estimate_depth(img_path, estimator, shape)
    b_img = binarize_image(img)
    
    features = np.zeros((shape[0], shape[1], img.shape[2] + 2 + h_img.shape[2]))
    features[:, :, :3] = img * weights[0]
    features[:, :, 3:6] = h_img * weights[1]
    features[:, :, 6] = d_img * weights[2]
    features[:, :, 7] = b_img * weights[3]
    
    img = np.float32(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(int)
    
    return features, (img, d_img, h_img, b_img)

if __name__ == '__main__':
    path = 'data/test_image_headmind/image-20210928-102713-12d2869d.jpg'
    estimator = load_depth_estimator()
    features, t = compose_features(path, estimator, shape=(256, 256), weights=[1, 1, 1, 1])
    
    fig, axes = plt.subplots(1, len(t), figsize=(12, 4))
    
    for i in range(len(t)):
        axes[i].imshow(t[i])
        axes[i].axis('off')
        axes[i].set_title(f'Feature {i}')
    
    plt.savefig('images/features.png')
    # plt.show()
