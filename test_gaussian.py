import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la

from skimage import morphology

img_path = 'lane.PNG'
#img = cv2.imread('lane.PNG', 0)

def gradient_thresh(img, thresh_min=25, thresh_max=100):

    img_gray = cv2.imread(img, 0)

    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

    Gx = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize = 3)
    Gy = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize = 3)

    img_addWeighted = cv2.addWeighted(Gx, 0.5, Gy, 0.5, 0)
    cvuint8 = cv2.convertScaleAbs(img_addWeighted)
    for i in range(0, np.shape(cvuint8)[0]):
        for j in range(0, np.shape(cvuint8)[1]):
            if cvuint8[i, j] < thresh_min or cvuint8[i, j] > thresh_max:
                cvuint8[i, j] = 0
            else:
                cvuint8[i, j] = 1
    return cvuint8

def color_thresh(img, thresh_min = 100, thresh_max = 255):
    img = cv2.imread(img, -1)
    img_HLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    size = img_HLS.shape
    w, h = size[0], size[1]
    color_biout = np.zeros((w, h))
    for i in range(w):
        for j in range(h):
            if img_HLS[w-1, h-1, 2] >= thresh_min and img_HLS[w-1, h-1, 2] <= 255:
                color_biout[i,j] = 1
            else:
                color_biout[i,j] = 0
    return color_biout

def combinedBinaryImage(img):
    SobelOutput = gradient_thresh(img)
    ColorOutput = color_thresh(img)

    binaryImage = np.zeros_like(SobelOutput)
    binaryImage[(ColorOutput == 1) | (SobelOutput == 1)] = 1
    # Remove noise from binary image
    binaryImage = morphology.remove_small_objects(binaryImage.astype('bool'), min_size=50, connectivity=2)

    return binaryImage

# res = combinedBinaryImage(img_path)
# plt.imshow(res, cmap = 'binary')
# plt.show()
# print(res)


def perspective_transform(img, verbose=False):
    img = cv2.imread(img, -1)
    # src_pts = np.float32([[273.1, 324.1], [491.2, 308.7], [324.2, 276.4], [418.9, 273.0]])
    # dst_pts = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
    src_pts = np.float32([[159.1, 378.8], [643.3, 365.3], [321.8, 263.6], [378.3, 255.3]])
    # src_pts = np.float32([[170, 374], [655, 371], [286, 277], [474, 279]])
    dst_pts = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    Minv = la.inv(M)
    warped_img = cv2.warpPerspective(img, M, (400, 400))
    return warped_img, M, Minv

ww,mm,mminv = perspective_transform(img_path, verbose=False)

print(ww)
plt.imshow(ww)
plt.show()

# image = combinedBinaryImage('img_path')
# plt.figure(2)
# plt.imshow(image, cmap='binary')
# plt.show()
# warp_img,M,Minv = perspective_transform('img_path')
# warp_im_rgb = cv2.cvtColor(warp_img, cv2.COLOR_BGR2RGB)
# plt.figure(2)
# plt.imshow(warp_im_rgb)
# plt.show()
