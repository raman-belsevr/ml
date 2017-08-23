from sklearn.base import TransformerMixin
from sklearn.pipeline import make_pipeline
import numpy as np
import cv2
from random import randint
from scipy import ndimage


class Crop(TransformerMixin):

    def __init__(self, x_limits, y_limits):
        self.x_limits = x_limits
        self.y_limits = y_limits

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        return X[self.x_limits[0]: self.x_limits[1], self.y_limits[0]: self.y_limits[1]]


class GrayScale(TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self

    def transform(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


class Channel(TransformerMixin):

    def __init__(self, channel):
        self.channel = channel

    def fit(self, X, y = None):
        return self

    def transform(self, image):
        return image[:, :, self.channel]


class Rotate(TransformerMixin):

    def __init__(self, angle):
        self.angle = angle

    def fit(self, X, y=None):
        return self

    def transform(self, image):
        x, y, c = image.shape
        image_center = tuple(np.array(image.shape)/2)
        rot_mat = cv2.getRotationMatrix2D(image_center, self.angle, 1.0)
        rotated = cv2.warpAffine(image, rot_mat, (y,x), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
        return rotated


class Shift(TransformerMixin):

    def __init__(self, shift_x, shift_y):
        self.shift_x = shift_x
        self.shift_y = shift_y

    def fit(self, X, y=None):
        return self

    def transform(self, image):
        delta = randint(3, 10)
        if randint(0,1) == 0:
           sign = 1
        else:
            sign = -1
        delta = sign * delta
        sequence = (0.0, delta, 0.0)
        return ndimage.shift(image, sequence, mode="nearest")


class WarpPerspective(TransformerMixin):

    def __init__(self, src_pts, dst_pts):
        self.src_pts = src_pts
        self.dst_pts = dst_pts
        self.M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)

    def fit(self, X, y=None):
        return self

    def transform(self, image):
        img_size = (image.shape[1], image.shape[0])
        warped_binary = cv2.warpPerspective(image, self.M, img_size, flags=cv2.INTER_LINEAR)
        return warped_binary
