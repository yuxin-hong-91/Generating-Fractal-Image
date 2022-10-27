import numpy as np
import cv2
from generateFractals import *

def autoCrop(fractal, x_min, x_max, y_min, y_max):
    minx, miny = fractal.shape
    maxx = 0
    maxy = 0
    thresh = 16

    for y in range(0, 640):
        if np.max(fractal[y, :]) > thresh:
            miny = y
            break

    y = 640
    while y > 0:
        y = y - 1
        if np.max(fractal[y, :]) > thresh:
            maxy = y
            break

    for x in range(0, 640):
        if np.max(fractal[:, x]) > thresh:
            minx = x
            break
    x = 640
    while x > 0:
        x = x - 1
        if np.max(fractal[:, x]) > thresh:
            maxx = x
            break

    x_min_new, x_max_new, y_min_new, y_max_new = update_region(x_min, x_max, y_min, y_max, minx, maxx, miny, maxy)

    return fractal[miny:maxy, minx:maxx], x_min_new, x_max_new, y_min_new, y_max_new


def update_region(x_min, x_max, y_min, y_max, minx, maxx, miny, maxy):
    delta_x = x_max - x_min
    delta_y = y_max - y_min
    # updata region
    x_min_new = x_min + minx * delta_x / 640
    x_max_new = x_min + maxx * delta_x / 640
    y_min_new = y_min + miny * delta_y / 640
    y_max_new = y_min + maxy * delta_y / 640

    return x_min_new, x_max_new, y_min_new, y_max_new


def convert_gray2rgb(fractal):
    img_rgb = cv2.cvtColor((fractal / 128 * 256).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    return img_rgb

def calc_occupancy(fractal):
    w, h = fractal.shape
    if w == 0:
        return 0
    if h == 0:
        return 0
    return len(fractal[fractal > 120]) / (w * h)


def generate_fractals(fractal_type, x_min, x_max, y_min, y_max, z=None, n=2, crop=True):
    if fractal_type == "Mandelbrot":
        generation_function = getMandelbrotEscapeTime
    elif fractal_type == "Multibrot":
        generation_function = getMultibrotEscapeTime
    elif fractal_type == "Burning_ship":
        generation_function = getBurningShipEscapeTime
    elif fractal_type == "Julia":
        generation_function = getJuliaEscapeTime
    else:
        print("please check fractal type!")

    def get_fractal_set(xFrom, xTo, yFrom, yTo, yN, xN):  # x_min, x_max, y_min, y_max,
        y, x = np.ogrid[yFrom:yTo:yN * 1j, xFrom:xTo:xN * 1j]
        c = x + y * 1j
        if fractal_type == "Multibrot":
            return np.frompyfunc(generation_function, 2, 1)(c, n).astype(np.float)
        elif fractal_type == "Julia":
            z_grid = np.array([np.array([z] * 640)] * 640)
            return np.frompyfunc(generation_function, 2, 1)(z_grid, c).astype(np.float)
        else:
            return np.frompyfunc(generation_function, 1, 1)(c).astype(np.float)

    fractal = get_fractal_set(x_min, x_max, y_min, y_max, 640, 640)
    if crop:
        fractal, x_min, x_max, y_min, y_max = autoCrop(fractal, x_min, x_max, y_min, y_max)
    return fractal, x_min, x_max, y_min, y_max

