import os
import cv2
import tqdm
from utils import convert_gray2rgb
import random

def load_img(path, shape):
    img_name = os.listdir(path)
    x = np.zeros(shape)
    for name in tqdm.tqdm(img_name):
        index = int(name.split('.')[0])
        gray_img = cv2.imread(os.path.join(path, name), cv2.IMREAD_GRAYSCALE)
        x[index] = gray_img.reshape((32,32,1)) # cv2.cvtColor(gray_img,cv2.COLOR_GRAY2RGB) #
    return x / 255.0

def rotation_aug(x):
    x_aug = np.zeros(x.shape)
    for i,img in tqdm.tqdm(enumerate(x)):
        x_aug[i] = np.rot90(img)
    return x_aug

def generate_negative(x_train_edter):
    x_train_neg = np.zeros(x_train_edter.shape)
    for i, img in tqdm.tqdm(enumerate(x_train_edter)):
        block = img.flatten()
        random.shuffle(block)
        x_train_neg[i] = block.reshape((32,32,1))
    return x_train_neg

def random_shuffle(x, y):
    index = [i for i in range(len(x))]
    random.shuffle(index)
    x = x[index]
    y = y[index]
    return x, y

class_dict = {100:'none',19: 'cattle', 29: 'dinosaur', 0: 'apple', 11: 'boy', 1: 'aquarium_fish', 86: 'telephone', 90: 'train', 28: 'cup', 23: 'cloud', 31: 'elephant', 39: 'keyboard', 96: 'willow_tree', 82: 'sunflower', 17: 'castle', 71: 'sea', 8: 'bicycle', 97: 'wolf', 80: 'squirrel', 74: 'shrew', 59: 'pine_tree', 70: 'rose', 87: 'television', 84: 'table', 64: 'possum', 52: 'oak_tree', 42: 'leopard', 47: 'maple_tree', 65: 'rabbit', 21: 'chimpanzee', 22: 'clock', 81: 'streetcar', 24: 'cockroach', 78: 'snake', 45: 'lobster', 49: 'mountain', 56: 'palm_tree', 76: 'skyscraper', 89: 'tractor', 73: 'shark', 14: 'butterfly', 9: 'bottle', 6: 'bee', 20: 'chair', 98: 'woman', 36: 'hamster', 55: 'otter', 72: 'seal', 43: 'lion', 51: 'mushroom', 35: 'girl', 83: 'sweet_pepper', 33: 'forest', 27: 'crocodile', 53: 'orange', 92: 'tulip', 50: 'mouse', 15: 'camel', 18: 'caterpillar', 46: 'man', 75: 'skunk', 38: 'kangaroo', 66: 'raccoon', 77: 'snail', 69: 'rocket', 95: 'whale', 99: 'worm', 93: 'turtle', 4: 'beaver', 61: 'plate', 94: 'wardrobe', 68: 'road', 34: 'fox', 32: 'flatfish', 88: 'tiger', 67: 'ray', 30: 'dolphin', 62: 'poppy', 63: 'porcupine', 40: 'lamp', 26: 'crab', 48: 'motorcycle', 79: 'spider', 85: 'tank', 54: 'orchid', 44: 'lizard', 7: 'beetle', 12: 'bridge', 2: 'baby', 41: 'lawn_mower', 37: 'house', 13: 'bus', 25: 'couch', 10: 'bowl', 57: 'pear', 5: 'bed', 60: 'plain', 91: 'trout', 3: 'bear', 58: 'pickup_truck', 16: 'can'}

def sampling(src_path, dst_path, size):
    for img_name in tqdm.tqdm(os.listdir(src_path)):
        # current_src_path = os.path.join(src_path, folder)
        # current_dst_path = os.path.join(dst_path, folder)
        # # 创建目标文件夹
        # os.makedirs(current_dst_path)
        # for img_name in os.listdir(current_src_path):
        img_path = os.path.join(src_path, img_name)
        target_path = os.path.join(dst_path, img_name)
        # grayscale
        img_src = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) #
        # img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
        factor = max(img_src.shape)/size
        img_gray_64 = cv2.resize(img_src, (int(img_src.shape[0]/factor), int(img_src.shape[1]/factor)))

        cv2.imwrite(target_path, img_gray_64)

def edter_txt(dst_path):
    '''Generate test.txt for given path'''
    with open(r'D:\Gnerating Fractal Image\EDTER-main\data\BSDS\ImageSets\test.txt', 'w') as file:
        # for folder in tqdm.tqdm(os.listdir(dst_path)):
        #     current_dst_path = os.path.join(dst_path, folder)
        for img_name in os.listdir(dst_path):
            name = os.path.join(dst_path, img_name)
            file.write(name.replace('\\','/'))
            file.write(' ')
            file.write('F:/dataset/coco2017/images/12003.mat')
            file.write('\n')

def gradient(src_path, dst_path):
    for img_name in os.listdir(src_path):
        img_path = os.path.join(src_path, img_name)
        target_path = os.path.join(dst_path, img_name)
        # grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) #
        dst = cv2.resize(apply_sobel(img),(32,32))
        factor = (255/np.max(dst))
        dst = dst * factor - 10
        # cv2.imshow("Result", dst)
        # cv2.waitKey(0)
        cv2.imwrite(target_path, dst)

def apply_sobel(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)
    # cv2.imshow("absX", absX)
    # cv2.imshow("absY", absY)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return dst


def apply_canny(region):
    gray_img = convert_gray2rgb(region)
    img_blur = cv2.GaussianBlur(gray_img, (3, 3), 0)
    canny = cv2.Canny(img_blur, 50, 150)
    return canny

import numpy as np
if __name__ == "__main__":
    mode = 'val'
    src_path = rf'F:\dataset\CIFAR100\cifar100_{mode}_gray'
    dst_path = rf'F:\dataset\CIFAR100\cifar100_{mode}_sobel_32'
    size = 32
    gradient(src_path, dst_path)
    # sampling(src_path, dst_path, size)
    # dst_path = r'F:\dataset\cifar100_val'
    # edter_txt(dst_path)