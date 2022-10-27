import matplotlib.pyplot as plt
import time
from generateFractals import *
from multiprocessing import Process
from utils import *
from yolo_utils import *

# yolo
import cv2
import numpy as np
from yolov6.utils.nms import non_max_suppression
from typing import List, Optional

checkpoint: str = "yolov6t_coco_edter"  # @param ["yolov6t_coco_edter", "yolov6t_aquarium_edter"]
yaml, class_names = load(checkpoint)
model, stride, device, img_size, half = load_model(checkpoint)
img_size = check_img_size(img_size, s=stride)
hide_labels: bool = False #@param {type:"boolean"}
hide_conf: bool = False #@param {type:"boolean"}
img_size:int = 640 #@param {type:"integer"}
conf_thres: float =.50 #@param {type:"number"}
iou_thres: float =.45 #@param {type:"number"}
max_det:int =  1000#@param {type:"integer"}
agnostic_nms: bool = False #@param {type:"boolean"}

def save_fractal(x_min, x_max, y_min, y_max, title, fractal_type, z, n, save_folder, show=False, save=True):
    fractal, x_min, x_max, y_min, y_max = generate_fractals(fractal_type,x_min, x_max, y_min, y_max, z, n)
    if (calc_occupancy(fractal) > 0.01) and (calc_occupancy(fractal) < 0.99):
        plt.figure()
        plt.imshow(fractal,
                   interpolation="bilinear",
                   cmap='gray',  # plt.cm.hot,
                   vmax=abs(fractal).max(),
                   vmin=abs(fractal).min(),
                   extent=[x_min, x_max, y_min, y_max])
        plt.axis('off')
        if save:
            plt.savefig('./object/'+save_folder+'/{}.png'.format(title), format="jpg", dpi=200, bbox_inches='tight',
                        pad_inches=0, transparent=False)
        if show:
            plt.imshow()
        plt.close()

def search_sub_yolo(x_min, x_max, y_min, y_max, region_box, times, fractal_type, n, z, apply_canny, save_folder):
    delta_x = x_max - x_min
    delta_y = y_max - y_min
    x_middle = ((x_min + x_max) / 2)
    y_middle = ((y_min + y_max) / 2)

    if delta_x / delta_y > 2:
        # left-right
        region_box = search_object_yolo(x_min, x_middle, y_min, y_max, region_box, times + 1, fractal_type, n, z, apply_canny, save_folder)
        region_box = search_object_yolo(x_middle, x_max, y_min, y_max, region_box, times + 1, fractal_type, n, z, apply_canny, save_folder)

    elif delta_y / delta_x > 2:
        # up-down
        region_box = search_object_yolo(x_min, x_max, y_min, y_middle, region_box, times + 1, fractal_type, n, z, apply_canny, save_folder)
        region_box = search_object_yolo(x_min, x_max, y_middle, y_max, region_box, times + 1, fractal_type, n, z, apply_canny, save_folder)

    else:
        # four pieces
        # region 1
        region_box = search_object_yolo(x_min, x_middle, y_min, y_middle, region_box, times + 1, fractal_type, n, z, apply_canny, save_folder)
        # region 2
        region_box = search_object_yolo(x_middle, x_max, y_min, y_middle, region_box, times + 1, fractal_type, n, z, apply_canny, save_folder)
        # region 3
        region_box = search_object_yolo(x_min, x_middle, y_middle, y_max, region_box, times + 1, fractal_type, n, z, apply_canny, save_folder)
        # region 4
        region_box = search_object_yolo(x_middle, x_max, y_middle, y_max, region_box, times + 1, fractal_type, n, z, apply_canny, save_folder)

    return region_box


def search_object_yolo(x_min, x_max, y_min, y_max, region_box, times, fractal_type, n, z, apply_canny, save_folder):
    if times > 9:
        return region_box

    region, x_min, x_max, y_min, y_max = generate_fractals(fractal_type, x_min, x_max, y_min, y_max, z, n)
    region, x_min, x_max, y_min, y_max = generate_fractals(fractal_type, x_min, x_max, y_min, y_max, z, n, crop=False)

    # empty region
    if len(region) < 0:
        return region_box
    if np.max(region) < 120:
        return region_box
    # sparse region
    if (calc_occupancy(region) < 0.01) or (calc_occupancy(region) > 0.99):
        return region_box

    ###############################
    #############detect############
    ###############################
    if apply_canny:
        gray_img = convert_gray2rgb(region)

        # apply corner detection
        img_blur = cv2.GaussianBlur(gray_img, (3, 3), 0)
        canny = cv2.Canny(img_blur, 50, 150)

        # detect object
        img, img_src = precess_image(canny, img_size, stride, half)
    else:
        # detect object
        img, img_src = precess_image(region, img_size, stride, half)

    img = img.to(device)
    if len(img.shape) == 3:
        img = img[None]
        # expand for batch dim
    pred_results = model(img)

    classes: Optional[List[int]] = None  # the classes to keep
    det = non_max_suppression(pred_results,
                              conf_thres,
                              iou_thres,
                              classes,
                              agnostic_nms,
                              max_det=max_det)[0]

    ###############################
    #############save##############
    ###############################
    if len(det) != 0:
        print(
            times * "    " + " -> Find " + str(len(det)) +
            " objects in resgion ", x_min, x_max, y_min, y_max, "!")

        for *xyxy, conf, cls in reversed(det):
            class_num = int(cls)
            minx, miny, maxx, maxy = np.array([np.array(num) for num in xyxy])
            x_min_new, x_max_new, y_min_new, y_max_new = update_region(
                x_min, x_max, y_min, y_max, minx, maxx, miny, maxy)
            region_box.append(
                ([class_names[class_num], x_min_new, x_max_new, y_min_new, y_max_new]))
            # save object fig
            title = f'{conf:.2f} {class_names[class_num]} {x_min_new, x_max_new, y_min_new, y_max_new}'
            save_fractal(x_min_new, x_max_new, y_min_new, y_max_new, title, fractal_type, z, n, save_folder)

    ###############################
    #########keep search###########
    ###############################

    # 搜索该区域的子区域
    region_box = search_sub_yolo(x_min, x_max, y_min, y_max, region_box, times, fractal_type, n, z, apply_canny, save_folder)

    return region_box

def multi_processor4(fractal_type, x_min, x_max, y_min, y_max, z, n, region_box, times, apply_canny, save_folder):
    region, x_min, x_max, y_min, y_max = generate_fractals(fractal_type, x_min, x_max, y_min, y_max, z, n)
    x_middle = ((x_min + x_max) / 2)
    y_middle = ((y_min + y_max) / 2)
    p1 = Process(target=search_object_yolo,args=(x_min, x_middle, y_min, y_middle, region_box, times + 1, fractal_type, n, z, apply_canny, save_folder))
    p2 = Process(target=search_object_yolo,args=(x_middle, x_max, y_min, y_middle, region_box, times + 1, fractal_type, n, z, apply_canny, save_folder))
    p3 = Process(target=search_object_yolo,args=(x_min, x_middle, y_middle, y_max, region_box, times + 1, fractal_type, n, z, apply_canny, save_folder))
    p4 = Process(target=search_object_yolo,args=(x_middle, x_max, y_middle, y_max, region_box, times + 1, fractal_type, n, z, apply_canny, save_folder))
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    return region_box

def multi_processor16(fractal_type, x_min, x_max, y_min, y_max, z, n, region_box, times, apply_canny, save_folder):
    region, x_min, x_max, y_min, y_max = generate_fractals(fractal_type, x_min, x_max, y_min, y_max, z, n)
    x_middle = ((x_min + x_max) / 2)
    y_middle = ((y_min + y_max) / 2)
    p1 = Process(target=multi_processor4,args=(fractal_type, x_min, x_middle, y_min, y_middle, z, n, region_box, times + 1, apply_canny, save_folder))
    p2 = Process(target=multi_processor4,args=(fractal_type, x_middle, x_max, y_min, y_middle, z, n, region_box, times + 1, apply_canny, save_folder))
    p3 = Process(target=multi_processor4,args=(fractal_type, x_min, x_middle, y_middle, y_max, z, n, region_box, times + 1, apply_canny, save_folder))
    p4 = Process(target=multi_processor4,args=(fractal_type, x_middle, x_max, y_middle, y_max, z, n, region_box, times + 1, apply_canny, save_folder))
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    return region_box


if __name__ == "__main__":
    # fractal type
    fractal_type = "Julia"  # @param ["Mandelbrot", "Burning_ship", "Julia", "Multibrot"]
    z = -0.4 + 0.6 * 1j  # only Julia need
    n = 3  # only Multibrot need

    # if canny detection on fractals when search
    apply_canny = False  #@param [True, False]

    # searching...
    save_folder = save_path(fractal_type, apply_canny, yaml, n, z)
    x_min, x_max, y_min, y_max = -4, 4, -4, 4
    region_box = []
    T1 = time.time()
    region_box = multi_processor16(fractal_type, x_min, x_max, y_min, y_max, z, n, region_box, 0, apply_canny, save_folder)
    T2 = time.time()
    print('program running time:%s h' % str(int(T2 - T1)/3600))
    with open('./object/'+save_folder+'_log.txt','w') as f:
        f.write('program running time:%s h' % str(int(T2 - T1)/3600))
