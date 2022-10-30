from imagenet_util import *
from utils import *
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from object_detection import save_fractal
from multiprocessing import Process
from tensorflow.keras.applications import (
        vgg16,
        resnet50,
        mobilenet,
        inception_v3
    )

vgg_model, inception_model, resnet_model, mobilenet_model = load_model()



def search_sub(x_min, x_max, y_min, y_max, region_box, times, fractal_type, n, z, save_folder):
    x_middle = ((x_min + x_max) / 2)
    y_middle = ((y_min + y_max) / 2)

    # four pieces
    # region 1
    region_box = search(x_min, x_middle, y_min, y_middle, region_box, times, fractal_type, n, z, save_folder)
    # region 2
    region_box = search(x_middle, x_max, y_min, y_middle, region_box, times, fractal_type, n, z, save_folder)
    # region 3
    region_box = search(x_min, x_middle, y_middle, y_max, region_box, times, fractal_type, n, z, save_folder)
    # region 4
    region_box = search(x_middle, x_max, y_middle, y_max, region_box, times, fractal_type, n, z, save_folder)

    # four middle region
    x_upper = ((x_min + x_middle) / 2)
    x_lower = ((x_middle + x_max) / 2)
    y_upper = ((y_min + y_middle) / 2)
    y_lower = ((y_middle + y_max) / 2)
    # region 5
    region_box = search(x_lower, x_upper, y_min, y_middle, region_box, times, fractal_type, n, z, save_folder)
    # region 6
    region_box = search(x_middle, x_max, y_lower, y_upper, region_box, times, fractal_type, n, z, save_folder)
    # region 7
    region_box = search(x_lower, x_upper, y_middle, y_max, region_box, times, fractal_type, n, z, save_folder)
    # region 8
    region_box = search(x_middle, x_max, y_lower, y_upper, region_box, times, fractal_type, n, z, save_folder)

    # center region 9
    region_box = search(x_lower, x_upper, y_lower, y_upper, region_box, times, fractal_type, n, z, save_folder)

    return region_box



def search(x_min, x_max, y_min, y_max, region_box, times, fractal_type, n, z,
           save_folder):
    if times > 7:
        return region_box

    region, x_min, x_max, y_min, y_max = generate_fractals(fractal_type,
                                                           x_min,
                                                           x_max,
                                                           y_min,
                                                           y_max,
                                                           z,
                                                           n,
                                                           crop=False)

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
    # detect object
    # VGG
    image_batch = fractal_preprocess(region)
    processed_image = vgg16.preprocess_input(image_batch.copy())
    predictions = vgg_model.predict(processed_image)
    label_vgg = decode_predictions(predictions)
    if label_vgg:
        top = label_vgg[0][0]
        class_name = top[1]
        conf = top[2]
        title = f'{conf:.2f} {class_name} {x_min, x_max, y_min, y_max}'
        if conf > 0.5:
            print(class_name,conf,'found in ',x_min, x_max, y_min, y_max)
            plt.figure()
            plt.imshow(region, cmap='gray')
            plt.axis('off')
            plt.savefig('G:/object/' + save_folder + '/{}.png'.format(title),
                        format="jpg",
                        dpi=200,
                        bbox_inches='tight',
                        pad_inches=0,
                        transparent=False)
            plt.close()
            region_box.append(([class_name, x_min, x_max, y_min, y_max]))

    ###############################
    #########keep search###########
    ###############################

    # 搜索该区域的子区域
    region_box = search_sub(x_min, x_max, y_min, y_max, region_box, times + 1, fractal_type, n, z, save_folder)

    return region_box

def multi_processor4(fractal_type, x_min, x_max, y_min, y_max, z, n, region_box, times, save_folder):
    x_middle = ((x_min + x_max) / 2)
    y_middle = ((y_min + y_max) / 2)
    p1 = Process(target=search,args=(x_min, x_middle, y_min, y_middle, region_box, times + 1, fractal_type, n, z, save_folder))
    p2 = Process(target=search,args=(x_middle, x_max, y_min, y_middle, region_box, times + 1, fractal_type, n, z, save_folder))
    p3 = Process(target=search,args=(x_min, x_middle, y_middle, y_max, region_box, times + 1, fractal_type, n, z, save_folder))
    p4 = Process(target=search,args=(x_middle, x_max, y_middle, y_max, region_box, times + 1, fractal_type, n, z, save_folder))
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    return region_box

def multi_processor16(fractal_type, x_min, x_max, y_min, y_max, z, n, region_box, times, save_folder):
    region, x_min, x_max, y_min, y_max = generate_fractals(fractal_type, x_min, x_max, y_min, y_max, z, n)
    x_middle = ((x_min + x_max) / 2)
    y_middle = ((y_min + y_max) / 2)
    p1 = Process(target=multi_processor4,args=(fractal_type, x_min, x_middle, y_min, y_middle, z, n, region_box, times + 1, save_folder))
    p2 = Process(target=multi_processor4,args=(fractal_type, x_middle, x_max, y_min, y_middle, z, n, region_box, times + 1, save_folder))
    p3 = Process(target=multi_processor4,args=(fractal_type, x_min, x_middle, y_middle, y_max, z, n, region_box, times + 1, save_folder))
    p4 = Process(target=multi_processor4,args=(fractal_type, x_middle, x_max, y_middle, y_max, z, n, region_box, times + 1, save_folder))
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
    # load model
    x_min, x_max, y_min, y_max = -2, 1.5, -2, 1.5
    region_box = []
    times = 0
    fractal_type = 'Mandelbrot'
    z = -0.4 + 0.6 * 1j  # only Julia need
    n = 3  # only Multibrot need
    save_folder = 'vgg/Mandelbrot'
    multi_processor4(fractal_type, x_min, x_max, y_min, y_max, z, n, region_box, times, save_folder)

