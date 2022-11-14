from utils import *
import matplotlib.pyplot as plt
from multiprocessing import Process
import numpy as np
from convert_gray import apply_sobel, apply_canny, class_dict
from keras.models import Model
import cifarEdgesModel
import tensorflow as tf
import tensorflow.keras.backend as K

model = cifarEdgesModel.create_model()
model.load_weights('./weights/lastest/weights-improvement-08-2.19.h5')
model = Model(inputs = model.input,outputs = model.layers[-1].output)
model.summary()
last_conv_layer = model.get_layer('conv_3_')
heatmap_model =tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])

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
    wax_lower = ((x_middle + x_max) / 2)
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
    region_32 = cv2.resize(region, (32, 32))
    # canny
    # canny = apply_canny(region)
    # sobel
    region_32 = apply_sobel(region_32)
    region_32 = region_32/255.0
    # inference

    with tf.GradientTape() as gtape:
        img = region_32.reshape((1, 32, 32, 1))
        conv_output, outcome = heatmap_model(img)
        prob = outcome[:, np.argmax(outcome[0])]  # 最大可能性类别的预测概率
        grads = gtape.gradient(prob, conv_output)  # 类别与卷积层的梯度 (1,14,14,512)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))  # 特征层梯度的全局平均代表每个特征层权重
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)  # 权重与特征层相乘，512层求和平均
    heatmap = np.maximum(heatmap, 0)
    max_heat = np.max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat

    outcome = model.predict(region_32.reshape((1, 32, 32, 1)))[0]
    ind = np.argmax(outcome)
    conf = outcome[ind]
    class_name = class_dict[ind]
    if conf > 0.4 and class_name != 'none' and class_name != 'sea':
        title = f'{conf:.2f} {class_name} {x_min, x_max, y_min, y_max}'
        print(f'save: {class_name} in {x_min}, {x_max}, {y_min}, {y_max}')
        original_img = convert_gray2rgb(region)
        heatmap1 = cv2.resize(heatmap[0], (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_CUBIC)
        heatmap1 = np.uint8(255 * heatmap1)
        heatmap1 = cv2.applyColorMap(heatmap1, cv2.COLORMAP_JET)
        frame_out = cv2.addWeighted(original_img, 0.75, heatmap1, 0.25, 0)
        cv2.imwrite(save_folder + '/{}.png'.format(title), frame_out)

        # plt.figure()
        # plt.imshow(region, cmap='gray')
        # plt.axis('off')
        # plt.savefig(save_folder + '/{}.png'.format(title),
        #             format="jpg",
        #             dpi=200,
        #             bbox_inches='tight',
        #             pad_inches=0,
        #             transparent=False)
        # plt.close()
        region_box.append(([class_name, x_min, x_max, y_min, y_max]))
    else:
        print(f'{class_name} {conf}')

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
    fractal_type = 'Burning_ship' # @param ["Mandelbrot", "Burning_ship", "Julia", "Multibrot"]
    z = -0.4 + 0.6 * 1j  # only Julia need
    n = 3  # only Multibrot need
    save_folder = 'G:/object/lastest/Burning_ship'
    # search(x_min, x_max, y_min, y_max, region_box, times, fractal_type, n, z, save_folder)
    multi_processor4(fractal_type, x_min, x_max, y_min, y_max, z, n, region_box, times, save_folder)

