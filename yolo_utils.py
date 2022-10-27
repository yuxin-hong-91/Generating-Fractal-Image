import math
from yolov6.layers.common import DetectBackend
from yolov6.data.data_augment import letterbox
from yolov6.utils.events import  load_yaml
import torch
from utils import *
import os

def check_img_size(img_size, s=32, floor=0):

    def make_divisible(x, divisor):
        # Upward revision the value x to make it evenly divisible by the divisor.
        return math.ceil(x / divisor) * divisor

    """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
    if isinstance(img_size, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(img_size, int(s)), floor)
    elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
        new_size = [max(make_divisible(x, int(s)), floor) for x in img_size]
    else:
        raise Exception(f"Unsupported type of img_size: {type(img_size)}")

    if new_size != img_size:
        print(
            f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}'
        )
    return new_size if isinstance(img_size, list) else [new_size] * 2

def precess_image(gray_image, img_size, stride, half):
    '''Process image before image inference.'''
    img_src = convert_gray2rgb(gray_image)
    image = letterbox(img_src, img_size, stride=stride)[0]

    # Convert
    image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    image = torch.from_numpy(np.ascontiguousarray(image))
    image = image.half() if half else image.float()  # uint8 to fp16/32
    image /= 255  # 0 - 255 to 0.0 - 1.0

    return image.clone().detach(), img_src

def load_model(checkpoint):
    device: str = "cpu"  # @param ["gpu", "cpu"]
    half: bool = False  # @param {type:"boolean"}
    model = DetectBackend(r"E:/YOLOv6-main/weights/"+checkpoint+".pt", device=device)
    stride = model.stride

    cuda = device != 'cpu' and torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')

    hide_labels: bool = False  # @param {type:"boolean"}
    hide_conf: bool = False  # @param {type:"boolean"}
    img_size: int = 640  # @param {type:"integer"}
    conf_thres: float = .05  # @param {type:"number"}
    iou_thres: float = .45  # @param {type:"number"}
    max_det: int = 1000  # @param {type:"integer"}
    agnostic_nms: bool = False  # @param {type:"boolean"}
    img_size = check_img_size(img_size, s=stride)

    if half & (device.type != 'cpu'):
        model.model.half()
    else:
        model.model.float()
        half = False

    if device.type != 'cpu':
        model(
            torch.zeros(1, 3, *img_size).to(device).type_as(
                next(model.model.parameters())))  # warmup
    return model, stride, device, img_size, half


def load(checkpoint):
    if checkpoint == "yolov6t_coco_edter":
        yaml: str = "./data/coco.yaml"
        class_names = load_yaml(yaml)['names']
    elif checkpoint == "yolov6t_aquarium_edter":
        yaml: str = "./data/aquarium.yaml"
        class_names = load_yaml(yaml)['names']
    else:
        yaml = None
        class_names = []
        print("please check yaml filename.")

    return yaml, class_names


def save_path(fractal_type, canny, yaml, n, z):
    if fractal_type == "Multibrot":
        save_folder = fractal_type + str(n)
    elif fractal_type == "Multibrot":
        save_folder = fractal_type + str(z)
    else:
        save_folder = fractal_type

    save_folder += '_' + yaml.split('.')[1].split('/')[-1]

    if canny:
        save_folder += '_canny'
    else:
        save_folder += '_solid'

    if not os.path.isdir("./object/" + save_folder):
        print("making folder: " + "./object/" + save_folder)
        os.makedirs("./object/" + save_folder)
    return save_folder