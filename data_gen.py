import math
import os
import random
import PIL.Image as pil
import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from config import im_size, unknown_code, fg_path, bg_path, a_path, num_valid                 # 原
# from config import im_size, lower_bound, upper_bound, fg_path, bg_path, a_path, num_valid # zc
from utils import safe_crop

# Data augmentation and normalization for training
# Just normalization for validation
"""
原
"""
# data_transforms = {
#     'train': transforms.Compose([
#         transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     ]),
#     'valid': transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }

"""
zc
"""
data_transforms = {
    'train': transforms.Compose([
        transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
with open('Combined_Dataset/Training_set/training_fg_names.txt') as f:
    fg_files = f.read().splitlines()
with open('Combined_Dataset/Training_set/training_bg_names.txt') as f:
    bg_files = f.read().splitlines()
with open('Combined_Dataset/Test_set/test_fg_names.txt') as f:
    fg_test_files = f.read().splitlines()
with open('Combined_Dataset/Test_set/test_bg_names.txt') as f:
    bg_test_files = f.read().splitlines()


def get_alpha(name):
    fg_i = int(name.split("_")[0])
    name = fg_files[fg_i]
    filename = os.path.join('data/mask', name)
    alpha = cv.imread(filename, 0)
    return alpha


def get_alpha_test(name):
    fg_i = int(name.split("_")[0])
    name = fg_test_files[fg_i]
    filename = os.path.join('data/mask_test', name)
    alpha = cv.imread(filename, 0)
    return alpha


def composite4(fg, bg, a, w, h):
    fg = np.array(fg, np.float32)
    bg_h, bg_w = bg.shape[:2]
    x = 0
    if bg_w > w:
        x = np.random.randint(0, bg_w - w)
    y = 0
    if bg_h > h:
        y = np.random.randint(0, bg_h - h)
    bg = np.array(bg[y:y + h, x:x + w], np.float32)
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a / 255.
    im = alpha * fg + (1 - alpha) * bg
    im = im.astype(np.uint8)
    return im, a, fg, bg


def process(im_name, bg_name):
    im = cv.imread(fg_path + im_name)
    a = cv.imread(a_path + im_name, 0)
    h, w = im.shape[:2]
    bg = cv.imread(bg_path + bg_name)
    bh, bw = bg.shape[:2]
    wratio = w / bw
    hratio = h / bh
    ratio = wratio if wratio > hratio else hratio
    if ratio > 1:
        bg = cv.resize(src=bg, dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)), interpolation=cv.INTER_CUBIC)

    return composite4(im, bg, a, w, h)


"""
原
"""
def gen_trimap(alpha):
    k_size = random.choice(range(1, 5))     # zc：随机选择一个膨胀和腐蚀的内核大小
    iterations = np.random.randint(1, 20)   # zc：随机生成膨胀和腐蚀的迭代次数
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_size, k_size))       # zc：创建膨胀和腐蚀的内核（用于膨胀和腐蚀操作）
    dilated = cv.dilate(alpha, kernel, iterations)
    eroded = cv.erode(alpha, kernel, iterations)
    trimap = np.zeros(alpha.shape)
    trimap.fill(128)
    trimap[eroded >= 255] = 255
    trimap[dilated <= 0] = 0
#
#     # cv.imshow('Dilated Image', dilated)
#     # cv.imshow('Eroded Image', eroded)
#     # cv.imshow('Trimap', trimap)
#     # cv.waitKey(0)
#     # cv.destroyAllWindows()
#
    return trimap


# from __future__ import absolute_import, division, print_function
# import os
# import numpy as np
# import torch
# from torchvision import transforms
"""
zc
"""
import depth_networks
# from depth_utils import download_model_if_doesnt_exist
# import matplotlib.pyplot as plt

# Randomly crop (image, trimap) pairs centered on pixels in the unknown regions.
"""原"""
def random_choice(trimap, crop_size=(320, 320)):
    crop_height, crop_width = crop_size
    y_indices, x_indices = np.where(trimap == unknown_code)
    num_unknowns = len(y_indices)
    x, y = 0, 0
    if num_unknowns > 0:
        ix = np.random.choice(range(num_unknowns))
        center_x = x_indices[ix]
        center_y = y_indices[ix]
        x = max(0, center_x - int(crop_width / 2))
        y = max(0, center_y - int(crop_height / 2))
    return x, y

"""zc"""
# def random_choice(alpha, crop_size=(320, 320)):
#     crop_height, crop_width = crop_size
#     y_indices, x_indices = np.where((alpha >= lower_bound) & (alpha <= upper_bound))
#     num_pixels = len(y_indices)
#     x, y = 0, 0
#     if num_pixels > 0:
#         ix = np.random.choice(range(num_pixels))
#         center_x = x_indices[ix]
#         center_y = y_indices[ix]
#         x = max(0, center_x - int(crop_width / 2))
#         y = max(0, center_y - int(crop_height / 2))
#     return x, y


'''zc增加'''
def gen_depth(img, encoder, depth_decoder, feed_height, feed_width):
    # 转换NumPy数组为PIL图像并进行预处理
    input_image = pil.fromarray(img)
    input_image_resized = input_image.resize((feed_width, feed_height), pil.Resampling.LANCZOS)
    input_image_tensor = transforms.ToTensor()(input_image_resized).unsqueeze(0)
    # input_image_tensor = input_image_tensor

    # 使用PyTorch模型进行预测
    with torch.no_grad():
        features = encoder(input_image_tensor)
        outputs = depth_decoder(features)

    disp = outputs[("disp", 0)]

    # 将深度信息调整为输入图像的尺寸
    depth_map = torch.nn.functional.interpolate(disp,
                                                (input_image.size[1], input_image.size[0]), mode="bilinear",
                                                align_corners=False)
    depth_map = depth_map.squeeze(1)

    # 将深度信息转换为NumPy数组
    # depth_map = depth_map.squeeze().cpu().numpy()

    return depth_map


class DIMDataset(Dataset):
    def __init__(self, split):
        self.split = split

        filename = '{}_names.txt'.format(split)
        with open(filename, 'r') as file:
            self.names = file.read().splitlines()
        self.transformer = data_transforms[split]

        # 加载深度模型    zc增加
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder, self.depth_decoder, self.feed_height, self.feed_width = self.load_depth_model()

    '''zc增加'''
    def load_depth_model(self):
        model_name = "mono_640x192"
        model_path = os.path.join("models", model_name)
        encoder_path = os.path.join(model_path, "encoder.pth")
        depth_decoder_path = os.path.join(model_path, "depth.pth")

        encoder = depth_networks.ResnetEncoder(18, False)
        depth_decoder = depth_networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

        loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
        encoder.load_state_dict(filtered_dict_enc)

        loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
        depth_decoder.load_state_dict(loaded_dict)

        # encoder.to(self.device)
        encoder.eval()
        # depth_decoder.to(self.device)
        depth_decoder.eval()

        feed_height = loaded_dict_enc['height']
        feed_width = loaded_dict_enc['width']

        return encoder, depth_decoder, feed_height, feed_width

    def __getitem__(self, i):
        name = self.names[i]
        fcount = int(name.split('.')[0].split('_')[0])
        bcount = int(name.split('.')[0].split('_')[1])
        im_name = fg_files[fcount]
        bg_name = bg_files[bcount]
        img, alpha, fg, bg = process(im_name, bg_name)

        # depth_map = gen_depth(img)                              # zc

        # cv.imshow('img', img)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        # crop size 320:640:480 = 1:1:1
        different_sizes = [(320, 320), (480, 480), (640, 640)]
        crop_size = random.choice(different_sizes)

        trimap = gen_trimap(alpha)                            # 原
        x, y = random_choice(trimap, crop_size)               # 原
        # x, y = random_choice(alpha, crop_size)  # zc
        img = safe_crop(img, x, y, crop_size)
        alpha = safe_crop(alpha, x, y, crop_size)

        # trimap = gen_trimap(alpha)                              # 原

        # Flip array left to right randomly (prob=1:1)
        if np.random.random_sample() > 0.5:
            img = np.fliplr(img)
            # trimap = np.fliplr(trimap)                        # 原
            alpha = np.fliplr(alpha)

        depth_map = gen_depth(img, self.encoder, self.depth_decoder, self.feed_height, self.feed_width)  # zc

        # depth_map = gen_depth(img)  # zc
        # cv.imwrite('ceshi/0/' + im_name, depth_map * 255)
        # depth_map[alpha == 0] = 0.0
        # depth_map[alpha == 255] = 1.0
        # cv.imwrite('ceshi/1/' + im_name, depth_map * 255)

        x = torch.zeros((4, im_size, im_size), dtype=torch.float)
        img = img[..., ::-1]  # RGB
        img = transforms.ToPILImage()(img)
        img = self.transformer(img)
        x[0:3, :, :] = img
        x[3, :, :] = depth_map
        # x[3, :, :] = torch.from_numpy(trimap.copy() / 255.)     # 原
        # x[3, :, :] = torch.from_numpy(depth_map.copy() / 255.)  # zc
        # depth_map = x[3, :, :]

        # y = np.empty((2, im_size, im_size), dtype=np.float32)     # 原
        y = np.empty((1, im_size, im_size), dtype=np.float32)  # zc
        y[0, :, :] = alpha / 255.
        # mask = np.equal(trimap, 128).astype(np.float32)           # 原
        # y[1, :, :] = mask                                         # 原
        # 每一条数据
        return x, y

    def __len__(self):
        return len(self.names)


def gen_names():
    num_fgs = 431
    num_bgs = 43100
    num_bgs_per_fg = 100

    names = []
    bcount = 0
    for fcount in range(num_fgs):
        for i in range(num_bgs_per_fg):
            names.append(str(fcount) + '_' + str(bcount) + '.png')
            bcount += 1

    valid_names = random.sample(names, num_valid)
    train_names = [n for n in names if n not in valid_names]

    with open('valid_names.txt', 'w') as file:
        file.write('\n'.join(valid_names))

    with open('train_names.txt', 'w') as file:
        file.write('\n'.join(train_names))


if __name__ == "__main__":
    gen_names()
