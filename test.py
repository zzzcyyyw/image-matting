import math

import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from config import device, fg_path_test, a_path_test, bg_path_test
from data_gen import data_transforms, fg_test_files, bg_test_files, gen_depth
from utils import compute_mse, compute_sad, AverageMeter, get_logger


def gen_test_names():
    num_fgs = 50
    num_bgs = 1000
    num_bgs_per_fg = 20

    names = []
    bcount = 0
    for fcount in range(num_fgs):
        for i in range(num_bgs_per_fg):
            names.append(str(fcount) + '_' + str(bcount) + '.png')
            bcount += 1

    return names


# def process_test(im_name, bg_name, trimap):         # 原
def process_test(im_name, bg_name):                   # zc
    # print(bg_path_test + bg_name)
    im = cv.imread(fg_path_test + im_name)
    a = cv.imread(a_path_test + im_name, 0)
    h, w = im.shape[:2]
    bg = cv.imread(bg_path_test + bg_name)
    bh, bw = bg.shape[:2]
    wratio = w / bw
    hratio = h / bh
    ratio = wratio if wratio > hratio else hratio
    if ratio > 1:
        bg = cv.resize(src=bg, dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)), interpolation=cv.INTER_CUBIC)

    # return composite4_test(im, bg, a, w, h, trimap)         # 原
    return composite4_test(im, bg, a, w, h)                   # zc


# def composite4_test(fg, bg, a, w, h):
#     fg = np.array(fg, np.float32)
#     bg_h, bg_w = bg.shape[:2]
#     x = max(0, int((bg_w - w)/2))
#     y = max(0, int((bg_h - h)/2))
#     bg = np.array(bg[y:y + h, x:x + w], np.float32)
#     alpha = np.zeros((h, w, 1), np.float32)
#     alpha[:, :, 0] = a / 255.
#     im = alpha * fg + (1 - alpha) * bg
#     im = im.astype(np.uint8)
#     print('im.shape: ' + str(im.shape))
#     print('a.shape: ' + str(a.shape))
#     print('fg.shape: ' + str(fg.shape))
#     print('bg.shape: ' + str(bg.shape))
#     return im, a, fg, bg


# def composite4_test(fg, bg, a, w, h, trimap):           # 原
def composite4_test(fg, bg, a, w, h):
    fg = np.array(fg, np.float32)
    bg_h, bg_w = bg.shape[:2]
    x = max(0, int((bg_w - w) / 2))
    y = max(0, int((bg_h - h) / 2))
    crop = np.array(bg[y:y + h, x:x + w], np.float32)
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a / 255.
    # trimaps = np.zeros((h, w, 1), np.float32)
    # trimaps[:,:,0]=trimap/255.

    im = alpha * fg + (1 - alpha) * crop
    im = im.astype(np.uint8)

    new_a = np.zeros((bg_h, bg_w), np.uint8)
    new_a[y:y + h, x:x + w] = a
    # new_trimap = np.zeros((bg_h, bg_w), np.uint8)                   # 原
    # new_trimap[y:y + h, x:x + w] = trimap                           # 原
    # cv.imwrite('images/test/new/' + trimap_name, new_trimap)        # 原
    new_im = bg.copy()
    new_im[y:y + h, x:x + w] = im
    # cv.imwrite('images/test/new_im/'+trimap_name,new_im)
    # return new_im, new_a, fg, bg, new_trimap                        # 原
    return new_im, new_a, fg, bg                                      # zc


import os
import depth_networks
import PIL.Image as pil
def depth_gen(img):
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

    # 转换NumPy数组为PIL图像并进行预处理
    input_image = pil.fromarray(img)
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
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


if __name__ == '__main__':
    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint)
    # model = checkpoint['model'].module
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    transformer = data_transforms['valid']

    names = gen_test_names()

    mse_losses = AverageMeter()
    sad_losses = AverageMeter()

    logger = get_logger()
    i = 0
    for name in tqdm(names):
        fcount = int(name.split('.')[0].split('_')[0])
        bcount = int(name.split('.')[0].split('_')[1])
        im_name = fg_test_files[fcount]
        # print(im_name)
        bg_name = bg_test_files[bcount]
        # trimap_name = im_name.split('.')[0] + '_' + str(i) + '.png'         # 原
        # print('trimap_name: ' + str(trimap_name))

        # trimap = cv.imread('data/Combined_Dataset/Test_set/Adobe-licensed images/trimaps/' + trimap_name, 0)        # 原
        # print('trimap: ' + str(trimap))

        # img, alpha, fg, bg, new_trimap = process_test(im_name, bg_name, trimap)         # 原
        img, alpha, fg, bg = process_test(im_name, bg_name)                           # zc
        h, w = img.shape[:2]
        depth_map = depth_gen(img)                                                    # zc
        depth_map_name = im_name.split('.')[0] + '_' + str(i) + '.png'                # zc
        # cv.imwrite('images/test/new2/' + depth_map_name, depth_map * 255)              # zc

        i += 1
        if i == 20:
            i = 0

        # mytrimap = gen_trimap(alpha)
        # cv.imwrite('images/test/new_im/'+trimap_name,mytrimap)

        x = torch.zeros((1, 4, h, w), dtype=torch.float)
        img = img[..., ::-1]  # RGB
        img = transforms.ToPILImage()(img)  # [3, 320, 320]
        img = transformer(img)  # [3, 320, 320]
        x[0:, 0:3, :, :] = img
        # x[0:, 3, :, :] = torch.from_numpy(new_trimap.copy() / 255.)             # 原
        x[0:, 3, :, :] = depth_map                   # zc

        # Move to GPU, if available
        x = x.type(torch.FloatTensor).to(device)  # [1, 4, 320, 320]
        alpha = alpha / 255.

        with torch.no_grad():
            pred = model(x)  # [1, 4, 320, 320]

        pred = pred.cpu().numpy()
        pred = pred.reshape((h, w))  # [320, 320]

        # pred[new_trimap == 0] = 0.0                                             # 原
        # pred[new_trimap == 255] = 1.0                                           # 原
        # cv.imwrite('images/test/out/' + trimap_name, pred * 255)                # 原
        cv.imwrite('images/test/out4/' + depth_map_name, pred * 255)        # zc

        # Calculate loss
        # loss = criterion(alpha_out, alpha_label)
        # mse_loss = compute_mse(pred, alpha, trimap)                             # 原
        mse_loss = compute_mse(pred, alpha)                                   # zc
        sad_loss = compute_sad(pred, alpha)

        # Keep track of metrics
        mse_losses.update(mse_loss.item())
        sad_losses.update(sad_loss.item())
        print("sad:{} mse:{}".format(sad_loss.item(), mse_loss.item()))
        print("avg_sad:{} avg_mse:{}".format(sad_losses.avg, mse_losses.avg))
    print("fin_sad:{} fin_mse:{}".format(sad_losses.avg, mse_losses.avg))
