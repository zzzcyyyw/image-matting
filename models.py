import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from config import device, im_size


class conv2DBatchNormRelu(nn.Module):
    def __init__(
            self,
            in_channels,
            n_filters,
            k_size,
            stride,
            padding,
            bias=True,
            dilation=1,
            with_bn=True,
            with_relu=True
    ):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(int(in_channels),
                             int(n_filters),
                             kernel_size=k_size,
                             padding=padding,
                             stride=stride,
                             bias=bias,
                             dilation=dilation, )

        if with_bn:
            if with_relu:
                self.cbr_unit = nn.Sequential(conv_mod, nn.BatchNorm2d(int(n_filters)), nn.ReLU(inplace=True))
            else:
                self.cbr_unit = nn.Sequential(conv_mod, nn.BatchNorm2d(int(n_filters)))
        else:
            if with_relu:
                self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))
            else:
                self.cbr_unit = nn.Sequential(conv_mod)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs

class segnetDown2(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetDown2, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, k_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, k_size=3, stride=1, padding=1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs2 = self.conv2(outputs)
        unpooled_shape = outputs2.size()
        outputs, indices = self.maxpool_with_argmax(outputs2)
        return outputs2, outputs, indices, unpooled_shape


class segnetDown3(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetDown3, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, k_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, k_size=3, stride=1, padding=1)
        self.conv3 = conv2DBatchNormRelu(out_size, out_size, k_size=3, stride=1, padding=1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs3 = self.conv3(outputs)
        unpooled_shape = outputs3.size()
        outputs, indices = self.maxpool_with_argmax(outputs3)
        return outputs3, outputs, indices, unpooled_shape


class segnetUp1(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetUp1, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        # self.conv = conv2DBatchNormRelu(in_size, out_size, k_size=5, stride=1, padding=2, with_relu=False)      # 原
        self.conv = conv2DBatchNormRelu(in_size, out_size, k_size=5, stride=1, padding=2)       # zc

    def forward(self, inputs, indices, output_shape, down_1):       # 增加down_1
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = outputs + down_1      # zc增加
        outputs = self.conv(outputs)
        return outputs


class DIMModel(nn.Module):
    def __init__(self, n_classes=1, in_channels=4, is_unpooling=True, pretrain=True):
        super(DIMModel, self).__init__()

        self.in_channels = in_channels
        self.is_unpooling = is_unpooling
        self.pretrain = pretrain

        self.down1 = segnetDown2(self.in_channels, 64)
        self.down2 = segnetDown2(64, 128)
        self.down3 = segnetDown3(128, 256)           # 原
        # self.down3 = segnetDown3(128+1, 256)          # zc
        self.down4 = segnetDown3(256, 512)
        self.down5 = segnetDown3(512, 512)           # 原
        # self.down5 = segnetDown3(512+1, 512)      # zc

        self.down6 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)     # zc
        self.up6 =nn.Conv2d(512, 512, kernel_size=1, bias=True)        # zc

        self.up5 = segnetUp1(512, 512)
        self.up4 = segnetUp1(512, 256)
        self.up3 = segnetUp1(256, 128)
        self.up2 = segnetUp1(128, 64)
        # self.up1 = segnetUp1(64, n_classes)       # 原
        self.up1 = segnetUp1(64, 64)        # zc

        self.up0 = nn.Conv2d(64, n_classes, kernel_size=5, padding=2, bias=True)     # zc

        # self.deconv5_1 = nn.Conv2d(512, 512, kernel_size=5, padding=2, bias=True)
        # self.deconv4_1 = nn.Conv2d(512, 256, kernel_size=5, padding=2, bias=True)
        # self.deconv3_1 = nn.Conv2d(256, 128, kernel_size=5, padding=2, bias=True)
        # self.deconv2_1 = nn.Conv2d(128, 64, kernel_size=5, padding=2, bias=True)
        # self.deconv1_1 = nn.Conv2d(64, 64, kernel_size=5, padding=2, bias=True)
        #
        # self.deconv1 = nn.Conv2d(64, n_classes, kernel_size=5, padding=2, bias=True)

        self.sigmoid = nn.Sigmoid()

        if self.pretrain:
            import torchvision.models as models
            vgg16 = models.vgg16()
            self.init_vgg16_params(vgg16)

    def forward(self, inputs):
        # inputs: [N, 4, 320, 320]
        # input_shape = inputs.shape       # zc
        # rgb_inputs = torch.zeros(inputs.shape, dtype=inputs.dtype, device=inputs.device)     # zc
        # rgb_inputs = inputs[:, :3, :, :]            # zc
        # depth_inputs = inputs[:, 3:4, :, :]          # zc

        down1_1, down1, indices_1, unpool_shape1 = self.down1(inputs)        # 原
        # down1_1, down1, indices_1, unpool_shape1 = self.down1(rgb_inputs)    # zc
        # depth_inputs_down1 = torch.nn.functional.interpolate(depth_inputs, size=down1.size()[2:], mode="bilinear", align_corners=False)  # zc
        # down1 = down1 + depth_inputs_down1  # zc

        down2_1, down2, indices_2, unpool_shape2 = self.down2(down1)
        # depth_inputs_down2 = torch.nn.functional.interpolate(depth_inputs, size=down2.size()[2:], mode="bilinear", align_corners=False)     # zc
        # depth_inputs_down2 = torch.nn.functional.interpolate(depth_inputs, size=down2.size()[2:])     # zc
        # down2 = torch.cat([down2, depth_inputs_down2], dim=1)            # zc
        # down2 = down2 + depth_inputs_down2      # zc

        down3_1, down3, indices_3, unpool_shape3 = self.down3(down2)
        # depth_inputs_down3 = torch.nn.functional.interpolate(depth_inputs, size=down3.size()[2:], mode="bilinear", align_corners=False)  # zc
        # down3 = torch.cat([down2, depth_inputs_down3], dim=1)  # zc
        # down3 = down3 + depth_inputs_down3  # zc

        down4_1, down4, indices_4, unpool_shape4 = self.down4(down3)
        # depth_inputs_down4 = torch.nn.functional.interpolate(depth_inputs, size=down4.size()[2:], mode="bilinear", align_corners=False)  # zc
        # depth_inputs_down4 = torch.nn.functional.interpolate(depth_inputs, size=down4.size()[2:])       # zc
        # down4 = torch.cat([down4, depth_inputs_down4], dim=1)       # zc
        # down4 = down4 + depth_inputs_down4

        down5_1, down5, indices_5, unpool_shape5 = self.down5(down4)
        # depth_inputs_down5 = torch.nn.functional.interpolate(depth_inputs, size=down5.size()[2:], mode="bilinear", align_corners=False)  # zc
        # down5 = down5 + depth_inputs_down5  # zc

        down6 = F.relu(self.down6(down5))      # zc
        up6 = F.relu(self.up6(down6))      # zc

        # up5 = self.up5(down5, indices_5, unpool_shape5)       # 原
        up5 = self.up5(up6, indices_5, unpool_shape5, down5_1)       # zc
        up4 = self.up4(up5, indices_4, unpool_shape4, down4_1)
        up3 = self.up3(up4, indices_3, unpool_shape3, down3_1)
        up2 = self.up2(up3, indices_2, unpool_shape2, down2_1)
        up1 = self.up1(up2, indices_1, unpool_shape1, down1_1)

        up0 = self.up0(up1)     # zc

        x = torch.squeeze(up0, dim=1)  # [N, 1, 320, 320] -> [N, 320, 320]        # 原
        x = self.sigmoid(x)              # 原

        return x


    def init_vgg16_params(self, vgg16):
        blocks = [self.down1, self.down2, self.down3, self.down4, self.down5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        vgg_layers = []
        for _layer in features:
            if isinstance(_layer, nn.Conv2d):
                vgg_layers.append(_layer)

        merged_layers = []
        for idx, conv_block in enumerate(blocks):
            if idx < 2:
                units = [conv_block.conv1.cbr_unit, conv_block.conv2.cbr_unit]
            else:
                units = [
                    conv_block.conv1.cbr_unit,
                    conv_block.conv2.cbr_unit,
                    conv_block.conv3.cbr_unit,
                ]
            for _unit in units:
                for _layer in _unit:
                    if isinstance(_layer, nn.Conv2d):
                        merged_layers.append(_layer)

        assert len(vgg_layers) == len(merged_layers)

        for l1, l2 in zip(vgg_layers, merged_layers):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                if l1.weight.size() == l2.weight.size() and l1.bias.size() == l2.bias.size():
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data


if __name__ == '__main__':
    model = DIMModel().to(device)

    summary(model, (4, im_size, im_size))
