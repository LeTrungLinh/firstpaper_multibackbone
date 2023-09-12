import torch
import torch.nn as nn
from models_custom.mobilevig import Stem, Grapher, FFN, InvertedResidual, Downsample
from timm.models.layers import DropPath

"""Custom Unet with mobiblevig as backbone
    for segmentation of images"""


class UpsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(UpsampleConv, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x

class UNetMobileVig(nn.Module):
    def __init__(self, local_channels, global_channels, drop_path) -> None:
        """Initialize the model
        Args: local_channels (list): list of channels for each layer
        """
        super(UNetMobileVig, self).__init__()

        self.stem = Stem(input_dim=3, output_dim=local_channels[0])

        self.stage1 = InvertedResidual(dim=local_channels[0], mlp_ratio=4, drop_path=drop_path)
        self.dow_stage1 = Downsample(local_channels[0], local_channels[1])

        self.stage2 = InvertedResidual(dim=local_channels[1], mlp_ratio=4, drop_path=drop_path)
        self.dow_stage2 = Downsample(local_channels[1], local_channels[2])

        self.stage3 = InvertedResidual(dim=local_channels[2], mlp_ratio=4, drop_path=drop_path)
        self.dow_stage3 = Downsample(local_channels[2], local_channels[3])

        self.stage4 = InvertedResidual(dim=local_channels[3], mlp_ratio=4, drop_path=drop_path)
        self.dow_stage4 = Downsample(local_channels[3], global_channels)

        self.bottom_neck = nn.ModuleList([])
        for j in range(2):
            self.bottom_neck += [nn.Sequential(
                                Grapher(global_channels, drop_path= 0, K=2),
                                FFN(global_channels, global_channels* 4, drop_path=drop_path))
                                ]

        # Upsamlping layers

        self.up_stage4 = UpsampleConv(global_channels, local_channels[3])
        self.up_conv_stage4 = InvertedResidual(dim=local_channels[3], mlp_ratio=4, drop_path=drop_path)

        self.up_stage3 = UpsampleConv(local_channels[3], local_channels[2])
        self.up_conv_stage3 = InvertedResidual(dim=local_channels[2], mlp_ratio=4, drop_path=drop_path)

        self.up_stage2 = UpsampleConv(local_channels[2], local_channels[1])
        self.up_conv_stage2 = InvertedResidual(dim=local_channels[1], mlp_ratio=4, drop_path=drop_path)

        self.up_stage1 = UpsampleConv(local_channels[1], local_channels[0])
        self.up_conv_stage1 = InvertedResidual(dim=local_channels[0], mlp_ratio=4, drop_path=drop_path)

        self.up_stage0 = UpsampleConv(local_channels[0], local_channels[0])
        self.up_conv_stage0 = InvertedResidual(dim=local_channels[0], mlp_ratio=4, drop_path= drop_path)
    
        self.final = nn.Conv2d(local_channels[0], 1, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        x = self.stem(x) # 1/2
        
        out = self.stage1(x) # 1/2
        out_stage1 = self.dow_stage1(out) # 1/4

        out = self.stage2(out_stage1) # 1/4
        out_stage2 = self.dow_stage2(out) # 1/8

        out = self.stage3(out_stage2) # 1/8
        out_stage3 = self.dow_stage3(out) # 1/16

        out = self.stage4(out_stage3) # 1/16
        out_stage4 = self.dow_stage4(out) # 1/32

        for i in range(len(self.bottom_neck)):
            out_bottom_neck = self.bottom_neck[i](out_stage4) # 1/32

        
        out = self.up_stage4(out_bottom_neck) # 1/16
        out_up_stage4 = self.up_conv_stage4(out) # 1/16

        out = self.up_stage3(out_up_stage4) # 1/8
        out_up_stage3 = self.up_conv_stage3(out) # 1/8

        out = self.up_stage2(out_up_stage3) # 1/4
        out_up_stage2 = self.up_conv_stage2(out) # 1/4

        out = self.up_stage1(out_up_stage2) # 1/2
        out_up_stage1 = self.up_conv_stage1(out) # 1/2

        out = self.up_stage0(out_up_stage1) # 1/1
        out_up_stage0 = self.up_conv_stage0(out) # 1/1


        out = self.final(out_up_stage0) # 1/1

        return out
    

if __name__ == "__main__":
    # random input to test model
    import torchsummary
    x = torch.randn(1, 3, 224, 224)
    model = UNetMobileVig(local_channels=[32, 64, 128, 256], global_channels=512, drop_path=0.1)
    torchsummary.summary(model, (3, 224, 224), device='cpu')
    model.eval()

    # y = model(x)
    # print(y.shape)