import torch
import torch.nn as nn

class Encoder_Block(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=3,padding=1,down=True):
        super().__init__()
        self.use_down = down
        self.down = nn.MaxPool2d(2)
        self.resblock = nn.Conv2d(in_channels,out_channels,kernel_size=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        if self.use_down:
            x = self.down(x)
        return self.resblock(x)+self.conv2(x)


class Decoder_Block(nn.Module):

    def __init__(self, in_channels, out_channels,kernel_size=3,padding=1,dilation=1, up=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding,dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
    def forward(self, x1, x2):
        return self.conv(torch.cat([x2, self.up(x1)], dim=1))

class Up(nn.Module):

    def __init__(self, in_channels, out_channels,kernel_size=3,padding=1,dilation=1, up=True):
        super().__init__()
        self.use_up = up
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding,dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def forward(self, x1):
        if self.use_up:
            x1 = self.up(x1)
        return self.conv(x1)


class Context_Exploration_Block(nn.Module):

    # adapt from "Camouflaged Object Segmentation with Distraction Mining“
    # github: https://github.com/Mhaiyang/CVPR2021_PFNet

    def __init__(self, input_channels):
        super().__init__()
        self.input_channels = input_channels
        self.channels_single = input_channels

        self.p1_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single),
            nn.ReLU(),
        )
        self.p2_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single),
            nn.ReLU(),
        )
        self.p3_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single),
            nn.ReLU(),
        )
        self.p4_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single),
            nn.ReLU(),
        )

        self.p1_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.channels_single),
            nn.ReLU(),
        )

        self.p2_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(self.channels_single),
            nn.ReLU(),
        )

        self.p3_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(self.channels_single),
            nn.ReLU(),
        )

        self.p4_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=7, dilation=7),
            nn.BatchNorm2d(self.channels_single),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(self.input_channels*4, self.input_channels, 1, 1, 0),
            nn.BatchNorm2d(self.input_channels),
            nn.ReLU()
        )

    def forward(self, x):
        p1 = self.p1_channel_reduction(x)
        p1_dc = self.p1_dc(p1)

        p2 = self.p2_channel_reduction(x)
        p2_dc = self.p2_dc(p2)

        p3 = self.p3_channel_reduction(x)
        p3_dc = self.p3_dc(p3)

        p4 = self.p4_channel_reduction(x)
        p4_dc = self.p3_dc(p4)

        ce = self.fusion(torch.cat((p1_dc, p2_dc, p3_dc, p4_dc), 1))

        return ce


class Mask_Correcting(nn.Module):
    def __init__(self, channel1, channel2):
        super().__init__()
        self.channel1 = channel1
        self.channel2 = channel2

        self.output_map = nn.Sequential(
            nn.Conv2d(self.channel1, 1, 5, 1, 2),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.ce_text = Context_Exploration_Block(self.channel1)
        self.ce_bg = Context_Exploration_Block(self.channel1)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

        self.bn1 = nn.BatchNorm2d(self.channel1)
        self.relu1 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(self.channel1)
        self.relu2 = nn.ReLU()

    def forward(self, x, mask):

        bg_feature = x * mask
        text_feature = x * (1 - mask)

        fn = self.ce_bg(bg_feature)
        fp = self.ce_text(text_feature)

        enhance = x - (self.alpha * fp)
        enhance = self.bn1(enhance)
        enhance = self.relu1(enhance)

        enhance = enhance + (self.beta * fn)
        enhance = self.bn2(enhance)
        enhance = self.relu2(enhance)

        return self.output_map(enhance)



class Text_Region_Position(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = Decoder_Block(512,256)
        self.conv2 = Decoder_Block(384,128)
        self.mask_get = nn.Sequential(
            nn.Conv2d(128,32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
    def forward(self, x4, x5,x6):
        x5 = self.conv1(x6,x5)
        x4 = self.conv2(x5,x4)
        return self.mask_get(x4)

class PSSTRModule(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder
        self.enc1 = Encoder_Block(3, 64,kernel_size=7,padding=3, down=False)
        self.enc2 = Encoder_Block(64, 128,kernel_size=5,padding=2)
        self.enc3 = Encoder_Block(128, 128)
        self.enc4 = Encoder_Block(128, 256)
        self.enc5 = Encoder_Block(256, 256)

        # text segmentation branch
        self.get_mask = Text_Region_Position()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.mask_correcting = Mask_Correcting(3, 3)

        # text removal branch
        self.dec1 = Decoder_Block(512, 256)
        self.dec2 = Decoder_Block(384, 128)
        self.dec3 = Decoder_Block(256, 64)
        self.dec4 = Decoder_Block(128, 64)
        self.out = nn.Conv2d(64,3,kernel_size=1)


    def forward(self, x_ori,f_in,mask_prev):

        x1 = self.enc1(f_in)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        # get mask (h/4,w/4)
        mask_now = self.get_mask(x3,x4,x5)
        # upsample (h,w)
        mask_now = self.upsample(self.upsample(mask_now))
        # mask merge
        mask_now = torch.where(mask_now < mask_prev, mask_now, mask_prev)
        # correct mask
        mask_now = self.mask_correcting(x_ori,mask_now)

        # text removal
        f = self.dec1(x5,x4)
        f = self.dec2(f,x3)
        f = self.dec3(f,x2)
        f = self.dec4(f,x1)
        f = self.out(f)

        # region-based modification strategy
        f = x_ori * mask_now + f * (1 - mask_now)
        return f, mask_now

class PSSTRNet(nn.Module):
    def __init__(self):
        super().__init__()
        # encode
        self.PSSTR = PSSTRModule()

    def forward(self, x):

        b, c, h, w = x.size()
        str_out_1, mask_out_1 = self.PSSTR(x, x, torch.ones((b, 1, h, w)).cuda())

        str_out_2, mask_out_2 = self.PSSTR(x, str_out_1, mask_out_1)

        str_out_3, mask_out_3 = self.PSSTR(x, str_out_2, mask_out_2)

        str_out_final = (str_out_1*(1-mask_out_1)+str_out_2*(1-mask_out_2)+str_out_3*(1-mask_out_3)+1e-8) / \
                        ((1-mask_out_1)+(1-mask_out_2)+(1-mask_out_3)+1e-8)
        mask_final = (mask_out_1 + mask_out_2 + mask_out_3)/3
        str_out_final = (1-mask_final)*str_out_final+mask_final*x

        return str_out_1, str_out_2, str_out_3, str_out_final, mask_out_1, mask_out_2, mask_out_3, mask_final

