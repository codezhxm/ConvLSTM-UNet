from models.unet_parts import Down, DoubleConv, Up, OutConv
from models.unet_parts_depthwise_separable import DoubleConvDS, UpDS, DownDS
from models.layers import CBAM,SELayer
from models.regression_lightning import Precip_regression_base
from torch import nn

class UNet(Precip_regression_base):
    def __init__(self, hparams):
        super(UNet, self).__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        self.bilinear = self.hparams.bilinear

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNetDS_Attention(Precip_regression_base):
    def __init__(self, hparams):
        super(UNetDS_Attention, self).__init__(hparams=hparams)
        self.n_channels = self.hparams.n_channels
        self.n_classes = self.hparams.n_classes
        self.bilinear = self.hparams.bilinear
        reduction_ratio = self.hparams.reduction_ratio
        kernels_per_layer = self.hparams.kernels_per_layer

        self.inc = DoubleConvDS(self.n_channels, 64, kernels_per_layer=kernels_per_layer)
        self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        self.cbam4 = CBAM(512, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam5 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4)
        x5Att = self.cbam5(x5)
        x = self.up1(x5Att, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits

class ConvLSTM-UNet(Precip_regression_base):  
    def __init__(self, hparams):
        super(ConvLSTM-UNet, self).__init__(hparams=hparams)
        self.n_channels = hparams.n_channels
        self.n_classes = hparams.n_classes
        self.inc1 = DoubleConvDS(self.n_channels, 48, 48)
        self.lstm1 = RNN([4, 4], 4, 288)
        self.down1 = DoubleConvDS(48, 96, 96)
        self.lstm2 = RNN([8, 8], 8, 144)
        self.down2 = DoubleConvDS(96, 192, 192)
        self.lstm3 = RNN([16, 16], 16, 72)
        self.down3 = DoubleConvDS(192, 768, 768)
        self.up1 = LSTMUpDS(576, 384, 384)
        self.up2 = LSTMUpDS(288, 192, 192)
        self.up3 = LSTMUpDS(144, 96, 48)
        self.outc = OutConv(48, self.n_classes)
    def forward(self, x):
        b, c, h, w = x.shape
        x1 = self.inc1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # print(x1.shape,x2.shape,x3.shape)
        tx1 = x1.reshape(b, 12, 4, w, h)
        tx2 = x2.reshape(b, 12, 8, int(w/2), int(h/2))
        tx3 = x3.reshape(b, 12, 16, int(w/4), int(h/4))
        lstm1 = self.lstm1(tx1)
        lstm2 = self.lstm2(tx2)
        lstm3 = self.lstm3(tx3)
        lstm1 = lstm1.reshape(b, 48, w, h)
        lstm2 = lstm2.reshape(b, 96, int(w/2), int(h/2))
        lstm3 = lstm3.reshape(b, 192, int(w/4), int(h/4))
        x = self.up1(x4, x3, lstm3)
        x = self.up2(x, x2, lstm2)
        x = self.up3(x, x1, lstm1)
        logits = self.outc(x)
        return logits
