# src/model.py
import torch
import torch.nn as nn

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNet3D, self).__init__()
        features = init_features
        self.encoder1 = self._block(in_channels, features)
        self.pool1    = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = self._block(features, features*2)
        self.pool2    = nn.MaxPool3d(kernel_size=2, stride=2)
        self.bottleneck = self._block(features*2, features*4)

        self.up2      = nn.ConvTranspose3d(features*4, features*2, kernel_size=2, stride=2, output_padding=1)
        self.decoder2 = self._block(features*2*2, features*2)
        self.up1      = nn.ConvTranspose3d(features*2, features, kernel_size=2, stride=2, output_padding=1)
        self.decoder1 = self._block(features*2, features)
        self.conv_last = nn.Conv3d(features, out_channels, kernel_size=1)

    def _block(self, in_ch, features):
        return nn.Sequential(
            nn.Conv3d(in_ch, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True),
            nn.Conv3d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))

        dec2 = self.up2(bottleneck)
        if dec2.shape != enc2.shape:
            diffZ = enc2.size(2) - dec2.size(2)
            diffY = enc2.size(3) - dec2.size(3)
            diffX = enc2.size(4) - dec2.size(4)
            dec2 = nn.functional.pad(dec2, [diffX//2, diffX-diffX//2,
                                            diffY//2, diffY-diffY//2,
                                            diffZ//2, diffZ-diffZ//2])
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.up1(dec2)
        if dec1.shape != enc1.shape:
            diffZ = enc1.size(2) - dec1.size(2)
            diffY = enc1.size(3) - dec1.size(3)
            diffX = enc1.size(4) - dec1.size(4)
            dec1 = nn.functional.pad(dec1, [diffX//2, diffX-diffX//2,
                                            diffY//2, diffY-diffY//2,
                                            diffZ//2, diffZ-diffZ//2])
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.conv_last(dec1)

