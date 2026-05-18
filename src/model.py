#src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, init_features=16):
        super(UNet3D, self).__init__()
        features = init_features

        self.encoder1 = DoubleConv3D(in_channels, features)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder2 = DoubleConv3D(features, features * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder3 = DoubleConv3D(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConv3D(features * 4, features * 8)

        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv3D(features * 8, features * 4)

        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv3D(features * 4, features * 2)

        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv3D(features * 2, features)

        self.final_conv = nn.Conv3d(features, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))

        bottleneck = self.bottleneck(self.pool3(enc3))

        dec3 = self.upconv3(bottleneck)
        if dec3.shape[2:] != enc3.shape[2:]:
            dec3 = F.interpolate(dec3, size=enc3.shape[2:], mode='trilinear', align_corners=True)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        if dec2.shape[2:] != enc2.shape[2:]:
            dec2 = F.interpolate(dec2, size=enc2.shape[2:], mode='trilinear', align_corners=True)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        if dec1.shape[2:] != enc1.shape[2:]:
            dec1 = F.interpolate(dec1, size=enc1.shape[2:], mode='trilinear', align_corners=True)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.sigmoid(self.final_conv(dec1))

class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm3d(out_channels),   # <-- FIX: was BatchNorm3d
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm3d(out_channels),   # <-- FIX
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class AttentionGate3D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate3D, self).__init__()
        # Gating signal (from deeper layer)
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(F_int)
        )

        # Lower-level features (from skip connection)
        self.W_l = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, l):
        g1 = self.W_g(g)
        l1 = self.W_l(l)

        # Spatial alignment for 3D tensors (D, H, W)
        if g1.size() != l1.size():
            g1 = F.interpolate(g1, size=l1.size()[2:], mode='trilinear', align_corners=True)

        q = self.relu(g1 + l1)
        attention_weights = self.psi(q)
        return l * attention_weights

class AttentionUNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, init_features=16):
        """
        in_channels = 4 (FLAIR, T1, T1ce, T2)
        out_channels = 3 (WT, TC, ET)
        init_features = 16 (Reduced to save VRAM in 3D)
        """
        super(AttentionUNet3D, self).__init__()
        f = init_features

        # Encoder Path
        self.encoder1 = DoubleConv3D(in_channels, f)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder2 = DoubleConv3D(f, f * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder3 = DoubleConv3D(f * 2, f * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConv3D(f * 4, f * 8)

        # Decoder Path with Attention Gates
        self.up3 = nn.ConvTranspose3d(f * 8, f * 4, kernel_size=2, stride=2)
        self.att3 = AttentionGate3D(F_g=f * 4, F_l=f * 4, F_int=f * 2)
        self.decoder3 = DoubleConv3D(f * 8, f * 4)

        self.up2 = nn.ConvTranspose3d(f * 4, f * 2, kernel_size=2, stride=2)
        self.att2 = AttentionGate3D(F_g=f * 2, F_l=f * 2, F_int=f)
        self.decoder2 = DoubleConv3D(f * 4, f * 2)

        self.up1 = nn.ConvTranspose3d(f * 2, f, kernel_size=2, stride=2)
        self.att1 = AttentionGate3D(F_g=f, F_l=f, F_int=f // 2)
        self.decoder1 = DoubleConv3D(f * 2, f)

        self.final_conv = nn.Conv3d(f, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))

        # Decoder 3
        d3 = self.up3(b)
        if d3.size() != e3.size():
            d3 = F.interpolate(d3, size=e3.size()[2:], mode='trilinear', align_corners=True)
        s3 = self.att3(g=d3, l=e3)
        d3 = torch.cat((s3, d3), dim=1)
        d3 = self.decoder3(d3)

        # Decoder 2
        d2 = self.up2(d3)
        if d2.size() != e2.size():
            d2 = F.interpolate(d2, size=e2.size()[2:], mode='trilinear', align_corners=True)
        s2 = self.att2(g=d2, l=e2)
        d2 = torch.cat((s2, d2), dim=1)
        d2 = self.decoder2(d2)

        # Decoder 1
        d1 = self.up1(d2)
        if d1.size() != e1.size():
            d1 = F.interpolate(d1, size=e1.size()[2:], mode='trilinear', align_corners=True)
        s1 = self.att1(g=d1, l=e1)
        d1 = torch.cat((s1, d1), dim=1)
        d1 = self.decoder1(d1)

        return self.sigmoid(self.final_conv(d1))