import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet25D(nn.Module):
    def __init__(self, in_channels=12, out_channels=3, init_features=16):
        super(UNet25D, self).__init__()
        features = init_features
        
        # Encoder Path (Swapped to 2D)
        self.encoder1 = DoubleConv2D(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = DoubleConv2D(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder3 = DoubleConv2D(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = DoubleConv2D(features * 4, features * 8)
        
        # Decoder Path (Swapped to 2D)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv2D(features * 8, features * 4) # 8 because of concatenation
        
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv2D(features * 4, features * 2)
        
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv2D(features * 2, features)
        
        # Final Layer - 2D output
        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool3(enc3))
        
        # Decoder with Skip Connections
        dec3 = self.upconv3(bottleneck)
        # Ensure spatial dimensions match before concatenation (handles odd input sizes)
        if dec3.size() != enc3.size():
            dec3 = F.interpolate(dec3, size=enc3.size()[2:], mode='bilinear', align_corners=True)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        if dec2.size() != enc2.size():
            dec2 = F.interpolate(dec2, size=enc2.size()[2:], mode='bilinear', align_corners=True)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        if dec1.size() != enc1.size():
            dec1 = F.interpolate(dec1, size=enc1.size()[2:], mode='bilinear', align_corners=True)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        out = self.final_conv(dec1)
        return self.sigmoid(out)

class DoubleConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): 
        return self.conv(x)

class AttentionGate2D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate2D, self).__init__()
        # Gating signal (from deeper layer)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Lower-level features (from skip connection)
        self.W_l = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, l):
        g1 = self.W_g(g)
        l1 = self.W_l(l)
        
        # Handle potential dimension mismatch during upsampling
        if g1.size() != l1.size():
            diffY = l1.size()[2] - g1.size()[2]
            diffX = l1.size()[3] - g1.size()[3]
            g1 = F.pad(g1, [diffX // 2, diffX - diffX // 2,
                           diffY // 2, diffY - diffY // 2])

        q = self.relu(g1 + l1)
        attention_weights = self.psi(q)
        return l * attention_weights

class AttentionUNet25D(nn.Module):
    def __init__(self, in_channels=12, out_channels=3, init_features=32):
        """
        in_channels = 12 (3 slices x 4 MRI modalities)
        out_channels = 3 (WT, TC, ET)
        """
        super(AttentionUNet25D, self).__init__()
        features = init_features

        # Encoder Path
        self.encoder1 = DoubleConv2D(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.encoder2 = DoubleConv2D(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.encoder3 = DoubleConv2D(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.bottleneck = DoubleConv2D(features * 4, features * 8)

        # Decoder Path with Attention Gates
        self.up3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.att3 = AttentionGate2D(F_g=features * 4, F_l=features * 4, F_int=features * 2)
        self.decoder3 = DoubleConv2D(features * 8, features * 4)

        self.up2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.att2 = AttentionGate2D(F_g=features * 2, F_l=features * 2, F_int=features)
        self.decoder2 = DoubleConv2D(features * 4, features * 2)

        self.up1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.att1 = AttentionGate2D(F_g=features, F_l=features, F_int=features // 2)
        self.decoder1 = DoubleConv2D(features * 2, features)

        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)
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
        # Match dimensions if necessary
        if d3.size() != e3.size():
            d3 = F.interpolate(d3, size=e3.size()[2:], mode='bilinear', align_corners=True)
        s3 = self.att3(g=d3, l=e3)
        d3 = torch.cat((s3, d3), dim=1)
        d3 = self.decoder3(d3)

        # Decoder 2
        d2 = self.up2(d3)
        if d2.size() != e2.size():
            d2 = F.interpolate(d2, size=e2.size()[2:], mode='bilinear', align_corners=True)
        s2 = self.att2(g=d2, l=e2)
        d2 = torch.cat((s2, d2), dim=1)
        d2 = self.decoder2(d2)

        # Decoder 1
        d1 = self.up1(d2)
        if d1.size() != e1.size():
            d1 = F.interpolate(d1, size=e1.size()[2:], mode='bilinear', align_corners=True)
        s1 = self.att1(g=d1, l=e1)
        d1 = torch.cat((s1, d1), dim=1)
        d1 = self.decoder1(d1)

        return self.sigmoid(self.final_conv(d1))