import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(Convolution3d => InstanceNorm3d => LeakyReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, init_features=16):
        super(UNet3D, self).__init__()
        features = init_features
        
        # Encoder Path
        self.encoder1 = DoubleConv(in_channels, features)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = DoubleConv(features, features * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = DoubleConv(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(features * 4, features * 8)
        
        # Decoder Path
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(features * 8, features * 4) # 8 because of concatenation
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(features * 4, features * 2)
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(features * 2, features)
        
        # Final Layer - 3 channels for WT, TC, ET
        self.final_conv = nn.Conv3d(features, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool3(enc3))
        
        # Decoder with Skip Connections
        # Using torch.cat to combine encoder details with upsampled features
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        out = self.final_conv(dec1)
        return self.sigmoid(out)
    
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
        q = self.relu(g1 + l1)
        attention_weights = self.psi(q)
        return l * attention_weights

class AttentionUNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, init_features=16):
        super(AttentionUNet3D, self).__init__()
        features = init_features

        # Encoder Path (Same as standard U-Net)
        self.encoder1 = DoubleConv(in_channels, features)
        self.pool1 = nn.MaxPool3d(2)
        self.encoder2 = DoubleConv(features, features * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.encoder3 = DoubleConv(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(2)

        self.bottleneck = DoubleConv(features * 4, features * 8)

        # Decoder Path with Attention Gates
        self.up3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.att3 = AttentionGate3D(F_g=features * 4, F_l=features * 4, F_int=features * 2)
        self.decoder3 = DoubleConv(features * 8, features * 4)

        self.up2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.att2 = AttentionGate3D(F_g=features * 2, F_l=features * 2, F_int=features)
        self.decoder2 = DoubleConv(features * 4, features * 2)

        self.up1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.att1 = AttentionGate3D(F_g=features, F_l=features, F_int=features // 2)
        self.decoder1 = DoubleConv(features * 2, features)

        self.final_conv = nn.Conv3d(features, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))
        
        b = self.bottleneck(self.pool3(e3))

        # Decoder 3 with Attention
        d3 = self.up3(b)
        s3 = self.att3(g=d3, l=e3) # g is gating, l is skip feature
        d3 = torch.cat((s3, d3), dim=1)
        d3 = self.decoder3(d3)

        # Decoder 2 with Attention
        d2 = self.up2(d3)
        s2 = self.att2(g=d2, l=e2)
        d2 = torch.cat((s2, d2), dim=1)
        d2 = self.decoder2(d2)

        # Decoder 1 with Attention
        d1 = self.up1(d2)
        s1 = self.att1(g=d1, l=e1)
        d1 = torch.cat((s1, d1), dim=1)
        d1 = self.decoder1(d1)

        return self.sigmoid(self.final_conv(d1))