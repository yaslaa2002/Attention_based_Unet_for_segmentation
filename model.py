# Importing libraries 
import torch
import torch.nn.functional as F
import torch.nn as nn

class UNet2d(nn.Module):
    def __init__(self, in_channels=1, out_channels=5):
        super(UNet2d, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Contracting path
        self.enc1 = self.contract_block(in_channels, 64)
        self.enc2 = self.contract_block(64, 128)
        self.enc3 = self.contract_block(128, 256)
        self.enc4 = self.contract_block(256, 512)

        # Bottleneck
        self.bottleneck = self.contract_block(512, 1024)

        # Expanding path
        self.dec4 = self.expand_block(1024 + 512, 512)
        self.dec3 = self.expand_block(512 + 256, 256)
        self.dec2 = self.expand_block(256 + 128, 128)
        self.dec1 = self.expand_block(128 + 64, 64)

        # Output layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def contract_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def expand_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def up_concat(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.size()[2:], mode='bilinear', align_corners=True)
        return torch.cat([x1, x2], dim=1)

    def forward(self, x):
        # Encode
        enc1 = self.enc1(x)
        enc1_pool = self.maxpool(enc1)

        enc2 = self.enc2(enc1_pool)
        enc2_pool = self.maxpool(enc2)

        enc3 = self.enc3(enc2_pool)
        enc3_pool = self.maxpool(enc3)

        enc4 = self.enc4(enc3_pool)
        enc4_pool = self.maxpool(enc4)

        # Bottleneck
        bottleneck = self.bottleneck(enc4_pool)

        # Decode
        dec4 = self.up_concat(bottleneck, enc4)
        dec4 = self.dec4(dec4)

        dec3 = self.up_concat(dec4, enc3)
        dec3 = self.dec3(dec3)

        dec2 = self.up_concat(dec3, enc2)
        dec2 = self.dec2(dec2)

        dec1 = self.up_concat(dec2, enc1)
        dec1 = self.dec1(dec1)

        dec1 = F.interpolate(dec1, size=(256, 256), mode='bilinear', align_corners=True)

        return self.final_conv(dec1)

class AttentionBlock2D(nn.Module):
    def __init__(self, F_g, F_l, n_coefficients):
        super(AttentionBlock2D, self).__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection, return_attention=False):
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        attentionMap=None
        if return_attention:
            attentionMap = psi  # Return the attention map
        
        out = skip_connection * psi
        return out,attentionMap


class UNetAug2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=5, n_coefficients=3):
        super(UNetAug2D, self).__init__()

        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Contracting path
        self.enc1 = self.contract_block(in_channels, 64)
        self.enc2 = self.contract_block(64, 128)
        self.enc3 = self.contract_block(128, 256)
        self.enc4 = self.contract_block(256, 512)
        self.enc5 = self.contract_block(512, 1024)

        # Expanding path
        self.dec5 = self.expand_block(1024, 512)
        self.att5 = AttentionBlock2D(F_g=512, F_l=512, n_coefficients=256)
        self.conv5 = self.contract_block(1024, 512)

        self.dec4 = self.expand_block(512, 256)
        self.att4 = AttentionBlock2D(F_g=256, F_l=256, n_coefficients=128)
        self.conv4 = self.contract_block(512, 256)

        self.dec3 = self.expand_block(256, 128)
        self.att3 = AttentionBlock2D(F_g=128, F_l=128, n_coefficients=64)
        self.conv3 = self.contract_block(256, 128)

        self.dec2 = self.expand_block(128, 64)
        self.att2 = AttentionBlock2D(F_g=64, F_l=64, n_coefficients=32)
        self.conv2 = self.contract_block(128, 64)

        # Output layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, padding=0)

    def contract_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def expand_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    
    def forward(self, x, return_attention=False):
        attention_maps = []  # List to store attention maps
        
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.MaxPool(enc1))
        enc3 = self.enc3(self.MaxPool(enc2))
        enc4 = self.enc4(self.MaxPool(enc3))
        enc5 = self.enc5(self.MaxPool(enc4))

        dec5 = self.dec5(enc5)
        att4,bis = self.att5(gate=dec5, skip_connection=enc4, return_attention=return_attention)
        if return_attention: 
            attention_maps.append(bis)
        dec5 = torch.cat((att4, dec5), dim=1)
        dec5 = self.conv5(dec5)

        dec4 = self.dec4(dec5)
        att3,bis2 = self.att4(gate=dec4, skip_connection=enc3, return_attention=return_attention)
        if return_attention: 
            attention_maps.append(bis2)
        dec4 = torch.cat((att3, dec4), dim=1)
        dec4 = self.conv4(dec4)

        dec3 = self.dec3(dec4)
        att2,bis3 = self.att3(gate=dec3, skip_connection=enc2, return_attention=return_attention)
        if return_attention: 
            attention_maps.append(bis3)
        dec3 = torch.cat((att2, dec3), dim=1)
        dec3 = self.conv3(dec3)

        dec2 = self.dec2(dec3)
        att1,bis4 = self.att2(gate=dec2, skip_connection=enc1, return_attention=return_attention)
        if return_attention: 
            attention_maps.append(bis4)
        dec2 = torch.cat((att1, dec2), dim=1)
        dec2 = self.conv2(dec2)

        if return_attention:
            self.final_conv(dec2), attention_maps  # Return the output + attention maps
        
        return self.final_conv(dec2)


