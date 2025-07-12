"""
Lighter U-net implementation that achieves same performance as the one reported in the paper: https://arxiv.org/abs/1505.04597
Main differences:
    a) U-net downblock has only 1 convolution instead of 2
    b) U-net upblock has only 1 convolution instead of 3
"""

import torch
import torch.nn as nn
import argparse
import torch.nn.functional as F

from core.base_temporal_model_ode import BaseTemporalModel

class UNetDownBlock(nn.Module):
    """
    Constructs a UNet downsampling block

       Parameters:
            input_nc (int)      -- the number of input channels
            output_nc (int)     -- the number of output channels
            norm_layer (str)    -- normalization layer
            down_type (str)     -- if we should use strided convolution or maxpool for reducing the feature map
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            user_dropout (bool) -- if use dropout layers.
            kernel_size (int)   -- convolution kernel size
            bias (boolean)      -- if convolution should use bias
    """
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, down_type='strideconv', outermost=False, innermost=False, dropout=0.2, kernel_size=4, bias=True):
        super(UNetDownBlock, self).__init__()
        self.innermost = innermost
        self.outermost = outermost
        self.use_maxpool = down_type == 'maxpool'

        stride = 1 if self.use_maxpool else 2
        kernel_size = 3 if self.use_maxpool else 4
        self.conv = nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, stride=stride, padding=1, bias=bias)
        self.relu = nn.LeakyReLU(0.2, True)
        self.maxpool = nn.MaxPool2d(2)
        self.norm = norm_layer(output_nc)
        self.dropout = nn.Dropout2d(dropout) if dropout else None

    def forward(self, x):
        if self.outermost:
            x = self.conv(x)
            x = self.norm(x)
        elif self.innermost:
            x = self.relu(x)
            if self.dropout: x = self.dropout(x)
            x = self.conv(x)
        else:
            x = self.relu(x)
            if self.dropout: x = self.dropout(x)
            x = self.conv(x)
            x = self.norm(x)

        return x

class UNetUpBlock(nn.Module):
    """
      Constructs a UNet upsampling block

         Parameters:
              input_nc (int)      -- the number of input channels
              output_nc (int)     -- the number of output channels
              norm_layer          -- normalization layer
              outermost (bool)    -- if this module is the outermost module
              innermost (bool)    -- if this module is the innermost module
              user_dropout (bool) -- if use dropout layers.
              kernel_size (int)   -- convolution kernel size
              remove_skip (bool)  -- if skip connections should be disabled or not
      """
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, outermost=False, innermost=False, dropout=0.2, kernel_size=4, remove_skip=0, use_bias=True):
        super(UNetUpBlock, self).__init__()
        self.innermost = innermost
        self.outermost = outermost
        self.remove_skip = remove_skip
        upconv_inner_nc = input_nc if self.remove_skip else input_nc * 2

        if self.innermost:
            self.conv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=kernel_size, stride=2, padding=1, bias=use_bias)
        elif self.outermost:
            self.conv = nn.ConvTranspose2d(upconv_inner_nc, output_nc, kernel_size=kernel_size, stride=2, padding=1)
        else:
            self.conv = nn.ConvTranspose2d(upconv_inner_nc, output_nc, kernel_size=kernel_size, stride=2, padding=1, bias=use_bias)

        self.norm = norm_layer(output_nc)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout) if dropout else None

    def forward(self, x):
        if self.outermost:
            x = self.relu(x)
            if self.dropout: x = self.dropout(x)
            x = self.conv(x)
        elif self.innermost:
            x = self.relu(x)
            if self.dropout: x = self.dropout(x)
            x = self.conv(x)
            x = self.norm(x)
        else:
            x = self.relu(x)
            if self.dropout: x = self.dropout(x)
            x = self.conv(x)
            x = self.norm(x)

        return x

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.max_pool=nn.AdaptiveMaxPool2d(1)
        self.fc=nn.Sequential(nn.Conv2d(in_planes, in_planes//ratio, 1, bias=True),
                              nn.ReLU(),
                              nn.Conv2d(in_planes//ratio, in_planes, 1, bias=True))
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        avg_out=self.fc(self.avg_pool(x))
        max_out=self.fc(self.max_pool(x))
        out=avg_out+max_out
        return self.sigmoid(out)
class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1=nn.Conv2d(2,1,kernel_size, padding=kernel_size//2, bias=True)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        avg_out=torch.mean(x,dim=1,keepdim=True)
        max_out,_ = torch.max(x, dim=1, keepdim=True)
        x=torch.cat([avg_out,max_out], dim=1)
        x=self.conv1(x)
        return self.sigmoid(x)
class DualAttention(nn.Module):
    def __init__(self,in_planes, kernel_size=7):
        super(DualAttention, self).__init__()
        self.ca=ChannelAttention(in_planes)
        self.sa=SpatialAttention(kernel_size)
    def forward(self,x):
        x=self.ca(x)*x
        x=self.sa(x)*x
        return x


class TemporalAttention(nn.Module):
    """A Temporal Attention block for Temporal Attention Unit"""

    def __init__(self, d_model, kernel_size=21, attn_shortcut=True):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)  # 1x1 conv
        self.activation = nn.GELU()  # GELU
        self.spatial_gating_unit = TemporalAttentionModule(d_model, kernel_size)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)  # 1x1 conv
        self.attn_shortcut = attn_shortcut

    def forward(self, x):
        if self.attn_shortcut:
            shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        if self.attn_shortcut:
            x = x + shortcut
        return x


class TemporalAttentionModule(nn.Module):
    """Large Kernel Attention for SimVP"""

    def __init__(self, dim, kernel_size, dilation=3, reduction=16):
        super().__init__()
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        dd_p = (dilation * (dd_k - 1) // 2)

        self.conv0 = nn.Conv2d(dim, dim, d_k, padding=d_p, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, dd_k, stride=1, padding=dd_p, groups=dim, dilation=dilation)
        self.conv1 = nn.Conv2d(dim, dim, 1)

        self.reduction = max(dim // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // self.reduction, bias=False),  # reduction
            nn.ReLU(True),
            nn.Linear(dim // self.reduction, dim, bias=False),  # expansion
            nn.Sigmoid()
        )

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)  # depth-wise conv
        attn = self.conv_spatial(attn)  # depth-wise dilation convolution
        f_x = self.conv1(attn)  # 1x1 conv
        # append a se operation
        b, c, _, _ = x.size()
        se_atten = self.avg_pool(x).view(b, c)
        se_atten = self.fc(se_atten).view(b, c, 1, 1)
        return se_atten * f_x * u


class TAUattention(nn.Module):
    """A TAUBlock (tau) for Temporal Attention Unit"""

    def __init__(self, dim, kernel_size=21, mlp_ratio=4.,
                 drop=0., drop_path=0.1, init_value=1e-2, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = TemporalAttention(dim, kernel_size)
    def forward(self, x):
        # x=self.norm1(x)
        x = self.attn(x)
        return x



class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
class Upwithoutskip(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Upwithoutskip, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.conv(x)
        return x
class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )

class UNet_ODE(BaseTemporalModel):
    """Create a Unet-based Fully Convolutional Network
          X -------------------identity----------------------
          |-- downsampling -- |submodule| -- upsampling --|

        Parameters:
            num_classes (int)      -- the number of channels in output images
            norm_layer             -- normalization layer
            input_nc               -- number of channels of input image

            Args:
            mode (str)             -- process single frames or sequence of frames
            timesteps (int)        --
            num_downs (int)        -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                      image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)              -- the number of filters in the last conv layer
            remove_skip (int [0,1])-- if skip connections should be disabled or not
            reconstruct (int [0,1])-- if we should reconstruct the next image or not
            sequence_model (str)   -- the sequence model that for the sequence mode []
            num_levels_tcn(int)    -- number of levels of the TemporalConvNet
      """

    def __init__(self, num_classes, args, norm_layer=nn.BatchNorm2d, input_nc=3):
        super(UNet_ODE, self).__init__(args)
        self.args=args
        self.base_c = args.base_c
        self.bilinear = True
        self.temporal_layer_list = args.temporal_layer_list
        self.hastemporal = {}
        for i in range(5):
            self.hastemporal[i] = True if i in self.temporal_layer_list else False


        self.remove_skip = args.remove_skip
        self.segmentation = args.segmentation
        self.reconstruct = args.reconstruct
        self.reconstruct_remove_skip = args.reconstruct_remove_skip
        self.largepic=args.largepic

        self.with_skip=args.with_skip
        self.with_attention = args.with_attention
        self.reduce_downsample = args.reduce_downsample


        self.encoder, self.enc_attention = self.build_encoder(input_nc, self.base_c, self.bilinear,self.largepic)
        if self.args.with_seq_attention:
            channel=512 if not self.reduce_downsample else 256
            self.seq_attention=TAUattention(self.args.timesteps*channel)

        self.encoder_sequence_models = self.get_skip_sequence_models(args, self.temporal_layer_list) if '+temporal_encoder' in args.sequence_model else None
        self.sequence_model = self.get_sequence_model(args) if args.hasTempBottle == "hasTempBottle" else None

        if args.segmentation:
            self.decoder, self.dec_attention = self.build_decoder(num_classes, self.base_c, self.bilinear,self.largepic)

        if self.reconstruct:
            self.reconstruction_decoder, self.rec_attention = self.build_recon_decoder(input_nc, self.base_c, self.bilinear,self.largepic)

    def build_encoder(self, in_channels=3, base_c=32, bilinear: bool = True,largepic=0):
        """Constructs a UNet downsampling encoder, consisting of $num_downs UNetDownBlocks

             Parameters:
                  num_downs (int)     -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                         image of size 128x128 will become of size 1x1 # at the bottleneck
                  input_nc (int)      -- the number of input channels
                  ngf (int)           -- the number of filters in the last conv layer
                  norm_layer (str)    -- normalization layer
                  down_type (str)     -- if we should use strided convolution or maxpool for reducing the feature map
             Returns:
                  nn.Sequential consisting of $num_downs UnetDownBlocks
        """
        factor = 2 if bilinear else 1
        layers = []
        attentions = []
        layers.append(DoubleConv(in_channels, base_c))
        layers.append(Down(base_c, base_c * 2))
        layers.append(Down(base_c * 2, base_c * 4))
        if self.with_attention:
            attentions.append(DualAttention(base_c))
            attentions.append(DualAttention(base_c*2))
            attentions.append(DualAttention(base_c*4))

        if not self.reduce_downsample:
            layers.append(Down(base_c * 4, base_c * 8))
            if self.with_attention:
                attentions.append(DualAttention(base_c*8))
            layers.append(Down(base_c * 8, base_c * 16))
        else:
            layers.append(Down(base_c * 4, base_c * 8))
        return nn.Sequential(*layers), nn.Sequential(*attentions)

    def build_decoder(self, num_classes, base_c, bilinear: bool = True,largepic=0):
        """Constructs a UNet downsampling encoder, consisting of $num_downs UNetUpBlocks

           Parameters:
                num_downs (int)     -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                       image of size 128x128 will become of size 1x1 # at the bottleneck
                num_classes (int)   -- number of classes to classify
                output_nc (int)     -- the number of output channels. outermost is ngf, innermost is ngf * 8
                norm_layer          -- normalization layer
                remove_skip (int)   -- if skip connections should be disabled or not

           Returns:
                nn.Sequential consisting of $num_downs UnetUpBlocks
        """
        factor = 2 if bilinear else 1
        layers = []
        attentions=[]
        if not self.with_skip:
            if not self.reduce_downsample:
                layers.append(Upwithoutskip(base_c * 16, base_c * 8, bilinear))
            layers.append(Upwithoutskip(base_c * 8, base_c * 4, bilinear))
            layers.append(Upwithoutskip(base_c * 4, base_c * 2, bilinear))
            layers.append(Upwithoutskip(base_c * 2, base_c, bilinear))
            layers.append(OutConv(base_c, num_classes))
        else:
            if not self.reduce_downsample:
                layers.append(Up(base_c * (16+8), base_c * 8, bilinear))
            layers.append(Up(base_c * (8+4), base_c * 4, bilinear))
            layers.append(Up(base_c * (4+2), base_c * 2, bilinear))
            layers.append(Up(base_c * (2+1), base_c, bilinear))
            layers.append(OutConv(base_c, num_classes))
        if self.with_attention:
            if not self.reduce_downsample:
                attentions.append(DualAttention(base_c*16))
            attentions.append(DualAttention(base_c * 8))
            attentions.append(DualAttention(base_c * 4))
            attentions.append(DualAttention(base_c * 2))
        return nn.Sequential(*layers), nn.Sequential(*attentions)

    def build_recon_decoder(self, num_classes, base_c, bilinear: bool = True,largepic=0):
        """Constructs a UNet downsampling encoder, consisting of $num_downs UNetUpBlocks

           Parameters:
                num_downs (int)     -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                       image of size 128x128 will become of size 1x1 # at the bottleneck
                num_classes (int)   -- number of classes to classify
                output_nc (int)     -- the number of output channels. outermost is ngf, innermost is ngf * 8
                norm_layer          -- normalization layer
                remove_skip (int)   -- if skip connections should be disabled or not

           Returns:
                nn.Sequential consisting of $num_downs UnetUpBlocks
        """
        factor = 2 if bilinear else 1
        layers = []
        attentions=[]
        if (not self.with_skip) or (self.args.onlysegskip):
            if not self.reduce_downsample:
                layers.append(Upwithoutskip(base_c * 16, base_c * 8, bilinear))
            layers.append(Upwithoutskip(base_c * 8, base_c * 4, bilinear))
            layers.append(Upwithoutskip(base_c * 4, base_c * 2, bilinear))
            layers.append(Upwithoutskip(base_c * 2, base_c, bilinear))
            layers.append(OutConv(base_c, num_classes))
        else:
            if not self.reduce_downsample:
                layers.append(Up(base_c * (16 + 8), base_c * 8, bilinear))
            layers.append(Up(base_c * (8 + 4), base_c * 4, bilinear))
            layers.append(Up(base_c * (4 + 2), base_c * 2, bilinear))
            layers.append(Up(base_c * (2 + 1), base_c, bilinear))
            layers.append(OutConv(base_c, num_classes))
        if self.with_attention:
            if not self.reduce_downsample:
                attentions.append(DualAttention(base_c*16))
            attentions.append(DualAttention(base_c * 8))
            attentions.append(DualAttention(base_c * 4))
            attentions.append(DualAttention(base_c * 2))
        return nn.Sequential(*layers), nn.Sequential(*attentions)


    def encoder_forward(self, x):
        self.enc_skip=[]
        self.num_down=3 if self.reduce_downsample else 4
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            if self.with_skip and i<self.num_down:
                self.enc_skip.append(x)
            if self.with_attention and i<self.num_down:
                x=self.enc_attention[i](x)
        return x

    def decoder_forward(self, x):
        for i in range(len(self.decoder)):
            if i<self.num_down:
                if self.with_attention:
                    x=self.dec_attention[i](x)
                if self.with_skip:
                    if self.args.skip_from=="enc":
                        x=self.decoder[i](x,self.enc_skip[self.num_down-1-i])
                    if self.args.skip_from=="rec":
                        x=self.decoder[i](x,self.rec_skip[i])
                else:
                    x=self.decoder[i](x)
            else:
                x = self.decoder[i](x)

        return x
    def reconstruction_decoder_forward(self, x):
        self.rec_skip=[]
        for i in range(len(self.reconstruction_decoder)):
            if i < self.num_down:
                if self.with_attention:
                    x = self.rec_attention[i](x)
                if self.with_skip and (not self.args.onlysegskip):
                    x = self.reconstruction_decoder[i](x, self.enc_skip[self.num_down - 1 - i])
                else:
                    x = self.reconstruction_decoder[i](x)
                self.rec_skip.append(x)
            else:
                x = self.reconstruction_decoder[i](x)

        return x

    # def reconstruction_decoder_forward(self, x, skip_connections):
    #     for i, up in enumerate(self.reconstruction_decoder):
    #         if not up.innermost:
    #             if not self.reconstruct_remove_skip:
    #                 skip = skip_connections[-i]
    #                 out = torch.cat([skip, out], 1)
    #             out = up(out)
    #         else:
    #             out = up(x)
    #
    #     return out

    def forward(self, input, pred_dur = False):
        x = self.remove_time_reshape(input)
        x = self.encoder_forward(x)
        if self.args.with_seq_attention:
            batch_size, C, H, W=int(x.size(0) / self.timesteps), int(x.size(1)), int(x.size(2)), int(x.size(3))
            # x = x.view(batch_size, self.timesteps,C, H, W)
            x = x.view(batch_size, self.timesteps*C, H, W)
            x=self.seq_attention(x)
            x = x.view(batch_size*self.timesteps, C, H, W)
        if self.sequence_model:
            x = self.temporal_forward(x, self.sequence_model, pred_dur = pred_dur)

        reconstruction_output = self.reconstruction_decoder_forward(x) if self.reconstruct else None
        segmentation_output = self.decoder_forward(x) if self.segmentation else None

        if 'sequence' in self.mode:
            segmentation_output = self.add_time_reshape(segmentation_output, pred_dur=pred_dur) if self.segmentation else None
            reconstruction_output = self.add_time_reshape(reconstruction_output,pred_dur = pred_dur) if self.reconstruct else None

        return segmentation_output, reconstruction_output

if __name__ == "__main__":
    b,t,c,h,w=8,4,32,112,112
    model=TAUattention(t*c)
    x=torch.randn((b,t*c,h,w))
    y=model(x)
    if 1:
        parser = argparse.ArgumentParser(description='PyTorch Semantic Video Segmentation training')
        parser.add_argument('--base_c', type=int, default=32)
        parser.add_argument('--temporal_layer_list', type=list, default=[0, 1, 2, 3, 4])

        parser.add_argument('--model', type=str, default='unet',
                            choices=['deeplab', 'deeplab-50', 'unet', 'unet_paper', 'unet_pytorch', 'pspnet'],
                            help='model name (default: deeplab)')
        parser.add_argument('--mode', type=str, default='sequence-1234',
                            choices=['fbf', 'fbf-1234', 'fbf-previous', 'sequence-1234'],
                            help='training type (default: frame by frame)')

        # unet specific
        parser.add_argument('--resize', type=str, default='512,256', help='image resize: h,w')
        parser.add_argument('--num_downs', type=int, default=7, help='number of unet encoder-decoder blocks')
        parser.add_argument('--ngf', type=int, default=128, help='# of gen filters in the last conv layer')
        parser.add_argument('--remove_skip', type=int, default=0,
                            help='if skip connections should be removed from the model')
        parser.add_argument('--sequence_model', type=str, default='convlstm_ode+temporal_encoder',
                            choices=['none', 'lstm', 'gru', 'convlstm', 'tcn', 'tcn2d', 'tcn2dhw',
                                     'tcn+temporal_skip', 'tcn+temporal_encoder', 'tcn+temporal_all',
                                     'tcn2d+temporal_skip', 'tcn2d+temporal_encoder', 'tcn2d+temporal_all',
                                     'tcn2dhw+temporal_skip', 'tcn2dhw+temporal_encoder', 'tcn2dhw+temporal_all',
                                     'convlstm+temporal_skip', 'convlstm+temporal_encoder', 'convlstm+temporal_all',
                                     'convlstm_ode', 'convlstm_ode+temporal_encoder'])
        # segmentation specific
        parser.add_argument('--segmentation', type=int, default=1, choices=[0, 1], help='segment image based on given mode')

        # reconstruction specific
        parser.add_argument('--reconstruct', type=int, default=0, choices=[0, 1], help='reconstruct future image')
        parser.add_argument('--reconstruct_remove_skip', type=int, default=0, choices=[0, 1],
                            help='if we should remvoe skip connections in the reconstruction head')
        parser.add_argument('--timesteps', type=int, default=4)
        parser.add_argument('--sequence_stacked_models', type=int, default=1, help='number of stacked sequence models')
        parser.add_argument('--down_type', type=str, default='maxpool', choices=['strideconv, maxpool'],
                            help='method to reduce feature map size')

        args = parser.parse_args()
    norm_layer = nn.BatchNorm2d
    net = UNet_ODE(num_classes=7, args=args, norm_layer=norm_layer)
    net = net.cuda()
    seq = torch.randn(1, 4, 3, 112, 112)#B, T, C, H, W
    seq = seq.cuda()
    import time
    st = time.time()
    out = net(seq)
    print(time.time()-st)
    print(out[0].size(), out[1])
    print("finishs")