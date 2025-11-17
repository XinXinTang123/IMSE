## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.deform_conv import DeformConv2d
from pdb import set_trace as stx
import numbers
import math
from einops import rearrange
import numpy as np
import torchvision

from torch.autograd import Function
import triton
import triton.language as tl
from torch.amp import custom_fwd, custom_bwd
import math

from torch.utils.checkpoint import checkpoint
from torch import Tensor
from typing import Tuple

freqs_dict = dict()

##########################################################################
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class RoPE(nn.Module):

    def __init__(self, embed_dim, num_heads):
        '''
        recurrent_chunk_size: (clh clw)
        num_chunks: (nch ncw)
        clh * clw == cl
        nch * ncw == nc

        default: clh==clw, clh != clw is not implemented
        '''
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 4))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.register_buffer('angle', angle)

    
    def forward(self, slen: Tuple[int]):
        '''
        slen: (h, w)
        h * w == l
        recurrent is not implemented
        '''
        # index = torch.arange(slen[0]*slen[1]).to(self.angle)
        index_h = torch.arange(slen[0]).to(self.angle)
        index_w = torch.arange(slen[1]).to(self.angle)
        # sin = torch.sin(index[:, None] * self.angle[None, :]) #(l d1)
        # sin = sin.reshape(slen[0], slen[1], -1).transpose(0, 1) #(w h d1)
        sin_h = torch.sin(index_h[:, None] * self.angle[None, :]) #(h d1//2)
        sin_w = torch.sin(index_w[:, None] * self.angle[None, :]) #(w d1//2)
        sin_h = sin_h.unsqueeze(1).repeat(1, slen[1], 1) #(h w d1//2)
        sin_w = sin_w.unsqueeze(0).repeat(slen[0], 1, 1) #(h w d1//2)
        sin = torch.cat([sin_h, sin_w], -1) #(h w d1)
        # cos = torch.cos(index[:, None] * self.angle[None, :]) #(l d1)
        # cos = cos.reshape(slen[0], slen[1], -1).transpose(0, 1) #(w h d1)
        cos_h = torch.cos(index_h[:, None] * self.angle[None, :]) #(h d1//2)
        cos_w = torch.cos(index_w[:, None] * self.angle[None, :]) #(w d1//2)
        cos_h = cos_h.unsqueeze(1).repeat(1, slen[1], 1) #(h w d1//2)
        cos_w = cos_w.unsqueeze(0).repeat(slen[0], 1, 1) #(h w d1//2)
        cos = torch.cat([cos_h, cos_w], -1) #(h w d1)

        retention_rel_pos = (sin.flatten(0, 1), cos.flatten(0, 1))

        return retention_rel_pos
    
def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)

def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)

class MagnitudeAwareLinearAttention(nn.Module):

    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkvo = nn.Conv2d(dim, dim * 4, 1)
        self.lepe = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.scale = self.head_dim ** -0.5
        self.elu = nn.ELU()

    def forward(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor):
        '''
        x: (b c h w)
        sin: ((h w) d1)
        cos: ((h w) d1)
        '''
        B, C, H, W = x.shape
        qkvo = self.qkvo(x) #(b 3*c h w)
        qkv = qkvo[:, :3*self.dim, :, :]
        o = qkvo[:, 3*self.dim:, :, :]
        lepe = self.lepe(qkv[:, 2*self.dim:, :, :]) # (b c h w)

        q, k, v = rearrange(qkv, 'b (m n d) h w -> m b n (h w) d', m=3, n=self.num_heads) # (b n (h w) d)

        q = self.elu(q) + 1
        k = self.elu(k) + 1

        z = q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) * self.scale

        q = theta_shift(q, sin, cos)
        k = theta_shift(k, sin, cos)

        kv = (k.transpose(-2, -1) * (self.scale / (H*W)) ** 0.5) @ (v * (self.scale / (H*W)) ** 0.5)

        res = q @ kv * (1 + 1/(z + 1e-6)) - z * v.mean(dim=2, keepdim=True)

        res = rearrange(res, 'b n (h w) d -> b (n d) h w', h=H, w=W)
        res = res + lepe
        return self.proj(res * o)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type,shared_refine_att=None,qk_norm=1):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.rope = RoPE(embed_dim=dim, num_heads=dim//8)
        self.mala = MagnitudeAwareLinearAttention(dim=dim, num_heads=dim//8, bias=bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):

        normalized_x = self.norm1(x)
        h, w = x.shape[2:]
        sin, cos = self.rope((h, w))
        x_mala = self.mala(normalized_x, sin, cos)
        x = x + x_mala 
        x = x + self.ffn(self.norm2(x))

        return x


class MFEncoder(nn.Module):
    """Multi-Head Convolutional self-Attention Encoder comprised of `MHCA`
    blocks."""

    def __init__(
            self,
            dim,
            num_layers=1,
            num_heads=8,
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type='BiasFree',
            qk_norm=1
    ):
        super().__init__()

        self.num_layers = num_layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                dim,
                num_heads=num_heads,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                qk_norm=qk_norm
            ) for idx in range(self.num_layers)
        ])

    def forward(self, x, size):
        """foward function"""
        H, W = size
        B = x.shape[0]

        # return x's shape : [B, N, C] -> [B, C, H, W]
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        for layer in self.transformer_layers:
            x = layer(x)

        return x


class ResBlock(nn.Module):
    """Residual block for convolutional local feature."""

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.Hardswish,
            norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = Conv2d_BN(in_features,
                               hidden_features,
                               act_layer=act_layer)
        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features,
            3,
            1,
            1,
            bias=False,
            groups=hidden_features,
        )
        self.act = act_layer()
        self.conv2 = Conv2d_BN(hidden_features, out_features)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """
        initialization
        """
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
                
    def forward(self, x):
        """foward function"""
        identity = x
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.act(feat)
        feat = self.conv2(feat)

        return identity + feat


class MF_stage(nn.Module):
    """Multi-Head Convolutional self-Attention stage comprised of `MHCAEncoder`
    layers."""

    def __init__(
            self,
            embed_dim,
            out_embed_dim,
            num_layers=1,
            num_heads=8,
            ffn_expansion_factor=2.66,
            num_path=4,
            bias=False,
            LayerNorm_type='BiasFree',
            qk_norm=1

    ):
        super().__init__()

        self.mhca_blks = nn.ModuleList([
            MFEncoder(
                embed_dim,
                num_layers,
                num_heads,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                qk_norm=qk_norm

            ) for _ in range(num_path)
        ])

        self.aggregate = SKFF(embed_dim, height=num_path)

    def forward(self, inputs):
        """foward function"""
        #att_outputs = [self.InvRes(inputs[0])]
        att_outputs = []

        for x, encoder in zip(inputs, self.mhca_blks):
            # [B, C, H, W] -> [B, N, C]
            _, _, H, W = x.shape
            x = x.flatten(2).transpose(1, 2).contiguous()
            att_outputs.append(encoder(x, size=(H, W)))

        #out_concat = torch.cat(att_outputs, dim=1)
        out = self.aggregate(att_outputs)

        return out


class Conv2d_BN(nn.Module):

    def __init__(
            self,
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            pad=0,
            dilation=1,
            groups=1,
            bn_weight_init=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=None,
    ):
        super().__init__()

        self.conv = torch.nn.Conv2d(in_ch,
                                    out_ch,
                                    kernel_size,
                                    stride,
                                    pad,
                                    dilation,
                                    groups,
                                    bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))

        self.act_layer = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x):
    
        x = self.conv(x)
        x = self.act_layer(x)

        return x


class SKFF(nn.Module):
    def __init__(self, in_channels, height=2, reduction=8, bias=False):
        super(SKFF, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.PReLU())

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V

class InceptionDWConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        gc = int(in_channels * branch_ratio)
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)
        self.pwconv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        x = self.pwconv(x)
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1, )


class MB_Deform_Embedding(nn.Module):

    def __init__(self,
                 in_chans=3,
                 embed_dim=768,
                 idx=0,
                 ):
        super().__init__()
        self.patch_conv = InceptionDWConv2d(in_chans, embed_dim) 

    def forward(self, x):
        """foward function"""
        x = self.patch_conv(x)

        return x


class Patch_Embed_stage(nn.Module):
    """Depthwise Convolutional Patch Embedding stage comprised of
    `DWCPatchEmbed` layers."""

    def __init__(self, in_chans, embed_dim, num_path=4, isPool=False,offset_clamp=(-1,1)):
        super(Patch_Embed_stage, self).__init__()

        self.patch_embeds = nn.ModuleList([
            MB_Deform_Embedding(
                in_chans=in_chans if idx == 0 else embed_dim,
                embed_dim=embed_dim,
                idx=idx,
            ) for idx in range(num_path)
        ])

    def forward(self, x):
        """foward function"""
        att_inputs = []
        for pe in self.patch_embeds:
            x = pe(x)
            att_inputs.append(x)

        return att_inputs


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, input_feat,out_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
            # dw
            nn.Conv2d(input_feat, input_feat, kernel_size=3, stride=1, padding=1, groups=input_feat, bias=False, ),
            # pw-linear
            nn.Conv2d(input_feat, out_feat // 4, 1, 1, 0, bias=False),
            nn.PixelUnshuffle(2))

    def forward(self, x):
        # if x.size(-2) % 2 != 0:
        #     x = x[:, :, :-1, :]  # 裁剪高度为奇数的输入图像
        # if x.size(-1) % 2 != 0:
        #     x = x[:, :, :, :-1]  # 裁剪高度为奇数的输入图像
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, input_feat, out_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            # dw
            nn.Conv2d(input_feat, input_feat, kernel_size=3, stride=1, padding=1, groups=input_feat, bias=False, ),
            # pw-linear
            nn.Conv2d(input_feat, out_feat * 4, 1, 1, 0, bias=False),
            nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class Multi_transformer(nn.Module):
    def __init__(self,
                 dense_channel=64,
                 inp_channels=2,
                 out_channels=2,
                 num_blocks=[4, 4, 4, 4],  #2334
                 dec_num_blocks=[4, 4, 4, 4],  # 2334
                 num_refinement_blocks=2,
                 heads=[1, 2, 4, 8],  #1248
                 ffn_expansion_factor=2.66,  #2.66
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 num_path=[1, 1, 1, 1],  #2222
                 dec_num_path=[1, 1, 1, 1],  # 2222
                 qk_norm=1,
                 offset_clamp=(-1, 1)
                 ):

        super(Multi_transformer, self).__init__()
        self.dim = [dense_channel, dense_channel*2, dense_channel*3, dense_channel*4]
        dim = self.dim
        # self.patch_embed = OverlapPatchEmbed(inp_channels, dim[0])
        self.patch_embed_encoder_level1 = Patch_Embed_stage(dim[0], dim[0], num_path=num_path[0], isPool=False,offset_clamp=offset_clamp)
        self.encoder_level1 = MF_stage(dim[0], dim[0], num_layers=num_blocks[0], num_heads=heads[0],
                                       ffn_expansion_factor=ffn_expansion_factor, num_path=num_path[0],
                                       bias=False, LayerNorm_type='BiasFree', qk_norm=qk_norm)
        
        self.down1_2 = Downsample(dim[0],dim[1])  ## From Level 1 to Level 2

        self.patch_embed_encoder_level2 = Patch_Embed_stage(dim[1], dim[1], num_path=num_path[1], isPool=False,offset_clamp=offset_clamp)
        self.encoder_level2 = MF_stage(dim[1], dim[1], num_layers=num_blocks[1], num_heads=heads[1],
                                       ffn_expansion_factor=ffn_expansion_factor,
                                       num_path=num_path[1], bias=False, LayerNorm_type='BiasFree', qk_norm=qk_norm)

        self.down2_3 = Downsample(dim[1],dim[2])  ## From Level 2 to Level 3

        # self.patch_embed_encoder_level3 = Patch_Embed_stage(dim[2], dim[2], num_path=num_path[2],
        #                                                     isPool=False,offset_clamp=offset_clamp)
        # self.encoder_level3 = MF_stage(dim[2], dim[2], num_layers=num_blocks[2], num_heads=heads[2],
        #                                ffn_expansion_factor=ffn_expansion_factor,
        #                                num_path=num_path[2], bias=False, LayerNorm_type='BiasFree', qk_norm=qk_norm)
        #
        # self.down3_4 = Downsample(dim[2],dim[3])  ## From Level 3 to Level 4

        self.patch_embed_latent = Patch_Embed_stage(dim[2], dim[2], num_path=num_path[3],
                                                    isPool=False,offset_clamp=offset_clamp)
        self.latent = MF_stage(dim[2], dim[2], num_layers=num_blocks[3], num_heads=heads[3],
                               ffn_expansion_factor=ffn_expansion_factor, num_path=num_path[3], bias=False,
                               LayerNorm_type='BiasFree', qk_norm=qk_norm)


        # self.up4_3 = Upsample(int(dim[3]),dim[2])  ## From Level 4 to Level 3
        # self.reduce_chan_level3 = nn.Sequential(
        #     nn.Conv2d(dim[2]*2, dim[2], 1, 1, 0, bias=bias),
        # )
        #
        # self.patch_embed_decoder_level3 = Patch_Embed_stage(dim[2], dim[2], num_path=num_path[2],
        #                                                     isPool=False,offset_clamp=offset_clamp)
        # self.decoder_level3 = MF_stage(dim[2], dim[2], num_layers=dec_num_blocks[2], num_heads=heads[2],
        #                                ffn_expansion_factor=ffn_expansion_factor, num_path=num_path[2], bias=False,
        #                                LayerNorm_type='BiasFree', qk_norm=qk_norm)

        self.up3_2 = Upsample(int(dim[2]),dim[1])  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Sequential(
            nn.Conv2d(dim[1]*2, dim[1], 1, 1, 0, bias=bias),
        )

        self.patch_embed_decoder_level2 = Patch_Embed_stage(dim[1], dim[1], num_path=dec_num_path[1],
                                                            isPool=False,offset_clamp=offset_clamp)
        self.decoder_level2 = MF_stage(dim[1], dim[1], num_layers=dec_num_blocks[1], num_heads=heads[1],
                                       ffn_expansion_factor=ffn_expansion_factor, num_path=dec_num_path[1], bias=False,
                                       LayerNorm_type='BiasFree', qk_norm=qk_norm)

        self.up2_1 = Upsample(int(dim[1]), dim[0])  ## From Level 2 to Level 1  (gave 1x1 conv to reduce channels)
        self.reduce_chan_level1 = nn.Sequential(
            nn.Conv2d(dim[0] * 2, dim[0], 1, 1, 0, bias=bias),
        )

        self.patch_embed_decoder_level1 = Patch_Embed_stage(dim[0], dim[0], num_path=dec_num_path[0],
                                                            isPool=False,offset_clamp=offset_clamp)
        self.decoder_level1 = MF_stage(dim[0], dim[0], num_layers=dec_num_blocks[0], num_heads=heads[0],
                                       ffn_expansion_factor=ffn_expansion_factor, num_path=dec_num_path[0], bias=False,
                                       LayerNorm_type='BiasFree', qk_norm=qk_norm)

        # 幅度
        self.mag_patch_embed_refinement = Patch_Embed_stage(dim[0], dim[0], num_path=dec_num_path[0],
                                                        isPool=False,offset_clamp=offset_clamp)
        self.mag_refinement = MF_stage(dim[0], dim[0], num_layers=dec_num_blocks[0], num_heads=heads[0],
                                   ffn_expansion_factor=ffn_expansion_factor, num_path=dec_num_path[0], bias=False,
                                   LayerNorm_type='BiasFree', qk_norm=qk_norm)

        self.mag_output = nn.Sequential(
            nn.Conv2d(dim[0], dim[0], kernel_size=3, stride=1, padding=1, bias=False, ),

        )

        # 相位
        self.pha_patch_embed_refinement = Patch_Embed_stage(dim[0], dim[0], num_path=dec_num_path[0],
                                                        isPool=False, offset_clamp=offset_clamp)
        self.pha_refinement = MF_stage(dim[0], dim[0], num_layers=dec_num_blocks[0], num_heads=heads[0],
                                   ffn_expansion_factor=ffn_expansion_factor, num_path=dec_num_path[0], bias=False,
                                   LayerNorm_type='BiasFree', qk_norm=qk_norm)

        self.pha_output = nn.Sequential(
            nn.Conv2d(dim[0], dim[0], kernel_size=3, stride=1, padding=1, bias=False, ),

        )

    def forward(self, inp_img):
        inp_enc_level1 = inp_img

        inp_enc_level1_list = self.patch_embed_encoder_level1(inp_enc_level1)

        out_enc_level1 = self.encoder_level1(inp_enc_level1_list) + inp_enc_level1

        inp_enc_level2 = self.down1_2(out_enc_level1)
        
        inp_enc_level2_list = self.patch_embed_encoder_level2(inp_enc_level2)
        out_enc_level2 = self.encoder_level2(inp_enc_level2_list) + inp_enc_level2
        inp_enc_level3 = self.down2_3(out_enc_level2)

        # inp_enc_level3_list = self.patch_embed_encoder_level3(inp_enc_level3)
        # out_enc_level3 = self.encoder_level3(inp_enc_level3_list) + inp_enc_level3
        # inp_enc_level4 = self.down3_4(out_enc_level3)

        inp_latent = self.patch_embed_latent(inp_enc_level3)
        latent = self.latent(inp_latent) + inp_enc_level3

        # inp_dec_level3 = self.up4_3(latent)
        # inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        # inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        # inp_dec_level3_list = self.patch_embed_decoder_level3(inp_dec_level3)
        # out_dec_level3 = self.decoder_level3(inp_dec_level3_list) + inp_dec_level3

        inp_dec_level2 = self.up3_2(latent)

        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)

        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        inp_dec_level2_list = self.patch_embed_decoder_level2(inp_dec_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2_list) + inp_dec_level2

        inp_dec_level1 = self.up2_1(out_dec_level2)

        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)

        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)

        inp_dec_level1_list = self.patch_embed_decoder_level1(inp_dec_level1)

        out_dec_level1 = self.decoder_level1(inp_dec_level1_list) + inp_dec_level1
        # 幅度
        mag_inp_latent_list = self.mag_patch_embed_refinement(out_dec_level1)

        mag_out_dec_level1 = self.mag_refinement(mag_inp_latent_list) + out_dec_level1

        mag_out_dec_level1 = self.mag_output(mag_out_dec_level1) + inp_enc_level1
        # 相位
        pha_inp_latent_list = self.pha_patch_embed_refinement(out_dec_level1)

        pha_out_dec_level1 = self.pha_refinement(pha_inp_latent_list) + out_dec_level1

        pha_out_dec_level1 = self.pha_output(pha_out_dec_level1) + inp_enc_level1

        return mag_out_dec_level1, pha_out_dec_level1


def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

