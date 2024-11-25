import time

import torch
from einops import rearrange
# from ptflops import get_model_complexity_info
from thop import profile, clever_format
from torch import nn
import torch.nn.functional as F

from utils import create_F


class SpatialMSA(nn.Module):
    def __init__(self, dim_vector, head_dim, heads_number):

        super(SpatialMSA, self).__init__()

        self.num_heads_column = head_dim
        self.heads_number = heads_number
        self.to_q = nn.Linear(dim_vector, head_dim * heads_number, bias=False)
        self.to_k = nn.Linear(dim_vector, head_dim * heads_number, bias=False)
        self.to_v = nn.Linear(dim_vector, head_dim * heads_number, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads_number, 1, 1))  # 权重参数*CORE
        self.proj = nn.Sequential(
            nn.Linear(head_dim * heads_number, dim_vector),  # 现在要映射（b c wh）   num_heads_column * heads_number=wh
            nn.GELU()
        )



    def forward(self, x_msi,x_fu):

        b,c,w,h= x_msi.shape
        x_msi = x_msi.reshape(b, c, w*h)
        x_fu = x_fu.reshape(b, c, w*h)

        q_inp = self.to_q(x_msi)
        k_inp = self.to_k(x_fu)
        v_inp = self.to_v(x_fu)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads_number),
                  (q_inp, k_inp, v_inp))
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, c, self.heads_number * self.num_heads_column)

        out = self.proj(x).reshape(b,c, w, h)


        return out




class SpectralMSA(nn.Module):
    def __init__(self, dim_vector, num_heads_column, heads_number):

        super(SpectralMSA, self).__init__()
        self.num_heads_column = num_heads_column
        self.heads_number = heads_number
        self.to_q = nn.Linear(dim_vector, num_heads_column * heads_number, bias=False)
        self.to_k = nn.Linear(dim_vector, num_heads_column * heads_number, bias=False)
        self.to_v = nn.Linear(dim_vector, num_heads_column * heads_number, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads_number, 1, 1))  # 权重参数*CORE
        self.proj = nn.Sequential(
            nn.Linear(num_heads_column * heads_number, dim_vector),  # 映射的x形状是（）
            nn.GELU()
        )

        self.pos_emb = nn.Sequential(
            nn.Conv2d(num_heads_column * heads_number, dim_vector,
                      kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.GELU(),
            nn.Conv2d(dim_vector, dim_vector,
                      kernel_size=(3, 3), stride=(1, 1), padding=1)
        )

    def forward(self, x_fu, x_hsi):

        b, h, w, c = x_fu.shape

        x = x_fu.reshape(b, h * w, c)
        x2 = x_hsi.reshape(b, h * w, c)

        q_inp = self.to_q(x2)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads_number),
                      (q_inp, k_inp, v_inp))      # (b wh c)-->(b heads wh d)    heads*d=c
        v = v

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, h * w, self.heads_number * self.num_heads_column)
        out_c = self.proj(x).contiguous().view(b, h, w, c)



        x_temp=x_fu.permute(0, 3, 2, 1)
        out_p = self.pos_emb(x_temp).permute(0, 3,2, 1)
        out=out_c+out_p
        return out



class Spatial_Learn(nn.Module):
    def __init__(self,c_in, w_cin, h_in, L, head_dim, heads_number):
        super(Spatial_Learn, self).__init__()
        self.to_s_hsi = nn.Parameter(torch.ones(c_in, L,  w_cin))
        self.to_a_hsi = nn.Parameter(torch.ones(c_in,  h_in, L))

        self.to_s_msi = nn.Parameter(torch.ones(c_in, L,  w_cin))
        self.to_a_msi = nn.Parameter(torch.ones(c_in, h_in, L))
        self.pos_emb = nn.Sequential(
            nn.Conv2d(c_in, c_in,
                      kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.GELU(),
            nn.Conv2d(c_in, c_in,
                      kernel_size=(3, 3), stride=(1, 1), padding=1)
        )
        self.msa_s=SpatialMSA(h_in * L, head_dim, heads_number)
        self.msa_a=SpatialMSA(h_in * L, head_dim, heads_number)

    def forward(self,x_msi,x_hsi):
        s_hsi = self.to_s_hsi @ x_hsi
        a_hsi = x_hsi @ self.to_a_hsi
        s_msi = self.to_s_msi @ x_msi
        a_msi = x_msi @ self.to_a_msi

        s = self.msa_s(s_msi,s_hsi)

        a = self.msa_a(a_msi,a_hsi)
        pos=self.pos_emb(x_hsi)
        out=a@s+pos

        return out


class FeedForward(nn.Module):
    def __init__(self,in_channels):
        super(FeedForward, self).__init__()
        self.ffn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (1, 1), (1, 1), padding=0, bias=False),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels, in_channels, (3, 3), (1, 1), padding=1, bias=False,groups=in_channels),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels, in_channels, (1, 1), (1, 1), padding=0, bias=False),
        )

    def forward(self,x):
        out=self.ffn(x)
        return out


class Guided_Spatial_SR(nn.Module):
    def __init__(self, c_in, w_in, h_in, L, head_dim, heads_number):
        super(Guided_Spatial_SR, self).__init__()
        self.sr=Spatial_Learn(c_in, w_in, h_in, L, head_dim, heads_number)
        self.msi_norm = nn.LayerNorm(w_in, h_in)
        self.hsi_norm = nn.LayerNorm(w_in, h_in)

        self.ffn=FeedForward(c_in)
        self.sr_hsi_norm= nn.LayerNorm(w_in, h_in)

    def forward(self,x_msi,x_hsi):
        x_msi_norm=self.msi_norm(x_msi)
        x_hsi_norm=self.hsi_norm(x_hsi)
        sr_hsi=self.sr(x_msi_norm,x_hsi_norm)
        sr_hsi=sr_hsi+x_hsi
        sr_hsi_norm=self.sr_hsi_norm(sr_hsi)
        out=self.ffn(sr_hsi_norm)+sr_hsi
        return out

class Guided_Spectral_SR(nn.Module):
    def __init__(self, dim_vector, num_heads_column, heads_number):
        super(Guided_Spectral_SR, self).__init__()
        self.sr = SpectralMSA(dim_vector, num_heads_column, heads_number)
        self.msi_norm = nn.LayerNorm(dim_vector)
        self.hsi_norm = nn.LayerNorm(dim_vector)

        self.ffn = FeedForward(dim_vector)
        self.sr_hsi_norm = nn.LayerNorm(dim_vector)

    def forward(self,x_msi,x_hsi):

        x_msi=x_msi.permute(0, 3, 2, 1)  # (b,c,w,h)-->(b,h,w,c)
        x_hsi = x_hsi.permute(0, 3, 2, 1)

        x_msi_norm=self.msi_norm(x_msi)
        x_hsi_norm=self.hsi_norm(x_hsi)
        sr_msi=self.sr(x_msi_norm,x_hsi_norm)
        sr_msi=sr_msi+x_msi
        sr_msi_norm=self.sr_hsi_norm(sr_msi).permute(0, 3, 2, 1)
        out=self.ffn(sr_msi_norm)+sr_msi.permute(0, 3, 2, 1)
        out=out
        return out




class SpatialAttention(nn.Module):
    def __init__(self, c_in, kernel_size1=3,kernel_size2=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(c_in, c_in, kernel_size1, padding=1, bias=False),
            nn.Conv2d(c_in, 1, 1, padding=0, bias=False),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c_in, c_in, kernel_size2, padding=3, bias=False),
            nn.Conv2d(c_in, 1, 1, padding=0, bias=False),
        )

        self.conv3=nn.Conv2d(2, 1, 1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):  # x.size() 30,40,50,30
        mask1=self.conv1(x)
        mask2=self.conv2(x)
        x = torch.cat([mask1, mask2], dim=1)
        x = self.conv3(x)  # 30,1,50,30
        return self.sigmoid(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, out_c,ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes, out_c, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        return self.sigmoid(avg_out)



class Self_mask_Spectral_MSA(nn.Module):
    def __init__(self, dim_vector, num_heads_column, heads_number,ratio):
        """
        :param dim_vector: 向量的维度
        :param num_heads_column: Wk矩阵的列——num_vector//heads_number
        :param heads_number: 多头的个数

        为了使用高光谱图像引导多光谱进行光谱超分，拟K和V来自rgb， Q来自高光谱。 交互注意力机制。
        """
        super(Self_mask_Spectral_MSA, self).__init__()
        self.num_heads_column = num_heads_column
        self.heads_number = heads_number
        self.to_q = nn.Linear(dim_vector, num_heads_column * heads_number, bias=False)
        self.to_k = nn.Linear(dim_vector, num_heads_column * heads_number, bias=False)
        self.to_v = nn.Linear(dim_vector, num_heads_column * heads_number, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads_number, 1, 1))  # 权重参数*CORE
        self.proj = nn.Sequential(
            nn.Linear(num_heads_column * heads_number, dim_vector),  # 映射的x形状是（）
            nn.GELU()
        )

        self.pos_emb = nn.Sequential(
            nn.Conv2d(num_heads_column * heads_number, dim_vector,
                      kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.GELU(),
            nn.Conv2d(dim_vector, dim_vector,
                      kernel_size=(3, 3), stride=(1, 1), padding=1)
        )


        self.avg_mask = ChannelAttention(dim_vector,dim_vector, ratio=ratio)
        self.spatial_mask=SpatialAttention(dim_vector)
        self.attention_conv=nn.Conv2d(dim_vector , dim_vector, (3, 3), (1, 1), padding=1, bias=False)

    def forward(self, x_fu, x_msi,x_hsi):

        b, h, w, c = x_fu.shape
        channel_mask = self.avg_mask(x_hsi.permute(0, 3, 2, 1)).contiguous().view(b, c, 1, 1)
        spatial_mask=self.spatial_mask(x_msi.permute(0, 3, 2, 1))

        x = x_fu.reshape(b, h * w, c)

        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        v_inp=v_inp.reshape(b, h, w, c).permute(0, 3, 2, 1)



        v_spec=v_inp.mul(spatial_mask)
        v_spec=v_spec.mul(channel_mask)
        v_spec=self.attention_conv(v_spec)
        v_spec=v_spec+x_hsi.permute(0, 3, 2, 1)
        v_spec=v_spec.permute(0, 3, 2, 1).reshape(b, h * w, c)

        # print(v_inp.shape)
        q, k, v1 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads_number),
                      (q_inp, k_inp, v_spec))
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v1 = v1.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        x1 = attn @ v1
        x1 = x1.permute(0, 3, 1, 2)
        x1 = x1.reshape(b, h * w, self.heads_number * self.num_heads_column)
        out1 = self.proj(x1).contiguous().view(b, h, w, c).permute(0, 3, 2, 1)

        out_p_sec=self.pos_emb(v_spec.reshape(b, h, w, c).permute(0, 3, 2, 1))   # (b,c,h,w)
        out=out1+out_p_sec
        return out,channel_mask,spatial_mask



class Self_mask_T(nn.Module):
    def __init__(self,dim_vector, num_heads_column, heads_number,ratio):
        super(Self_mask_T, self).__init__()
        self.msa=Self_mask_Spectral_MSA(dim_vector, num_heads_column, heads_number,ratio)
        self.ffn=FeedForward(dim_vector)
        self.fu_norm1 = nn.LayerNorm(dim_vector)
        self.fu_norm2 = nn.LayerNorm(dim_vector)

    def forward(self,x_fu,x_msi,x_hsi):
        x_fu=x_fu.permute(0, 3, 2, 1)
        x_msi=x_msi.permute(0, 3, 2, 1)
        x_hsi = x_hsi.permute(0, 3, 2, 1)
        x_fu_norm=self.fu_norm1(x_fu)

        sr_hsi,ca,sa=self.msa(x_fu_norm,x_msi,x_hsi)
        sr_hsi=sr_hsi.permute(0, 3, 2, 1)+x_fu
        sr_hsi_norm=self.fu_norm2(sr_hsi).permute(0, 3, 2, 1)
        out=self.ffn(sr_hsi_norm)+sr_hsi.permute(0, 3, 2, 1)
        return out,ca,sa


class Cross_Guide_Fusion(nn.Module):
    def __init__(self,in_c,in_w,in_h, L):
        super(Cross_Guide_Fusion, self).__init__()
        self.min_w=in_w//2
        self.min_c=in_c

        # stage1
        self.spa_sr1=Guided_Spatial_SR(in_c,in_w,in_h,L,self.min_w*L,in_w//self.min_w)
        self.hsi_up1=nn.Upsample(scale_factor=8, mode='bilinear')
        self.spec_sr1=Guided_Spectral_SR(in_c,self.min_c,in_c//self.min_c)

        self.hsi_down_sample1=nn.Conv2d(in_c,in_c*2,(4,4),(2,2), 1,bias=False)
        self.msi_down_sample1=nn.Conv2d(in_c,in_c*2,(4,4),(2,2), 1,bias=False)

        self.msi_channels_up1=nn.Sequential(
            nn.Conv2d(3, in_c, (3, 3), (1, 1), 1, bias=False),
        )
        self.hsi_channels_up1= nn.Sequential(
            nn.Conv2d(in_c, in_c, (3, 3), (1, 1), 1, bias=False),
        )


        in_c, in_w, in_h=in_c*2, in_w//2, in_h//2    # 62*32*32

        self.spa_sr2 = Guided_Spatial_SR( in_c, in_w, in_h, L, self.min_w * L, in_w // self.min_w)
        self.hsi_up2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.spec_sr2=Guided_Spectral_SR(in_c,self.min_c,in_c//self.min_c)

        self.hsi_down_sample2=nn.Conv2d(in_c,in_c*2,(4,4),(2,2), 1,bias=False)
        self.msi_down_sample2=nn.Conv2d(in_c,in_c*2,(4,4),(2,2), 1,bias=False)

        self.msi_channels_up2=nn.Sequential(
            nn.Conv2d(3, in_c, (3, 3), (1, 1), 1, bias=False),
        )
        self.hsi_channels_up2=nn.Sequential(
            nn.Conv2d(in_c//2, in_c, (3, 3), (1, 1), 1, bias=False),
        )

        in_c, in_w, in_h = in_c * 2, in_w // 2, in_h // 2  #248*8*8

        # cat
        self.hsi_up3 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.hsi_channels_up3= nn.Sequential(
            nn.Conv2d(in_c//2//2, in_c, (3, 3), (1, 1), 1, bias=False),
        )
        self.msi_channels_up3=nn.Sequential(
            nn.Conv2d(3, in_c, (3, 3), (1, 1), 1, bias=False),
        )
        self.layer_cat=nn.Sequential(
            nn.Conv2d(in_c * 2, in_c, (1, 1), (1, 1), padding=0, bias=False),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_c, in_c, (1, 1), (1, 1), padding=0, bias=False),
        )

        self.mask_t1=Self_mask_T(in_c,in_c//4,4,2)

        # fusion
        in_c, in_w, in_h=in_c//2, in_w*2, in_h*2
        self.connect2 = nn.Conv2d(in_c*3,in_c,1,1,0,bias=False)
        self.mask_t2=Self_mask_T(in_c,self.min_c,in_c//self.min_c,2)
        # print(self.min_c,in_c//self.min_c)
        self.fu_up2= nn.Sequential(
            nn.Conv2d(in_c*2, in_c*2*2, kernel_size=5, stride=1,
                                     padding=5 // 2),
        )

        # fusion
        in_c, in_w, in_h=in_c//2, in_w*2, in_h*2
        self.connect3 = nn.Conv2d(in_c*3,in_c,1,1,0,bias=False)
        self.mask_t3=Self_mask_T(in_c,self.min_c,in_c//self.min_c,2)
        # print(self.min_c,in_c//self.min_c)
        self.fu_up3= nn.Sequential(
            nn.Conv2d(in_c*2, in_c*2*2, kernel_size=5, stride=1,
                                     padding=5 // 2),
        )

        # out_layer
        self.connect4 = nn.Sequential(
            nn.Conv2d(in_c*3,in_c,1,1,0,bias=False)
        )


        self.ps=nn.PixelShuffle(2)

    def forward(self,x_hsi,x_msi):
        x_msi_64=x_msi
        x_msi_32_0=F.interpolate(x_msi,scale_factor=0.5,mode='bilinear')
        x_msi_16_0 = F.interpolate(x_msi, scale_factor=0.25, mode='bilinear')


        x_hsi_16=self.hsi_up3(x_hsi)
        x_hsi_32=self.hsi_up2(x_hsi)
        x_hsi_64=self.hsi_up1(x_hsi)


        x_hsi_64=self.hsi_channels_up1(x_hsi_64)
        x_hsi_32=self.hsi_channels_up2(x_hsi_32)
        x_hsi_16=self.hsi_channels_up3(x_hsi_16)



        x_msi_64=self.msi_channels_up1(x_msi_64)
        x_msi_32=self.msi_channels_up2(x_msi_32_0)
        x_msi_16=self.msi_channels_up3(x_msi_16_0)

        # stage1   # 31*64*64       # RGB 原大小
        sr_hsi1=self.spa_sr1(x_msi_64,x_hsi_64)
        sr_msi1=self.spec_sr1(x_msi_64,x_hsi_64)
                 # 62*32*32
        sr_hsi1_d=self.hsi_down_sample1(sr_hsi1)
        sr_msi1_d=self.msi_down_sample1(sr_msi1)

        # stage2     # 62*32*32
        sr_hsi2=self.spa_sr2(x_msi_32, sr_hsi1_d)
        sr_msi2=self.spec_sr2(sr_msi1_d,x_hsi_32)
                    # 128*16*16
        sr_hsi2_d=self.hsi_down_sample2(sr_hsi2)
        sr_msi2_d=self.msi_down_sample2(sr_msi2)


        # cat
        fea_fu=torch.cat((sr_hsi2_d,sr_msi2_d),dim=1)
        fea_fu=self.layer_cat(fea_fu)
        fea_fu,ca1,sa1=self.mask_t1(fea_fu,x_msi_16,x_hsi_16)

        # fusion2      #124*16*16 --> 62*32*32
        fea_fu = self.fu_up2(fea_fu)
        fea_fu=self.ps(fea_fu)
        fea_fu=self.connect2(torch.cat((fea_fu,sr_hsi2,sr_msi2),dim=1))
        fea_fu,ca2,sa2=self.mask_t2(fea_fu,x_msi_32,x_hsi_32)

        # fusion3
        fea_fu = self.fu_up3(fea_fu)  # 62*32*32--》31*64*64
        fea_fu=self.ps(fea_fu)

        fea_fu=self.connect3(torch.cat((fea_fu,sr_hsi1,sr_msi1),dim=1))
        fea_fu,ca3,sa3 = self.mask_t3(fea_fu,x_msi_64,x_hsi_64)
        fea_fu=self.connect4(torch.cat((fea_fu,x_hsi_64,x_msi_64),dim=1))+x_hsi_64

        return fea_fu


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")


    R = create_F()
    print("测试 Cross_Guide_Fusion")
    a,b=64,64
    hsi = torch.randn((1, 31,a//8,b//8), device=device)
    msi = torch.randn((1, 3, a,b), device=device)

    model2 = Cross_Guide_Fusion(31, a,b,1).cuda()
    flops, params,DIC= profile(model2, inputs=(hsi,msi),ret_layer_info=True)
    print(flops, params)
    flops, params= clever_format([flops, params], "%.3f")
    print(flops, params)


    for i in range(10):
        time1 = time.time()
        out = model2(hsi, msi)
        time2 = time.time()
        print(time2 - time1)



