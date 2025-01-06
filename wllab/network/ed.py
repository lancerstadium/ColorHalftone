import torch
import torch.nn as nn


from ..common.network import calc_normal


class EnDecoder(nn.Module):
    def __init__(self,cfg):
        super(EnDecoder, self).__init__()
        self.feature = self.make_layers(cfg)

    def make_layers(self, cfg):
        layers = []
        if 'M' in cfg: # encoder
            in_channels = 3
        else:          # decoder
            in_channels = 52
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)]
            elif v == 'U':
                layers += [nn.Upsample(scale_factor=2, mode='nearest')]
            elif v == 'RP':
                layers += [nn.ReflectionPad2d((1, 1, 1, 1))]
            elif v == 'Re': 
                layers += [nn.ReLU()]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3,  bias=False)
                layers += [conv2d, nn.BatchNorm2d(v)] 
                in_channels = v
        k = 0
        if 'M' in cfg: 
            for i in range(len(layers)):
                if isinstance(layers[i], nn.Conv2d):
                    k = i
                    break
            out_channel = layers[k].out_channels
            layers.remove(layers[k])
            layers.insert(k,nn.Conv2d(3, out_channel, kernel_size=1,  bias=False))
        return nn.Sequential(*layers[:-1]) # remove the last BN layer of encoder and decoder

    def forward(self, x):
        return self.feature(x)


class MCCNet(nn.Module):
    def __init__(self, in_dim=52):
        super(MCCNet, self).__init__()
        self.f = nn.Conv2d(in_dim , int(in_dim ), (1,1))
        self.g = nn.Conv2d(in_dim , int(in_dim ) , (1,1))
        self.h = nn.Conv2d(in_dim  ,int(in_dim ) , (1,1))
        self.out_conv = nn.Conv2d(int(in_dim ), in_dim, (1, 1))
        self.fc = nn.Linear(in_dim, in_dim)

    def forward(self,content_feat,style_feat):
        B,C,H,W = content_feat.size()
        F_Fc_norm  = self.f(calc_normal(content_feat))

        B,C,H,W = style_feat.size()
        G_Fs_norm =  self.g(calc_normal(style_feat)).view(-1,1,H*W) 
        # G_Fs = self.h(style_feat).view(-1,1,H*W) 
        G_Fs_sum = G_Fs_norm.view(B,C,H*W).sum(-1)
        FC_S = torch.bmm(G_Fs_norm,G_Fs_norm.permute(0,2,1)).view(B,C) /G_Fs_sum  #14
        FC_S = self.fc(FC_S).view(B,C,1,1)

        out = F_Fc_norm*FC_S
        B,C,H,W = content_feat.size()
        out = out.contiguous().view(B,-1,H,W)
        out = self.out_conv(out)
        out = content_feat + out
        return out # , FC_S


class MCCN(nn.Module):
    def __init__(self, in_dim=52):
        super(MCCN, self).__init__()
        self.MCCN = MCCNet(in_dim)
    
    def forward(self, x, y):
        self.MCCN(x, y)


class EnDeNet(nn.Module):
    def __init__(self, 
                 en_cfg=[3, 'RP', 6, 'Re', 'RP', 6, 'Re', 'M', 'RP', 12, 'Re', 'RP', 12, 'Re', 'M', 'RP', 25, 'Re', 'RP', 25, 'Re', 'RP', 25, 'Re', 'RP', 25, 'Re', 'M', 'RP', 52], 
                 de_cfg=['RP', 25, 'Re', 'U', 'RP', 25, 'Re', 'RP', 25, 'Re', 'RP', 25, 'Re', 'RP', 12, 'Re', 'U', 'RP', 12, 'Re', 'RP', 6, 'Re', 'U', 'RP', 6, 'Re', 'RP', 3], 
                 SA=MCCN, 
                 alpha=1.0):
        super(EnDeNet, self).__init__()
        self.alpha   = alpha
        self.encoder = EnDecoder(en_cfg)
        self.sa      = SA(en_cfg[-1])
        self.decoder = EnDecoder(de_cfg)

    def load(self, en_path, de_path, sa_path):
        self.encoder.feature.load_state_dict(torch.load(en_path))
        self.decoder.feature.load_state_dict(torch.load(de_path))
        self.sa.load_state_dict(torch.load(sa_path))

    def forward(self, content, style):
        content_feat = self.encoder(content)
        style_feat = self.encoder(style)
        fccc = self.sa(content_feat, content_feat)
        feat = self.sa(content_feat, style_feat)
        feat = self.alpha * feat + (1 - self.alpha) * fccc 
        out = self.decoder(feat)
        return out
