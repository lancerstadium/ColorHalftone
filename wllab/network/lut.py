import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from ..common.network import *

mode_pad_dict = {"s": 1, "d": 2, "y": 2, "e": 3, "h": 3, "o": 3}





def round_func(input):
    # Backward Pass Differentiable Approximation (BPDA)
    # This is equivalent to replacing round function (non-differentiable)
    # with an identity function (differentiable) only when backward,
    forward_value = torch.round(input)
    out = input.clone()
    out.data = forward_value.data
    return out

class BaseSRNets(nn.Module):
    """ A MuLUT network"""

    def __init__(self, nf=64, scale=4, modes=['s', 'd', 'y'], stages=2):
        super(BaseSRNets, self).__init__()
        self.modes = modes
        self.stages = stages

        for s in range(stages):  # 2-stage
            if (s + 1) == stages:
                upscale = scale
                flag = "N"
            else:
                upscale = None
                flag = "1"
            for mode in modes:
                self.add_module("s{}_{}".format(str(s + 1), mode),
                                SRNet("{}x{}".format(mode.upper(), flag), nf=nf, upscale=upscale))

    def forward(self, x, phase='train'):
        modes, stages = self.modes, self.stages
        # Stage 1
        for s in range(stages):
            pred = 0
            for mode in modes:
                sub_module = getattr(self, "s{}_{}".format(str(s + 1), mode))

                pad = mode_pad_dict[mode]
                for r in [0, 1, 2, 3]:
                    pred += round_func(torch.rot90(
                        torch.tanh(sub_module(F.pad(torch.rot90(x, r, [2, 3]), (0, pad, 0, pad), mode='replicate'))),
                        (4 - r) % 4, [2, 3]) * 127)
            if s + 1 == stages:
                avg_factor, bias, norm = len(modes), 0, 1
                x = round_func((pred / avg_factor) + bias)
                if phase == "train":
                    x = x / 255.0
            else:
                avg_factor, bias, norm = len(modes) * 4, 127, 255.0
                x = round_func(torch.clamp((pred / avg_factor) + bias, 0, 255)) / norm
        return x

class MuLUT(nn.Module):
    """ PyTorch version of MuLUT for LUT-aware fine-tuning. """

    def __init__(self, lut_folder, stages, modes, lutName, upscale, interval, phase=None, **kwargs):
        super(MuLUT, self).__init__()
        self.interval = interval
        self.upscale = upscale
        self.modes = modes
        self.stages = stages
        L = 2 ** (8 - interval) + 1

        for s in range(stages):
            stage = s + 1
            scale = upscale if stage == stages else 1
            for mode in modes:
                # lut_path = os.path.join(lut_folder, 'LUT_x4_4bit_int8_s{}_{}.npy'.format(str(stage), mode))
                lut_path = os.path.join(lut_folder, '{}_s{}_{}.npy'.format(lutName, str(stage), mode))
                key = "s{}_{}".format(str(stage), mode)
                lut_arr = np.load(lut_path).reshape(-1, scale * scale).astype(np.float32) / 127.0
                self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

    @staticmethod
    def round_func(input):
        # Backward Pass Differentiable Approximation (BPDA)
        # This is equivalent to replacing round function (non-differentiable)
        # with an identity function (differentiable) only when backward
        forward_value = torch.round(input)
        out = input.clone()
        out.data = forward_value.data
        return out

    def InterpTorchBatch(self,weight_c2, upscale, mode, img_in, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        weight_c2 = weight_c2 * 127
        weight_c2 = self.round_func(weight_c2)
        weight_c2 = torch.clamp(weight_c2, -127, 127)

        interval = self.interval
        q = 2 ** interval  # 16
        L = 2 ** (8 - interval) + 1  # 17

        img_abcd = torch.floor_divide(img_in, q).type(torch.int64)
        fabcd = img_in % q

        if mode == "s":
            # pytorch 1.5 dont support rounding_mode, use // equavilent
            # https://pytorch.org/docs/1.5.0/torch.html#torch.div
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 1:1 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 0:0 + w]
            fd = fabcd[:, :, 1:1 + h, 1:1 + w]

        elif mode == "d":
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 2:2 + w]
            img_c1 = img_abcd[:, :, 2:2 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 2:2 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 2:2 + w]
            fc = fabcd[:, :, 2:2 + h, 0:0 + w]
            fd = fabcd[:, :, 2:2 + h, 2:2 + w]

        elif mode == "y":
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 1:1 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 2:2 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 1:1 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 2:2 + w]
            fd = fabcd[:, :, 2:2 + h, 1:1 + w]
        else:
            # more sampling modes can be implemented similarly
            raise ValueError("Mode {} not implemented.".format(mode))
        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        p0000 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],img_a1.shape[3], upscale, upscale))
        p0001 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],img_a1.shape[3], upscale, upscale))
        p0010 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],img_a1.shape[3], upscale, upscale))
        p0011 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],img_a1.shape[3], upscale, upscale))
        p0100 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],img_a1.shape[3], upscale, upscale))
        p0101 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],img_a1.shape[3], upscale, upscale))
        p0110 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],img_a1.shape[3], upscale, upscale))
        p0111 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],img_a1.shape[3], upscale, upscale))

        p1000 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],img_a1.shape[3], upscale, upscale))
        p1001 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],img_a1.shape[3], upscale, upscale))
        p1010 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],img_a1.shape[3], upscale, upscale))
        p1011 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],img_a1.shape[3], upscale, upscale))
        p1100 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],img_a1.shape[3], upscale, upscale))
        p1101 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],img_a1.shape[3], upscale, upscale))
        p1110 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],img_a1.shape[3], upscale, upscale))
        p1111 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],img_a1.shape[3], upscale, upscale))

        out = torch.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],
                           img_a1.shape[3], upscale, upscale), dtype=weight_c2.dtype).to(device=weight_c2.device)
        sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2] * img_a1.shape[3]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Fix the overflow bug in SR-LUT's implementation, should compare fd with fa first!
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out / q

        out = out.reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],img_a1.shape[3], upscale, upscale))
        out = out.permute(0, 1, 2, 4, 3, 5).reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2] * upscale, img_a1.shape[3] * upscale))
        return out

    def forward(self, x, phase='train'):
        x = torch.clamp(x, 0, 1)
        x = x * 255.0

        modes, stages = self.modes, self.stages
        for s in range(stages):
            pred = 0
            stage = s + 1
            if stage == stages:
                avg_factor, bias, norm = len(modes), 0, 1
                scale = self.upscale
            else:
                avg_factor, bias, norm = len(modes) * 4, 127, 255.0
                scale = 1

            for mode in modes:
                pad = mode_pad_dict[mode]
                key = "s{}_{}".format(str(stage), mode)
                weight = getattr(self, "weight_" + key)

                for r in [0, 1, 2, 3]:
                    tmp = torch.rot90(
                        self.InterpTorchBatch(weight, scale, mode, F.pad(torch.rot90(x, r, [
                            2, 3]), (0, pad, 0, pad), mode='replicate'), pad), (4 - r) % 4, [2, 3])
                    pred += tmp
                    pred = self.round_func(pred)
                    print(pred.max(), pred.min(), s, mode, r)
                    print(pred.shape)
                    print(pred)
        
            x = self.round_func(torch.clamp((pred / avg_factor) + bias, 0, 255))
            if stage != stages:
                print(x.shape)
                print(x)
        # print('*'*10)

        if phase == 'train':
            x = x / 255.0
        return x

# lut_folder, stages, modes, lutName, upscale, interval, phase=None, **kwargs
class BaseMuLUT_DFC(nn.Module):
    """ PyTorch version of MuLUT for LUT-aware fine-tuning. """

    def __init__(self, lut_folder, stages, modes, lutName, upscale, interval, compressed_dimensions, diagonal_width, sampling_interval, phase = 'train'):
        super(BaseMuLUT_DFC, self).__init__()
        self.interval = interval
        self.upscale = upscale
        self.modes = modes
        self.stages = stages
        self.d = diagonal_width
        L = 2 ** (8 - interval) + 1

        self.compression_type = compressed_dimensions
        self.sampling_interval = sampling_interval

        if os.path.exists(os.path.join(lut_folder,'ref2index_{}{}i{}.npy'.format(compressed_dimensions, diagonal_width, sampling_interval))):
            self.ref2index = np.load(os.path.join(lut_folder, 'ref2index_{}{}i{}.npy'.format(compressed_dimensions, diagonal_width, sampling_interval)))
            self.ref2index = torch.Tensor(self.ref2index).type(torch.int64)
        else:
            self.ref2index = None


        for s in range(stages):
            stage = s + 1
            scale = upscale if stage == stages else 1
            for mode in modes:
                lut_path = os.path.join(lut_folder, '{}_s{}_{}_compress1.npy'.format(lutName, str(stage), mode))
                key = "s{}_{}_compress1".format(str(stage), mode)
                if compressed_dimensions=='xy':
                    lut_arr = np.load(lut_path).reshape(-1, L * L, scale * scale).astype(np.float32) / 127.0
                elif compressed_dimensions=='xyz':
                    lut_arr = np.load(lut_path).reshape(-1, L, scale * scale).astype(np.float32) / 127.0
                elif compressed_dimensions=='xyzt':
                    lut_arr = np.load(lut_path).reshape(-1, scale * scale).astype(np.float32) / 127.0
                else:
                    raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')

                self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

                lut_path = os.path.join(lut_folder, '{}_s{}_{}_compress2.npy'.format(lutName, str(stage), mode))
                key = "s{}_{}_compress2".format(str(stage), mode)
                lut_arr = np.load(lut_path).reshape(-1, scale * scale).astype(np.float32) / 127.0
                self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

    @staticmethod
    def round_func(input):
        # Backward Pass Differentiable Approximation (BPDA)
        # This is equivalent to replacing round function (non-differentiable)
        # with an identity function (differentiable) only when backward
        forward_value = torch.round(input)
        out = input.clone()
        out.data = forward_value.data
        return out

    def InterpTorchBatch_compress1_xy(self, weight_c1, upscale, mode, img_in, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        weight_c1 = weight_c1 * 127
        weight_c1 = self.round_func(weight_c1)
        weight_c1 = torch.clamp(weight_c1, -127, 127)

        interval = self.interval
        q = 2 ** interval  # 16
        L = 2 ** (8 - interval) + 1  # 17

        diag = 2 * self.d + 1
        N = diag * L + (1 - diag ** 2) // 4

        # img_x = img_in[:, :, 0:0 + h, 0:0 + w] / float(q)
        # img_y = img_in[:, :, 0:0 + h, 1:1 + w] / float(q)
        # index_flag = (torch.abs(img_x - img_y) <= self.d)

        if mode == "s":
            # pytorch 1.5 dont support rounding_mode, use // equavilent
            # https://pytorch.org/docs/1.5.0/torch.html#torch.div
            img_x = img_in[:, :, 0:0 + h, 0:0 + w] / float(q)
            img_y = img_in[:, :, 0:0 + h, 1:1 + w] / float(q)
            index_flag = (torch.abs(img_x - img_y) <= self.d)

            img_a1 = torch.floor_divide(img_in[:, :, 0:0 + h, 0:0 + w], q).type(torch.int64)
            img_b1 = torch.floor_divide(img_in[:, :, 0:0 + h, 1:1 + w], q).type(torch.int64)
            img_c1 = torch.floor_divide(img_in[:, :, 1:1 + h, 0:0 + w], q).type(torch.int64)
            img_d1 = torch.floor_divide(img_in[:, :, 1:1 + h, 1:1 + w], q).type(torch.int64)

            # Extract LSBs
            fa = img_in[:, :, 0:0 + h, 0:0 + w] % q
            fb = img_in[:, :, 0:0 + h, 1:1 + w] % q
            fc = img_in[:, :, 1:1 + h, 0:0 + w] % q
            fd = img_in[:, :, 1:1 + h, 1:1 + w] % q

        elif mode == "d":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w] / float(q)
            img_y = img_in[:, :, 0:0 + h, 2:2 + w] / float(q)
            index_flag = (torch.abs(img_x - img_y) <= self.d)

            img_a1 = torch.floor_divide(img_in[:, :, 0:0 + h, 0:0 + w], q).type(torch.int64)
            img_b1 = torch.floor_divide(img_in[:, :, 0:0 + h, 2:2 + w], q).type(torch.int64)
            img_c1 = torch.floor_divide(img_in[:, :, 2:2 + h, 0:0 + w], q).type(torch.int64)
            img_d1 = torch.floor_divide(img_in[:, :, 2:2 + h, 2:2 + w], q).type(torch.int64)

            # Extract LSBs
            fa = img_in[:, :, 0:0 + h, 0:0 + w] % q
            fb = img_in[:, :, 0:0 + h, 2:2 + w] % q
            fc = img_in[:, :, 2:2 + h, 0:0 + w] % q
            fd = img_in[:, :, 2:2 + h, 2:2 + w] % q

        elif mode == "y":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w] / float(q)
            img_y = img_in[:, :, 1:1 + h, 1:1 + w] / float(q)
            index_flag = (torch.abs(img_x - img_y) <= self.d)

            img_a1 = torch.floor_divide(img_in[:, :, 0:0 + h, 0:0 + w], q).type(torch.int64)
            img_b1 = torch.floor_divide(img_in[:, :, 1:1 + h, 1:1 + w], q).type(torch.int64)
            img_c1 = torch.floor_divide(img_in[:, :, 1:1 + h, 2:2 + w], q).type(torch.int64)
            img_d1 = torch.floor_divide(img_in[:, :, 2:2 + h, 1:1 + w], q).type(torch.int64)

            # Extract LSBs
            fa = img_in[:, :, 0:0 + h, 0:0 + w] % q
            fb = img_in[:, :, 1:1 + h, 1:1 + w] % q
            fc = img_in[:, :, 1:1 + h, 2:2 + w] % q
            fd = img_in[:, :, 2:2 + h, 1:1 + w] % q
        else:
            # more sampling modes can be implemented similarly
            raise ValueError("Mode {} not implemented.".format(mode))

        img_a1 = img_a1[index_flag].flatten()
        img_b1 = img_b1[index_flag].flatten()
        img_c1 = img_c1[index_flag].flatten()
        img_d1 = img_d1[index_flag].flatten()

        fa = fa[index_flag].flatten()
        fb = fb[index_flag].flatten()
        fc = fc[index_flag].flatten()
        fd = fd[index_flag].flatten()

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        k00 = self.ref2index[img_a1, img_b1 - img_a1]
        k01 = self.ref2index[img_a1, img_b2 - img_a1]
        k10 = self.ref2index[img_a2, img_b1 - img_a2]
        k11 = self.ref2index[img_a2, img_b2 - img_a2]

        p0000 = weight_c1[k00, img_c1 * L + img_d1].reshape((-1, upscale, upscale))
        p0001 = weight_c1[k00, img_c1 * L + img_d2].reshape((-1, upscale, upscale))
        p0010 = weight_c1[k00, img_c2 * L + img_d1].reshape((-1, upscale, upscale))
        p0011 = weight_c1[k00, img_c2 * L + img_d2].reshape((-1, upscale, upscale))
        p0100 = weight_c1[k01, img_c1 * L + img_d1].reshape((-1, upscale, upscale))
        p0101 = weight_c1[k01, img_c1 * L + img_d2].reshape((-1, upscale, upscale))
        p0110 = weight_c1[k01, img_c2 * L + img_d1].reshape((-1, upscale, upscale))
        p0111 = weight_c1[k01, img_c2 * L + img_d2].reshape((-1, upscale, upscale))

        p1000 = weight_c1[k10, img_c1 * L + img_d1].reshape((-1, upscale, upscale))
        p1001 = weight_c1[k10, img_c1 * L + img_d2].reshape((-1, upscale, upscale))
        p1010 = weight_c1[k10, img_c2 * L + img_d1].reshape((-1, upscale, upscale))
        p1011 = weight_c1[k10, img_c2 * L + img_d2].reshape((-1, upscale, upscale))
        p1100 = weight_c1[k11, img_c1 * L + img_d1].reshape((-1, upscale, upscale))
        p1101 = weight_c1[k11, img_c1 * L + img_d2].reshape((-1, upscale, upscale))
        p1110 = weight_c1[k11, img_c2 * L + img_d1].reshape((-1, upscale, upscale))
        p1111 = weight_c1[k11, img_c2 * L + img_d2].reshape((-1, upscale, upscale))

        out = torch.zeros((img_a1.shape[0], upscale, upscale), dtype=weight_c1.dtype).to(device=weight_c1.device)
        sz = img_a1.shape[0]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Fix the overflow bug in SR-LUT's implementation, should compare fd with fa first!
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out.reshape((-1, upscale * upscale))

        out = out / q
        return out, index_flag

    def InterpTorchBatch_compress1_xyz(self, weight_c1, upscale, mode, img_in, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        weight_c1 = weight_c1 * 127
        weight_c1 = self.round_func(weight_c1)
        weight_c1 = torch.clamp(weight_c1, -127, 127)

        interval = self.interval
        q = 2 ** interval  # 16

        img_abcd = torch.floor_divide(img_in, q).type(torch.int64)
        fabcd = img_in % q

        if mode == "s":
            # pytorch 1.5 dont support rounding_mode, use // equavilent
            # https://pytorch.org/docs/1.5.0/torch.html#torch.div

            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 1:1 + w]
            img_z = img_in[:, :, 1:1 + h, 0:0 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d*q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d*q)
            index_flag = index_flag_xy & index_flag_xz

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 1:1 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 0:0 + w]
            fd = fabcd[:, :, 1:1 + h, 1:1 + w]

        elif mode == "d":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 2:2 + w]
            img_z = img_in[:, :, 2:2 + h, 0:0 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d*q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d*q)
            index_flag = index_flag_xy & index_flag_xz

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 2:2 + w]
            img_c1 = img_abcd[:, :, 2:2 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 2:2 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 2:2 + w]
            fc = fabcd[:, :, 2:2 + h, 0:0 + w]
            fd = fabcd[:, :, 2:2 + h, 2:2 + w]

        elif mode == "y":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 1:1 + h, 1:1 + w]
            img_z = img_in[:, :, 1:1 + h, 2:2 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d*q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d*q)
            index_flag = index_flag_xy & index_flag_xz

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 1:1 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 2:2 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 1:1 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 2:2 + w]
            fd = fabcd[:, :, 2:2 + h, 1:1 + w]
        else:
            # more sampling modes can be implemented similarly
            raise ValueError("Mode {} not implemented.".format(mode))
        img_a1 = img_a1[index_flag].flatten()
        img_b1 = img_b1[index_flag].flatten()
        img_c1 = img_c1[index_flag].flatten()
        img_d1 = img_d1[index_flag].flatten()

        fa = fa[index_flag].flatten()
        fb = fb[index_flag].flatten()
        fc = fc[index_flag].flatten()
        fd = fd[index_flag].flatten()

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        k000 = self.ref2index[img_a1, img_b1 - img_a1, img_c1 - img_a1]
        k001 = self.ref2index[img_a1, img_b1 - img_a1, img_c2 - img_a1]
        k010 = self.ref2index[img_a1, img_b2 - img_a1, img_c1 - img_a1]
        k011 = self.ref2index[img_a1, img_b2 - img_a1, img_c2 - img_a1]

        k100 = self.ref2index[img_a2, img_b1 - img_a2, img_c1 - img_a2]
        k101 = self.ref2index[img_a2, img_b1 - img_a2, img_c2 - img_a2]
        k110 = self.ref2index[img_a2, img_b2 - img_a2, img_c1 - img_a2]
        k111 = self.ref2index[img_a2, img_b2 - img_a2, img_c2 - img_a2]

        p0000 = weight_c1[k000, img_d1].reshape((-1, upscale, upscale))
        p0001 = weight_c1[k000, img_d2].reshape((-1, upscale, upscale))
        p0010 = weight_c1[k001, img_d1].reshape((-1, upscale, upscale))
        p0011 = weight_c1[k001, img_d2].reshape((-1, upscale, upscale))
        p0100 = weight_c1[k010, img_d1].reshape((-1, upscale, upscale))
        p0101 = weight_c1[k010, img_d2].reshape((-1, upscale, upscale))
        p0110 = weight_c1[k011, img_d1].reshape((-1, upscale, upscale))
        p0111 = weight_c1[k011, img_d2].reshape((-1, upscale, upscale))

        p1000 = weight_c1[k100, img_d1].reshape((-1, upscale, upscale))
        p1001 = weight_c1[k100, img_d2].reshape((-1, upscale, upscale))
        p1010 = weight_c1[k101, img_d1].reshape((-1, upscale, upscale))
        p1011 = weight_c1[k101, img_d2].reshape((-1, upscale, upscale))
        p1100 = weight_c1[k110, img_d1].reshape((-1, upscale, upscale))
        p1101 = weight_c1[k110, img_d2].reshape((-1, upscale, upscale))
        p1110 = weight_c1[k111, img_d1].reshape((-1, upscale, upscale))
        p1111 = weight_c1[k111, img_d2].reshape((-1, upscale, upscale))

        out = torch.zeros((img_a1.shape[0], upscale, upscale), dtype=weight_c1.dtype).to(device=weight_c1.device)
        sz = img_a1.shape[0]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Fix the overflow bug in SR-LUT's implementation, should compare fd with fa first!
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out.reshape((-1, upscale * upscale))

        out = out / q
        return out, index_flag

    def InterpTorchBatch_compress1_xyzt(self, weight_c1, upscale, mode, img_in, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        weight_c1 = weight_c1 * 127
        weight_c1 = self.round_func(weight_c1)
        weight_c1 = torch.clamp(weight_c1, -127, 127)

        interval = self.interval
        q = 2 ** interval  # 16

        img_abcd = torch.floor_divide(img_in, q).type(torch.int64)
        fabcd = img_in % q

        if mode == "s":
            # pytorch 1.5 dont support rounding_mode, use // equavilent
            # https://pytorch.org/docs/1.5.0/torch.html#torch.div

            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 1:1 + w]
            img_z = img_in[:, :, 1:1 + h, 0:0 + w]
            img_t = img_in[:, :, 1:1 + h, 1:1 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d * q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d * q)
            index_flag_xt = (torch.abs(img_x - img_t) <= self.d * q)
            index_flag = (index_flag_xy & index_flag_xz) & index_flag_xt

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 1:1 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 0:0 + w]
            fd = fabcd[:, :, 1:1 + h, 1:1 + w]

        elif mode == "d":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 2:2 + w]
            img_z = img_in[:, :, 2:2 + h, 0:0 + w]
            img_t = img_in[:, :, 2:2 + h, 2:2 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d * q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d * q)
            index_flag_xt = (torch.abs(img_x - img_t) <= self.d * q)
            index_flag = (index_flag_xy & index_flag_xz) & index_flag_xt

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 2:2 + w]
            img_c1 = img_abcd[:, :, 2:2 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 2:2 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 2:2 + w]
            fc = fabcd[:, :, 2:2 + h, 0:0 + w]
            fd = fabcd[:, :, 2:2 + h, 2:2 + w]

        elif mode == "y":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 1:1 + h, 1:1 + w]
            img_z = img_in[:, :, 1:1 + h, 2:2 + w]
            img_t = img_in[:, :, 2:2 + h, 1:1 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d * q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d * q)
            index_flag_xt = (torch.abs(img_x - img_t) <= self.d * q)
            index_flag = (index_flag_xy & index_flag_xz) & index_flag_xt

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 1:1 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 2:2 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 1:1 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 2:2 + w]
            fd = fabcd[:, :, 2:2 + h, 1:1 + w]
        else:
            # more sampling modes can be implemented similarly
            raise ValueError("Mode {} not implemented.".format(mode))
        img_a1 = img_a1[index_flag].flatten()
        img_b1 = img_b1[index_flag].flatten()
        img_c1 = img_c1[index_flag].flatten()
        img_d1 = img_d1[index_flag].flatten()

        fa = fa[index_flag].flatten()
        fb = fb[index_flag].flatten()
        fc = fc[index_flag].flatten()
        fd = fd[index_flag].flatten()

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        k0000 = self.ref2index[img_a1, img_b1 - img_a1, img_c1 - img_a1, img_d1 - img_a1]
        k0001 = self.ref2index[img_a1, img_b1 - img_a1, img_c1 - img_a1, img_d2 - img_a1]
        k0010 = self.ref2index[img_a1, img_b1 - img_a1, img_c2 - img_a1, img_d1 - img_a1]
        k0011 = self.ref2index[img_a1, img_b1 - img_a1, img_c2 - img_a1, img_d2 - img_a1]
        k0100 = self.ref2index[img_a1, img_b2 - img_a1, img_c1 - img_a1, img_d1 - img_a1]
        k0101 = self.ref2index[img_a1, img_b2 - img_a1, img_c1 - img_a1, img_d2 - img_a1]
        k0110 = self.ref2index[img_a1, img_b2 - img_a1, img_c2 - img_a1, img_d1 - img_a1]
        k0111 = self.ref2index[img_a1, img_b2 - img_a1, img_c2 - img_a1, img_d2 - img_a1]

        k1000 = self.ref2index[img_a2, img_b1 - img_a2, img_c1 - img_a2, img_d1 - img_a2]
        k1001 = self.ref2index[img_a2, img_b1 - img_a2, img_c1 - img_a2, img_d2 - img_a2]
        k1010 = self.ref2index[img_a2, img_b1 - img_a2, img_c2 - img_a2, img_d1 - img_a2]
        k1011 = self.ref2index[img_a2, img_b1 - img_a2, img_c2 - img_a2, img_d2 - img_a2]
        k1100 = self.ref2index[img_a2, img_b2 - img_a2, img_c1 - img_a2, img_d1 - img_a2]
        k1101 = self.ref2index[img_a2, img_b2 - img_a2, img_c1 - img_a2, img_d2 - img_a2]
        k1110 = self.ref2index[img_a2, img_b2 - img_a2, img_c2 - img_a2, img_d1 - img_a2]
        k1111 = self.ref2index[img_a2, img_b2 - img_a2, img_c2 - img_a2, img_d2 - img_a2]

        p0000 = weight_c1[k0000].reshape((-1, upscale * upscale))
        p0001 = weight_c1[k0001].reshape((-1, upscale * upscale))
        p0010 = weight_c1[k0010].reshape((-1, upscale * upscale))
        p0011 = weight_c1[k0011].reshape((-1, upscale * upscale))
        p0100 = weight_c1[k0100].reshape((-1, upscale * upscale))
        p0101 = weight_c1[k0101].reshape((-1, upscale * upscale))
        p0110 = weight_c1[k0110].reshape((-1, upscale * upscale))
        p0111 = weight_c1[k0111].reshape((-1, upscale * upscale))

        p1000 = weight_c1[k1000].reshape((-1, upscale * upscale))
        p1001 = weight_c1[k1001].reshape((-1, upscale * upscale))
        p1010 = weight_c1[k1010].reshape((-1, upscale * upscale))
        p1011 = weight_c1[k1011].reshape((-1, upscale * upscale))
        p1100 = weight_c1[k1100].reshape((-1, upscale * upscale))
        p1101 = weight_c1[k1101].reshape((-1, upscale * upscale))
        p1110 = weight_c1[k1110].reshape((-1, upscale * upscale))
        p1111 = weight_c1[k1111].reshape((-1, upscale * upscale))

        out = torch.zeros((img_a1.shape[0], upscale * upscale), dtype=weight_c1.dtype).to(
            device=weight_c1.device)
        sz = img_a1.shape[0]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Fix the overflow bug in SR-LUT's implementation, should compare fd with fa first!
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out.reshape((-1, upscale * upscale))

        out = out / q
        return out, index_flag


    def InterpTorchBatch(self, weight_c1, weight_c2, upscale, mode, img_in, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        if self.compression_type == 'xy':
            out_compress1, index_flag = self.InterpTorchBatch_compress1_xy(weight_c1, upscale, mode, img_in, bd)
        elif self.compression_type == 'xyz':
            out_compress1, index_flag = self.InterpTorchBatch_compress1_xyz(weight_c1, upscale, mode, img_in, bd)
        elif self.compression_type == 'xyzt':
            out_compress1, index_flag = self.InterpTorchBatch_compress1_xyzt(weight_c1, upscale, mode, img_in, bd)
        else:
            raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')

        index_flag = index_flag.flatten()

        weight_c2 = weight_c2 * 127
        weight_c2 = self.round_func(weight_c2)
        weight_c2 = torch.clamp(weight_c2, -127, 127)

        interval = self.sampling_interval
        q = 2 ** interval 
        L = 2 ** (8 - interval) + 1

        img_abcd = torch.floor_divide(img_in, q).type(torch.int64)
        fabcd = img_in % q

        if mode == "s":
            # pytorch 1.5 dont support rounding_mode, use // equavilent
            # https://pytorch.org/docs/1.5.0/torch.html#torch.div
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 1:1 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 0:0 + w]
            fd = fabcd[:, :, 1:1 + h, 1:1 + w]

        elif mode == "d":
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 2:2 + w]
            img_c1 = img_abcd[:, :, 2:2 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 2:2 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 2:2 + w]
            fc = fabcd[:, :, 2:2 + h, 0:0 + w]
            fd = fabcd[:, :, 2:2 + h, 2:2 + w]

        elif mode == "y":
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 1:1 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 2:2 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 1:1 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 2:2 + w]
            fd = fabcd[:, :, 2:2 + h, 1:1 + w]
        else:
            # more sampling modes can be implemented similarly
            raise ValueError("Mode {} not implemented.".format(mode))
        sz0, sz1, sz2, sz3 = img_a1.shape
        out = torch.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],
                           img_a1.shape[3], upscale, upscale), dtype=weight_c2.dtype).to(
            device=weight_c2.device)
        sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2] * img_a1.shape[3]
        out_all = out.reshape(sz, -1)
        out = out_all[~index_flag]
        sz = out.size(0)
        if sz == 0:
            out_all[index_flag] = out_compress1
            out_all = out_all.reshape((sz0, sz1, sz2, sz3, upscale, upscale))
            out_all = out_all.permute(0, 1, 2, 4, 3, 5).reshape(
                (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2] * upscale, img_a1.shape[3] * upscale))
            return out_all
        img_a1 = img_a1.flatten()[~index_flag]
        img_b1 = img_b1.flatten()[~index_flag]
        img_c1 = img_c1.flatten()[~index_flag]
        img_d1 = img_d1.flatten()[~index_flag]

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        p0000 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (sz, upscale, upscale))
        p0001 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (sz, upscale, upscale))
        p0010 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (sz, upscale, upscale))
        p0011 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (sz, upscale, upscale))
        p0100 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (sz, upscale, upscale))
        p0101 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (sz, upscale, upscale))
        p0110 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (sz, upscale, upscale))
        p0111 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (sz, upscale, upscale))

        p1000 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (sz, upscale, upscale))
        p1001 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (sz, upscale, upscale))
        p1010 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (sz, upscale, upscale))
        p1011 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (sz, upscale, upscale))
        p1100 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (sz, upscale, upscale))
        p1101 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (sz, upscale, upscale))
        p1110 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (sz, upscale, upscale))
        p1111 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (sz, upscale, upscale))

        # out = torch.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],
        #                    img_a1.shape[3], upscale, upscale), dtype=weight_c2.dtype).to(device=weight_c2.device)
        # sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2] * img_a1.shape[3]
        # out_all = out.reshape(sz, -1)
        # out = out_all[~index_flag]

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)[~index_flag]

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)[~index_flag]
        fc = fc.reshape(-1, 1)[~index_flag]

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)[~index_flag]

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Fix the overflow bug in SR-LUT's implementation, should compare fd with fa first!
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out / q
        out_all[index_flag] = out_compress1
        out_all[~index_flag] = out
        out_all = out_all.reshape((sz0, sz1, sz2, sz3, upscale, upscale))
        out_all = out_all.permute(0, 1, 2, 4, 3, 5).reshape(
            (sz0, sz1, sz2 * upscale, sz3 * upscale))
        return out_all

    def forward(self, x, phase='train'):
        x = torch.clamp(x, 0, 1)
        x = x * 255.0

        if self.ref2index is not None:
            self.ref2index = self.ref2index.to(x.device)

        modes, stages = self.modes, self.stages
        for s in range(stages):
            pred = 0
            stage = s + 1
            if stage == stages:
                avg_factor, bias, norm = len(modes), 0, 1
                scale = self.upscale
            else:
                avg_factor, bias, norm = len(modes) * 4, 127, 255.0
                scale = 1

            for mode in modes:
                pad = mode_pad_dict[mode]
                key = "s{}_{}_compress1".format(str(stage), mode)
                weight_c1 = getattr(self, "weight_" + key)
                key = "s{}_{}_compress2".format(str(stage), mode)
                weight_c2 = getattr(self, "weight_" + key)
                for r in [0, 1, 2, 3]:
                    pred += torch.rot90(
                        self.InterpTorchBatch(weight_c1, weight_c2, scale, mode, F.pad(torch.rot90(x, r, [
                            2, 3]), (0, pad, 0, pad), mode='replicate'), pad), (4 - r) % 4, [2, 3])
                    pred = self.round_func(pred)

            x = self.round_func(torch.clamp((pred / avg_factor) + bias, 0, 255))

        if phase == 'train':
            x = x / 255.0
        return x


def identity(input):
    return input


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, scale=None, output_quant=False, modes=['s', 'd', 'y'], nf=64):
        super(ConvBlock, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.modes = modes
        self.module_dict = dict()
        self.upscale = scale
        self.output_quant = output_quant

        scale_factor = 1 if scale is None else scale ** 2
        for c in range(in_c):
            for mode in modes:
                self.module_dict['DepthwiseBlock{}_{}'.format(c, mode)] = MuLUTConv('{}x{}'.format(mode.upper(), 'N'),
                                                                                    nf=nf, out_c=out_c * scale_factor,
                                                                                    stride=1)
        self.module_dict = nn.ModuleDict(self.module_dict)
        if scale is None:
            self.pixel_shuffle = identity
        else:
            self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        modes = self.modes

        x_out = 0
        for c in range(self.in_c):
            x_c = x[:, c:c + 1, :, :]
            pred = 0
            for mode in modes:
                pad = mode_pad_dict[mode]
                sub_module = self.module_dict['DepthwiseBlock{}_{}'.format(c, mode)]
                for r in [0, 1, 2, 3]:
                    pred += round_func(torch.tanh(torch.rot90(self.pixel_shuffle(
                        sub_module(F.pad(torch.rot90(x_c, r, [2, 3]), (0, pad, 0, pad), mode='replicate'))),
                        (4 - r) % 4, [2, 3])) * 127)

            x_out += pred
        if self.output_quant:
            avg_factor = len(modes) * 4 * self.in_c
            x = round_func(torch.clamp(x_out / avg_factor, -1, 1) * 127) / 127
        else:
            x = x_out / self.in_c

        return x


class SPF_LUT_net(nn.Module):
    def __init__(self, nf=32, scale=4, modes=['s', 'd', 'y'], stages=2):
        super(SPF_LUT_net, self).__init__()
        self.upscale = scale
        self.modes = modes

        self.convblock1 = ConvBlock(1, 2, scale=None, output_quant=False, modes=modes, nf=nf)
        self.convblock2 = ConvBlock(1, 2, scale=None, output_quant=False, modes=modes, nf=nf)
        self.convblock3 = ConvBlock(1, 2, scale=None, output_quant=False, modes=modes, nf=nf)
        self.convblock4 = ConvBlock(1, 1, scale=None, output_quant=False, modes=modes, nf=nf)
        self.ChannelConv = MuLUTcUnit(in_c=4, out_c=4, mode='1x1', nf=nf)
        self.upblock = ConvBlock(4, 1, scale=scale, output_quant=False, modes=modes, nf=nf)


    def forward(self, x, phase='train'):
        B, C, H, W = x.size()
        x = x.reshape((B * C, 1, H, W))

        refine_list = []

        # block1
        x = self.convblock1(x)
        avg_factor, bias, norm = len(self.modes) * 4, 127, 255.0
        x = round_func(torch.clamp((x / avg_factor) + bias, 0, 255)) / norm

        refine_list.append(x[:, 0:1, :, :])
        x = x[:, 1:, :, :]

        # block2
        x = self.convblock2(x)
        avg_factor, bias, norm = len(self.modes) * 4, 127, 255.0
        x = round_func(torch.clamp((x / avg_factor) + bias, 0, 255)) / norm

        refine_list.append(x[:, 0:1, :, :])
        x = x[:, 1:, :, :]

        # block3
        x = self.convblock3(x)
        avg_factor, bias, norm = len(self.modes) * 4, 127, 255.0
        x = round_func(torch.clamp((x / avg_factor) + bias, 0, 255)) / norm

        refine_list.append(x[:, 0:1, :, :])
        x = x[:, 1:, :, :]

        # block4
        x = self.convblock4(x)
        avg_factor, bias, norm = len(self.modes) * 4, 127, 255.0
        x = round_func(torch.clamp((x / avg_factor) + bias, 0, 255)) / norm
        refine_list.append(x)

        x = torch.cat(refine_list, dim=1)
        x = round_func(torch.tanh(self.ChannelConv(x)) * 127.0)
        x = round_func(torch.clamp(x + 127, 0, 255)) / 255.0

        x = self.upblock(x)
        avg_factor, bias, norm = len(self.modes), 0, 1
        x = round_func((x / avg_factor) + bias)

        if phase == 'train':
            x = x / 255.0
        x = x.reshape((B, C, self.upscale * H, self.upscale * W))

        return x

class SPF_LUT(nn.Module):
    """ PyTorch version of MuLUT for LUT-aware fine-tuning. """

    def __init__(self, lut_folder, stages, modes, lutName, upscale, interval, phase=None, **kwargs):
        super(SPF_LUT, self).__init__()
        self.interval = interval
        self.upscale = upscale
        self.modes = modes
        self.stages = stages

        L = 2 ** (8 - interval) + 1


        for mode in modes:
            # conv1
            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}.npy'.format(lutName, 1, mode))
            # lut_path = os.path.join(lut_folder, 'LUT_x4_4bit_int8_s{}c0_{}.npy'.format(1, mode))
            key = "s{}c0_{}".format(1, mode)
            lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            # conv2
            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}.npy'.format(lutName, 2, mode))
            # lut_path = os.path.join(lut_folder, 'LUT_x4_4bit_int8_s{}c0_{}.npy'.format(2, mode))
            key = "s{}c0_{}".format(2, mode)
            lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            # conv3
            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}.npy'.format(lutName, 3, mode))
            # lut_path = os.path.join(lut_folder, 'LUT_x4_4bit_int8_s{}c0_{}.npy'.format(3, mode))
            key = "s{}c0_{}".format(3, mode)
            lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            # conv4
            lut_path = os.path.join(lut_folder, '{}_s{}c0_{}.npy'.format(lutName, 4, mode))
            # lut_path = os.path.join(lut_folder, 'LUT_x4_4bit_int8_s{}c0_{}.npy'.format(4, mode))
            key = "s{}c0_{}".format(4, mode)
            lut_arr = np.load(lut_path).reshape((-1, 1)).astype(np.float32) / 127.0
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            for c in range(4):
                # conv6
                lut_path = os.path.join(lut_folder, '{}_s{}c{}_{}.npy'.format(lutName, 6,c, mode))
                # lut_path = os.path.join(lut_folder, 'LUT_x4_4bit_int8_s{}c{}_{}.npy'.format(6,c, mode))
                key = "s{}c{}_{}".format(6,c, mode)
                lut_arr = np.load(lut_path).reshape((-1, self.upscale * self.upscale)).astype(np.float32) / 127.0
                self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

        # conv5
        lut_path = os.path.join(lut_folder, '{}_s{}_channel.npy'.format(lutName, 5))
        # lut_path = os.path.join(lut_folder, 'LUT_x4_4bit_int8_s{}_channel.npy'.format(5))
        key = "s{}_channel".format(5)
        lut_arr = np.load(lut_path).reshape((-1, 4)).astype(np.float32) / 127.0
        self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

    @staticmethod
    def round_func(input):
        # Backward Pass Differentiable Approximation (BPDA)
        # This is equivalent to replacing round function (non-differentiable)
        # with an identity function (differentiable) only when backward
        forward_value = torch.round(input)
        out = input.clone()
        out.data = forward_value.data
        return out

    def InterpTorchBatch(self, weight, upscale,out_c, mode, img_in, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        weight = weight * 127
        weight = self.round_func(weight)
        weight = torch.clamp(weight, -127, 127)

        interval = self.interval
        q = 2 ** interval  # 16
        L = 2 ** (8 - interval) + 1  # 17

        img_abcd = torch.floor_divide(img_in, q).type(torch.int64)
        fabcd = img_in % q

        if mode == "s":
            # pytorch 1.5 dont support rounding_mode, use // equavilent
            # https://pytorch.org/docs/1.5.0/torch.html#torch.div
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 1:1 + h, 1:1 + w]
            # img_a1 = torch.floor_divide(img_in[:, :, 0:0 + h, 0:0 + w], q).type(torch.int64)
            # img_b1 = torch.floor_divide(img_in[:, :, 0:0 + h, 1:1 + w], q).type(torch.int64)
            # img_c1 = torch.floor_divide(img_in[:, :, 1:1 + h, 0:0 + w], q).type(torch.int64)
            # img_d1 = torch.floor_divide(img_in[:, :, 1:1 + h, 1:1 + w], q).type(torch.int64)

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 0:0 + w]
            fd = fabcd[:, :, 1:1 + h, 1:1 + w]
            # fa = img_in[:, :, 0:0 + h, 0:0 + w] % q
            # fb = img_in[:, :, 0:0 + h, 1:1 + w] % q
            # fc = img_in[:, :, 1:1 + h, 0:0 + w] % q
            # fd = img_in[:, :, 1:1 + h, 1:1 + w] % q

        elif mode == "d":
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 2:2 + w]
            img_c1 = img_abcd[:, :, 2:2 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 2:2 + w]
            # img_a1 = torch.floor_divide(img_in[:, :, 0:0 + h, 0:0 + w], q).type(torch.int64)
            # img_b1 = torch.floor_divide(img_in[:, :, 0:0 + h, 2:2 + w], q).type(torch.int64)
            # img_c1 = torch.floor_divide(img_in[:, :, 2:2 + h, 0:0 + w], q).type(torch.int64)
            # img_d1 = torch.floor_divide(img_in[:, :, 2:2 + h, 2:2 + w], q).type(torch.int64)

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 2:2 + w]
            fc = fabcd[:, :, 2:2 + h, 0:0 + w]
            fd = fabcd[:, :, 2:2 + h, 2:2 + w]
            # fa = img_in[:, :, 0:0 + h, 0:0 + w] % q
            # fb = img_in[:, :, 0:0 + h, 2:2 + w] % q
            # fc = img_in[:, :, 2:2 + h, 0:0 + w] % q
            # fd = img_in[:, :, 2:2 + h, 2:2 + w] % q

        elif mode == "y":
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 1:1 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 2:2 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 1:1 + w]
            # img_a1 = torch.floor_divide(img_in[:, :, 0:0 + h, 0:0 + w], q).type(torch.int64)
            # img_b1 = torch.floor_divide(img_in[:, :, 1:1 + h, 1:1 + w], q).type(torch.int64)
            # img_c1 = torch.floor_divide(img_in[:, :, 1:1 + h, 2:2 + w], q).type(torch.int64)
            # img_d1 = torch.floor_divide(img_in[:, :, 2:2 + h, 1:1 + w], q).type(torch.int64)

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 1:1 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 2:2 + w]
            fd = fabcd[:, :, 2:2 + h, 1:1 + w]
            # fa = img_in[:, :, 0:0 + h, 0:0 + w] % q
            # fb = img_in[:, :, 1:1 + h, 1:1 + w] % q
            # fc = img_in[:, :, 1:1 + h, 2:2 + w] % q
            # fd = img_in[:, :, 2:2 + h, 1:1 + w] % q
        else:
            # more sampling modes can be implemented similarly
            raise ValueError("Mode {} not implemented.".format(mode))

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        p0000 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p0001 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p0010 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p0011 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p0100 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p0101 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p0110 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p0111 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))

        p1000 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p1001 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p1010 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p1011 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p1100 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p1101 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p1110 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))
        p1111 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c*upscale*upscale))

        out = torch.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],
                           img_a1.shape[3], out_c*upscale*upscale), dtype=weight.dtype).to(device=weight.device)
        sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2] * img_a1.shape[3]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Fix the overflow bug in SR-LUT's implementation, should compare fd with fa first!
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out / q
        out = out.reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],
                                   img_a1.shape[3], out_c,upscale, upscale))
        out = out.permute(0,1,4,2,5,3,6).reshape(
            (img_a1.shape[0], img_a1.shape[1]*out_c, img_a1.shape[2] * upscale, img_a1.shape[3] * upscale))
        return out

    def InterpTorchBatch_channel(self, weight,out_c, img_in):

        weight = weight * 127
        weight = self.round_func(weight)
        weight = torch.clamp(weight, -127, 127)

        interval = self.interval
        q = 2 ** interval  # 16
        L = 2 ** (8 - interval) + 1  # 17

        # pytorch 1.5 dont support rounding_mode, use // equavilent
        # https://pytorch.org/docs/1.5.0/torch.html#torch.div
        img_a1,img_b1,img_c1,img_d1 = torch.chunk(torch.floor_divide(img_in, q).type(torch.int64),4,1)

        # Extract LSBs
        fa,fb,fc,fd = torch.chunk(img_in%q,4,1)


        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        p0000 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0001 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0010 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0011 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0100 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0101 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0110 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0111 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))

        p1000 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1001 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1010 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1011 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1100 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1101 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1110 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1111 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))

        out = torch.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],
                           img_a1.shape[3], out_c), dtype=weight.dtype).to(device=weight.device)
        sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2] * img_a1.shape[3]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Fix the overflow bug in SR-LUT's implementation, should compare fd with fa first!
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out / q
        out = out.reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],
                                   img_a1.shape[3], out_c))
        out = out.permute(0,1,4,2,3).reshape(
            (img_a1.shape[0], img_a1.shape[1]*out_c, img_a1.shape[2], img_a1.shape[3]))
        return out

    def forward(self, x, phase='train'):
        B,C,H,W = x.shape
        x = x.reshape((B*C,1,H,W))
        x = torch.clamp(x, 0, 1)
        x = x * 255.0

        out_c_list = [2,2,2,1]
        refine_list = []
        # conv1~4
        for s in range(4):
            stage = s+1
            pred = 0
            for mode in self.modes:
                pad = mode_pad_dict[mode]
                key = "s{}c0_{}".format(str(stage), mode)
                weight = getattr(self, "weight_" + key)
                scale =1
                for r in [0, 1, 2, 3]:
                    pred += torch.rot90(
                        self.InterpTorchBatch(weight, scale,out_c_list[s], mode, F.pad(torch.rot90(x, r, [
                            2, 3]), (0, pad, 0, pad), mode='replicate'), pad), (4 - r) % 4, [2, 3])
                    pred = self.round_func(pred)
            avg_factor, bias, norm = len(self.modes) * 4, 127, 255.0
            x = self.round_func(torch.clamp((pred / avg_factor) + bias, 0, 255))
            if out_c_list[s]==2:
                x1, x2 = torch.chunk(x, out_c_list[s], 1)
                x = x2
                refine_list.append(x1)
            else:
                refine_list.append(x)

        x = torch.cat(refine_list, dim=1)
        # conv5
        key = "s{}_channel".format(5)
        weight = getattr(self, "weight_" + key)
        x = self.InterpTorchBatch_channel(weight,4,x)
        x = self.round_func(torch.clamp(x + 127, 0, 255))

        # conv6
        pred = 0
        for c in range(4):
            x_c = x[:,c:c+1,:,:]
            for mode in self.modes:
                pad = mode_pad_dict[mode]
                key = "s{}c{}_{}".format(6,c, mode)
                weight = getattr(self, "weight_" + key)
                scale = self.upscale
                for r in [0, 1, 2, 3]:
                    pred += torch.rot90(
                        self.InterpTorchBatch(weight, scale, 1, mode,
                                              F.pad(torch.rot90(x_c, r, [2, 3]), (0, pad, 0, pad),
                                                    mode='replicate'), pad), (4 - r) % 4,[2, 3])
                    pred = self.round_func(pred)
                    # print(pred.max(), pred.min(), c, mode, r)
        # exit()
        pred = pred / 4
        avg_factor, bias, norm = len(self.modes), 0, 1
        x = self.round_func((pred / avg_factor) + bias)
        x = x.reshape((B,C,H*self.upscale,W*self.upscale))
        if phase == 'train':
            x = x / 255.0
        return x

# lut_folder, stages, modes, lutName, upscale, interval, compressed_dimensions, diagonal_width, sampling_interval, phase = 'train'
class SPF_LUT_DFC(nn.Module):
    """ PyTorch version of MuLUT for LUT-aware fine-tuning. """

    def __init__(self, lut_folder, stages, modes, lutName, upscale, interval, compressed_dimensions, diagonal_width, sampling_interval, phase = 'train'):
        super(SPF_LUT_DFC, self).__init__()
        self.interval = interval
        self.upscale = upscale
        self.modes = modes
        self.stages = stages
        self.d = diagonal_width
        L = 2 ** (8 - interval) + 1
        self.compression_type = compressed_dimensions
        # self.diagonal_width = diagonal_width
        self.sampling_interval = sampling_interval

        if os.path.exists(os.path.join(lut_folder,'ref2index_{}{}i{}.npy'.format(compressed_dimensions, diagonal_width, sampling_interval))):
            self.ref2index = np.load(os.path.join(lut_folder, 'ref2index_{}{}i{}.npy'.format(compressed_dimensions, diagonal_width, sampling_interval)))
            self.ref2index = torch.Tensor(self.ref2index).type(torch.int64)
        else:
            self.ref2index = None

        for mode in modes:
            # conv1
            if phase=='train':
                lut_path = os.path.join(lut_folder, '{}_x{}_4bit_int8_s{}c0_{}_compress1.npy'.format(lutName, upscale, 1, mode))
            else:
                lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress1.npy'.format(lutName, 1, mode))
            key = "s{}c0_{}_compress1".format(1, mode)
            if compressed_dimensions=='xy':
                lut_arr = np.load(lut_path).reshape((-1, L * L, 2)).astype(np.float32) / 127.0
            elif compressed_dimensions=='xyz':
                lut_arr = np.load(lut_path).reshape((-1, L, 2)).astype(np.float32) / 127.0
            elif compressed_dimensions=='xyzt':
                lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            else:
                raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            if phase=='train':
                lut_path = os.path.join(lut_folder, '{}_x{}_4bit_int8_s{}c0_{}_compress2.npy'.format(lutName, upscale, 1, mode))
            else:
                lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress2.npy'.format(lutName, 1, mode))
            key = "s{}c0_{}_compress2".format(1, mode)
            lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            # conv2
            if phase == 'train':
                lut_path = os.path.join(lut_folder, '{}_x{}_4bit_int8_s{}c0_{}_compress1.npy'.format(lutName, upscale, 2, mode))
            else:
                lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress1.npy'.format(lutName, 2, mode))
            key = "s{}c0_{}_compress1".format(2, mode)
            if compressed_dimensions=='xy':
                lut_arr = np.load(lut_path).reshape((-1, L * L, 2)).astype(np.float32) / 127.0
            elif compressed_dimensions=='xyz':
                lut_arr = np.load(lut_path).reshape((-1, L, 2)).astype(np.float32) / 127.0
            elif compressed_dimensions=='xyzt':
                lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            else:
                raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            if phase == 'train':
                lut_path = os.path.join(lut_folder, '{}_x{}_4bit_int8_s{}c0_{}_compress2.npy'.format(lutName, upscale, 2, mode))
            else:
                lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress2.npy'.format(lutName, 2, mode))
            key = "s{}c0_{}_compress2".format(2, mode)
            lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            # conv3
            if phase == 'train':
                lut_path = os.path.join(lut_folder, '{}_x{}_4bit_int8_s{}c0_{}_compress1.npy'.format(lutName, upscale, 3, mode))
            else:
                lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress1.npy'.format(lutName, 3, mode))
            key = "s{}c0_{}_compress1".format(3, mode)
            if compressed_dimensions=='xy':
                lut_arr = np.load(lut_path).reshape((-1, L * L, 2)).astype(np.float32) / 127.0
            elif compressed_dimensions=='xyz':
                lut_arr = np.load(lut_path).reshape((-1, L, 2)).astype(np.float32) / 127.0
            elif compressed_dimensions=='xyzt':
                lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            else:
                raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            if phase == 'train':
                lut_path = os.path.join(lut_folder, '{}_x{}_4bit_int8_s{}c0_{}_compress2.npy'.format(lutName, upscale, 3, mode))
            else:
                lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress2.npy'.format(lutName, 3, mode))
            key = "s{}c0_{}_compress2".format(3, mode)
            lut_arr = np.load(lut_path).reshape((-1, 2)).astype(np.float32) / 127.0
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            # conv4
            if phase == 'train':
                lut_path = os.path.join(lut_folder, '{}_x{}_4bit_int8_s{}c0_{}_compress1.npy'.format(lutName, upscale, 4, mode))
            else:
                lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress1.npy'.format(lutName, 4, mode))
            key = "s{}c0_{}_compress1".format(4, mode)
            if compressed_dimensions=='xy':
                lut_arr = np.load(lut_path).reshape((-1, L * L, 1)).astype(np.float32) / 127.0
            elif compressed_dimensions=='xyz':
                lut_arr = np.load(lut_path).reshape((-1, L, 1)).astype(np.float32) / 127.0
            elif compressed_dimensions=='xyzt':
                lut_arr = np.load(lut_path).reshape((-1, 1)).astype(np.float32) / 127.0
            else:
                raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            if phase == 'train':
                lut_path = os.path.join(lut_folder, '{}_x{}_4bit_int8_s{}c0_{}_compress2.npy'.format(lutName, upscale, 4, mode))
            else:
                lut_path = os.path.join(lut_folder, '{}_s{}c0_{}_compress2.npy'.format(lutName, 4, mode))
            key = "s{}c0_{}_compress2".format(4, mode)
            lut_arr = np.load(lut_path).reshape((-1, 1)).astype(np.float32) / 127.0
            self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

            for c in range(4):
                # conv6
                if phase == 'train':
                    lut_path = os.path.join(lut_folder, '{}_x{}_4bit_int8_s{}c{}_{}_compress1.npy'.format(lutName, upscale, 6, c, mode))
                else:
                    lut_path = os.path.join(lut_folder, '{}_s{}c{}_{}_compress1.npy'.format(lutName, 6, c, mode))
                key = "s{}c{}_{}_compress1".format(6,c, mode)
                if compressed_dimensions=='xy':
                    lut_arr = np.load(lut_path).reshape((-1, L * L, self.upscale * self.upscale)).astype(np.float32) / 127.0
                elif compressed_dimensions=='xyz':
                    lut_arr = np.load(lut_path).reshape((-1, L, self.upscale * self.upscale)).astype(np.float32) / 127.0
                elif compressed_dimensions=='xyzt':
                    lut_arr = np.load(lut_path).reshape((-1, self.upscale * self.upscale)).astype(np.float32) / 127.0
                else:
                    raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')
                self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

                if phase == 'train':
                    lut_path = os.path.join(lut_folder, '{}_x{}_4bit_int8_s{}c{}_{}_compress2.npy'.format(lutName, upscale, 6, c, mode))
                else:
                    lut_path = os.path.join(lut_folder, '{}_s{}c{}_{}_compress2.npy'.format(lutName, 6, c, mode))
                key = "s{}c{}_{}_compress2".format(6,c, mode)
                lut_arr = np.load(lut_path).reshape((-1, self.upscale * self.upscale)).astype(np.float32) / 127.0
                self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

        # conv5
        if phase == 'train':
            lut_path = os.path.join(lut_folder, '{}_x{}_4bit_int8_s{}_channel.npy'.format(lutName, upscale, 5))
        else:
            lut_path = os.path.join(lut_folder, '{}_s{}_channel.npy'.format(lutName, 5))
        key = "s{}_channel".format(5)
        lut_arr = np.load(lut_path).reshape((-1, 4)).astype(np.float32) / 127.0
        self.register_parameter(name="weight_" + key, param=torch.nn.Parameter(torch.Tensor(lut_arr)))

    @staticmethod
    def round_func(input):
        # Backward Pass Differentiable Approximation (BPDA)
        # This is equivalent to replacing round function (non-differentiable)
        # with an identity function (differentiable) only when backward
        forward_value = torch.round(input)
        out = input.clone()
        out.data = forward_value.data
        return out

    def InterpTorchBatch_compress1_xy(self, weight_c1, upscale,out_c, mode, img_in, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        weight_c1 = weight_c1 * 127
        weight_c1 = self.round_func(weight_c1)
        weight_c1 = torch.clamp(weight_c1, -127, 127)

        interval = self.interval
        q = 2 ** interval  # 16
        L = 2 ** (8 - interval) + 1  # 17

        diag = 2 * self.d + 1
        N = diag * L + (1 - diag ** 2) // 4

        img_abcd = torch.floor_divide(img_in, q).type(torch.int64)
        fabcd = img_in % q

        if mode == "s":
            # pytorch 1.5 dont support rounding_mode, use // equavilent
            # https://pytorch.org/docs/1.5.0/torch.html#torch.div
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 1:1 + w]
            index_flag = (torch.abs(img_x - img_y) <= self.d*q)

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 1:1 + h, 1:1 + w]
            # img_a1 = torch.floor_divide(img_in[:, :, 0:0 + h, 0:0 + w], q).type(torch.int64)
            # img_b1 = torch.floor_divide(img_in[:, :, 0:0 + h, 1:1 + w], q).type(torch.int64)
            # img_c1 = torch.floor_divide(img_in[:, :, 1:1 + h, 0:0 + w], q).type(torch.int64)
            # img_d1 = torch.floor_divide(img_in[:, :, 1:1 + h, 1:1 + w], q).type(torch.int64)

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 0:0 + w]
            fd = fabcd[:, :, 1:1 + h, 1:1 + w]
            # fa = img_in[:, :, 0:0 + h, 0:0 + w] % q
            # fb = img_in[:, :, 0:0 + h, 1:1 + w] % q
            # fc = img_in[:, :, 1:1 + h, 0:0 + w] % q
            # fd = img_in[:, :, 1:1 + h, 1:1 + w] % q

        elif mode == "d":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 2:2 + w]
            index_flag = (torch.abs(img_x - img_y) <= self.d*q)

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 2:2 + w]
            img_c1 = img_abcd[:, :, 2:2 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 2:2 + w]
            # img_a1 = torch.floor_divide(img_in[:, :, 0:0 + h, 0:0 + w], q).type(torch.int64)
            # img_b1 = torch.floor_divide(img_in[:, :, 0:0 + h, 2:2 + w], q).type(torch.int64)
            # img_c1 = torch.floor_divide(img_in[:, :, 2:2 + h, 0:0 + w], q).type(torch.int64)
            # img_d1 = torch.floor_divide(img_in[:, :, 2:2 + h, 2:2 + w], q).type(torch.int64)

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 2:2 + w]
            fc = fabcd[:, :, 2:2 + h, 0:0 + w]
            fd = fabcd[:, :, 2:2 + h, 2:2 + w]
            # fa = img_in[:, :, 0:0 + h, 0:0 + w] % q
            # fb = img_in[:, :, 0:0 + h, 2:2 + w] % q
            # fc = img_in[:, :, 2:2 + h, 0:0 + w] % q
            # fd = img_in[:, :, 2:2 + h, 2:2 + w] % q

        elif mode == "y":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 1:1 + h, 1:1 + w]
            index_flag = (torch.abs(img_x - img_y) <= self.d*q)

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 1:1 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 2:2 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 1:1 + w]
            # img_a1 = torch.floor_divide(img_in[:, :, 0:0 + h, 0:0 + w], q).type(torch.int64)
            # img_b1 = torch.floor_divide(img_in[:, :, 1:1 + h, 1:1 + w], q).type(torch.int64)
            # img_c1 = torch.floor_divide(img_in[:, :, 1:1 + h, 2:2 + w], q).type(torch.int64)
            # img_d1 = torch.floor_divide(img_in[:, :, 2:2 + h, 1:1 + w], q).type(torch.int64)

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 1:1 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 2:2 + w]
            fd = fabcd[:, :, 2:2 + h, 1:1 + w]
            # fa = img_in[:, :, 0:0 + h, 0:0 + w] % q
            # fb = img_in[:, :, 1:1 + h, 1:1 + w] % q
            # fc = img_in[:, :, 1:1 + h, 2:2 + w] % q
            # fd = img_in[:, :, 2:2 + h, 1:1 + w] % q
        else:
            # more sampling modes can be implemented similarly
            raise ValueError("Mode {} not implemented.".format(mode))

        img_a1 = img_a1[index_flag].flatten()
        img_b1 = img_b1[index_flag].flatten()
        img_c1 = img_c1[index_flag].flatten()
        img_d1 = img_d1[index_flag].flatten()

        fa = fa[index_flag].flatten()
        fb = fb[index_flag].flatten()
        fc = fc[index_flag].flatten()
        fd = fd[index_flag].flatten()

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        k00 = self.get_1d_index(img_a1, img_b1, N, L, diag)
        k01 = k00 + 1
        k10 = self.get_1d_index(img_a2, img_b1, N, L, diag)
        k11 = k10 + 1

        p0000 = weight_c1[k00, img_c1 * L + img_d1].reshape((-1, out_c*upscale*upscale))
        p0001 = weight_c1[k00, img_c1 * L + img_d2].reshape((-1, out_c*upscale*upscale))
        p0010 = weight_c1[k00, img_c2 * L + img_d1].reshape((-1, out_c*upscale*upscale))
        p0011 = weight_c1[k00, img_c2 * L + img_d2].reshape((-1, out_c*upscale*upscale))
        p0100 = weight_c1[k01, img_c1 * L + img_d1].reshape((-1, out_c*upscale*upscale))
        p0101 = weight_c1[k01, img_c1 * L + img_d2].reshape((-1, out_c*upscale*upscale))
        p0110 = weight_c1[k01, img_c2 * L + img_d1].reshape((-1, out_c*upscale*upscale))
        p0111 = weight_c1[k01, img_c2 * L + img_d2].reshape((-1, out_c*upscale*upscale))

        p1000 = weight_c1[k10, img_c1 * L + img_d1].reshape((-1, out_c*upscale*upscale))
        p1001 = weight_c1[k10, img_c1 * L + img_d2].reshape((-1, out_c*upscale*upscale))
        p1010 = weight_c1[k10, img_c2 * L + img_d1].reshape((-1, out_c*upscale*upscale))
        p1011 = weight_c1[k10, img_c2 * L + img_d2].reshape((-1, out_c*upscale*upscale))
        p1100 = weight_c1[k11, img_c1 * L + img_d1].reshape((-1, out_c*upscale*upscale))
        p1101 = weight_c1[k11, img_c1 * L + img_d2].reshape((-1, out_c*upscale*upscale))
        p1110 = weight_c1[k11, img_c2 * L + img_d1].reshape((-1, out_c*upscale*upscale))
        p1111 = weight_c1[k11, img_c2 * L + img_d2].reshape((-1, out_c*upscale*upscale))

        out = torch.zeros((img_a1.shape[0], out_c*upscale*upscale), dtype=weight_c1.dtype).to(device=weight_c1.device)
        sz = img_a1.shape[0]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Fix the overflow bug in SR-LUT's implementation, should compare fd with fa first!
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out.reshape((-1, out_c*upscale*upscale))

        out = out / q
        return out, index_flag

    def InterpTorchBatch_compress1_xyz(self, weight_c1, upscale, out_c, mode, img_in, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        weight_c1 = weight_c1 * 127
        weight_c1 = self.round_func(weight_c1)
        weight_c1 = torch.clamp(weight_c1, -127, 127)

        interval = self.interval
        q = 2 ** interval  # 16

        img_abcd = torch.floor_divide(img_in, q).type(torch.int64)
        fabcd = img_in % q

        if mode == "s":
            # pytorch 1.5 dont support rounding_mode, use // equavilent
            # https://pytorch.org/docs/1.5.0/torch.html#torch.div

            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 1:1 + w]
            img_z = img_in[:, :, 1:1 + h, 0:0 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d*q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d*q)
            index_flag = index_flag_xy & index_flag_xz

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 1:1 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 0:0 + w]
            fd = fabcd[:, :, 1:1 + h, 1:1 + w]

        elif mode == "d":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 2:2 + w]
            img_z = img_in[:, :, 2:2 + h, 0:0 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d*q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d*q)
            index_flag = index_flag_xy & index_flag_xz

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 2:2 + w]
            img_c1 = img_abcd[:, :, 2:2 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 2:2 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 2:2 + w]
            fc = fabcd[:, :, 2:2 + h, 0:0 + w]
            fd = fabcd[:, :, 2:2 + h, 2:2 + w]

        elif mode == "y":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 1:1 + h, 1:1 + w]
            img_z = img_in[:, :, 1:1 + h, 2:2 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d*q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d*q)
            index_flag = index_flag_xy & index_flag_xz

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 1:1 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 2:2 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 1:1 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 2:2 + w]
            fd = fabcd[:, :, 2:2 + h, 1:1 + w]
        else:
            # more sampling modes can be implemented similarly
            raise ValueError("Mode {} not implemented.".format(mode))
        img_a1 = img_a1[index_flag].flatten()
        img_b1 = img_b1[index_flag].flatten()
        img_c1 = img_c1[index_flag].flatten()
        img_d1 = img_d1[index_flag].flatten()

        fa = fa[index_flag].flatten()
        fb = fb[index_flag].flatten()
        fc = fc[index_flag].flatten()
        fd = fd[index_flag].flatten()

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        k000 = self.ref2index[img_a1, img_b1 - img_a1, img_c1 - img_a1]
        k001 = self.ref2index[img_a1, img_b1 - img_a1, img_c2 - img_a1]
        k010 = self.ref2index[img_a1, img_b2 - img_a1, img_c1 - img_a1]
        k011 = self.ref2index[img_a1, img_b2 - img_a1, img_c2 - img_a1]

        k100 = self.ref2index[img_a2, img_b1 - img_a2, img_c1 - img_a2]
        k101 = self.ref2index[img_a2, img_b1 - img_a2, img_c2 - img_a2]
        k110 = self.ref2index[img_a2, img_b2 - img_a2, img_c1 - img_a2]
        k111 = self.ref2index[img_a2, img_b2 - img_a2, img_c2 - img_a2]

        p0000 = weight_c1[k000, img_d1].reshape((-1, out_c*upscale*upscale))
        p0001 = weight_c1[k000, img_d2].reshape((-1, out_c*upscale*upscale))
        p0010 = weight_c1[k001, img_d1].reshape((-1, out_c*upscale*upscale))
        p0011 = weight_c1[k001, img_d2].reshape((-1, out_c*upscale*upscale))
        p0100 = weight_c1[k010, img_d1].reshape((-1, out_c*upscale*upscale))
        p0101 = weight_c1[k010, img_d2].reshape((-1, out_c*upscale*upscale))
        p0110 = weight_c1[k011, img_d1].reshape((-1, out_c*upscale*upscale))
        p0111 = weight_c1[k011, img_d2].reshape((-1, out_c*upscale*upscale))

        p1000 = weight_c1[k100, img_d1].reshape((-1, out_c*upscale*upscale))
        p1001 = weight_c1[k100, img_d2].reshape((-1, out_c*upscale*upscale))
        p1010 = weight_c1[k101, img_d1].reshape((-1, out_c*upscale*upscale))
        p1011 = weight_c1[k101, img_d2].reshape((-1, out_c*upscale*upscale))
        p1100 = weight_c1[k110, img_d1].reshape((-1, out_c*upscale*upscale))
        p1101 = weight_c1[k110, img_d2].reshape((-1, out_c*upscale*upscale))
        p1110 = weight_c1[k111, img_d1].reshape((-1, out_c*upscale*upscale))
        p1111 = weight_c1[k111, img_d2].reshape((-1, out_c*upscale*upscale))

        out = torch.zeros((img_a1.shape[0], out_c*upscale*upscale), dtype=weight_c1.dtype).to(device=weight_c1.device)
        sz = img_a1.shape[0]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Fix the overflow bug in SR-LUT's implementation, should compare fd with fa first!
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out.reshape((-1, out_c*upscale*upscale))

        out = out / q
        return out, index_flag

    def InterpTorchBatch_compress1_xyzt(self, weight_c1, upscale, out_c, mode, img_in, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        weight_c1 = weight_c1 * 127
        weight_c1 = self.round_func(weight_c1)
        weight_c1 = torch.clamp(weight_c1, -127, 127)

        interval = self.interval
        q = 2 ** interval  # 16

        img_abcd = torch.floor_divide(img_in, q).type(torch.int64)
        fabcd = img_in % q

        if mode == "s":
            # pytorch 1.5 dont support rounding_mode, use // equavilent
            # https://pytorch.org/docs/1.5.0/torch.html#torch.div

            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 1:1 + w]
            img_z = img_in[:, :, 1:1 + h, 0:0 + w]
            img_t = img_in[:, :, 1:1 + h, 1:1 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d*q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d*q)
            index_flag_xt = (torch.abs(img_x - img_t) <= self.d * q)
            index_flag = (index_flag_xy & index_flag_xz) & index_flag_xt

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 1:1 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 0:0 + w]
            fd = fabcd[:, :, 1:1 + h, 1:1 + w]

        elif mode == "d":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 0:0 + h, 2:2 + w]
            img_z = img_in[:, :, 2:2 + h, 0:0 + w]
            img_t = img_in[:, :, 2:2 + h, 2:2 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d*q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d*q)
            index_flag_xt = (torch.abs(img_x - img_t) <= self.d * q)
            index_flag = (index_flag_xy & index_flag_xz) & index_flag_xt

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 2:2 + w]
            img_c1 = img_abcd[:, :, 2:2 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 2:2 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 2:2 + w]
            fc = fabcd[:, :, 2:2 + h, 0:0 + w]
            fd = fabcd[:, :, 2:2 + h, 2:2 + w]

        elif mode == "y":
            img_x = img_in[:, :, 0:0 + h, 0:0 + w]
            img_y = img_in[:, :, 1:1 + h, 1:1 + w]
            img_z = img_in[:, :, 1:1 + h, 2:2 + w]
            img_t = img_in[:, :, 2:2 + h, 1:1 + w]
            index_flag_xy = (torch.abs(img_x - img_y) <= self.d*q)
            index_flag_xz = (torch.abs(img_x - img_z) <= self.d*q)
            index_flag_xt = (torch.abs(img_x - img_t) <= self.d * q)
            index_flag = (index_flag_xy & index_flag_xz) & index_flag_xt

            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 1:1 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 2:2 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 1:1 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 2:2 + w]
            fd = fabcd[:, :, 2:2 + h, 1:1 + w]
        else:
            # more sampling modes can be implemented similarly
            raise ValueError("Mode {} not implemented.".format(mode))
        img_a1 = img_a1[index_flag].flatten()
        img_b1 = img_b1[index_flag].flatten()
        img_c1 = img_c1[index_flag].flatten()
        img_d1 = img_d1[index_flag].flatten()

        fa = fa[index_flag].flatten()
        fb = fb[index_flag].flatten()
        fc = fc[index_flag].flatten()
        fd = fd[index_flag].flatten()

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        k0000 = self.ref2index[img_a1, img_b1 - img_a1, img_c1 - img_a1, img_d1 - img_a1]
        k0001 = self.ref2index[img_a1, img_b1 - img_a1, img_c1 - img_a1, img_d2 - img_a1]
        k0010 = self.ref2index[img_a1, img_b1 - img_a1, img_c2 - img_a1, img_d1 - img_a1]
        k0011 = self.ref2index[img_a1, img_b1 - img_a1, img_c2 - img_a1, img_d2 - img_a1]
        k0100 = self.ref2index[img_a1, img_b2 - img_a1, img_c1 - img_a1, img_d1 - img_a1]
        k0101 = self.ref2index[img_a1, img_b2 - img_a1, img_c1 - img_a1, img_d2 - img_a1]
        k0110 = self.ref2index[img_a1, img_b2 - img_a1, img_c2 - img_a1, img_d1 - img_a1]
        k0111 = self.ref2index[img_a1, img_b2 - img_a1, img_c2 - img_a1, img_d2 - img_a1]

        k1000 = self.ref2index[img_a2, img_b1 - img_a2, img_c1 - img_a2, img_d1 - img_a2]
        k1001 = self.ref2index[img_a2, img_b1 - img_a2, img_c1 - img_a2, img_d2 - img_a2]
        k1010 = self.ref2index[img_a2, img_b1 - img_a2, img_c2 - img_a2, img_d1 - img_a2]
        k1011 = self.ref2index[img_a2, img_b1 - img_a2, img_c2 - img_a2, img_d2 - img_a2]
        k1100 = self.ref2index[img_a2, img_b2 - img_a2, img_c1 - img_a2, img_d1 - img_a2]
        k1101 = self.ref2index[img_a2, img_b2 - img_a2, img_c1 - img_a2, img_d2 - img_a2]
        k1110 = self.ref2index[img_a2, img_b2 - img_a2, img_c2 - img_a2, img_d1 - img_a2]
        k1111 = self.ref2index[img_a2, img_b2 - img_a2, img_c2 - img_a2, img_d2 - img_a2]

        p0000 = weight_c1[k0000].reshape((-1, out_c*upscale*upscale))
        p0001 = weight_c1[k0001].reshape((-1, out_c*upscale*upscale))
        p0010 = weight_c1[k0010].reshape((-1, out_c*upscale*upscale))
        p0011 = weight_c1[k0011].reshape((-1, out_c*upscale*upscale))
        p0100 = weight_c1[k0100].reshape((-1, out_c*upscale*upscale))
        p0101 = weight_c1[k0101].reshape((-1, out_c*upscale*upscale))
        p0110 = weight_c1[k0110].reshape((-1, out_c*upscale*upscale))
        p0111 = weight_c1[k0111].reshape((-1, out_c*upscale*upscale))

        p1000 = weight_c1[k1000].reshape((-1, out_c*upscale*upscale))
        p1001 = weight_c1[k1001].reshape((-1, out_c*upscale*upscale))
        p1010 = weight_c1[k1010].reshape((-1, out_c*upscale*upscale))
        p1011 = weight_c1[k1011].reshape((-1, out_c*upscale*upscale))
        p1100 = weight_c1[k1100].reshape((-1, out_c*upscale*upscale))
        p1101 = weight_c1[k1101].reshape((-1, out_c*upscale*upscale))
        p1110 = weight_c1[k1110].reshape((-1, out_c*upscale*upscale))
        p1111 = weight_c1[k1111].reshape((-1, out_c*upscale*upscale))

        out = torch.zeros((img_a1.shape[0], out_c*upscale*upscale), dtype=weight_c1.dtype).to(device=weight_c1.device)
        sz = img_a1.shape[0]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Fix the overflow bug in SR-LUT's implementation, should compare fd with fa first!
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out.reshape((-1, out_c*upscale*upscale))

        out = out / q
        return out, index_flag

    def InterpTorchBatch(self, weight_c1, weight_c2, upscale,out_c, mode, img_in, bd):
        _, _, h, w = img_in.shape
        h -= bd
        w -= bd

        if self.compression_type == 'xy':
            out_compress1, index_flag = self.InterpTorchBatch_compress1_xy(weight_c1, upscale,out_c, mode, img_in, bd)
        elif self.compression_type == 'xyz':
            out_compress1, index_flag = self.InterpTorchBatch_compress1_xyz(weight_c1, upscale,out_c, mode, img_in, bd)
        elif self.compression_type == 'xyzt':
            out_compress1, index_flag = self.InterpTorchBatch_compress1_xyzt(weight_c1, upscale,out_c, mode, img_in, bd)
        else:
            raise ValueError('Wrong Compressed Dimensions (should be xy, xyz, or xyzt).')
        index_flag = index_flag.flatten()

        weight_c2 = weight_c2 * 127
        weight_c2 = self.round_func(weight_c2)
        weight_c2 = torch.clamp(weight_c2, -127, 127)

        interval = self.sampling_interval
        q = 2 ** interval 
        L = 2 ** (8 - interval) + 1 

        img_abcd = torch.floor_divide(img_in, q).type(torch.int64)
        fabcd = img_in % q

        if mode == "s":
            # pytorch 1.5 dont support rounding_mode, use // equavilent
            # https://pytorch.org/docs/1.5.0/torch.html#torch.div
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 1:1 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 0:0 + w]
            fd = fabcd[:, :, 1:1 + h, 1:1 + w]

        elif mode == "d":
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 0:0 + h, 2:2 + w]
            img_c1 = img_abcd[:, :, 2:2 + h, 0:0 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 2:2 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 0:0 + h, 2:2 + w]
            fc = fabcd[:, :, 2:2 + h, 0:0 + w]
            fd = fabcd[:, :, 2:2 + h, 2:2 + w]

        elif mode == "y":
            img_a1 = img_abcd[:, :, 0:0 + h, 0:0 + w]
            img_b1 = img_abcd[:, :, 1:1 + h, 1:1 + w]
            img_c1 = img_abcd[:, :, 1:1 + h, 2:2 + w]
            img_d1 = img_abcd[:, :, 2:2 + h, 1:1 + w]

            # Extract LSBs
            fa = fabcd[:, :, 0:0 + h, 0:0 + w]
            fb = fabcd[:, :, 1:1 + h, 1:1 + w]
            fc = fabcd[:, :, 1:1 + h, 2:2 + w]
            fd = fabcd[:, :, 2:2 + h, 1:1 + w]
        else:
            # more sampling modes can be implemented similarly
            raise ValueError("Mode {} not implemented.".format(mode))
        sz0, sz1, sz2, sz3 = img_a1.shape
        out = torch.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],
                           img_a1.shape[3], out_c * upscale * upscale), dtype=weight_c2.dtype).to(
            device=weight_c2.device)
        sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2] * img_a1.shape[3]
        out_all = out.reshape(sz, -1)
        out = out_all[~index_flag]
        sz = out.size(0)
        if sz == 0:
            out_all[index_flag] = out_compress1
            out_all = out_all.reshape((sz0, sz1, sz2, sz3, out_c, upscale, upscale))
            out_all = out_all.permute(0, 1, 4, 2, 5, 3, 6).reshape(
                (img_a1.shape[0], img_a1.shape[1] * out_c, img_a1.shape[2] * upscale, img_a1.shape[3] * upscale))
            return out_all
        img_a1 = img_a1.flatten()[~index_flag]
        img_b1 = img_b1.flatten()[~index_flag]
        img_c1 = img_c1.flatten()[~index_flag]
        img_d1 = img_d1.flatten()[~index_flag]

        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        p0000 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0001 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0010 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0011 = weight_c2[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0100 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0101 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0110 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p0111 = weight_c2[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))

        p1000 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1001 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1010 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1011 = weight_c2[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1100 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1101 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1110 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (sz, out_c*upscale*upscale))
        p1111 = weight_c2[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (sz, out_c*upscale*upscale))

        # out = torch.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],
        #                    img_a1.shape[3], out_c*upscale*upscale), dtype=weight_c2.dtype).to(device=weight_c2.device)
        # sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2] * img_a1.shape[3]
        # out_all = out.reshape(sz, -1)
        # out = out_all[~index_flag]

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)[~index_flag]

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)[~index_flag]
        fc = fc.reshape(-1, 1)[~index_flag]

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)[~index_flag]

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Fix the overflow bug in SR-LUT's implementation, should compare fd with fa first!
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out / q
        out_all[index_flag] = out_compress1
        out_all[~index_flag] = out
        out_all = out_all.reshape((sz0,sz1,sz2,sz3, out_c,upscale, upscale))
        out_all = out_all.permute(0,1,4,2,5,3,6).reshape(
            (sz0, sz1*out_c, sz2 * upscale, sz3 * upscale))
        return out_all

    def InterpTorchBatch_channel(self, weight,out_c, img_in):

        weight = weight * 127
        weight = self.round_func(weight)
        weight = torch.clamp(weight, -127, 127)

        interval = self.interval
        q = 2 ** interval  # 16
        L = 2 ** (8 - interval) + 1  # 17

        # pytorch 1.5 dont support rounding_mode, use // equavilent
        # https://pytorch.org/docs/1.5.0/torch.html#torch.div
        img_a1,img_b1,img_c1,img_d1 = torch.chunk(torch.floor_divide(img_in, q).type(torch.int64),4,1)

        # Extract LSBs
        fa,fb,fc,fd = torch.chunk(img_in%q,4,1)


        img_a2 = img_a1 + 1
        img_b2 = img_b1 + 1
        img_c2 = img_c1 + 1
        img_d2 = img_d1 + 1

        p0000 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0001 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0010 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0011 = weight[
            img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0100 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0101 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0110 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p0111 = weight[
            img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))

        p1000 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1001 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1010 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1011 = weight[
            img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1100 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1101 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1110 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))
        p1111 = weight[
            img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
            (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], out_c))

        out = torch.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],
                           img_a1.shape[3], out_c), dtype=weight.dtype).to(device=weight.device)
        sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2] * img_a1.shape[3]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        # Fix the overflow bug in SR-LUT's implementation, should compare fd with fa first!
        # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
        # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        out = out / q
        out = out.reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2],
                                   img_a1.shape[3], out_c))
        out = out.permute(0,1,4,2,3).reshape(
            (img_a1.shape[0], img_a1.shape[1]*out_c, img_a1.shape[2], img_a1.shape[3]))
        return out

    def forward(self, x, phase='train'):
        B,C,H,W = x.shape
        x = x.reshape((B*C,1,H,W))
        x = torch.clamp(x, 0, 1)
        x = x * 255.0
        if self.ref2index is not None:
            self.ref2index = self.ref2index.to(x.device)

        out_c_list = [2,2,2,1]
        refine_list = []
        # conv1~4
        for s in range(4):
            stage = s+1
            pred = 0
            for mode in self.modes:
                pad = mode_pad_dict[mode]
                key = "s{}c0_{}_compress1".format(str(stage), mode)
                weight_c1 = getattr(self, "weight_" + key)
                key = "s{}c0_{}_compress2".format(str(stage), mode)
                weight_c2 = getattr(self, "weight_" + key)
                scale =1
                for r in [0, 1, 2, 3]:
                    pred += torch.rot90(
                        self.InterpTorchBatch(weight_c1, weight_c2, scale,out_c_list[s], mode, F.pad(torch.rot90(x, r, [
                            2, 3]), (0, pad, 0, pad), mode='replicate'), pad), (4 - r) % 4, [2, 3])
                    pred = self.round_func(pred)
            avg_factor, bias, norm = len(self.modes) * 4, 127, 255.0
            x = self.round_func(torch.clamp((pred / avg_factor) + bias, 0, 255))
            if out_c_list[s]==2:
                x1, x2 = torch.chunk(x, out_c_list[s], 1)
                x = x2
                refine_list.append(x1)
            else:
                refine_list.append(x)

        x = torch.cat(refine_list, dim=1)
        # conv5
        key = "s{}_channel".format(5)
        weight = getattr(self, "weight_" + key)
        x = self.InterpTorchBatch_channel(weight,4,x)
        x = self.round_func(torch.clamp(x + 127, 0, 255))
        # conv6
        pred = 0
        for c in range(4):
            x_c = x[:,c:c+1,:,:]
            for mode in self.modes:
                pad = mode_pad_dict[mode]
                key = "s{}c{}_{}_compress1".format(6,c, mode)
                weight_c1 = getattr(self, "weight_" + key)
                key = "s{}c{}_{}_compress2".format(6,c, mode)
                weight_c2 = getattr(self, "weight_" + key)
                scale = self.upscale
                for r in [0, 1, 2, 3]:
                    pred += torch.rot90(
                        self.InterpTorchBatch(weight_c1, weight_c2, scale, 1, mode,
                                              F.pad(torch.rot90(x_c, r, [2, 3]), (0, pad, 0, pad),
                                                    mode='replicate'), pad), (4 - r) % 4,[2, 3])
                    pred = self.round_func(pred)
        pred = pred / 4
        avg_factor, bias, norm = len(self.modes), 0, 1
        x = self.round_func((pred / avg_factor) + bias)
        x = x.reshape((B,C,H*self.upscale,W*self.upscale))
        if phase == 'train':
            x = x / 255.0
        return x



class FloorSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # 前向传播直接返回 floor(x)
        return torch.floor(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播时，直接返回原始梯度（绕过 floor 的梯度）
        return grad_output.clone()

# 封装成可调用的函数
def floor_ste(x):
    return FloorSTE.apply(x)


class DepthwiseLUT(nn.Module):
    def __init__(self, kernel_size=3, out_channels=16, dense=True):
        super(DepthwiseLUT, self).__init__()
        # 生成block_idx的向量化版本
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.dense = dense
        if self.kernel_size % 2 == 0:
            self.pad = 1
        else:
            self.pad = self.kernel_size // 2
        idx = torch.arange(kernel_size**2)
        block_idx = torch.zeros(kernel_size**2, kernel_size, kernel_size)
        block_idx[idx, idx//kernel_size, idx%kernel_size] = 1
        # 创建组合权重张量 (kernel_size² * out_channels, 1, kernel_size, kernel_size)
        self.wegt = nn.Parameter(
            block_idx.unsqueeze(1)          # [K², 1, K, K]
            .repeat(1, out_channels, 1, 1)       # [K², out_channels, K, K] 
            .view(-1, 1, kernel_size, kernel_size),  # [K²*out_channels, 1, K, K]
            requires_grad=True
        )
        # 创建固定掩码 (注册为buffer)
        self.register_buffer(
            "mask",
            block_idx.unsqueeze(1)          # [K², 1, K, K]
            .repeat(1, self.out_channels, 1, 1)       # [K², out_channels, K, K] 
            .view(-1, 1, kernel_size, kernel_size),  # [K²*out_channels, 1, K, K]
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # 组合式卷积操作
        y = F.conv2d(x, self.wegt * self.mask, padding=self.pad, groups=C)
        # 目标维度 [B, K²*out_channels, H', W']
        # # 每 K² 个取平均值 -> [B, out_channels, H', W']
        y = y.view(B, -1, self.out_channels, y.shape[-2], y.shape[-1])
        if self.dense:
            y = floor_ste(y.mean(dim=1)).clamp(-128,127) + x
        else:
            y = floor_ste(y.mean(dim=1)).clamp(-128,127)
        return y



class PointwiseONE(nn.Module):
    def __init__(self, upscale=4, n_feature=64):
        super(PointwiseONE, self).__init__()
        self.conv1 = nn.Conv2d(1, n_feature, 1, stride=1, padding=0, dilation=1)
        self.conv2 = nn.Conv2d(n_feature, n_feature, 1, stride=1, padding=0, dilation=1)
        self.conv3 = nn.Conv2d(n_feature, upscale * upscale, 1, stride=1, padding=0, dilation=1)
        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = self.conv3(x)
        return floor_ste(x)


class PointwiseLUT(nn.Module):
    def __init__(self, upscale=4, n_feature=64, dense=True, ratio=0.8):
        super(PointwiseLUT, self).__init__()
        self.dense = dense
        self.upscale = upscale
        self.Convs = nn.ModuleList()
        base_scale = torch.ones([1, upscale * upscale]) * ratio
        self.scale = nn.Parameter(base_scale)
        for i in range(upscale * upscale):
            self.Convs.append(PointwiseONE(upscale=upscale, n_feature=n_feature))

    def forward(self, x, idx=-1):
        if idx > 0:
            if self.dense:
                return self.Convs[idx](x[:, idx:idx+1, :, :] * self.scale[0, i]) + x
            else:
                return self.Convs[idx](x[:, idx:idx+1, :, :] * self.scale[0, i])
        else:
            y = []
            for i in range(self.upscale * self.upscale):
                y.append(floor_ste(self.Convs[i](x[:, i:i + 1, :, :] * self.scale[0, i])).clamp(-128,127))
            if self.dense:
                return floor_ste(torch.mean(torch.stack(y, 1), 1)).clamp(-128,127) + x
            else:
                return floor_ste(torch.mean(torch.stack(y, 1), 1)).clamp(-128,127)
        

class LogicLUTNet(nn.Module):
    def __init__(self, kernel_size=3, upscale=4, n_feature=64):
        super(LogicLUTNet, self).__init__()
        self.kernel_size = kernel_size
        self.upscale = upscale
        self.scale = nn.Parameter(torch.ones([1, self.upscale * self.upscale, 1, 1]) * 0.1)
        self.dw_msb = DepthwiseLUT(kernel_size=kernel_size, out_channels=upscale ** 2)
        self.dw_lsb = DepthwiseLUT(kernel_size=kernel_size, out_channels=upscale ** 2)
        self.pw_msb = PointwiseLUT(upscale=upscale, n_feature=n_feature)
        self.pw_lsb = PointwiseLUT(upscale=upscale, n_feature=n_feature)
        self.enhance = PointwiseLUT(upscale=1, n_feature=n_feature // 4, ratio=1)

    def extract(self):
        # <<<< MSB >>>>
        base = torch.arange(0, 64, 1)
        L = base.size(0)

        # input
        first_ = base.cuda().unsqueeze(1)
        first__ = torch.cat([first_, first_], 1)
        first___ = torch.cat([first__, first__], 1)
        first_8 = torch.cat([first___, first___], 1)
        first_9 = torch.cat([first_8, first_], 1)

        # Depthwise
        # Rearange input: [N, 4] -> [N, C=1, H=2, W=2]
        input_tensor = first_9.unsqueeze(1).unsqueeze(1).reshape(-1,1,3,3).float() - 32.0
        outputs = []
        for i in range(9):
            batch_output = self.dw_msb(input_tensor)
            results = torch.clamp(batch_output[i], -128,127).floor().cpu().data.numpy().astype(np.int8)
            outputs.append(results)
        results_cat = np.concatenate(outputs, 0)
        print("Resulting LUT_cat size: ", results_cat.shape)

        # Pointwise
        outputs = []
        for j in range(self.upscale * self.upscale):
            base = torch.arange(-32, 32)
            first_ = (base.cuda() * self.pw_msb.scale[0,j].item()).floor().unique() 
            first_ = torch.arange(first_.min(), first_.max() + 1)
            input_tensor = first_.unsqueeze(1).repeat(1, self.kernel_size**2).unsqueeze(-1).unsqueeze(-1).float()
            batch_output = self.pw_msb(input_tensor, j)
            results = torch.clamp(batch_output, -128,127).floor().cpu().data.numpy().astype(np.int8)
            outputs.append(results)
        results_cat = np.concatenate(outputs, 0)
        print("Resulting LUT_cat size: ", results_cat.shape)


    def seg(self, x, interval = 2):
        msb = floor_ste(torch.remainder(x, (interval ** 2)))
        lsb = floor_ste(torch.div(x, interval ** 2))
        return msb, lsb

    def forward(self, x):
        # Channel to batch: [N, C, H, W] -> [N * C, 1, H, W]
        is_trs = x.max() <= 1
        if is_trs:
            x = x * 255
        x = floor_ste(x - 128).clamp(-128, 127)
        C = x.size(1)
        x = x.view(-1, 1, x.size(2), x.size(3))
        msb, lsb = self.seg(x)
        msb1 = self.dw_msb(msb).clamp(-32, 31)
        lsb1 = self.dw_lsb(lsb).clamp(0, 3)
        msb2 = self.pw_msb(msb1).clamp(-32, 31)
        lsb2 = self.pw_lsb(lsb1).clamp(0, 3)
        res1 = floor_ste((msb2 * 4 + lsb2).clamp(-128, 127) * self.scale).clamp(-128, 127)
        res2 = floor_ste(x * (1 - self.scale)).clamp(-128, 127)
        res = nn.PixelShuffle(self.upscale)((res1 + res2).clamp(-128, 127))
        res = self.enhance(res).clamp(-128, 127)
        # Batch to channel: [N * C, 1, H, W] -> [N, C, H, W]
        res = res.view(-1, C, res.size(2), res.size(3))
        res = res + 128
        if is_trs:
            res = res / 255
        return res



class XQuantize(Function):
    @staticmethod
    def forward(ctx, x):
        x = torch.round(x)
        return x
    
    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None
    

class PointOneChannel(torch.nn.Module):
    def __init__(self):
        super(PointOneChannel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 1, stride=1, padding=0, dilation=1)
        self.conv2 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv3 = nn.Conv2d(64, 16, 1, stride=1, padding=0, dilation=1)
        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif classname.find('bn') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
    def forward(self,x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = self.conv3(x)
        return XQuantize.apply(x)

class UpOneChannel(torch.nn.Module):
    def __init__(self):
        super(UpOneChannel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 1, stride=1, padding=0, dilation=1)
        self.conv2 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv3 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv4 = nn.Conv2d(64, 64, 1, stride=1, padding=0, dilation=1)
        self.conv5 = nn.Conv2d(64, 16, 1, stride=1, padding=0, dilation=1)
        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif classname.find('bn') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
    def forward(self,x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        x = self.conv5(x)
        return XQuantize.apply(x)


class DepthWise(torch.nn.Module):
    def __init__(self):
        super(DepthWise, self).__init__()
        nkernel = 3
        blist = np.zeros((nkernel**2,nkernel,nkernel))
        for l in range(blist.shape[0]):
            blist[l][l//nkernel][l%nkernel] = 1  
        channle_ = 16
        " High "
        kernel11 = blist[0]
        kernel11 = torch.FloatTensor(kernel11).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.weight11 = nn.Parameter(data = kernel11, requires_grad=True)
        mask11 = blist[0]
        mask11 = torch.FloatTensor(mask11).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.mask11 = nn.Parameter(data = mask11, requires_grad=False)
        
        kernel21 = blist[1]
        kernel21 = torch.FloatTensor(kernel21).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.weight21 = nn.Parameter(data = kernel21, requires_grad=True)
        mask21 = blist[1]
        mask21 = torch.FloatTensor(mask21).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.mask21 = nn.Parameter(data = mask21, requires_grad=False)
        
        kernel31 = blist[2]
        kernel31 = torch.FloatTensor(kernel31).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.weight31 = nn.Parameter(data = kernel31, requires_grad=True)
        mask31 = blist[2]
        mask31 = torch.FloatTensor(mask31).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.mask31 = nn.Parameter(data = mask31, requires_grad=False)
        
        kernel41 = blist[3]
        kernel41 = torch.FloatTensor(kernel41).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.weight41 = nn.Parameter(data = kernel41, requires_grad=True)
        mask41 = blist[3]
        mask41 = torch.FloatTensor(mask41).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.mask41 = nn.Parameter(data = mask41, requires_grad=False)
        
        kernel51 = blist[4]
        kernel51 = torch.FloatTensor(kernel51).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.weight51 = nn.Parameter(data = kernel51, requires_grad=True)
        mask51 = blist[4]
        mask51 = torch.FloatTensor(mask51).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.mask51 = nn.Parameter(data = mask51, requires_grad=False)
        
        kernel61 = blist[5]
        kernel61 = torch.FloatTensor(kernel61).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.weight61 = nn.Parameter(data = kernel61, requires_grad=True)
        mask61 = blist[5]
        mask61 = torch.FloatTensor(mask61).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.mask61 = nn.Parameter(data = mask61, requires_grad=False)
                
        kernel71 = blist[6]
        kernel71 = torch.FloatTensor(kernel71).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.weight71 = nn.Parameter(data = kernel71, requires_grad=True)
        mask71 = blist[6]
        mask71 = torch.FloatTensor(mask71).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.mask71 = nn.Parameter(data = mask71, requires_grad=False)
        
        kernel81 = blist[7]
        kernel81 = torch.FloatTensor(kernel81).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.weight81 = nn.Parameter(data = kernel81, requires_grad=True)
        mask81 = blist[7]
        mask81 = torch.FloatTensor(mask81).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.mask81 = nn.Parameter(data = mask81, requires_grad=False)
        
        kernel91 = blist[8]
        kernel91 = torch.FloatTensor(kernel91).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.weight91 = nn.Parameter(data = kernel91, requires_grad=True)
        mask91 = blist[8]
        mask91 = torch.FloatTensor(mask91).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.mask91 = nn.Parameter(data = mask91, requires_grad=False)
        
        " Low "
        
        kernel111 = blist[0]
        kernel111 = torch.FloatTensor(kernel111).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.weight111 = nn.Parameter(data = kernel111, requires_grad=True)
        mask111 = blist[0]
        mask111 = torch.FloatTensor(mask111).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.mask111 = nn.Parameter(data = mask111, requires_grad=False)
        
        kernel211 = blist[1]
        kernel211 = torch.FloatTensor(kernel211).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.weight211 = nn.Parameter(data = kernel211, requires_grad=True)
        mask211 = blist[1]
        mask211 = torch.FloatTensor(mask211).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.mask211 = nn.Parameter(data = mask211, requires_grad=False)
        
        kernel311 = blist[2]
        kernel311 = torch.FloatTensor(kernel311).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.weight311 = nn.Parameter(data = kernel311, requires_grad=True)
        mask311 = blist[2]
        mask311 = torch.FloatTensor(mask311).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.mask311 = nn.Parameter(data = mask311, requires_grad=False)
        
        kernel411 = blist[3]
        kernel411 = torch.FloatTensor(kernel411).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.weight411 = nn.Parameter(data = kernel411, requires_grad=True)
        mask411 = blist[3]
        mask411 = torch.FloatTensor(mask411).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.mask411 = nn.Parameter(data = mask411, requires_grad=False)
        
        kernel511 = blist[4]
        kernel511 = torch.FloatTensor(kernel511).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.weight511 = nn.Parameter(data = kernel511, requires_grad=True)
        mask511 = blist[4]
        mask511 = torch.FloatTensor(mask511).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.mask511 = nn.Parameter(data = mask511, requires_grad=False)
        
        kernel611 = blist[5]
        kernel611 = torch.FloatTensor(kernel611).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.weight611 = nn.Parameter(data = kernel611, requires_grad=True)
        mask611 = blist[5]
        mask611 = torch.FloatTensor(mask611).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.mask611 = nn.Parameter(data = mask611, requires_grad=False)
                
        kernel711 = blist[6]
        kernel711 = torch.FloatTensor(kernel711).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.weight711 = nn.Parameter(data = kernel711, requires_grad=True)
        mask711 = blist[6]
        mask711 = torch.FloatTensor(mask711).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.mask711 = nn.Parameter(data = mask711, requires_grad=False)
        
        kernel811 = blist[7]
        kernel811 = torch.FloatTensor(kernel811).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.weight811 = nn.Parameter(data = kernel811, requires_grad=True)
        mask811 = blist[7]
        mask811 = torch.FloatTensor(mask811).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.mask811 = nn.Parameter(data = mask811, requires_grad=False)
        
        kernel911 = blist[8]
        kernel911 = torch.FloatTensor(kernel911).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.weight911 = nn.Parameter(data = kernel911, requires_grad=True)
        mask911 = blist[8]
        mask911 = torch.FloatTensor(mask911).unsqueeze(0).unsqueeze(0).repeat(channle_,1,1,1)
        self.mask911 = nn.Parameter(data = mask911, requires_grad=False)
       
    def forward(self, xh, xl, h):
        B,C,H,W = xh.size()
        
        x1 = F.conv2d(xh, self.weight11*self.mask11, padding=0, groups=C)
        x1 = x1.clamp(-128,127)
        
        x2 = F.conv2d(xh, self.weight21*self.mask21, padding=0, groups=C)
        x2 = x2.clamp(-128,127)
        
        x3 = F.conv2d(xh, self.weight31*self.mask31, padding=0, groups=C)
        x3 = x3.clamp(-128,127)
        
        x4 = F.conv2d(xh, self.weight41*self.mask41, padding=0, groups=C)
        x4 = x4.clamp(-128,127)
        
        x5 = F.conv2d(xh, self.weight51*self.mask51, padding=0, groups=C)
        x5 = x5.clamp(-128,127)
        
        x6 = F.conv2d(xh, self.weight61*self.mask61, padding=0, groups=C)
        x6 = x6.clamp(-128,127)
        
        x7 = F.conv2d(xh, self.weight71*self.mask71, padding=0, groups=C)
        x7 = x7.clamp(-128,127)
        
        x8 = F.conv2d(xh, self.weight81*self.mask81, padding=0, groups=C)
        x8 = x8.clamp(-128,127)
        
        x9 = F.conv2d(xh, self.weight91*self.mask91, padding=0, groups=C)
        x9 = x9.clamp(-128,127)
        
        xh = [XQuantize.apply(x1), XQuantize.apply(x2), XQuantize.apply(x3), XQuantize.apply(x4), XQuantize.apply(x5), XQuantize.apply(x6), XQuantize.apply(x7), XQuantize.apply(x8), XQuantize.apply(x9)]
       
        
        x11 = F.conv2d(xl, self.weight111*self.mask111, padding=0, groups=C)
        x11 = x11.clamp(-128,127)
        
        x21 = F.conv2d(xl, self.weight211*self.mask211, padding=0, groups=C)
        x21 = x21.clamp(-128,127)
        
        x31 = F.conv2d(xl, self.weight311*self.mask311, padding=0, groups=C)
        x31 = x31.clamp(-128,127)
        
        x41 = F.conv2d(xl, self.weight411*self.mask411, padding=0, groups=C)
        x41 = x41.clamp(-128,127)
        
        x51 = F.conv2d(xl, self.weight511*self.mask511, padding=0, groups=C)
        x51 = x51.clamp(-128,127)
        
        x61 = F.conv2d(xl, self.weight611*self.mask611, padding=0, groups=C)
        x61 = x61.clamp(-128,127)
        
        x71 = F.conv2d(xl, self.weight711*self.mask711, padding=0, groups=C)
        x71 = x71.clamp(-128,127)
        
        x81 = F.conv2d(xl, self.weight811*self.mask811, padding=0, groups=C)
        x81 = x81.clamp(-128,127)
        
        x91 = F.conv2d(xl, self.weight911*self.mask911, padding=0, groups=C)
        x91 = x91.clamp(-128,127)
        
        xhl = [XQuantize.apply(x11), XQuantize.apply(x21), XQuantize.apply(x31), XQuantize.apply(x41), XQuantize.apply(x51), XQuantize.apply(x61), XQuantize.apply(x71), XQuantize.apply(x81), XQuantize.apply(x91)]

        if h:
            return xh
        else:
            return xhl


class PointConv(torch.nn.Module):
    def __init__(self):
        super(PointConv, self).__init__()
        self.HConv = nn.ModuleList()
        self.LConv = nn.ModuleList()
        
        for i in range(16):
            self.HConv.append(PointOneChannel())
            self.LConv.append(PointOneChannel())
            
    def forward(self,xh,xl,h,s,l):
        if s:
            xhout = []
            xlout = []
            for i in range(16):
                xhout.append(XQuantize.apply((self.HConv[i](xh[:,i:i+1,:,:])).clamp(-128,127)))
                xlout.append(XQuantize.apply((self.LConv[i](xl[:,i:i+1,:,:])).clamp(-128,127)))
                
            if h:
                return xhout
            else:
                return xlout
        else:
            xhl = self.HConv[l](xh).clamp(-128,127)
            xll = self.LConv[l](xl).clamp(-128,127)
            if h:
                return xhl
            else:
                return xll


class UpConv(torch.nn.Module):
    def __init__(self):
        super(UpConv, self).__init__()
        self.HConv = nn.ModuleList()
        self.LConv = nn.ModuleList()
        for i in range(16):
            self.HConv.append(UpOneChannel())
            self.LConv.append(UpOneChannel())
    def forward(self,xh,xl,h,s,l):
        if s:
            xhout = []
            xlout = []
            for i in range(16):
                xhout.append(XQuantize.apply((self.HConv[i](xh[:,i:i+1,:,:])).clamp(-128,127)))
                xlout.append(XQuantize.apply((self.LConv[i](xl[:,i:i+1,:,:])).clamp(-128,127)))
                
            if h:
                return xhout
            else:
                return xlout
        else:
            xhl = self.HConv[l](xh).clamp(-128,127)
            xll = self.LConv[l](xl).clamp(-128,127)
            if h:
                return xhl
            else:
                return xll



class TinyLUTNet(torch.nn.Module):
    def __init__(self, upscale=4):
        super(TinyLUTNet, self).__init__()
        self.depthconv1 = DepthWise()
        self.pointconv1 = PointConv()
        self.upconv = UpConv()
        self.upscale = upscale
        clist = np.ones((1,16,1,1))
        cl = torch.FloatTensor(clist*0.8)
        self.clip_hh1 = nn.Parameter(data = cl, requires_grad=False)
        self.clip_hl1 = nn.Parameter(data = cl, requires_grad=False)
        
        self.clip_hl2 = nn.Parameter(data = cl, requires_grad=False)
        self.clip_hh2 = nn.Parameter(data = cl, requires_grad=False)
        
    @staticmethod
    def low_high(image):
        xl = torch.remainder(image, 4)
        xh = torch.div(image, 4)
        xl_ = image.clone()
        xh_ = image.clone()
        xl_.data = xl.data
        xh_.data = xh.data
        return xl_.type(torch.float32), xh_.type(torch.float32)
    
    def forward(self, x):
        # Channel to batch: [N, C, H, W] -> [N * C, 1, H, W]
        is_trs = x.max() <= 1
        if is_trs:
            x = x * 255
        x = (x - 128).clamp(-128, 127)
        C = x.size(1)
        x = x.view(-1, 1, x.size(2), x.size(3))
        xh, xl = self.low_high(x)
        # layer 1
        batch_xh = torch.cat(self.depthconv1(xh, xl, True) , dim=1).view(-1, 9, 16, xh.size(2) - 2, xh.size(3) - 2).sum(dim=1)
        batch_xl = torch.cat(self.depthconv1(xh, xl, False), dim=1).view(-1, 9, 16, xl.size(2) - 2, xl.size(3) - 2).sum(dim=1)
        # quantize
        xh = (XQuantize.apply(batch_xh / 9) + xh[:,:,2:,2:]).clamp(-32,31)
        xl = (XQuantize.apply(batch_xl / 9) + xl[:,:,2:,2:]).clamp(0, 3)
        # mean: list to tensor
        # layer 2
        batch_xh = torch.cat(self.pointconv1(xh * self.clip_hh1, xl, True, True, 0) , dim=1).view(-1, 16, 16, xh.size(2), xh.size(3)).sum(dim=1)
        batch_xl = torch.cat(self.pointconv1(xh, xl * self.clip_hl1, False, True, 0), dim=1).view(-1, 16, 16, xl.size(2), xl.size(3)).sum(dim=1)
        xh = (XQuantize.apply(batch_xh / 16) + xh).clamp(-32,31)
        xl = (XQuantize.apply(batch_xl / 16) + xl).clamp(0, 3)
        # up
        batch_xh = torch.cat(self.upconv(xh * self.clip_hh2, xl, True, True, 0) , dim=1).view(-1, 16, 16, xh.size(2), xh.size(3)).sum(dim=1)
        batch_xl = torch.cat(self.upconv(xh, xl * self.clip_hl2, False, True, 0), dim=1).view(-1, 16, 16, xl.size(2), xl.size(3)).sum(dim=1)
        xh = (XQuantize.apply(batch_xh / 16) + xh).clamp(-128, 127)
        xl = (XQuantize.apply(batch_xl / 16) + xl).clamp(-128, 127)
        res = (xh + xl).clamp(-128, 127)
        res = nn.PixelShuffle(self.upscale)(res)
        # Batch to channel: [N * C, 1, H, W] -> [N, C, H, W]
        res = res.view(-1, C, res.size(2), res.size(3))
        res = res + 128
        if is_trs:
            res = res / 255
        return res


class DepthWiseOpt(torch.nn.Module):
    def __init__(self, channel=16, is_pad=False):
        super().__init__()
        self.channel = channel
        self.pad = 1 if is_pad else 0
        # 合并高低分支权重 [2, 9*C, 1, 3, 3]
        base_kernel = torch.eye(9, dtype=torch.float32).view(1, 9, 1, 3, 3)
        self.weights = nn.Parameter(base_kernel.repeat(2, channel, 1, 1, 1))  # [2,9C,1,3,3]
        self.register_buffer('mask', (self.weights != 0).float())
        # 高低都有bias
        self.bias = nn.Parameter(torch.zeros(2, 9, channel))

    def forward(self, xh, xl, h):
        B, C, H, W = xh.size() if h else xl.size()
        idx = 0 if h else 1
        
        # 合并卷积计算
        x = xh if h else xl
        weights = self.weights[idx].view(-1, 1, 3, 3)  # [9C,1,3,3]
        mask = self.mask[idx].view(-1, 1, 3, 3)
        
        # 高效卷积实现（保持 clamp 顺序）
        outputs = F.conv2d(
            x.repeat_interleave(9, dim=1),  # [B, 9C, H, W]
            weights * mask,
            bias=self.bias[idx].view(-1),
            padding=self.pad,
            groups=9*C
        ).view(B, 9, self.channel, H-(2 - self.pad * 2), W-(2 - self.pad * 2)).clamp(-128, 127)  # 保留原有 clamp
        return [XQuantize.apply(outputs[:,i]) for i in range(9)]  # 保持量化顺序


class PointOneChannelOpt(nn.Module):
    def __init__(self, in_ch=1, out_ch=16, n_feature=32, shared_module : nn.Module = None):
        super().__init__()
        # 优化1：减少通道数(64->32)和层数
        self.bias = nn.Parameter(torch.zeros(out_ch))
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, n_feature, 1),
            nn.ReLU(inplace=True),
            shared_module if shared_module else nn.Sequential(
                nn.Conv2d(n_feature, out_ch, 1)
            ),
        )

    def forward(self, x):
        # 保留量化逻辑但优化执行顺序
        return XQuantize.apply(self.conv(x)).clamp(-128, 127)

# class PointConvOpt(nn.Module):
#     def __init__(self, upscale=4, out_ch=16, n_feature=32):
#         super().__init__()
#         self.out_ch = out_ch
#         self.upscale = upscale
#         # 优化2：共享基础卷积层减少参数
#         self.msb_conv = nn.ModuleList()
#         self.lsb_conv = nn.ModuleList()
#         for i in range(upscale ** 2):
#             self.msb_conv.append(PointOneChannelOpt(out_ch=out_ch, n_feature=n_feature))
#             self.lsb_conv.append(PointOneChannelOpt(out_ch=out_ch, n_feature=n_feature))
        
#     def forward(self, xh, xl, h, s, l):
#         if s:
#             # 优化3：并行处理所有通道
#             x = xh if h else xl
#             B, C, H, W = x.shape
#             # 合并批次和通道维度进行批量处理
#             x = x.view(B*C, 1, H, W)
#             outputs = []
#             for i in range(self.upscale ** 2):
#                 outputs.append((self.msb_conv[i](x) if h else self.lsb_conv[i](x)).view(B, C, self.out_ch, H, W).clamp(-128, 127))
#             outputs = torch.mean(torch.stack(outputs, dim=1), dim=1)
#             return outputs
#         else:
#             return (self.msb_conv[l](xh) if h else self.lsb_conv[l](xl)).view(B, C, self.out_ch, H, W).clamp(-128, 127)
        

class PointConvOpt(nn.Module):
    def __init__(self, upscale=4, out_ch=16, n_feature=32, inner_shared=False):
        super().__init__()
        self.out_ch = out_ch
        self.upscale = upscale
        # 使用ModuleList存储卷积模块
        self.msb_conv = nn.ModuleList()
        self.lsb_conv = nn.ModuleList()
        self.shard_module = nn.Sequential(
            nn.Conv2d(n_feature, n_feature, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feature, out_ch, 1)
        ) if inner_shared else None

        for _ in range(upscale ** 2):
            self.msb_conv.append(PointOneChannelOpt(out_ch=out_ch, n_feature=n_feature, shared_module=self.shard_module))
            self.lsb_conv.append(PointOneChannelOpt(out_ch=out_ch, n_feature=n_feature, shared_module=self.shard_module))
        
    def forward(self, xh, xl, h, s, l):
        if s:
            x = xh if h else xl
            B, C, H, W = x.shape
            x = x.view(B*C, 1, H, W)
            
            output = None
            num_modules = self.upscale ** 2
            for i in range(num_modules):
                # 逐个处理每个卷积模块并累加结果
                out_i = self.msb_conv[i](x) if h else self.lsb_conv[i](x)
                out_i = out_i.view(B, C, self.out_ch, H, W).clamp(-128, 127)
                if output is None:
                    output = out_i
                else:
                    output += out_i
                # 及时释放中间变量（可选）
                del out_i
                torch.cuda.empty_cache()
            
            # 计算均值并确保结果在正确设备上
            output = XQuantize.apply(output / num_modules).clamp(-128, 127)
            return output
        else:
            # 处理单一路径
            x = xh if h else xl
            B, C, H, W = x.shape
            x = x.view(B*C, 1, H, W)
            out = self.msb_conv[l](x) if h else self.lsb_conv[l](x)
            return out.view(B, C, self.out_ch, H, W).clamp(-128, 127)


class UpOneChannelOpt(nn.Module):
    def __init__(self, in_ch=1, out_ch=16, n_feature=32):
        super().__init__()
        # 优化1：减少通道数(64->32)和层数
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, n_feature, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feature, n_feature, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feature, n_feature, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feature, n_feature, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feature, out_ch, 1)
        )

    def forward(self, x):
        # 保持量化但优化内存顺序
        return XQuantize.apply(self.conv(x).clamp(-128, 127))

class UpConvOpt(nn.Module):
    def __init__(self, n_feature=32):
        super().__init__()
        self.msb_conv = UpOneChannelOpt(n_feature=n_feature)
        self.lsb_conv = UpOneChannelOpt(n_feature=n_feature)
        
    def forward(self, xh, xl, h, s, l):
        if s:
            x = xh if h else xl
            B, C, H, W = x.shape
            # 合并维度进行批量处理
            x = x.view(B*C, 1, H, W)
            outputs = self.msb_conv(x) if h else self.lsb_conv(x)
            return outputs.view(B, C, 16, H, W)
        else:
            return self.msb_conv(xh) if h else self.lsb_conv(xl)

class TinyLUTNetOpt(nn.Module):
    def __init__(self, upscale=4, n_feature=32):
        super().__init__()
        # 初始化各模块
        self.down = DepthWise()
        self.depthconv = DepthWiseOpt(is_pad=True)
        self.pointconv = PointConvOpt(upscale=4, out_ch=16,n_feature=n_feature // 2, inner_shared=True)
        self.depthwise = DepthWiseOpt(is_pad=True)
        self.pointwise = PointConvOpt(upscale=1, out_ch=16,n_feature=n_feature, inner_shared=True)
        self.updepth = DepthWiseOpt(is_pad=True)
        self.upconv = UpConvOpt(n_feature=n_feature)
        self.upscale = upscale
        
        # 量化参数（减少参数数量）
        self.clip_params = nn.Parameter(torch.full((11, upscale * upscale, 1, 1), 0.8))
        
    @staticmethod
    def low_high(image):
        xl = torch.remainder(image, 4)
        xh = torch.div(image, 4)
        return xl, xh

    def forward(self, x):
        # 启用混合精度训练
        # with torch.cuda.amp.autocast():
        with torch.amp.autocast('cuda'):
            # 输入预处理
            is_trs = x.max() <= 1
            x = x * 255 if is_trs else x
            x = (x - 128).clamp(-128, 127)
            
            # 通道维度处理
            B, C, H, W = x.size()
            x = x.view(B*C, 1, H, W)
            xl, xh = self.low_high(x)
            xll = xl[:, :, 2:, 2:]
            xhl = xh[:, :, 2:, 2:]

            # Layer 0: Down
            xH = torch.stack(self.down(xh, xl, h=True), dim=1).sum(dim=1)
            xL = torch.stack(self.down(xh, xl, h=False), dim=1).sum(dim=1)
            
            # 及时释放中间变量
            xh = (XQuantize.apply(xH / 9) + xh[:,:,2:,2:]).clamp(-32, 31)
            xl = (XQuantize.apply(xL / 9) + xl[:,:,2:,2:]).clamp(0, 3)
            del xH, xL
            torch.cuda.empty_cache()

            # Layer 1: DepthConv
            xH = torch.stack(self.depthconv(xh, xl, h=True), dim=1).sum(dim=1)
            xL = torch.stack(self.depthconv(xh, xl, h=False), dim=1).sum(dim=1)
            
            # 及时释放中间变量
            xh = (XQuantize.apply(xH / 9) + xh).clamp(-32, 31)
            xl = (XQuantize.apply(xL / 9) + xl).clamp(0, 3)
            del xH, xL
            torch.cuda.empty_cache()

            # Layer 2: PointConv
            xH = self.pointconv(xh * self.clip_params[0], xl, h=True, s=True, l=0).sum(dim=1)
            xL = self.pointconv(xh, xl * self.clip_params[1], h=False, s=True, l=0).sum(dim=1)
            
            xh = (XQuantize.apply(xH / 16) + xh).clamp(-32, 31)
            xl = (XQuantize.apply(xL / 16) + xl).clamp(0, 3)
            del xH, xL
            torch.cuda.empty_cache()

            # Concat ResBlock
            xh = XQuantize.apply(xh * (1 - self.clip_params[6]) + xhl * self.clip_params[6]).clamp(-32, 31)
            xl = XQuantize.apply(xl * (1 - self.clip_params[7]) + xll * self.clip_params[7]).clamp(0, 3)
            xll = xl
            xhl = xh

            # Layer 3: Depthwise
            xH = torch.stack(self.depthwise(xh, xl, h=True), dim=1).sum(dim=1)
            xL = torch.stack(self.depthwise(xh, xl, h=False), dim=1).sum(dim=1)
            
            # 及时释放中间变量
            xh = (XQuantize.apply(xH / 9) + xh).clamp(-32, 31)
            xl = (XQuantize.apply(xL / 9) + xl).clamp(0, 3)
            del xH, xL
            torch.cuda.empty_cache()

            # Layer 4: Pointwise
            xH = self.pointwise(xh * self.clip_params[2], xl, h=True, s=True, l=0).sum(dim=1)
            xL = self.pointwise(xh, xl * self.clip_params[3], h=False, s=True, l=0).sum(dim=1)
            
            xh = (XQuantize.apply(xH / 16) + xh).clamp(-32, 31)
            xl = (XQuantize.apply(xL / 16) + xl).clamp(0, 3)
            del xH, xL
            torch.cuda.empty_cache()

            # Concat ResBlock
            xh = XQuantize.apply(xh * (1 - self.clip_params[8]) + xhl * self.clip_params[8]).clamp(-32, 31)
            xl = XQuantize.apply(xl * (1 - self.clip_params[9]) + xll * self.clip_params[9]).clamp(0, 3)
            del xhl, xll

            # Layer 6: UpDepth
            xH = torch.stack(self.updepth(xh, xl, h=True), dim=1).sum(dim=1)
            xL = torch.stack(self.updepth(xh, xl, h=False), dim=1).sum(dim=1)
            
            # 及时释放中间变量
            xh = (XQuantize.apply(xH / 9) + xh).clamp(-32, 31)
            xl = (XQuantize.apply(xL / 9) + xl).clamp(0, 3)
            del xH, xL
            torch.cuda.empty_cache()

            # Layer 7: UpConv
            xH = self.upconv(xh * self.clip_params[4], xl, h=True, s=True, l=0).sum(dim=1)
            xL = self.upconv(xh, xl * self.clip_params[5], h=False, s=True, l=0).sum(dim=1)
            
            xh = (XQuantize.apply(xH / 16) + xh).clamp(-32, 31)
            xl = (XQuantize.apply(xL / 16) + xl).clamp(0, 3)
            del xH, xL
            torch.cuda.empty_cache()

            # Accumulate ResBlock
            res = XQuantize.apply(((xh * 4 + xl).clamp(-128, 127) + x[:,:,2:,2:]) * self.clip_params[10]).clamp(-128, 127)

            # 上采样与后处理
            res = nn.PixelShuffle(self.upscale)(res)
            res = res.view(B, C, (H - 2)*self.upscale, (W - 2)*self.upscale)

        return (res + 128) / 255 if is_trs else res + 128