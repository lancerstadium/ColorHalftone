from torchsummary import summary

from wllab.network.lut import SRNet, SPF_LUT_net, print_network


def model_summary(model):
    summary(model, (1, 48, 48))


if __name__ == "__main__":
    model = SRNet(mode='SxN', nf=64, upscale=4, dense=True)
    print_network(model)