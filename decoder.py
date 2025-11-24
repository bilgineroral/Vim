import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, img_size: int = 512, in_channels: int = 1024, out_channels: int = 2):
        super().__init__()

        self.in_channels = in_channels
        self.final_H = img_size
        self.final_W = img_size
        layer_info = [
            # (in_ch, out_ch), stride, kernel
            ((in_channels, 512), 1, 3),
            ((512, 256), 2, 4),
            ((256, 128), 2, 4),
            ((128, 128), 2, 4),
            ((128, 64),  2, 4),
            ((64, 64),   1, 3),
            ((64, out_channels), 1, 3)
        ]
        blocks = []
        for (in_ch, out_ch), stride, kernel_size in layer_info:
            blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_ch, out_ch,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1,
                        output_padding=0,
                    ),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )
        self.deconv_blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.deconv_blocks:
            x = block(x)
        W_hat = x
        return W_hat


if __name__ == "__main__":
    decoder = Decoder(in_channels=1024, out_channels=2)
    x = torch.randn(4, 1024, 32, 32)
    W_hat = decoder(x)
    print("Output shape:", W_hat.shape)

    print("Number of parameters: {:.2f}M".format(
        sum(p.numel() for p in decoder.parameters() if p.requires_grad) / 1e6
    ))
