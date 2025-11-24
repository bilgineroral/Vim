from decoder import Decoder
from vim.models_mamba import VisionMamba
import torch
import torch.nn as nn

class VisionMambaDecoder(nn.Module):
    def __init__(self, img_size=512, patch_size=16):
        super().__init__()
        self.encoder = VisionMamba(
            img_size=img_size,
            patch_size=patch_size,
            stride=patch_size,
            depth=24,
            embed_dim=512,
            d_state=16,
            channels=2,
            num_classes=0,
            bimamba_type="v2",
            if_cls_token=False,
            final_pool_type='all',
            # leave other params as default
        )
        self.proj = nn.Linear(self.encoder.embed_dim, 2*self.encoder.embed_dim)
        self.decoder = Decoder(img_size=img_size, in_channels=self.proj.out_features, out_channels=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z = self.proj(z)
        z = z.transpose(1, 2).reshape(z.size(0), -1, self.encoder.feat_gran, self.encoder.feat_gran)
        W_hat = self.decoder(z)
        return W_hat

if __name__ == "__main__":
    model = VisionMambaDecoder(img_size=512, patch_size=16).to('cuda')
    x = torch.randn(1, 2, 512, 512).to('cuda')
    W_hat = model(x)
    print("Final output shape:", W_hat.shape)

    print("Number of parameters (VMD): {:.2f}M".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    ))