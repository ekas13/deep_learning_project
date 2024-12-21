import torch
import torch.nn as nn
import torch.nn.functional as F


class score_network_0(torch.nn.Module):
    # takes an input image and time, returns the score function
    def __init__(self):
        super().__init__()
        nch = 2
        chs = [32, 64, 128, 256, 256]
        self._convs = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(2, chs[0], kernel_size=3, padding=1),  # (batch, ch, 28, 28)
                torch.nn.LogSigmoid(),  # (batch, 8, 28, 28)
            ),
            torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, ch, 14, 14)
                torch.nn.Conv2d(chs[0], chs[1], kernel_size=3, padding=1),  # (batch, ch, 14, 14)
                torch.nn.LogSigmoid(),  # (batch, 16, 14, 14)
            ),
            torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, ch, 7, 7)
                torch.nn.Conv2d(chs[1], chs[2], kernel_size=3, padding=1),  # (batch, ch, 7, 7)
                torch.nn.LogSigmoid(),  # (batch, 32, 7, 7)
            ),
            torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # (batch, ch, 4, 4)
                torch.nn.Conv2d(chs[2], chs[3], kernel_size=3, padding=1),  # (batch, ch, 4, 4)
                torch.nn.LogSigmoid(),  # (batch, 64, 4, 4)
            ),
            torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, ch, 2, 2)
                torch.nn.Conv2d(chs[3], chs[4], kernel_size=3, padding=1),  # (batch, ch, 2, 2)
                torch.nn.LogSigmoid(),  # (batch, 64, 2, 2)
            ),
        ])
        self._tconvs = torch.nn.ModuleList([
            torch.nn.Sequential(
                # input is the output of convs[4]
                torch.nn.ConvTranspose2d(chs[4], chs[3], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, 64, 4, 4)
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[3]
                torch.nn.ConvTranspose2d(chs[3] * 2, chs[2], kernel_size=3, stride=2, padding=1, output_padding=0),  # (batch, 32, 7, 7)
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[2]
                torch.nn.ConvTranspose2d(chs[2] * 2, chs[1], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, chs[2], 14, 14)
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[1]
                torch.nn.ConvTranspose2d(chs[1] * 2, chs[0], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, chs[1], 28, 28)
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[0]
                torch.nn.Conv2d(chs[0] * 2, chs[0], kernel_size=3, padding=1),  # (batch, chs[0], 28, 28)
                torch.nn.LogSigmoid(),
                torch.nn.Conv2d(chs[0], 1, kernel_size=3, padding=1),  # (batch, 1, 28, 28)
            ),
        ])

    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        # x: (..., ch0 * 28 * 28), t: (..., 1)
        # print(f"Input shape before reshape: {x.shape}")
        # print(f"Expected shape: {(*x.shape[:-1], 1, 28, 28)}")
        x2 = torch.reshape(x, (*x.shape[:-1], 1, 28, 28))  # (..., ch0, 28, 28)
        
        # tt = t[..., None, None].expand(*t.shape[:-1], 1, 28, 28)  # (..., 1, 28, 28)
        # if labels is not None:
        #     labels = labels[..., None, None].expand(*labels.shape[:-1], 1, 28, 28)
        #     tt = tt + labels
        
        # Expand time tensor `t` to match spatial dimensions
        if t.shape[1:] == (1, 28):  # Ensure compatibility
            tt = t.unsqueeze(-1).expand(-1, 1, 28, 28)  # Shape: (batch, 1, 28, 28)
        else:
            raise ValueError(f"Unexpected shape for t: {t.shape}")

        # Expand labels to match spatial dimensions
        if labels is not None:
            if labels.ndim == 2 and labels.shape[1] == 28:  # Ensure compatibility
                labels = labels.unsqueeze(1).unsqueeze(-1).expand(-1, 1, 28, 28)  # Shape: (batch, 1, 28, 28)
            else:
                raise ValueError(f"Unexpected shape for labels: {labels.shape}")

            # Combine time and label embeddings
            tt = tt + labels
        
        x2t = torch.cat((x2, tt), dim=-3)
        signal = x2t
        signals = []
        for i, conv in enumerate(self._convs):
            signal = conv(signal)
            if i < len(self._convs) - 1:
                signals.append(signal)

        for i, tconv in enumerate(self._tconvs):
            if i == 0:
                signal = tconv(signal)
            else:
                signal = torch.cat((signal, signals[-i]), dim=-3)
                signal = tconv(signal)
        signal = torch.reshape(signal, (*signal.shape[:-3], -1))  # (..., 1 * 28 * 28)
        return signal



class score_network_1(torch.nn.Module):
    # takes an input image and time, returns the score function
    def __init__(self):
        super().__init__()
        nch = 2
        chs = [32, 64, 128, 256, 256]
        self._convs = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(2, chs[0], kernel_size=3, padding=1),  # (batch, ch, 28, 28)
                torch.nn.LogSigmoid(),  # (batch, 8, 28, 28)
            ),
            torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, ch, 14, 14)
                torch.nn.Conv2d(chs[0] + 1, chs[1], kernel_size=3, padding=1),  # (batch, ch, 14, 14)
                torch.nn.LogSigmoid(),  # (batch, 16, 14, 14)
            ),
            torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, ch, 7, 7)
                torch.nn.Conv2d(chs[1] + 1, chs[2], kernel_size=3, padding=1),  # (batch, ch, 7, 7)
                torch.nn.LogSigmoid(),  # (batch, 32, 7, 7)
            ),
            torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # (batch, ch, 4, 4)
                torch.nn.Conv2d(chs[2] + 1, chs[3], kernel_size=3, padding=1),  # (batch, ch, 4, 4)
                torch.nn.LogSigmoid(),  # (batch, 64, 4, 4)
            ),
            torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2),  # (batch, ch, 2, 2)
                torch.nn.Conv2d(chs[3] + 1, chs[4], kernel_size=3, padding=1),  # (batch, ch, 2, 2)
                torch.nn.LogSigmoid(),  # (batch, 64, 2, 2)
            ),
        ])
        self._tconvs = torch.nn.ModuleList([
            torch.nn.Sequential(
                # input is the output of convs[4]
                torch.nn.ConvTranspose2d(chs[4] + 1, chs[3], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, 64, 4, 4)
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[3]
                torch.nn.ConvTranspose2d(chs[3] * 2 + 1, chs[2], kernel_size=3, stride=2, padding=1, output_padding=0),  # (batch, 32, 7, 7)
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[2]
                torch.nn.ConvTranspose2d(chs[2] * 2 + 1, chs[1], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, chs[2], 14, 14)
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[1]
                torch.nn.ConvTranspose2d(chs[1] * 2 + 1, chs[0], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, chs[1], 28, 28)
                torch.nn.LogSigmoid(),
            ),
            torch.nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[0]
                torch.nn.Conv2d(chs[0] * 2 + 1, chs[0], kernel_size=3, padding=1),  # (batch, chs[0], 28, 28)
                torch.nn.LogSigmoid(),
                torch.nn.Conv2d(chs[0], 1, kernel_size=3, padding=1),  # (batch, 1, 28, 28)
            ),
        ])

    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        # x: (..., ch0 * 28 * 28), t: (..., 1)
        x2 = torch.reshape(x, (*x.shape[:-1], 1, 28, 28))  # (..., ch0, 28, 28)
        
        # Expand time tensor `t` to match spatial dimensions
        if t.shape[1:] == (1, 28):  # Ensure compatibility
            tt = t.unsqueeze(-1).expand(-1, 1, 28, 28)  # Shape: (batch, 1, 28, 28)
        else:
            raise ValueError(f"Unexpected shape for t: {t.shape}")

        # Expand labels to match spatial dimensions
        if labels is not None:
            if labels.ndim == 2 and labels.shape[1] == 28:  # Ensure compatibility
                labels = labels.unsqueeze(1).unsqueeze(-1).expand(-1, 1, 28, 28)  # Shape: (batch, 1, 28, 28)
            else:
                raise ValueError(f"Unexpected shape for labels: {labels.shape}")

            # Combine time and label embeddings
            tt = tt + labels
        
        paddings = [0, 0, 1, 0]
        x2t = torch.cat((x2, tt), dim=-3)
        signal = x2t
        signals = []
        for i, conv in enumerate(self._convs):
            signal = conv(signal)
            if i < len(self._convs) - 1:    # dont append the final signal
                signals.append(signal)
                signal = torch.cat((signal, tt), dim=-3)
                tt = torch.nn.functional.avg_pool2d(tt, kernel_size=2, stride=2, padding=paddings[i])
        
        for i, tconv in enumerate(self._tconvs):
            if i == 0:
                signal = torch.cat((signal, tt), dim=-3)
            else:
                signal = torch.cat((signal, signals[-i], tt), dim=-3)
            signal = tconv(signal)
            tt = torch.nn.functional.interpolate(tt, size=signal.shape[-2:], mode='bilinear', align_corners=True)
        signal = torch.reshape(signal, (*signal.shape[:-3], -1))  # (..., 1 * 28 * 28)
        return signal


#####################################################################################################################################################################

class TimeEmbedding(nn.Module):
    """
    Expands [batch, time_dim] -> [batch, time_dim*4] via an MLP.
    E.g., if time_dim=128, output is 512-dimensional.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4)
        )

    def forward(self, t):
        # Expects t of shape [batch, time_dim]
        return self.mlp(t)


class ResidualBlock(nn.Module):
    """
    A typical 2-conv residual block with a time-based bias.
    Uses GroupNorm for better training stability.
    """
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.gn1   = nn.GroupNorm(num_groups=8, num_channels=out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.gn2   = nn.GroupNorm(num_groups=8, num_channels=out_channels)

        # Project time embedding down to 'out_channels' so we can add it into the conv activations
        self.time_mlp = nn.Linear(time_dim, out_channels)

        # 1×1 conv if in/out channels differ, for the skip connection
        if in_channels != out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x, t):
        """
        x: [batch, in_channels, H, W]
        t: [batch, time_dim]  (already processed by TimeEmbedding, e.g. shape=512)
        """
        # 1) Project time embedding to out_channels
        time_emb = self.time_mlp(t)           # [batch, out_channels]
        time_emb = time_emb[:, :, None, None] # [batch, out_channels, 1, 1]

        # 2) First conv
        h = self.conv1(x)
        h = self.gn1(h)
        h = F.silu(h + time_emb)

        # 3) Second conv
        h = self.conv2(h)
        h = self.gn2(h)
        h = F.silu(h)

        # Residual
        return h + self.res_conv(x)


#####################################################################################################################################################################

class SimpleUNet(nn.Module):
    def __init__(self, time_dim=128):
        super().__init__()

        # Our channel configuration at each level
        self.dims = [128, 256, 512]

        # We'll embed time from [batch, time_dim] -> [batch, time_dim*4]
        # so the ResidualBlock expects time_dim*4
        self.time_dim   = time_dim
        self.time_embed = TimeEmbedding(time_dim)  # final output = time_dim*4

        # ----------  Down / Encode  ----------
        # initial conv from RGB=3 to 128
        self.init_conv = nn.Conv2d(3, self.dims[0], 3, padding=1)

        # down blocks
        #   block0: 128->128
        #   downsample to 256
        #   block1: 256->256
        #   downsample to 512
        #   block2: 512->512
        self.down_block0 = ResidualBlock(self.dims[0], self.dims[0], time_dim=self.time_dim*4)
        self.down_block1 = ResidualBlock(self.dims[1], self.dims[1], time_dim=self.time_dim*4)
        self.down_block2 = ResidualBlock(self.dims[2], self.dims[2], time_dim=self.time_dim*4)

        # 2x2 stride-2 conv for downsampling
        self.downsample0 = nn.Conv2d(self.dims[0], self.dims[1], 4, 2, 1)
        self.downsample1 = nn.Conv2d(self.dims[1], self.dims[2], 4, 2, 1)

        # ----------  Up / Decode  ----------
        # We'll go back up:
        #   block2_up, upsample to 256
        #   block1_up, upsample to 128
        #   block0_up
        self.up_block2 = ResidualBlock(self.dims[2], self.dims[2], time_dim=self.time_dim*4)
        
        # After we upsample from 512->256, we concatenate skip (256) => total=512
        # Then up_block1 expects in_channels=512, out_channels=256
        self.up_block1 = ResidualBlock(self.dims[2], self.dims[1], time_dim=self.time_dim*4)
        
        # Similarly, after we upsample from 256->128, we concatenate skip (128) => total=256
        # up_block0 expects 256->128
        self.up_block0 = ResidualBlock(self.dims[1], self.dims[0], time_dim=self.time_dim*4)
        
        # transposed conv for upsampling
        self.upsample2 = nn.ConvTranspose2d(self.dims[2], self.dims[1], 4, 2, 1)
        self.upsample1 = nn.ConvTranspose2d(self.dims[1], self.dims[0], 4, 2, 1)

        # final output conv: 128->3 (RGB)
        self.final_conv = nn.Conv2d(self.dims[0], 3, kernel_size=3, padding=1)

    def forward(self, x, t):
        """
        x: [batch, 3, 32, 32]   (CIFAR-10 resolution)
        t: [batch, time_dim]    (e.g., 128)
        """
        # 1) time embedding => shape [batch, time_dim*4]
        emb = self.time_embed(t)

        # 2) Down/encode
        # initial conv
        h0 = self.init_conv(x)                              # [B, 128, 32, 32]
        h0 = self.down_block0(h0, emb)                      # [B, 128, 32, 32]

        h1 = self.downsample0(h0)                           # [B, 256, 16, 16]
        h1 = self.down_block1(h1, emb)                      # [B, 256, 16, 16]

        h2 = self.downsample1(h1)                           # [B, 512, 8, 8]
        h2 = self.down_block2(h2, emb)                      # [B, 512, 8, 8]

        # 3) "Bottom" block (optionally more res blocks here)
        h2u = self.up_block2(h2, emb)                       # [B, 512, 8, 8]

        # 4) Up/decode
        # upsample from 512->256
        h1u = self.upsample2(h2u)                           # [B, 256, 16, 16]
        # concatenate skip from h1: shape [B, 256, 16, 16]
        h1u = torch.cat([h1u, h1], dim=1)                   # => [B, 512, 16, 16]
        # now pass that to up_block1, which expects 512->256
        h1u = self.up_block1(h1u, emb)                      # => [B, 256, 16, 16]

        # upsample from 256->128
        h0u = self.upsample1(h1u)                           # => [B, 128, 32, 32]
        # skip from h0: shape [B, 128, 32, 32]
        h0u = torch.cat([h0u, h0], dim=1)                   # => [B, 256, 32, 32]
        # pass to up_block0, which expects 256->128
        h0u = self.up_block0(h0u, emb)                      # => [B, 128, 32, 32]

        # 5) final conv
        out = self.final_conv(h0u)                          # => [B, 3, 32, 32]
        return out


# if __name__ == "__main__":
#     # Quick shape test
#     net = SimpleUNet(time_dim=128)
#     x   = torch.randn(4, 3, 32, 32)   # batch=4
#     t   = torch.randn(4, 128)         # time embedding input
#     out = net(x, t)
#     print("Output:", out.shape)       # should be [4, 3, 32, 32]


#####################################################################################################################################################################


class ImprovedCIFAR10ScoreNetwork(nn.Module):
    """
    A deeper U-Net that goes down 4 times:
    3 -> 128 -> 256 -> 384 -> 512 -> 768
    and then back up. Ideal for 32x32 input.

    Each "down_blockX" keeps the same channels, 
    each "downsampleX" strides from dims[i] to dims[i+1].

    Then we do the reverse in the up path.
    """
    def __init__(self, time_embedding_dim=128):
        super().__init__()
        
        # Dimensions for each level
        self.dims = [128, 256, 384, 512, 768]
        
        # Time embedding
        self.time_dim = time_embedding_dim
        self.time_embed = TimeEmbedding(self.time_dim)
        
        # Initial conv from 3-channel RGB to 128
        self.init_conv = nn.Conv2d(in_channels=3, out_channels=self.dims[0], kernel_size=3, padding=1)
        

        # ----------  Residual blocks  ----------
        #   e.g.
        #   block0: 128->128
        #   downsample to 256
        #   block1: 256->256
        #   downsample to 512
        #   block2: 512->512
        #   ...
        self.down_block0 = ResidualBlock(self.dims[0], self.dims[0], time_dim=self.time_dim*4)
        self.down_block1 = ResidualBlock(self.dims[1], self.dims[1], time_dim=self.time_dim*4)
        self.down_block2 = ResidualBlock(self.dims[2], self.dims[2], time_dim=self.time_dim*4)
        self.down_block3 = ResidualBlock(self.dims[3], self.dims[3], time_dim=self.time_dim*4)
        self.down_block4 = ResidualBlock(self.dims[4], self.dims[4], time_dim=self.time_dim*4)
        
        # ----------  Downsampling layers  ----------
        # 2x2 stride-2 conv for downsampling
        # kernel_size = 4, stride = 2, padding = 1 ===> halves H and W
        self.downsample0 = nn.Conv2d(self.dims[0], self.dims[1], 4, 2, 1)
        self.downsample1 = nn.Conv2d(self.dims[1], self.dims[2], 4, 2, 1)
        self.downsample2 = nn.Conv2d(self.dims[2], self.dims[3], 4, 2, 1)
        self.downsample3 = nn.Conv2d(self.dims[3], self.dims[4], 4, 2, 1)
        

        # ----------  Up / Decode  ----------
        # We'll go back up:
        # We have 5 up-blocks, but we only need 4 upsample layers,
        # because the bottom block is "down_block4" => skip4 

        self.up_block4 = ResidualBlock(self.dims[4], self.dims[4], self.time_dim*4)
        self.upsample4 = nn.ConvTranspose2d(self.dims[4], self.dims[3], kernel_size=4, stride=2, padding=1)

        self.up_block3 = ResidualBlock(self.dims[3] + self.dims[3], self.dims[3], self.time_dim*4)
        self.upsample3 = nn.ConvTranspose2d(self.dims[3], self.dims[2], kernel_size=4, stride=2, padding=1)

        self.up_block2 = ResidualBlock(self.dims[2] + self.dims[2], self.dims[2], self.time_dim*4)
        self.upsample2 = nn.ConvTranspose2d(self.dims[2], self.dims[1], kernel_size=4, stride=2, padding=1)

        self.up_block1 = ResidualBlock(self.dims[1] + self.dims[1], self.dims[1], self.time_dim*4)
        self.upsample1 = nn.ConvTranspose2d(self.dims[1], self.dims[0], kernel_size=4, stride=2, padding=1)
        
        # Finally, after we cat skip0 => shape=128+128=256
        self.up_block0 = ResidualBlock(self.dims[0] + self.dims[0], self.dims[0], self.time_dim*4)

        
        # final output conv: 128->3 (RGB)
        self.final_conv = nn.Conv2d(self.dims[0], 3, kernel_size=3, padding=1)


    def forward(self, x, t, labels=None):
        """
        x: [batch, 3, 32, 32]
        t: [batch, time_dim], e.g. [batch, 128]

        Returns: [batch, 3, 32, 32]
        """

        # 1) Time embedding => [batch, time_dim*4]
        
        emb = self.time_embed(t)



        # 2) Down / Encode

        h0 = self.init_conv(x)                # => [B, 128, 32, 32]
        h0 = self.down_block0(h0, emb)        # => [B, 128, 32, 32]
        skip0 = h0

        h1 = self.downsample0(h0)             # => [B, 256, 16, 16]
        h1 = self.down_block1(h1, emb)        # => [B, 256, 16, 16]
        skip1 = h1

        h2 = self.downsample1(h1)             # => [B, 384, 8, 8]
        h2 = self.down_block2(h2, emb)        # => [B, 384, 8, 8]
        skip2 = h2

        h3 = self.downsample2(h2)             # => [B, 512, 4, 4]
        h3 = self.down_block3(h3, emb)        # => [B, 512, 4, 4]
        skip3 = h3

        h4 = self.downsample3(h3)             # => [B, 768, 2, 2]
        h4 = self.down_block4(h4, emb)        # => [B, 768, 2, 2]
        skip4 = h4  # The bottom-most features



        # 3) Up / Decode
  
        h4u = self.up_block4(skip4, emb)      # => [B, 768, 2, 2]
        # upsample to match skip3 resolution
        h3u = self.upsample4(h4u)             # => [B, 512, 4, 4]
        # cat skip3 => shape [B, 512+512=1024, 4, 4]
        h3u = torch.cat([h3u, skip3], dim=1)
        h3u = self.up_block3(h3u, emb)        # => [B, 512, 4, 4]

        # upsample3 => from 512->384
        h2u = self.upsample3(h3u)             # => [B, 384, 8, 8]
        # cat skip2 => [B, 384+384=768, 8, 8]
        h2u = torch.cat([h2u, skip2], dim=1)
        h2u = self.up_block2(h2u, emb)        # => [B, 384, 8, 8]

        # upsample2 => from 384->256
        h1u = self.upsample2(h2u)             # => [B, 256, 16, 16]
        # cat skip1 => [B, 256+256=512, 16, 16]
        h1u = torch.cat([h1u, skip1], dim=1)
        h1u = self.up_block1(h1u, emb)        # => [B, 256, 16, 16]

        # upsample1 => from 256->128
        h0u = self.upsample1(h1u)             # => [B, 128, 32, 32]
        # cat skip0 => [B, 128+128=256, 32, 32]
        h0u = torch.cat([h0u, skip0], dim=1)
        h0u = self.up_block0(h0u, emb)        # => [B, 128, 32, 32]

        # final 3x3 conv => 3 channels
        out = self.final_conv(h0u)            # => [B, 3, 32, 32]
        
        
        return out


# if __name__ == "__main__":
#     net = ImprovedCIFAR10ScoreNetwork(time_embedding_dim=128)

#     # Example input: batch_size=4, 3-channel image, 32x32
#     x   = torch.randn(4, 3, 32, 32)
#     # time-embedding input (e.g., if T=1000, you might use sinusoidal or something)
#     t   = torch.randn(4, 128)

#     out = net(x, t)
#     print("Output shape:", out.shape)  # Should be [4, 3, 32, 32]
#####################################################################################################################################################################

class CIFAR10ScoreNetwork(torch.nn.Module):
  def __init__(self):
    super().__init__()

    n_of_channels = 3  # RGB
    dimensions = [64, 128, 256, 512, 512] # appropriate dimensions

    self._convs = torch.nn.ModuleList([
      torch.nn.Sequential(
        torch.nn.Conv2d(3, dimensions[0], kernel_size=3, padding=1),  # Input channels: 3 for RGB
        torch.nn.ReLU(),
      ),
      torch.nn.Sequential(
        torch.nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample to 16x16
        torch.nn.Conv2d(dimensions[0], dimensions[1], kernel_size=3, padding=1),
        torch.nn.ReLU(),
      ),
      torch.nn.Sequential(
        torch.nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample to 8x8
        torch.nn.Conv2d(dimensions[1], dimensions[2], kernel_size=3, padding=1),
        torch.nn.ReLU(),
      ),
      torch.nn.Sequential(
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # Downsample to 4x4
        torch.nn.Conv2d(dimensions[2], dimensions[3], kernel_size=3, padding=1),
        torch.nn.ReLU(),
      ),
      torch.nn.Sequential(
        torch.nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample to 2x2
        torch.nn.Conv2d(dimensions[3], dimensions[4], kernel_size=3, padding=1),
        torch.nn.ReLU(),
      ),
    ])

    self._tconvs = torch.nn.ModuleList([
        torch.nn.Sequential(
            torch.nn.ConvTranspose2d(dimensions[4], dimensions[3], kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(),
        ),
        torch.nn.Sequential(
            torch.nn.ConvTranspose2d(dimensions[3] * 2, dimensions[2], kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(),
        ),
        torch.nn.Sequential(
            torch.nn.ConvTranspose2d(dimensions[2] * 2, dimensions[1], kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(),
        ),
        torch.nn.Sequential(
            torch.nn.ConvTranspose2d(dimensions[1] * 2, dimensions[0], kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(),
        ),
        torch.nn.Sequential(
            torch.nn.Conv2d(dimensions[0] * 2, dimensions[0], kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(dimensions[0], 3, kernel_size=3, padding=1),  # Output 3 channels for RGB !!!
        ),
      ])



  def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
      x = x.view(-1, 3, 32, 32)  # Ensure correct input shape for CIFAR-10

      # print(f"Input t shape in forward: {t.shape}")
      if t.ndim == 2:  # Expected input shape: [batch_size, embedding_dim]
        tt = t[:, :1, None, None].expand(-1, 3, 32, 32)  # Add spatial dimensions + use only the first channel of the embedding
      else:
        raise ValueError(f"Unexpected shape for t: {t.shape}")
      
      # print(tt.shape) # should be [batch_size, 1, 32, 32]

      # BUG FIX
      x2t = x + tt
      
      # Debugging print statements
      # print(f"x.shape: {x.shape}")       # Expect: [128, 3, 32, 32]
      # print(f"tt.shape: {tt.shape}")     # Expect: [128, 3, 32, 32]
      # print(f"x2t.shape: {x2t.shape}")   # Expect: [128, 3, 32, 32]

      # if labels is not None:
      #     labels = labels[..., None, None].expand(-1, 1, 32, 32)
      #     tt = tt + labels
      # # x2t = torch.cat((x, tt), dim=1)
      # print(x2t.shape)  # should be [batch_size, 4, 32, 32]   (4 bcs 3 from RGB + 1 for time channel)
      signal = x2t
      signals = []

      for i, conv in enumerate(self._convs):
          signal = conv(signal)
          if i < len(self._convs) - 1:
              signals.append(signal)
      
      # print(f"signal.shape: {signal.shape}, signals[-{i}].shape: {signals[-i].shape}")
      for i, tconv in enumerate(self._tconvs):
          if i == 0:
            signal = tconv(signal)
          else:
            # Ensure spatial dimensions match
            if signal.shape[-2:] != signals[-1].shape[-2:]:
              signals[-i] = F.interpolate(signals[-i], size=signal.shape[-2:], mode='bilinear', align_corners=True)
      
            signal = torch.cat((signal, signals[-i]), dim=1)
            signal = tconv(signal)
      
      return signal.view(-1, 3, 32, 32)