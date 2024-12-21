import torch
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