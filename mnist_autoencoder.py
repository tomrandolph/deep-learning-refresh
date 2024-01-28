from torch import nn


class Encoder(nn.Module):
    def __init__(self) -> None:
        super(Encoder, self).__init__()

    def forward(self, x):
        return x


class Decoder(nn.Module):
    def __init__(self) -> None:
        super(Decoder, self).__init__()

    def forward(self, x):
        return x


class AutoEncoder(nn.Module):
    def __init__(self) -> None:
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)
