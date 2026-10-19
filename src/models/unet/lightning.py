import torch

from src.models.lightning import LModule
from src.models.unet.unet_parts import DoubleConv, Down, OutConv, Up
from src.utils.lightning_utils import calc_concat_shape_dict


class model(LModule):
    def __init__(
        self,
        input_shape_dict,
        target_shape_dict,
        learning_rate: float = 0.0001,
        weight_decay: float = 0.0,
        bilinear: bool = True,
        **kwargs,
    ):
        super().__init__(
            input_shape_dict,
            target_shape_dict,
            learning_rate=learning_rate,
            **kwargs,
        )
        self.save_hyperparameters()

        self.channels_in = calc_concat_shape_dict(input_shape_dict)[0]
        self.channels_out = calc_concat_shape_dict(target_shape_dict)[0]

        self.lr = learning_rate
        self.weight_decay = weight_decay

        self.bilinear = bilinear

        self.inc = DoubleConv(self.channels_in, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)
        self.outc = OutConv(64, self.channels_out)

    def forward(self, x, cond=None):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return [opt], []
