import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models.fno.fno_module import FNO2d
from src.models.lightning import LModule
from src.utils.lightning_utils import calc_concat_shape_dict


class model(LModule):
    def __init__(
        self,
        input_shape_dict,
        target_shape_dict,
        target_shape_dict_val,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        n_epochs: int = 100,
        modes: int = 16,
        width: int = 64,
        n_layers: int = 4,
        padding: int = 8,
        use_grid: bool = True,
        **kwargs,
    ):
        super().__init__(
            input_shape_dict,
            target_shape_dict,
            target_shape_dict_val,
            learning_rate=learning_rate,
            **kwargs,
        )
        self.save_hyperparameters()

        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs

        # inputs are concatenated along time; prediction covers the first target only
        self.channels_in = calc_concat_shape_dict(input_shape_dict)[0]
        self.channels_out = list(target_shape_dict.values())[0][0]

        self.fno = FNO2d(
            in_channels=self.channels_in,
            out_channels=self.channels_out,
            modes1=modes,
            modes2=modes,
            width=width,
            n_layers=n_layers,
            padding=padding,
            use_grid=bool(use_grid),
        )

    def forward(self, x, cond=None):
        return self.fno(x)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(opt, T_max=self.n_epochs)
        return [opt], [scheduler]
