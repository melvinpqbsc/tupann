import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models.afno.afno_module import AFNONet
from src.models.lightning import LModule
from src.utils.lightning_utils import calc_concat_shape_dict


class model(LModule):
    def __init__(
        self,
        input_shape_dict,
        target_shape_dict,
        target_shape_dict_val,
        learning_rate: float = 0.0005,
        weight_decay: float = 0.0,
        n_epochs: int = 100,
        patch_size: int = 8,
        embed_dim: int = 256,
        depth: int = 8,
        mlp_ratio: float = 4.0,
        num_blocks: int = 8,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
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
        input_shape = calc_concat_shape_dict(input_shape_dict)
        self.channels_in = input_shape[0]
        self.channels_out = list(target_shape_dict.values())[0][0]

        self.afno = AFNONet(
            img_size=tuple(input_shape[-2:]),
            patch_size=patch_size,
            in_channels=self.channels_in,
            out_channels=self.channels_out,
            embed_dim=embed_dim,
            depth=depth,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            num_blocks=num_blocks,
            sparsity_threshold=sparsity_threshold,
            hard_thresholding_fraction=hard_thresholding_fraction,
        )

    def forward(self, x, cond=None):
        return self.afno(x)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(opt, T_max=self.n_epochs)
        return [opt], [scheduler]
