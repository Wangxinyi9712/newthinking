from src.utils.config import load_config
from src.data.datasets import build_dataloaders
from src.models.seg_model import HybridUNet
from src.engine.trainer import MeanTeacherTrainer
from src.utils.seed import set_seed

def _build_model(cfg):

    model = HybridUNet(
        in_channels=cfg.model.get("in_channels", 4),
        out_channels=cfg.model.get("out_channels", 1),
        channels=cfg.model.get("channels", [16, 32, 64, 128]),
        use_transformer=cfg.model.get("use_transformer", False),
    )

    return model

def main():

    cfg = load_config("src/configs/brats_group_e.yaml")

    set_seed(cfg.train.get("seed", [0])[0])

    loaders = build_dataloaders(cfg.data)

    model = _build_model(cfg)

    trainer = MeanTeacherTrainer(model, cfg)

    for epoch in range(cfg.train["epochs"]):

        for batch_l, batch_u in zip(loaders["labeled"], loaders["unlabeled"]):

            loss = trainer.train_step(batch_l, batch_u)

        print("epoch:", epoch, "loss:", loss)


if __name__ == "__main__":
    main()