import random
import os
import torch
import wandb

import numpy as np

from torchvision.models import efficientnet_b0
from data_utils import CustomDataset, get_transforms
from training import train


def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')

    model_params = {}

    config = dict(
        seed=0xbebebe,
        lr=0.001,
        n_epochs=24,
        batch_size=64,
        model_params=model_params,
        device=device,
        log_iters=200,
        augmentations=get_transforms('train'),
    )
    run = wandb.init(
        project='DL01-XRay',
        config=config,
    )
    seed_everything(wandb.config.seed)

    cfg = wandb.config
    model = efficientnet_b0(num_classes=5)
    model.to(device)
    train_dataset = CustomDataset('data/dev_train.csv', get_transforms('train'))
    val_dataset = CustomDataset('data/dev_val.csv', get_transforms('val'))
    dataloaders = dict(
        train=torch.utils.data.DataLoader(train_dataset, cfg.batch_size, shuffle=True, pin_memory=True, num_workers=8),
        val=torch.utils.data.DataLoader(val_dataset, cfg.batch_size, shuffle=False, num_workers=8)
    )

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr,
        pct_start=0.2,
        total_steps=cfg.n_epochs * (len(train_dataset.data) // cfg.batch_size + 1)
    )
    
    train(
        model=model,
        dataloaders=dataloaders,
        optimizer=optimizer,
        criterion=criterion,
        n_epochs=cfg.n_epochs,
        scheduler=scheduler,
        device=device,
    )

    run.finish()
