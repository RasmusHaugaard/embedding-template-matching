import numpy as np
import cv2
import torch.utils.data
import pytorch_lightning as pl

from dataset import Dataset
from model import Model
import utils

name = 'big_pulley'

rgba_template = utils.load_rgba_template(name)

train_data = Dataset(name=name, data_slice=slice(20))
val_data = Dataset(name=name, data_slice=slice(20, 23), tfms=False)

loader_kwargs = dict(batch_size=1, num_workers=5, persistent_workers=True,
                     worker_init_fn=lambda *_: np.random.seed())
train_loader = torch.utils.data.DataLoader(dataset=train_data, shuffle=True, **loader_kwargs)
valid_loader = torch.utils.data.DataLoader(dataset=val_data, **loader_kwargs)

model = Model(rgba_template=rgba_template)

trainer = pl.Trainer(
    gpus=[0],
    callbacks=[
        pl.callbacks.EarlyStopping(monitor='val_loss', patience=10),
        pl.callbacks.ModelCheckpoint(monitor='val_loss')
    ],
)
trainer.fit(model=model, train_dataloader=train_loader, val_dataloaders=valid_loader)
