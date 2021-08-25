import cv2
import torch.utils.data
import pytorch_lightning as pl

from dataset import Dataset
from model import Model

name = 'big_pulley'

rgba_template = cv2.imread(f'templates/{name}.png', cv2.IMREAD_UNCHANGED)

train_data = Dataset(name=name, data_slice=slice(6))
val_data = Dataset(name=name, data_slice=slice(6, 7), tfms=False)

loader_kwargs = dict(batch_size=1, num_workers=5, persistent_workers=True)
train_loader = torch.utils.data.DataLoader(dataset=train_data, shuffle=True, **loader_kwargs)
valid_loader = torch.utils.data.DataLoader(dataset=val_data, **loader_kwargs)

model = Model(rgba_template=rgba_template)

trainer = pl.Trainer(
    gpus=[0],
    callbacks=[
        pl.callbacks.EarlyStopping(monitor='val_loss', patience=3),
        pl.callbacks.ModelCheckpoint(monitor='val_loss')
    ],
)
trainer.fit(model=model, train_dataloader=train_loader, val_dataloaders=valid_loader)
