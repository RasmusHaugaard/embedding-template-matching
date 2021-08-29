import argparse

import numpy as np
import torch.utils.data
import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from .dataset import Dataset
from .model import Model
from . import utils
from .camera import CameraInfo

parser = argparse.ArgumentParser()
parser.add_argument('object_name')
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dataset-min-size', type=int, default=10)
parser.add_argument('--val-ratio', type=float, default=0.1)
parser.add_argument('--val-min', type=int, default=5)
args = parser.parse_args()
object_name = args.object_name
dataset_min_size = args.dataset_min_size
val_ratio = args.val_ratio
val_min = args.val_min

rgba_template, table_offset, obj_t_template, sym = utils.load_current_template(object_name)
annotation_fps, image_fps = utils.load_valid_annotation_and_image_fps(object_name)
cam_info = CameraInfo.load()

if len(annotation_fps) < dataset_min_size:
    print(f'Please annotate at least {dataset_min_size} images')
    quit()

# random train/val split
idxs = np.arange(len(annotation_fps))
np.random.shuffle(idxs)
annotation_fps, image_fps = np.array(annotation_fps)[idxs], np.array(image_fps)[idxs]
n_valid = max(val_min, int(np.ceil(len(annotation_fps) * val_ratio)))

data_kwargs = dict(obj_t_template=obj_t_template, K=cam_info.K)
train_data = Dataset(annotation_fps=annotation_fps[n_valid:], image_fps=image_fps[n_valid:], **data_kwargs)
val_data = Dataset(annotation_fps=annotation_fps[:n_valid], image_fps=image_fps[:n_valid], tfms=False, **data_kwargs)

loader_kwargs = dict(batch_size=1, num_workers=5, persistent_workers=True,
                     worker_init_fn=lambda *_: np.random.seed())
train_loader = torch.utils.data.DataLoader(dataset=train_data, shuffle=True, **loader_kwargs)
valid_loader = torch.utils.data.DataLoader(dataset=val_data, **loader_kwargs)

model = Model(rgba_template=rgba_template, sym=sym)

trainer = pl.Trainer(
    logger=TensorBoardLogger(utils.get_current_template_folder(object_name), 'models'),
    gpus=[args.gpu],
    callbacks=[
        pl.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience),
        pl.callbacks.ModelCheckpoint(monitor='val_loss')
    ],
)
trainer.fit(model=model, train_dataloader=train_loader, val_dataloaders=valid_loader)

# TODO: set current model to new model
