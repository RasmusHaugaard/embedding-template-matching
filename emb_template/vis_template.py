import argparse

from . import vis
from .model import Model
from . import utils

parser = argparse.ArgumentParser()
parser.add_argument('object_name')
args = parser.parse_args()
object_name = args.object_name

rgba_template, *_, sym = utils.load_current_template(object_name)

model = Model.load_from_checkpoint(
    # TODO: load *current* model
    utils.latest_checkpoint(object_name),
    rgba_template=rgba_template, sym=sym
)
vis.show_template(model, rotate=True)
