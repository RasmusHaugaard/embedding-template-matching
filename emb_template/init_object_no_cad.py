from pathlib import Path
import argparse

import trimesh
from transform3d import Transform

parser = argparse.ArgumentParser()
parser.add_argument('object_name')
parser.add_argument('--scale', type=float, required=True)
args = parser.parse_args()
object_name = args.object_name
scale = args.scale

obj_folder = Path('objects') / object_name
if obj_folder.exists():
    print('Object with that name already exists')
    quit()

mesh = trimesh.load_mesh(Path(__file__).parent.parent / 'arrow.stl')  # type: trimesh.Trimesh
mesh.apply_scale(scale)

obj_folder.mkdir(parents=True)
mesh.export(obj_folder / 'cad.stl')
Transform().save(obj_folder / 'table_t_obj_stable.txt')

print()
print(f'"{object_name}" initialized')
