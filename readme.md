# Tabletop (3-dof) pose-estimation <br/> with latent space template matching

## install

## workflow

* configure scene
* configure objects
* collect initial dataset and train initial model
* run model (data is stored)
* improve model by annotating stored data

## thoughts and design decisions

**Objects (and thus templates) of different sizes**
are currently handled by employing a stride as a fixed ratio (1/20) of the template size. This makes the pose resolution
relative to the object size, which may or may not be desired.

**Angle resolution** is currently fixed at 45 discrete angles, (8 deg resolution).

**Full image coverage** is achieved by padding the embedding images, based on the template size. Padding with big
templates and the following convolution requires more memory and computation.

## TODO:

* get actual poses
    * scene config
        * object config
            * template height from table
            * rotation symmetry
            * template_t_obj
                * choose stable pose
* refactor to module
* training on multiple objects
* explicit template rotation symmetry
* ros action server (obj_id -> pose)
    * store all
* refinement search
    * maybe global / local template
* try smaller backbone
* augmentations
    * continuous rotation
    * cut-n-paste
    * random overlays
* write documentation