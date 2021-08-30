# Tabletop (3-dof) pose-estimation <br/> with latent space template matching

## install

## workflow

* configure camera, table and objects
* collect initial dataset
* train initial models
* improve models as necessary
    * run model and save images of failure cases
    * retrain model(s) and repeat

## scene folder structure

All files can be initialized using the scripts in this project, or manually.

* [scene folder]
    * cam_info.json (image topic, camera calibration)
    * cam_t_table.txt (table calibration)
    * images
        * [datetime].png
    * objects
        * [object name]
            * cad.stl
            * table_t_obj_stable.txt (stable pose)
            * current_template.txt (name of current template)
            * annotations
                * [datetime].txt (cam_t_obj for the corresponding image)
            * templates
                * [template name]
                    * obj_t_template.txt
                    * rgba_template.png
                    * sym.txt
                    * current_model.txt (version of current model)
                    * models
                        * [model version].ckpt (parameters of trained model)

## thoughts and design decisions

**Objects (and thus templates) of different sizes**
are currently handled by employing a stride as a fixed ratio (1/20) of the template size. This makes the pose resolution
relative to the object size, which may or may not be desired.

**Angle resolution** is currently fixed at 45 discrete angles, (8 deg resolution).

**Full image coverage** is achieved by padding the embedding images, based on the template size. Padding with big
templates and the following convolution requires more memory and computation.

## todo

* per object logs (also from infer) ?
    * use these in annotate
* allow to select a roi
    * project wide? / per object?
* allow updating the camera
    * new camera / camera calibration should not affect older annotations
    * should be able to use old annotations, as long as scale has not changed
* script to delete images / log images with no annotations
* monitor and log average recall
* better documentation

Maybe:

* multi-instance
* use same model for multiple objects
* explicit template rotation symmetry
* refinement search
    * maybe global / local template
* smaller backbone
* bias towards template smoothness (reg. template 2nd derivative)
* augmentations
    * cut-n-paste
    * random overlays
    * small scale variation
* sample validation set in a clever way
* proper gui