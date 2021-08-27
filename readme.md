# Tabletop (3-dof) pose-estimation <br/> with latent space template matching

## install

## workflow

* configure camera, table and objects
* collect initial dataset
* train initial models
* improve models as necessary
    * run model and save images of failure cases
    * retrain model(s) and repeat

## thoughts and design decisions

**Objects (and thus templates) of different sizes**
are currently handled by employing a stride as a fixed ratio (1/20) of the template size. This makes the pose resolution
relative to the object size, which may or may not be desired.

**Angle resolution** is currently fixed at 45 discrete angles, (8 deg resolution).

**Full image coverage** is achieved by padding the embedding images, based on the template size. Padding with big
templates and the following convolution requires more memory and computation.

## todo

mvp:

* get 3D poses
    * object config
    * template_t_obj
* ros action server (obj_name -> cam_t_pose)
    * dynamically load model
    * store all images and predictions
* script to delete (old) images with no annotations

later:

* multi-instance
* same model for multiple objects
* explicit template rotation symmetry
* refinement search
    * maybe global / local template
* try smaller backbone
* try bias towards template smoothness (reg. template 2nd derivative)
* augmentations
    * continuous rotation
    * cut-n-paste
    * random overlays
* better documentation