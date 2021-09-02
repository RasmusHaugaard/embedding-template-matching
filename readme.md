# Tabletop (3-dof) pose-estimation <br/> with embedding template matching

## workflow overview

* initialize camera and table
* collect initial images (~ 15 images)
* for each object
    * annotate object in initial images
    * train model
    * repeat until satisfactory performance
        * run model
        * annotate failure cases
        * train new model

## install

Make sure [ROS](http://wiki.ros.org/ROS/Installation) is installed. Then:

```
$ cd emb-template
$ pip install -r requiremenst.txt
$ pip install git+https://github.com/eric-wieser/ros_numpy.git
$ pip install -e .
```

To be able to run the ROS action server, add `emb-template/ros/src/emb_template_ros` to your catkin workspace.

## getting started

A scene refers to a camera/table combination, a scene has its own folder, and all commands are expected to be run from
the scene folder.

### command overview

For all commands, add `--help` to see more options.

```
$ python -m emb_template.init_cam [ros image topic]
$ python -m emb_template.init_table
$ python -m emb_template.collect_images [image folder]
$ python -m emb_template.init_object [object name] [cad file] 
$ python -m emb_template.init_object_no_cad [object name] 
$ python -m emb_template.annotate [image folder] [object name] 
$ python -m emb_template.init_template [object name] [template name]
$ python -m emb_template.vis_data [object name]
$ python -m emb_template.train [object name]
$ python -m emb_template.infer [object name] 
$ python -m emb_template.action_server [server name]
$ python -m emb_template.action_server_test [server name] [object_name]
```

## scene folder structure

All files can be initialized manually or using commands.

* [scene folder]
    * cam_info.json
    * cam_t_table.txt
    * images
        * [datetime].png
    * objects
        * [object name]
            * cad.stl
            * table_t_obj_stable.txt
            * current_template.txt
            * annotations
                * [image dir]/[image datetime].txt
            * templates
                * [template name]
                    * obj_t_template.txt
                    * rgba_template.png
                    * sym.txt
                    * table_offset.txt
                    * models
    * logs
        * [object name]
            * [datetime].png
            * [datetime].[object name].png

## todo

* better documentation
* allow to select a roi (project wide/per object?)
* allow updating the camera
    * new camera / camera calibration should not affect older annotations
    * should be able to use old annotations, as long as scale has not changed
* script to delete images / log images with no annotations
* neighbourhood confidence (sum locally by 3x3x3 ones kernel with padding=0 for xy and padding=repeat for theta)
* better template center annotation (currently highly limited by pixel resolution)
    * only matters for
* annotation could use a faster renderer (not offline renderer)

Maybe:

* self-supervised representation learning on initial dataset as starting point
* multi-instance (detection)
* use same model for multiple objects
* explicit template rotation symmetry
* refinement
    * global / local template
    * regression
    * iterative refinement
* smaller backbone
* bias towards template smoothness (reg. template 2nd derivative)
* 3d-aware templates from cad models
* augmentations
    * cut-n-paste
    * random overlays
    * small scale variation
* sample validation set in a clever way
* proper gui