# DROW: Deep Multiclass Detection in 2D Range ("Laser") Data

All code related to our work on detection in laser (aka lidar aka 2D range) data, covering both of the following papers:

- [DROW: Real-Time Deep Learning based Wheelchair Detection in 2D Range Data](http://arxiv.org/abs/1603.02636), henceforth called "v1".
- [Deep Person Detection in 2D Range Data](https://arxiv.org/abs/1804.02463), henceforth called "v2".

If you use anything provided here, please cite both papers in your work, see [citations below](#citations-and-thanks) below for the citation format.

# DROW v2 Detector Training and Evaluation

Code for training and evaluating DROW (v2) resides in various notebooks in the `v2` subfolder.
All notebooks are highly similar, and each notebook is used for obtaining one different curve in the paper.
Our final best model was obtained in `v2/Clean Final* [T=5,net=drow3xLF2p,odom=rot,trainval].ipynb`.

## What's new in v2?

Our second paper ("Deep Person Detection in 2D Range Data") adds the following:

- Annotations of persons in the dataset.
- Inclusion of odometry in the dataset.
- New network architecture.
- Publishing of pre-trained weights.
- Temporal integration of intormation in the model while respecting odometry.
- Comparison to well-tuned competing state-of-the-art person detectors. (In the paper only, not this repo.)

## Pre-trained weights

For DROW v2, we are able to provide the weights of the various models used in the paper here on GitHub in the [releases section](https://github.com/VisualComputingInstitute/DROW/releases).
The names correspond to the notebooks in the `v2` subfolder which were used to obtain these models.

When trying to load these weights, you might encounter the following error:
```
cuda runtime error (10) : invalid device ordinal
```
which can easily be solved by adding `map_location={'cuda:1': 'cuda:0'}` to the `load()` call, [additional details here](https://discuss.pytorch.org/t/saving-and-loading-torch-models-on-2-machines-with-different-number-of-gpu-devices/6666).


# DROW v1 Training and Evaluation

All code for training and evaluating DROW (v1) resides in the `v1/train-eval.ipynb` notebook, which you can open here on github or run for yourself.
Most, but not all, of this notebook was used during actual training of the final model for the paper.
While it was not intended to facilitate training your own model, it could be used for that after some careful reading.


## DROW v1 Detector ROS Node

A ROS detector node that can be used with a trained model and outputs standard `PoseArray` messages can be found in the [STRANDS repositories](https://github.com/strands-project/strands_perception_people/tree/indigo-devel/wheelchair_detector).


# DROW Laser Dataset

You can obtain our full dataset here on GitHub in the [releases section](https://github.com/VisualComputingInstitute/DROW/releases),
specifically the file [DROWv2-data.zip](https://github.com/VisualComputingInstitute/DROW/releases/download/v2/DROWv2-data.zip).

PLEASE read the v1 paper carefully before asking about the dataset, as we describe it at length in Section III.A.
Further details about the data storage format are given below in this README.

## Citations and Thanks

If you use this dataset or code in your work, please cite **both** the following papers:

> Beyer*, L., Hermans*, A., Leibe, B. (2017). DROW: Real-Time Deep Learning-Based Wheelchair Detection in 2-D Range Data. IEEE Robotics and Automation Letters, 2(2), 585-592.

BibTex:

```
@article{BeyerHermans2016RAL,
  title   = {{DROW: Real-Time Deep Learning based Wheelchair Detection in 2D Range Data}},
  author  = {Beyer*, Lucas and Hermans*, Alexander and Leibe, Bastian},
  journal = {{IEEE Robotics and Automation Letters (RA-L)}},
  year    = {2016}
}
```

> Beyer, L., Hermans, A., Linder T., Arras O.K., Leibe, B. (2018). Deep Person Detection in 2D Range Data. IEEE Robotics and Automation Letters, 3(3), 2726-2733.

BibTex:

```
@article{Beyer2018RAL,
  title   = {{Deep Person Detection in 2D Range Data}},
  author  = {Beyer, Lucas and Hermans, Alexander and Linder Timm and Arras Kai O. and Leibe, Bastian},
  journal = {{IEEE Robotics and Automation Letters (RA-L)}},
  year    = {2018}
}
```

Walker and wheelchair annotations by Lucas Beyer (@lucasb-eyer) and Alexander Hermans (@Pandoro),
and huge thanks to Supinya Beyer (@SupinyaMay) who created the person annotations for v2.

## License

The whole dataset is published under the MIT license, [roughly meaning](https://tldrlegal.com/license/mit-license) you can use it for whatever you want as long as you credit us.
However, we encourage you to contribute any extensions back, so that this repository stays the central place.

One exception to this licensing terms is the `reha` subset of the dataset, which we have converted from TU Ilmenau's data.
The [original dataset](https://www.tu-ilmenau.de/de/neurob/data-sets-code/people-detection-in-2d-laser-range-data/) was released under [CC-BY-NC-SA 3.0 Unported License](http://creativecommons.org/licenses/by-nc-sa/3.0/), and our conversion of it included herein keeps that license.

## Data Recording Setup

The exact recording setup is described in Section III.A of our v1 paper.
In short, it was recorded using a SICK S300 spanning 225° in 450 poins at 37cm height.
Recording happened in an elderly care facility, the test-set is completely disjoint from the train and validation sets, as it was recorded in a different aisle of the facility.

## Data Annotation Setup

Again, the exact setup is described in the v1-paper.
We used [this annotator](https://github.com/lucasb-eyer/laser-detection-annotator) to create the annotations.
Instead of all the laser scans, we annotate small batches throughout every sequence as follows:
A batch consists of 100 frames, out of which we annotate every 5th frame, resulting in 20 annotated frames per batch.
Within a sequence, we only annotate every 4th batch, leading to a total of 5 % of the laser scans being annotated.

## Dataset Use and Format

We highly recommend you use the `load_scan`, `load_dets`, and `load_odom` functions in `utils.py` for loading raw laser scans, detection annotations, and odometry data, respectively.
Please see the code's doc-comments or the DROW reference code for details on how to use them.
Please note that each scan (or frame), as well as detections and odometry, comes with a **sequence number that is only unique within a file, but not across files**.

### Detailed format description

If you want to load the files yourself regardless, this is their format:

One recording consists of a `.csv` file which contains all raw laser-scans, and one file per type of annotation, currently `.wc` for wheelchairs, `.wa` for walking-aids, and `.wp` for persons.

The `.csv` files contain one line per scan, the first value is the sequence number of that scan, followed by 450 floating-point values representing the distance at which the laser-points hit something.
There is at least one "magic value" for that distance at `29.96` which means N/A.
Note that the laser values go from left-to-right, i.e. the first value corresponds to the leftmost laser point, from the robot's point of view.

The files `.wa`/`.wc` again contain one line per frame and start with a sequence number which should be used to match the detections to the scan **in the corresponding `.csv` file only**.
Then follows a json-encoded list of `(r,φ)` pairs, which are the detections in polar coordinates.
For each detection, `r` represents the distance from the laser scanner and `φ ∈ [-π,π]` the angle in radians, zero being right in the front centered of the scanner ("up"), positive values going to the left and negative ones to the right.
There's an important difference between an empty frame and an un-annotated one:
An empty frame is present in the data as `123456,[]` and means that no detection of that type (person/wheelchair/walker) is present in the frame, whereas an un-annotated frame is simply not present in the file: the sequence number is skipped.

Finally, the `.odom2` files again contain one line per frame and start with a sequence number which should be used to match the odometry data to the scan **in the corresponding `.csv` file only**.
Then follows a comma-separated sequence of floating points, which correspond to `time` in seconds, `Tx` and `Ty` translation in meters, and `φ ∈ [-π,π]` orientation in radians of the robot's scanner.
These values are all relative to some arbitrary initial value which is not provided, so one should only work with differences.
