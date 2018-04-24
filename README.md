***IMPORTANT NOTE:*** The extension to persons for our new paper [Deep Person Detection in 2D Range Data](https://arxiv.org/abs/1804.02463) will be added at latest upon acceptance.

# DROW
All code related to the ["DROW: Real-Time Deep Learning based Wheelchair Detection in 2D Range Data"](http://arxiv.org/abs/1603.02636) paper.


# DROW Detector ROS Node

We will add here a ROS detector node that can be used with a trained model and outputs standard `PoseArray` messages.
Until we add it here, you can already get a sneak-peek in the [STRANDS repositories](https://github.com/strands-project/strands_perception_people/tree/indigo-devel/wheelchair_detector).


# DROW Training and Evaluation

All code for training and evaluating DROW resides in the `train-eval.ipynb` notebook, which you can open here on github or run for yourself.
Most, but not all, of this notebook was used during actual training of the final model for the paper.
While it was not intended to facilitate training your own model, it could be used for that after some careful reading.


# DROW Laser Dataset

You can obtain our full dataset here on GitHub in the [releases section](https://github.com/VisualComputingInstitute/DROW/releases), specifically the file [DROW-data.zip](https://github.com/VisualComputingInstitute/DROW/releases/download/data/DROW-data.zip).

The laser-based detection dataset released with the paper "DROW: Real-Time Deep Learning based Wheelchair Detection in 2D Range Data" available at https://arxiv.org/abs/1603.02636 and published at ICRA'17.

PLEASE read the paper carefully before asking about the dataset, as we describe it at length in Section III.A.

## Citations

If you use this dataset in your work, please cite the following:

> Beyer, L., Hermans, A., & Leibe, B. (2017). DROW: Real-Time Deep Learning-Based Wheelchair Detection in 2-D Range Data. IEEE Robotics and Automation Letters, 2(2), 585-592.

BibTex:

```
@article{BeyerHermans2016RAL,
  title   = {{DROW: Real-Time Deep Learning based Wheelchair Detection in 2D Range Data}},
  author  = {Beyer*, Lucas and Hermans*, Alexander and Leibe, Bastian},
  journal = {{IEEE Robotics and Automation Letters (RA-L)}},
  year    = {2016}
}
```

## License

The whole dataset is published under the MIT license, [roughly meaning](https://tldrlegal.com/license/mit-license) you can use it for whatever you want as long as you credit us.
However, we encourage you to contribute any extensions back, so that this repository stays the central place.

One exception to this licensing terms is the `reha` subset of the dataset, which we have converted from TU Ilmenau's data.
The [original dataset](https://www.tu-ilmenau.de/de/neurob/data-sets-code/people-detection-in-2d-laser-range-data/) was released under [CC-BY-NC-SA 3.0 Unported License](http://creativecommons.org/licenses/by-nc-sa/3.0/), and our conversion of it included herein keeps that license.

## Data Recording Setup

The exact recording setup is described in Section III.A of our paper.
In short, it was recorded using a SICK S300 spanning 225° in 450 poins at 37cm height.
Recording happened in an elderly care facility, the test-set is completely disjoint from the train and validation sets, as it was recorded in a different aisle of the facility.

## Data Annotation Setup

Again, the exact setup is described in the paper.
We used [this annotator](https://github.com/lucasb-eyer/laser-detection-annotator) to create the annotations.
Instead of all the laser scans, we annotate small batches throughout every sequence as follows:
A batch consists of 100 frames, out of which we annotate every 5th frame, resulting in 20 annotated frames per batch.
Within a sequence, we only annotate every 4th batch, leading to a total of 5 % of the laser scans being annotated.

## Dataset Use and Format

We highly recommend you use the `load_scan` and `load_dets` functions in `utils.py` for loading raw laser scans and detection annotations, respectively.
Please see the code's doc-comments or the DROW reference code for details on how to use them.
Please note that each scan (or frame), as well as detections, comes with a **sequence number that is only unique within a file, but not across files**.

### Detailed format description

If you want to load the files yourself regardless, this is their format:

One recording consists of a `.csv` file which contains all raw laser-scans, and one file per type of annotation, currently `.wc` for wheelchairs and `.wa` for walking-aids, with more to come.

The `.csv` files contain one line per scan, the first value is the sequence number of that scan, followed by 450 floating-point values representing the distance at which the laser-points hit something.
There is at least one "magic value" for that distance at `29.96` which means N/A.
Note that the laser values go from left-to-right, i.e. the first value corresponds to the leftmost laser point, from the robot's point of view.

The files `.wa`/`.wc` again contain one line per frame and start with a sequence number which should be used to match the detections to the scan **in the corresponding `.csv` file only**.
Then follows a json-encoded list of `(r,φ)` pairs, which are the detections in polar coordinates.
For each detection, `r` represents the distance from the laser scanner and `φ ∈ [-π,π]` the angle in radians, zero being right in the front centered of the scanner ("up"), positive values going to the left and negative ones to the right.
There's an important difference between an empty frame and an un-annotated one:
An empty frame is present in the data as `123456,[]` and means that no detection of that type (person/wheelchair/walker) is present in the frame, whereas an un-annotated frame is simply not present in the file: the sequence number is skipped.
