# Hand Gesture Recognition 
Using HoloLens 2 and NVIDIA Jetson

## Project Overview
This repository involves research artifacts for the paper _“Did I Do Well? Personalized Assessment of Trainees' Performance in Augmented Reality-assisted Neurosurgical Training”_, submitted to **the 5th Annual Workshop on 3D Content Creation for Simulated Training in eXtended Reality, co-located with IEEE VR, 2024** by [Sarah Eom](https://sites.duke.edu/sangjuneom/), [Tiffany Ma](https://sites.duke.edu/tiffanyma/), [Tianyi Hu](http://hutianyi.tech/), Neha Vutakuri, Joshua Jackson, and [Maria Gorlatova](https://maria.gorlatova.com/). The repository contains implementations of collecting hand landmarks using HoloLens 2 and processing them through a deep learning model on an NVIDIA Jetson board for hand gesture recognition. This README provides information on how to set up and use the project.

## Table of Contents
- [Installation](#installation)
- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Data Collection](#data-collection)
- [Model Training](#model-training)
- [Model Inference](#model-inference)
- [Acknowledgments](#acknowledgments)

## Installation

Instructions for installing the project. Include steps for cloning the repo, setting up environments, etc.

```bash
git clone git@github.com:Duke-I3T-Lab/Hand-gesture-recognition.git
```

## Hardware Requirements

- HoloLens 2
- Edge server
    - Ubuntu PC with Nvidia GPU
    - NVIDIA Jetson (Tested on Xavier NX)

## Software Requirements
- Hololens2
    - Unity
    - Microsoft MixedReality Toolkit

- Edge server
    - python3 (tested on 3.8)
    - numpy (tested on 1.24.4)
    - pandas (tested on 2.0.3)
    - pytorch (tested on 2.0.1)

## Data Collection

To collect and label hand gesture data for training your customized hand gesture recognition model, you need to:
    Step 1
    Step 2
    ...

## Model Training

    Step 1
    Step 2
    ...

## Model Inference

    Step 1
    Step 2
    ...

# Citation
Please cite the following paper in your publications if this code helps your research.
```
@inproceedings{Eom24ARNeuro,
  title={Did I Do Well? Personalized Assessment of Trainees' Performance in Augmented Reality-assisted Neurosurgical Training},
  author={Eom, Sangjun and Ma, Tiffany and Hu, Tianyi and Vutakuri, Neha and Jackson, Joshua and Gorlatova, Maria},
  booktitle={Proc. IEEE Conference on Virtual Reality and 3D User Interfaces Abstracts and Workshops (VRW)},
  year={2024},
  organization={IEEE}
}
```
# Acknowledgments
The contributors of the code are [Tianyi Hu](http://hutianyi.tech/) and [Maria Gorlatova](https://maria.gorlatova.com/). For questions on this repository or the related paper, please contact Tianyi Hu at tianyi.hu [AT] duke [DOT] edu.

This work was supported in part by NSF grants CNS-2112562 and CNS-1908051, NSF CAREER Award IIS-2046072, and by a Thomas Lord Educational Innovation Grant.