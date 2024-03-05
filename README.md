# Hand Gesture Recognition 
Using HoloLens 2 and NVIDIA Jetson

## Project Overview
The repository contains implementations of customizable hand gesture recognition using a HoloLens2 and an edge server (e.g. Nvidia Jetson).
The HoloLens2 collects hand landmarks and send them to the edge server, where we convert the hand landmarks as multivariate time-series and use a deep learning model to classify the hand gestures.

This repository is developed from research artifacts for the paper _“Did I Do Well? Personalized Assessment of Trainees' Performance in Augmented Reality-assisted Neurosurgical Training”_, submitted to **the 5th Annual Workshop on 3D Content Creation for Simulated Training in eXtended Reality, co-located with IEEE VR, 2024** by [Sarah Eom](https://sites.duke.edu/sangjuneom/), [Tiffany Ma](https://sites.duke.edu/tiffanyma/), [Tianyi Hu](http://hutianyi.tech/), Neha Vutakuri, Joshua Jackson, and [Maria Gorlatova](https://maria.gorlatova.com/).  This README provides information on how to set up and use the project.

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
    - a NVIDIA Jetson (Tested on Xavier NX) 
    - Or, a Ubuntu PC with Nvidia GPU

## Software Requirements
- Hololens2
    - Unity
    - Microsoft MixedReality Toolkit

- Edge server
    - python3 (tested on 3.8)
    - numpy (tested on 1.24.4)
    - pandas (tested on 2.0.3)
    - pytorch (tested on 2.0.1)
    - scikit-learn (tested on 1.2.2)

## Data Collection

To collect and label hand gesture data for training your customized hand gesture recognition model, you need two devices and two personals. One (the user) wears the HoloLens 2 and performs the desired hand gestures, the other (the observer) observes the hand gestures and provide the ground truth labels for supervied model training.
    
- HoloLens2
    - Create a Unity project for HoloLens2
    - Add ```DataCollection/Hololens2/TCP_TrackingLogger.cs``` to a game object in Unity
    - Modify the ip address in ```DataCollection/Hololens2/TCP_TrackingLogger.cs``` based on your network settings.
    - Deploy the Unity project to your HoloLens2
    - Once you run the Unity app on HoloLens, it will record hand landmarks in the backend and saving them into a .csv file.
    - Meanwhile, the HoloLens2 will try to obtain ground truth labels from a edge server.

- Edge Server
    - Run ```python3 DataCollection/send_label.py``` on edge server before the user run the Unity app.
    - While the user wearing HoloLens2 is performing the desired gestures, the observer need to press the keyboard connected to the edge server for logging the groundtruth labels. 


## Model Training

- Put the labeled .csv files under ```ModelTraining/dataset/labeled```, and remove the first row of each csv file/
- Modify the python notebook ```ModelTraining/dataset/DataPreprocessing.ipynb``` based on your own label values.
- Run the python notebook ```ModelTraining/dataset/DataPreprocessing.ipynb``` to preprocess the dataset.
- Run the python notebook ```ModelTraining/dataset/GestureRecognition-Train.ipynb``` to train the model.

## Model Inference
- Create a Unity project for HoloLens2
- Add ```ModelInference/Hololens2/UDP_TrackingLogger.cs``` to a game object in Unity
- Modify the ip address in ```ModelInference/Hololens2/UDP_TrackingLogger.cs``` based on your network settings.
- Deploy the Unity project to your HoloLens2
- Put the model weights from ```ModelTraining/saved_model``` to ```ModelInference/Jetson/saved_model``` on your edge server
- Run ```ModelInference/Jetson/main_multiprocess``` on the edge server
- Once you run the Unity app on HoloLens, it will keep sending hand landmarks to an edge server over UDP.
- The edge server will return the estimated labels of the hand gestures to the HoloLens 2.

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
The contributors of the code are [Tianyi Hu](http://hutianyi.tech/) and [Maria Gorlatova](https://maria.gorlatova.com/). For questions on this repository or the related paper, please directly submit an issue on GitHub or contact Tianyi Hu at tianyi.hu [AT] duke [DOT] edu.

This work was supported in part by NSF grants CNS-2112562 and CNS-1908051, NSF CAREER Award IIS-2046072, and by a Thomas Lord Educational Innovation Grant.