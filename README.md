# VNet Tensorflow
Tensorflow implementation of the V-Net architecture for medical imaging segmentation.

## Tensorflow implementation of V-Net
This is a Tensorflow implementation of the [V-Net](https://arxiv.org/abs/1606.04797) architecture used for 3D medical imaging segmentation. This code adopts the tensorflow graph from https://github.com/MiguelMonteiro/VNet-Tensorflow. The repository covers training, evaluation and prediction modules for the (multimodal) 3D medical image segmentation in multiple classes.

### Visual Representation of Network
Here is an example graph of network this code implements. Channel depth may change owning to change in modality number and class number.
![VNetDiagram](VNetDiagram.png)

## Content
- [Features](#features)
- [Development Progress](#development-progress)
- [Usage](#usage)
  - [Required Libraries](#required-libraries)
  - [Folder Hierarchy](#folder-hierarchy)
  - [Training](#training)
    - [Image Batch Preparation](#image-batch-preparation)
    - [Tensorboard](#tensorboard)
    - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Evaluation](#evaluation)
- [Citations](#citations)
- [References](#references)

### Features
- 2D and 3D data processing ready
- Augmented patching technique, requires less image input for training
- Multichannel input and multiclass output
- Generic image reader with SimpleITK support (Currently only support .nii/.nii.gz format for convenience, easy to expand to DICOM, tiff and jpg format)
- Medical image pre-post processing with SimpleITK filters
- Easy network replacement structure
- Sørensen and Jaccard similarity measurement as golden standard in medical image segmentation benchmarking
- Utilizing medical image headers to retrive space and orientation info after passthrough the network

## Development Progress

- [x] Training
- [x] Tensorboard visualization and logging
- [x] Resume training from checkpoint
- [x] Epoch training
- [x] Evaluation from single data
- [x] Multichannel input
- [x] Multiclass output
- [x] Weighted DICE loss
- [x] C++ inference (Deprecated)
- [x] Preprocessing pipeline from external file
- [x] Docker image run
- [ ] Postprocessing pipeline after evaluation
- [ ] Hyperparameter tuning

## Usage
### Native Python Runtime
Tensorflow 1 is no longer supported officially by Google. If you wish to go for native run check the installation instruction given by [Nvidia-tensorflow](https://github.com/NVIDIA/tensorflow).

We recommend [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) for native python environment control as a faster replacement of Anaconda. Make sure you have install the Nvidia driver and accessible to GPU beforehand.

Quick checklist for the GPU versions:
- Ubuntu 20.04 or later (64-bit)
- Python 3.8
- Nvidia-Tensorflow 1.15.5
- CUDA 10.0

```bash
mamba create -n vnet-tensorflow python=3.8 -y
mamba activate vnet-tensorflow
pip install --user nvidia-pyindex
pip install --user nvidia-tensorflow[horovod]

git clone git@github.com:jackyko1991/vnet-tensorflow.git
cd vnet-tensorflow
pip install -r requirements.txt
```

### Docker Container
We also provide Dockerfile for quick setup purpose.
```bash
# pull docker file
docker pull vnet-tensorflow:latest

# training settings
DATA_DIR=<data-dir>
CONFIG_JSON=<config-file>
PIPELINE=<pipeline-file>
LOG_DIR=<log-dir>
CKPT_DIR=<ckpt-dir>

# run docker image
docker run -u $(id -u):$(id -g) --rm -p 6006:6006/tcp -v $DATA_DIR:/app/data -v $CONFIG_JSON:/app/configs -v $PIPELINE:/app/configs -v $LOG_DIR:/app/log -v $CKPT_DIR:/app/ckpt -it --entrypoint /bin/bash vnet-tensorflow:latest

# run training command
python main.py -p train --config_json /app/configs/config_docker.json
```

Change `<data-dir>`, `<config-file>`, `<pipeline-file>`, `<log-dir>` and  `<ckpt-dir>` accordingly. Sample configuration file can be found in [.configs/config_docker.json](./configs/config_docker.json) 

Container port 6006 is exposed for Tensorboard. If you wish to start Tensorboard service during training you may reuse the same image with a new running container:
```bash
LOG_DIR=<log-dir>

# run docker image
docker run -u $(id -u):$(id -g) --rm -p 6006:6006/tcp -v $LOG_DIR:/app/log-it --entrypoint /bin/bash vnet-tensorflow:latest

# run tensorboard
tensorboard --logdir /app/log
```

### Folder Hierarchy
All training, testing and evaluation data should put in `./data`

    .
    ├── ...
    ├── data                      # All data
    │   ├── testing               # Put all testing data here
    |   |   ├── case1            
    |   |   |   ├── img.nii.gz    # Image for testing
    |   |   |   └── label.nii.gz  # Corresponding label for testing
    |   |   ├── case2
    |   |   ├──...
    │   ├── training              # Put all training data here
    |   |   ├── case1             # foldername for the cases is arbitary
    |   |   |   ├── img.nii.gz    # Image for training
    |   |   |   └── label.nii.gz  # Corresponding label for training
    |   |   ├── case2
    |   |   ├──...
    │   └── evaluation            # Put all evaluation data here
    |   |   ├── case1             # foldername for the cases is arbitary
    |   |   |   └── img.nii.gz    # Image for evaluation
    |   |   ├── case2
    |   |   ├──...
    ├── tmp
    |   ├── cktp                  # Tensorflow checkpoints
    |   └── log                   # Tensorboard logging folder
    ├── ...
    
If you wish to use image and label with filename other than `img.nii.gz` and `label.nii.gz`, please change the following values in `config.json`

```json
"ImageFilenames": ["img.nii.gz"],
"LabelFilename": "label.nii.gz"
```

The network will automatically select 2D/3D mode by the length of `PatchShape` in `config.json`

In segmentation tasks, image and label are always in pair, missing either one would terminate the training process.

The code has been tested with [LiTS dataset](http://academictorrents.com/details/27772adef6f563a1ecc0ae19a528b956e6c803ce)

### Training

You may run train.py with commandline arguments. To check usage, type ```python main.py -h``` in terminal to list all possible training parameters.

Available training parameters
```console
  -h, --help            show this help message and exit
  -v, --verbose         Show verbose output
  -p [train evaluate], --phase [train evaluate]
                        Training phase (default= train)
  --config_json FILENAME
                        JSON file for model configuration
  --gpu GPU_IDs         Select GPU device(s) (default = 0)
 ```

The program will read the configuration from `config.json`. Modify the necessary hyperparameters to suit your dataset.

Note: You should always set label 0 as the first `SegmentationClasses` in `config.json`. Current model will only run properly with at least 2 classes.

The software will automatically determine run in 2D or 3D mode according to rank of `PatchShape` in `config.json`

#### Image batch preparation
Typically medical image is large in size when comparing with natural images (height x width x layers x modality), where number of layers could up to hundred or thousands of slices. Also medical images are not bounded to unsigned char pixel type but accepts short, double or even float pixel type. This will consume large amount of GPU memories, which is a great barrier limiting the application of neural network in medical field.

Here we introduce serveral data augmentation skills that allow users to normalize and resample medical images in 3D sense. In `train.py`, you can access to `trainTransforms`/`testTransforms`. For general purpose we combine the advantage of tensorflow dataset api and SimpleITK (SITK) image processing toolkit together. Following is the preprocessing pipeline in SITK side to facilitate image augmentation with limited available memories.

1. Image Normalization (fit to 0-255)
2. Isotropic Resampling (adjustable size, in mm)
3. Padding (allow input image batch smaller than network input size to be trained)
4. Random Crop (randomly select a zone in the 3D medical image in exact size as network input)
5. Gaussian Noise

The preprocessing pipeline can easily be adjusted with following example code in `./pipeline/pipeline2D.yaml`:
```yaml
train:
  # it is possible to first perform normalization on 3D image first, e.g. image normalization using statistical values 
  3D:

  2D:
    - name: "ManualNormalization"
      variables: 
        windowMin: 0
        windowMax: 600
    - name: "Resample"
      variables: 
        voxel_size: [0.75, 0.75]
    - name: "Padding"
      variables: 
        output_size: [256,256]
    - name: "RandomCrop"
      variables: 
        output_size: [256,256]
```

For 2D image training mode, you need to provide transforms in 2D and 3D separately.

To write you own preprocessing pipeline, you need to modify the preprocessing classes in `./pipeline/NiftiDataset2D.py` or `./pipeline/NiftiDataset3D.py`.

Additional preprocessing classes (incomplete list, check `./pipeline/NiftiDataset2D.py` or `./pipeline/NiftiDataset3D.py` for full list):
- StatisticalNormalization
- Reorient (take care on the direction when performing evaluation)
- Invert
- ConfidenceCrop (for very small volume like cerebral microbleeds, alternative of RandomCrop)
- Deformations:
  The deformations are following SITK deep learning data augmentation documentations, will be expand soon.
  Now contains:
  - BSplineDeformation 

  **Hint: Directly apply deformation is slow. Instead you can first perform cropping with a larger than patch size region then with deformation, then crop to actual patch size. If you apply deformation to exact training size region, it will create black zone which may affect final training accuracy.**
  
#### Tensorboard
In training stage, result can be visualized via Tensorboard. Run the following command:
```console
tensorboard --logdir=./tmp/log
```

Once TensorBoard is running, navigate your web browser to ```localhost:6006``` to view the TensorBoard.

Note: ```localhost``` may need to change to localhost name by your own in newer version of Tensorboard.

#### Hyperparameter Tuning

### Evaluation
To evaluate image data, first place the data in folder ```./data/evaluate```. Each image data should be placed in separate folder as indicated in the folder hierarchy

There are several parameters you need to set in order manually in `EvaluationSetting` session of `./config/config.json`.

Run `main.py -p evaluate --config_json ./config/config.json` after you have modified the corresponding variables. All data in `./data/evaluate` will be iterated. Segmented label is named specified ini `LabelFilename` and output in same folder of the respective input image files.

Note that you should keep preprocessing pipeline similar to the one during training, but without random cropping and noise. You may take reference to `evaluate` session in `./pipeline/pipeline2D.yaml`.
#### Post Processing (To be updated)

## C++ Inference (Deprecated for newer version of Tensorflow)
We provide a C++ inference example under directory [cxx](./cxx). For C++ implementation, please follow the guide [here](./cxx/README.md)

## Citations
Use the following Bibtex if you need to cite this repository:
```bibtex
@misc{jackyko1991_vnet_tensorflow,
  author = {Jacky KL Ko},
  title = {Implementation of vnet in tensorflow for medical image segmentation},
  howpublished = {\url{https://github.com/jackyko1991/vnet-tensorflow}},
  year = {2018},
  publisher={Github},
  journal={GitHub repository},
}

@inproceedings{milletari2016v,
  title={V-net: Fully convolutional neural networks for volumetric medical image segmentation},
  author={Milletari, Fausto and Navab, Nassir and Ahmadi, Seyed-Ahmad},
  booktitle={3D Vision (3DV), 2016 Fourth International Conference on},
  pages={565--571},
  year={2016},
  organization={IEEE}
}

@misc{MiguelMonteiro_VNet_Tensorflow,
  author = {Miguel Monteiro},
  title = {VNet-Tensorflow: Tensorflow implementation of the V-Net architecture for medical imaging segmentation.},
  howpublished = {\url{https://github.com/MiguelMonteiro/VNet-Tensorflow}},
  year = {2018},
  publisher={Github},
  journal={GitHub repository},
}
```

## References:
- SimpleITK guide on deep learning data augmentation:
https://simpleitk.readthedocs.io/en/master/Documentation/docs/source/fundamentalConcepts.html
https://simpleitk.github.io/SPIE2018_COURSE/data_augmentation.pdf
https://simpleitk.github.io/SPIE2018_COURSE/spatial_transformations.pdf

## Author
Jacky Ko jackkykokoko@gmail.com
