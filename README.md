# VNet Tensorflow
Tensorflow implementation of the V-Net architecture for medical imaging segmentation.

## Tensorflow implementation of V-Net
This is a Tensorflow implementation of the [V-Net](https://arxiv.org/abs/1606.04797) architecture used for 3D medical imaging segmentation. This code adopts the tensorflow graph from https://github.com/MiguelMonteiro/VNet-Tensorflow. The whole code covers training, evaluation and prediction modules for a 3D medical image segmentation.

### Visual Represetation of Network
This is an example of network this code implements.
![VNetDiagram](VNetDiagram.png)

### Features
- 3D data processing ready
- Augumented patching technique, required less image input for training
- Multichannel input and multiclass output
- Generic image reader with SimpleITK support (Currently only support .nii/.nii.gz format for convenience, easy to expand to DICOM)
- Medical image pre-post processing with SimpleITK filters
- Easy network replacement structure
- Sørensen and Jaccard similarity measurement as golden standard in medical image segmentation benchmarking
- Utilizing medical image headers to retrive space and orentation info after passthrough the network

## Development Progress

- [x] Training
- [x] Tensorboard visualization and logging
- [x] Resume training from checkpoint
- [x] Epoch training
- [ ] Evaluation from single data
- [ ] Multichannel input
- [x] Multiclass output
- [ ] C++ inference

## Usage
### Required Libraries
Known good dependencies
- Python 3.6
- Tensorflow 1.5 or above
- SimpleITK

### Folder Hierarchy
All training, testing and evaluation data should put in `./data`

    .
    ├── ...
    ├── data                      # All data
    │   ├── testing               # Put all testing data here
    |   |   ├── case1            
    |   |   |   ├── img.nii.gz    # Image For testing
    |   |   |   └── label.nii.gz  # Corresponding label for testing
    |   |   ├── case2
    |   |   ├──...
    │   ├── training              # Put all training data here
    |   |   ├── case1             # foldername for the cases is arbitar
    |   |   |   ├── img.nii.gz    # Image For testing
    |   |   |   └── label.nii.gz  # Corresponding label for testing
    |   |   ├── case2
    |   |   ├──...
    │   └── evaluation            # Put all evaluation data here
    ├── tmp
    |   ├── cktp                  # Tensorflow checkpoints
    |   └── log                   # Tensorboard logging folder
    ├── ...
    
If you wish to use image and label with filename other than `img.nii.gz` and `label.nii.gz`, please change the following values in `train.py`

```
image_filename = 'img.nii.gz'
label_filename = 'label.nii.gz'
```

In segmentation tasks, images and labels are always in pair, missing either one would terminate the training process.

### Training

You may run train.py with commandline arguments. To check usage, type ```python train.py -h``` in terminal to list all possible training parameters.

Available training parameters
```
  --batch_size: Size of batch
    (default: '1')
    (an integer)
  --checkpoint_dir: Directory where to write checkpoint
    (default: './tmp/ckpt')
  --data_dir: Directory of stored data.
    (default: './data')
  --decay_factor: Exponential decay learning rate factor
    (default: '0.01')
    (a number) (not implemented)
  --decay_steps: Number of epoch before applying one learning rate decay
    (default: '100')
    (an integer) (not implemented)
  --display_step: Display and logging interval (train steps)
    (default: '10')
    (an integer)
  --drop_ratio: Probability to drop a cropped area if the label is empty. All
    empty patches will be droped for 0 and accept all cropped patches if set to
    1
    (default: '0.5')
    (a number)
  --epochs: Number of epochs for training
    (default: '2000')
    (an integer)
  --init_learning_rate: Initial learning rate
    (default: '0.1')
    (a number)
  --min_pixel: Minimum non-zero pixels in the cropped label
    (default: '10')
    (an integer)
  --patch_layer: Number of layers in data patch
    (default: '128')
    (an integer)
  --patch_size: Size of a data patch
    (default: '128')
    (an integer)
  --save_interval: Checkpoint save interval (epochs)
    (default: '1')
    (an integer)
  --shuffle_buffer_size: Number of elements used in shuffle buffer
    (default: '5')
    (an integer)
  --tensorboard_dir: Directory where to write tensorboard summary
    (default: './tmp/tensorboard') (deprecated)
  --train_dir: Directory where to write training event logs
    (default: './tmp/train_log') (deprecated)
 ```

#### Image batch preparation
Typically medical image is large in size when comparing to natural images (height x widht x layers x modilty), where number of layers could up to hundred or thousands of slices. Also medical images are not bounded to unsigned char pixel type but accepts short, double or even float pixel type. This will consume large amount of GPU memories, which is a great barrier limiting the application of neural network in medical field.

Here we introduced a data augmentation skills that allows users to normalize and resample medical images in 3D sense. In train.py, you can access to trainTransforms/ testTransforms. Here we combined the advantage of tensorflow dataset api and SimpleITK (SITK) image processing toolkit together. Following is the preprocessing pipeline in SITK side to faciliate image augumentation with limited available memories.

1. Image Normalization (fit to 0-255)
2. Isotropic Resampling (adjustable size, in mm)
3. Paddinig (allow input image batch smaller than network input size to be trained)
4. Random Crop (randomly select a zone in the 3D medical image in exact size as network input)
5. Gaussian Noise

The preprocessing pipeline can be easily adjustable with following example code in `train.py`:
```
trainTransforms = [
                NiftiDataset.Normalization(),
                NiftiDataset.Resample(0.4356),
                NiftiDataset.Padding((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer)),
                NiftiDataset.RandomCrop((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer),FLAGS.drop_ratio,FLAGS.min_pixel),
                NiftiDataset.RandomNoise()
                ]
```

To write you own preprocessing pipeline, you need to modify the preprocessing classes in `NiftiDataset.py`

#### Tensorboard
In training stage, result can be visualized via Tensorboard. Run the following command:
```
tensorboard --logdir=./tmp/log
```

Once TensorBoard is running, navigate your web browser to ```localhost:6006``` to view the TensorBoard.

Note: ```localhost``` may need to change to localhost name by your own in newer version of Tensorboard.

### Evaluation
Under development

## Author
Jacky Ko jackkykokoko@gmail.com
