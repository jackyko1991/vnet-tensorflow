# Draw Bounding Box on segmentation data

Draw bounding box enclosing the segmentation label.

![Bounding box with segmentation label](./bbox.png)

## Usage
To check the usage of the file, use `-h` argument:
```bash
$ python main.py -h
```

Quick example:
```bash
$ python main.py image.nii.gz label.nii.gz ./output -o 0.7 -m 0 -M 80
```

## To-do-list
- [ ]Sagittal and coronal slicing