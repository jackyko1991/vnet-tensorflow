# Draw Bounding Box on segmentation data

Draw bounding box enclosing the segmentation label.

![Bounding box with segmentation label](./bbox.png)

## Usage
To check the usage of the file, use `-h` argument:
```bash
$ python main.py -h
```

### Quick example
The following command will remap the image intensity from (0-80) to (0-255), with opacity 0.7 for segmentation label overlay
```bash
$ python main.py image.nii.gz label.nii.gz ./output -o 0.7 -m 0 -M 80
```

### Load class names from json file
To display the class name next to corresponding bounding box, you have to prepare the json file as following:

```json
// classnames.json
{
	"0": "background",
	"1": "ich"
}
```

**Note: Label 0 is a special value for background, where the bounding box will not be displayed.**

```bash
$ python main.py image.nii.gz label.nii.gz ./output -o 0.7 -m 0 -M 80 -c classnames.json
```

## To-do-list
- [ ]Sagittal and coronal slicing