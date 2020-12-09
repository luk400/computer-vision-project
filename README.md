### TODO:

- [ ] improve image integration
- [ ] add more functions for data augmentation
- [X] modify bad labels using relabel_data.m
- [x] remove images with none labels from dataloaders (e.g. instances_train.json)
- [x] relabeling feature
- [ ] ~~fix data augmentation~~

### Steps to use it

The EfficientDet model [code](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch), [paper]().

```
# this will clone EfficientDet too
git clone --recursive  https://github.com/luk400/computer-vision-project.git

# download pretrained weights into Yet-Another-EfficientDet-Pytorch/weights
python download_pretrained_weights.py

```

Download the relabelled data [here](https://drive.google.com/file/d/1NCCOX-WBd89zz3MrhGSlILYIt56ktio1/view?usp=sharing)

### Data augmentation, relabelling, etc ...

Simply put the data-folder for the CV-project, then execute the preprocess_data.m script with matlab. 
This should create the folder Yet-Another-EfficientDet-Pytorch/datasets/ with all the appropriate subfolders, images and json-files needed for training. 

To start training, after installing the necessary dependencies, you will also need to create the yml file needed for training inside the projects folder (Yet-Another-EfficientDet-Pytorch/projects/cv_project.yml) and edit it appropriately, in particular, for train.py to run, you will need to include: 

```
project_name: cv_project
train_set: train
val_set: val
obj_list: ['person']
...
```

### Train and test

You should then be able to train the model using, e.g.:
```
python train.py -c 1 -p cv_project --batch_size 8 --lr 1e-5
```

and evaluate it on test data using:
```
python coco_eval.py -p cv_project -c 1 -w ./logs/cv_project/<your-saved-weights>.pth --cuda False
```


### Files description:

```
. - root dir computer-vision-project
├── assets
├── custom_effientdet_train.py
├── cv_eval.py
├── data -> ../data
├── debug_data.py
├── download_pretrained_weights.sh
├── drone_dataset.py
├── helpers.py
├── logs
├── modified_labels
├── preprocess_data.asv
├── preprocess_data.m
├── README.md
├── relabel_data.m - Modifies the label file in the results folder"
├── results - This folder stores all data and it's used for data agmentation
├── results_relabelled.zip
├── util
├── visualize_dataset.py - Shows the data used for training (you can see the test too)
├── visualize_test.py
└── Yet-Another-EfficientDet-Pytorch

```

