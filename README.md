Don't forget the recursive :) This is to clone submodules, in our case the EfficientDet implementation we are going to use, taken from [here](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch).

```
git clone --recursive  https://github.com/luk400/computer-vision-project.git
```

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


You should then be able to train the model using, e.g.:
```
python train.py -c 1 -p cv_project --batch_size 8 --lr 1e-5
```

and evaluate it on test data using:
```
python coco_eval.py -p cv_project -c 1 -w ./logs/cv_project/<your-saved-weights>.pth --cuda False
```

NOTE: currently the validation set used when training is the provided test set, this should of course not be used as a validation set and is just temporary, because I needed to specify a validation set to try out if the train.py works. 
Also, currently when running coco_eval.py, I get the error 'model does not provide any valid output', which (after looking at the code) means that the model doesn't predict any bounding boxes. I'm not yet sure if this is just because I didn't even train for a single epoch yet and because the model parameters are completely arbitrary, or because there's a problem with the dataset specification. This will be the next step to figure out.
