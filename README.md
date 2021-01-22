### This project is about ...

### The EfficientDet model [code](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch), [paper](https://arxiv.org/abs/1911.09070v7).

```
# this will clone EfficientDet too
git clone --recursive  https://github.com/luk400/computer-vision-project.git

# download pretrained weights into Yet-Another-EfficientDet-Pytorch/weights
python download_pretrained_weights.py

```

### Data 
**1.** Download the original data - it has to be in a folder data/ in this repo
* [data_SAR.zip(JKUDrive)](https://drive.jku.at/filr/public-link/file-download/ff8080827595a35701759e6ca83d481f/22192/-5528057403698270347/data_SAR.zip)
* [data_SAR.zip(WeTransfer)](https://wetransfer.com/downloads/cecc0a101b4dab1827aa7bedd3f640c820201106160324/e48651)
    
**2.** Create all the data necessary (images and json) + augmentation for training
```
# saves it by default under Yet-Another-EfficientDet-Pytorch/datasets/
run matlab preprocess_data.m
```

**3.** To start training, after installing the necessary dependencies, you will also need to create the yaml file needed for training inside the projects folder (Yet-Another-EfficientDet-Pytorch/projects/cv_project.yml) and edit it appropriately, in particular, for train.py to run, you will need to include: 

for a sample check this one [here](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/blob/master/projects/coco.yml)
```
touch Yet-Another-EfficientDet-Pytorch/projects/cv_project.yml

# required fields to be specified
project_name: cv_project
train_set: train
eval_set: eval
obj_list: ['person']
```

### Train 

```
python thermal_efficientdet_train.py --project cv_project --compound_coeff 0 --batch_size 8 --weights ./Yet-Another-EfficientDet-Pytorch/weights/efficientdet-d0.pth
```

### Test
```
python generate_evaluation_json.py --weights PATH_TO_CHECKPOINT

# this outputs the mAP for the jaon generated above
run matlab evaluate.m to 
```

### Visualize output
```
python visualize_test.py --weights ./Yet-Another-EfficientDet-Pytorch/logs/cv_project/weights-training-c2/efficientdet-d2_116_29500.pth 
```

### Record Results:

500 epochs on not relabelled data (but augmented) [tensorboard](https://tensorboard.dev/experiment/7rzp1jdRQlamQVo5IK759g/#scalars).

500 epochs on relabelled and augmented data [tensorboard]().

Last training on D1 and D2:{
"20210104-204331/Loss/train" resolution c==1, not so good as expected, it overfits - batch size 4 

[check this tensorboard plot](https://tensorboard.dev/experiment/bfokc6DfRbu7FpquNtGuaA/#scalars&runSelectionState=eyIyMDIwMTIwOS0xMjQ3NDIiOnRydWUsIjIwMjAxMjA5LTEyNDc0Mi9DbGFzc2ZpY2F0aW9uX2xvc3MvdHJhaW4iOnRydWUsIjIwMjAxMjA5LTEyNDc0Mi9DbGFzc2ZpY2F0aW9uX2xvc3MvdmFsIjp0cnVlLCIyMDIwMTIwOS0xMjQ3NDIvTG9zcy90cmFpbiI6dHJ1ZSwiMjAyMDEyMDktMTI0NzQyL0xvc3MvdmFsIjp0cnVlLCIyMDIwMTIwOS0xMjQ3NDIvUmVncmVzc2lvbl9sb3NzL3RyYWluIjp0cnVlLCIyMDIwMTIwOS0xMjQ3NDIvUmVncmVzc2lvbl9sb3NzL3ZhbCI6dHJ1ZSwiMjAyMTAxMDQtMjAyMTM2Ijp0cnVlLCIyMDIxMDEwNC0yMDIxMzYvQ2xhc3NmaWNhdGlvbl9sb3NzL3RyYWluIjp0cnVlLCIyMDIxMDEwNC0yMDIxMzYvQ2xhc3NmaWNhdGlvbl9sb3NzL3ZhbCI6dHJ1ZSwiMjAyMTAxMDQtMjAyMTM2L0xvc3MvdHJhaW4iOnRydWUsIjIwMjEwMTA0LTIwMjEzNi9Mb3NzL3ZhbCI6dHJ1ZSwiMjAyMTAxMDQtMjAyMTM2L1JlZ3Jlc3Npb25fbG9zcy90cmFpbiI6dHJ1ZSwiMjAyMTAxMDQtMjAyMTM2L1JlZ3Jlc3Npb25fbG9zcy92YWwiOnRydWUsIjIwMjEwMTA0LTIwNDEyOCI6dHJ1ZSwiMjAyMTAxMDQtMjA0MTI4L0NsYXNzZmljYXRpb25fbG9zcy90cmFpbiI6dHJ1ZSwiMjAyMTAxMDQtMjA0MTI4L0xvc3MvdHJhaW4iOnRydWUsIjIwMjEwMTA0LTIwNDEyOC9SZWdyZXNzaW9uX2xvc3MvdHJhaW4iOnRydWUsIjIwMjEwMTA0LTIwNDE1NCI6dHJ1ZSwiMjAyMTAxMDQtMjA0MTU0L0NsYXNzZmljYXRpb25fbG9zcy90cmFpbiI6dHJ1ZSwiMjAyMTAxMDQtMjA0MTU0L0xvc3MvdHJhaW4iOnRydWUsIjIwMjEwMTA0LTIwNDE1NC9SZWdyZXNzaW9uX2xvc3MvdHJhaW4iOnRydWUsIjIwMjEwMTA0LTIwNDMzMSI6dHJ1ZSwiMjAyMTAxMDQtMjA0MzMxL0NsYXNzZmljYXRpb25fbG9zcy90cmFpbiI6dHJ1ZSwiMjAyMTAxMDQtMjA0MzMxL0NsYXNzZmljYXRpb25fbG9zcy92YWwiOnRydWUsIjIwMjEwMTA0LTIwNDMzMS9Mb3NzL3RyYWluIjp0cnVlLCIyMDIxMDEwNC0yMDQzMzEvTG9zcy92YWwiOnRydWUsIjIwMjEwMTA0LTIwNDMzMS9SZWdyZXNzaW9uX2xvc3MvdHJhaW4iOnRydWUsIjIwMjEwMTA0LTIwNDMzMS9SZWdyZXNzaW9uX2xvc3MvdmFsIjp0cnVlfQ%3D%3D)

[check also this tensorboard](https://tensorboard.dev/experiment/V7w94kumQGyGU0SpPQqg5w/#scalars&runSelectionState=eyIyMDIwMTIwOS0xMjQ3NDIiOmZhbHNlLCIyMDIwMTIwOS0xMjQ3NDIvQ2xhc3NmaWNhdGlvbl9sb3NzL3RyYWluIjpmYWxzZSwiMjAyMDEyMDktMTI0NzQyL0NsYXNzZmljYXRpb25fbG9zcy92YWwiOmZhbHNlLCIyMDIwMTIwOS0xMjQ3NDIvTG9zcy90cmFpbiI6dHJ1ZSwiMjAyMDEyMDktMTI0NzQyL0xvc3MvdmFsIjp0cnVlLCIyMDIwMTIwOS0xMjQ3NDIvUmVncmVzc2lvbl9sb3NzL3RyYWluIjpmYWxzZSwiMjAyMDEyMDktMTI0NzQyL1JlZ3Jlc3Npb25fbG9zcy92YWwiOmZhbHNlLCIyMDIxMDEwNC0yMDQzMzEiOmZhbHNlLCIyMDIxMDEwNC0yMDQzMzEvQ2xhc3NmaWNhdGlvbl9sb3NzL3RyYWluIjpmYWxzZSwiMjAyMTAxMDQtMjA0MzMxL0NsYXNzZmljYXRpb25fbG9zcy92YWwiOmZhbHNlLCIyMDIxMDEwNC0yMDQzMzEvTG9zcy90cmFpbiI6dHJ1ZSwiMjAyMTAxMDQtMjA0MzMxL0xvc3MvdmFsIjp0cnVlLCIyMDIxMDEwNC0yMDQzMzEvUmVncmVzc2lvbl9sb3NzL3RyYWluIjpmYWxzZSwiMjAyMTAxMDQtMjA0MzMxL1JlZ3Jlc3Npb25fbG9zcy92YWwiOmZhbHNlLCIyMDIxMDEwNC0yMzM3NDkiOmZhbHNlLCIyMDIxMDEwNC0yMzM3NDkvQ2xhc3NmaWNhdGlvbl9sb3NzL3RyYWluIjpmYWxzZSwiMjAyMTAxMDQtMjMzNzQ5L0NsYXNzZmljYXRpb25fbG9zcy92YWwiOmZhbHNlLCIyMDIxMDEwNC0yMzM3NDkvTG9zcy90cmFpbiI6dHJ1ZSwiMjAyMTAxMDQtMjMzNzQ5L0xvc3MvdmFsIjp0cnVlLCIyMDIxMDEwNC0yMzM3NDkvUmVncmVzc2lvbl9sb3NzL3RyYWluIjpmYWxzZSwiMjAyMTAxMDQtMjMzNzQ5L1JlZ3Jlc3Npb25fbG9zcy92YWwiOmZhbHNlfQ%3D%3D)
}

| coefficient | pth_download | mAP (61 testset images) |
| :-------------: | :----------: |:-----------: |
| D0 | [efficientdet-d0_236_22500.pth](https://drive.google.com/file/d/1L6B7UIoqTOTTR6PVB56hEsqncJ_ds6Sx/view?usp=sharing) | 0.70 
| D1 | [efficientdet-d1_102_19500.pth](https://drive.google.com/file/d/17qVN4W4jZiDhPoOtdDODGNSuDIcR8MmU/view?usp=sharing) | 0.87 
| D2 | [efficientdet-d2_116_29500.pth](https://drive.google.com/file/d/1_dyAQgBbaPbkR6AJgiFbl6j-vYKnTevB/view?usp=sharing) | 0.87 


### TODO:

- [x] **improve data augmentation**
- [ ] git repo refactoring
- [X] do training for the others D*
- [X] make a matlab evaluation script for python results json comparison
- [X] modify bad labels using relabel_data.m
- [x] remove images with none labels from dataloaders (e.g. instances_train.json)
- [x] relabeling feature
- [x] improve image integration
- [x] fix data augmentation
