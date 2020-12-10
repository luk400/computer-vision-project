%# %%
% --------------- Example How to Load and Work with our thermal data ------
% This scripts computes the average precision and FP and TP for our
% testscenes!
addpath 'util'
clear all; clc; close all; % clean up!
iou_threshold = .10;
conf_threshold = .09;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BASELINE RESULTS OF PROFESSOR

results = readJSON('./assets/yolov4-tiny_integral_results.json');
[ filenames, detections, gts, ious, gtids] = parseResults( results );

averagePrecision = evaluateDetectionPrecision(detections,gts,iou_threshold);
[FP, TP, GT] = computeFpTpFn( detections, gts, iou_threshold, conf_threshold );

%%%%%%%%%%%%%
% OUR RESULTS

img_folder = './results/';
predictions_path = './assets/val_bbox_results.json';
annotations_path = './Yet-Another-EfficientDet-Pytorch/datasets/cv_project/annotations/instances_val.json';

[detections, gts] = create_detections_and_gts_tables(img_folder, predictions_path, annotations_path);

our_averagePrecision = evaluateDetectionPrecision(detections,gts,iou_threshold);
[our_FP, our_TP, our_GT] = computeFpTpFn( detections, gts, iou_threshold, conf_threshold );


%%%%%%%%%%%%
% comparison

sprintf("Baseline results of Professor\n AP: %.2f | FP/TP/GT: %d/%d/%d",...
    averagePrecision, FP, TP, GT)

sprintf("Our results\n AP: %.2f | FP/TP/GT: %d/%d/%d",...
    our_averagePrecision, our_FP, our_TP, our_GT)

%# %%
