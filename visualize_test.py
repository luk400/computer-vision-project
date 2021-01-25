import argparse
import os
import shutil
from pathlib import Path

import cv2
import moviepy.video.io.ImageSequenceClip
import numpy as np
import torch
from torch.backends import cudnn

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import aspectaware_resize_padding
from utils.utils import invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, \
    plot_one_box

__all__ = ["preprocess", "display", "get_testset_imgs_filepaths", "coefficient_from_weights_filepath",
           "prepare_inputs_for_model", "eval", "read_images"]


def preprocess(image_filepath, max_size=512, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) -> (
        list, list, list):
    ori_imgs = [cv2.imread(img_path)[..., ::-1] for img_path in image_filepath]
    normalized_imgs = [(img / 255 - mean) / std for img in ori_imgs]
    imgs_meta = [aspectaware_resize_padding(img, max_size, max_size, means=None) for img in normalized_imgs]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_imgs, framed_imgs, framed_metas


def display(preds, imgs, compound_coef, obj_list=None, imshow=True, imwrite=False, debug=False):
    if obj_list is None:
        obj_list = ['person']
    color_list = standard_to_bgr(STANDARD_COLORS)

    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            if debug:
                cv2.imshow('img', imgs[i])
                cv2.waitKey(0)
            continue

        imgs[i] = imgs[i].copy()

        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])
            plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj, score=score,
                         color=color_list[get_index_label(obj, obj_list)])

        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)

        if imwrite:
            os.system("mkdir -p ./assets/predictions")
            cv2.imwrite(f'./assets/predictions/img_inferred_d{compound_coef}_this_repo_{i}.jpg', imgs[i])

    if imwrite:
        image_folder = './assets/predictions'
        image_files = [image_folder + '/' + img for img in os.listdir(image_folder) if img.endswith(".jpg")]
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=1)
        clip.write_videofile('./assets/predictions_testset.mp4')


def get_testset_imgs_filepaths(project_testset_folder_path):
    imgs_filepath = [os.path.join(project_testset_folder_path, i) for i in os.listdir(project_testset_folder_path)]
    return imgs_filepath


def coefficient_from_weights_filepath(weight_filepath: str) -> (int, str):
    """ example" split this path efficientdet-d2_216_54901.pth" to get the <2>"""
    _, tail_path = os.path.split(weight_filepath)
    weights_identifier = tail_path.split("-")[1]
    rm_extention_identifier = weights_identifier.split(".")[0]
    coefficient = rm_extention_identifier[1]
    return int(coefficient), rm_extention_identifier


def prepare_inputs_for_model(use_cuda: bool):
    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    all_testset_imgs_preprocessed = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    split_input_images = all_testset_imgs_preprocessed.split(3)

    return split_input_images


def eval(pretrained_weights: Path, inputs_splitted_into_lists: list, compound_coef: int, use_cuda: bool) -> list:
    threshold = 0.2
    iou_threshold = 0.2
    # replace this part with your project's anchor config
    anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=1, ratios=anchor_ratios, scales=anchor_scales)
    model.load_state_dict(torch.load(pretrained_weights, map_location='cpu'))
    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model = model.cuda()
    if use_float16:
        model = model.half()

    predictions = []

    for inputs_split in inputs_splitted_into_lists:
        with torch.no_grad():
            features, regression, classification, anchors = model(inputs_split)

            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()

            out = postprocess(inputs_split,
                              anchors, regression, classification,
                              regressBoxes, clipBoxes,
                              threshold, iou_threshold)

            predictions += out

    return predictions


def read_images(imgs_filepath, compound_coefficient):
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = input_sizes[compound_coefficient]
    ori_imgs, framed_imgs, framed_metas = preprocess(imgs_filepath, max_size=input_size)
    return ori_imgs, framed_imgs, framed_metas


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--project', type=str, default='cv_project', help='project file that contains parameters')
    ap.add_argument('-w', '--weights', type=str, default=None, help='/path/to/weights')
    ap.add_argument('-d', '--data-folder', default="./Yet-Another-EfficientDet-Pytorch/datasets/cv_project/eval")
    ap.add_argument('--nms_threshold', type=float, default=0.2, help='nms threshold, change for testing purposes only')
    ap.add_argument('--cuda', default=True)
    ap.add_argument('--float16', default=False)
    ap.add_argument('--imshow', action="store_true")
    ap.add_argument('--imwrite',action="store_true")
    ap.add_argument('--debug', action="store_true")

    args = ap.parse_args()

    weights = Path(args.weights)

    if args.debug:
        shutil.rmtree("./assets/predictions")

    use_cuda = args.cuda
    use_float16 = args.float16
    if use_cuda:
        cudnn.fastest = True
        cudnn.benchmark = True

    imgs_filepath = get_testset_imgs_filepaths(args.data_folder)
    print("# images for inference:", len(imgs_filepath))

    compound_coefficient, _ = coefficient_from_weights_filepath(weights)
    ori_imgs, framed_imgs, framed_metas = read_images(imgs_filepath, compound_coefficient)
    splitted_testset = prepare_inputs_for_model(use_cuda)
    predictions = eval(weights, splitted_testset, compound_coefficient, use_cuda)

    predictions_postrocessed = invert_affine(framed_metas, predictions)

    display(predictions_postrocessed, ori_imgs, compound_coefficient, imshow=args.imshow, imwrite=args.imwrite)
