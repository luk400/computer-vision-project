import os

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


def preprocess(image_path, max_size=512, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    ori_imgs = [cv2.imread(img_path)[..., ::-1] for img_path in image_path]
    normalized_imgs = [(img / 255 - mean) / std for img in ori_imgs]
    imgs_meta = [aspectaware_resize_padding(img, max_size, max_size, means=None) for img in normalized_imgs]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_imgs, framed_imgs, framed_metas


compound_coef = 0
force_input_size = None  # set None to use default size
# img_path = ['test/img.png']
project_folder_path = "./Yet-Another-EfficientDet-Pytorch/datasets/cv_project/val"

img_path = [os.path.join(project_folder_path, i) for i in os.listdir(project_folder_path)]

# img_path = img_path[:50]

print("# images for inference:", len(img_path))

# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = 0.2
iou_threshold = 0.2

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

obj_list = ['person']

model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                             ratios=anchor_ratios, scales=anchor_scales)
pretrained_weights = "./logs/cv_project/efficientdet-d0_236_22500.pth"
model.load_state_dict(torch.load(pretrained_weights, map_location='cpu'))
model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()

color_list = standard_to_bgr(STANDARD_COLORS)
# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)

if use_cuda:
    x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
else:
    x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

with torch.no_grad():
    features, regression, classification, anchors = model(x)

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    out = postprocess(x,
                      anchors, regression, classification,
                      regressBoxes, clipBoxes,
                      threshold, iou_threshold)


def display(preds, imgs, imshow=True, imwrite=False, debug=False):
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
            os.system("mkdir -p ./predictions")
            cv2.imwrite(f'./predictions/img_inferred_d{compound_coef}_this_repo_{i}.jpg', imgs[i])

    if imwrite:
        image_folder = './predictions'
        image_files = [image_folder + '/' + img for img in os.listdir(image_folder) if img.endswith(".jpg")]
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=1)
        clip.write_videofile('./assets/predictions_testset.mp4')


out = invert_affine(framed_metas, out)
display(out, ori_imgs, imshow=False, imwrite=True)
