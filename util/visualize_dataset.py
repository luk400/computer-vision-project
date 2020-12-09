import cv2
from torch.utils.data import DataLoader
import sys
sys.path.append('../Yet-Another-EfficientDet-Pytorch')

from efficientdet.dataset import CocoDataset as ThermalDataset

if __name__ == "__main__":
    root_dir = "../Yet-Another-EfficientDet-Pytorch/datasets/cv_project"
    training_set = ThermalDataset(root_dir=root_dir, set="train")
    training_generator = DataLoader(training_set)
    for data in training_generator:
        img_from_torch_to_numpy = data["img"][0].numpy()
        annotations = data["annot"][0]
        for anno in annotations:
            x1, y1, x2, y2, _ = anno
            cv2.rectangle(img_from_torch_to_numpy, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.imshow("img", img_from_torch_to_numpy)
        cv2.waitKey(0)

