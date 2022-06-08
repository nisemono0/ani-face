import argparse

import torch
from PIL import Image

import config.config as cfg
from model import YOLOv1
from utils import draw_pred_image, get_single_bboxes, load_model_only


def main(image_path, model_path, iou_threshold, threshold, border_thickness, save_image):
    model = YOLOv1(in_channels=3, split_size=7, num_boxes=2, num_classes=1).to(cfg.DEVICE)

    load_model_only(torch.load(model_path, map_location=cfg.DEVICE), model)

    image = Image.open(image_path).convert("RGB")
    image, _ = cfg.TEST_TRANSFORMS(image, None)

    boxes = get_single_bboxes(image, model, iou_threshold=iou_threshold, threshold=threshold, box_format="midpoint")
    draw_pred_image(image_path=image_path, thickness=border_thickness, save_image=save_image, boxes=boxes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single image through the network", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("image_path", metavar="IMAGE", type=str, help="Path to an image file")
    parser.add_argument("-m", "--model", metavar="MODEL", type=str, default=cfg.MODEL_FILE, dest="model_path", help="Path to saved model")
    parser.add_argument("-i", "--iou", metavar="IOU_THRESHOLD", type=float, default=0.4, dest="iou_threshold", help="Percent IoU threshold at which boxes are dropped")
    parser.add_argument("-d", "--drop", metavar="DROP_THRESHOLD", type=float, default=0.4, dest="drop_threshold", help="Percent threshold at which a box is dropped")
    parser.add_argument("-b", "--border", metavar="BORDER", type=int, default=7, dest="border_thickness", help="Thickness of the drawn border")
    parser.add_argument("-s", "--save", metavar="SAVE", type=bool, default=False, dest="save_image", help="Save predicted image")
    args = parser.parse_args()

    image_path = args.image_path
    model_path = args.model_path
    iou_threshold = args.iou_threshold
    threshold = args.drop_threshold
    border_thickness = args.border_thickness
    save_image = args.save_image
    
    main(image_path, model_path, iou_threshold, threshold, border_thickness, save_image)
