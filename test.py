from sys import argv

import torch
from torch.utils.data import DataLoader

import config.config as cfg
from dataset import AniDataset
from model import YOLOv1
from utils import get_bboxes, load_model_only, mean_average_precision


def main(model_path):
    model = YOLOv1(in_channels=3, split_size=7, num_boxes=2, num_classes=1).to(cfg.DEVICE)

    load_model_only(torch.load(model_path, map_location=cfg.DEVICE), model)
    
    test_dataset = AniDataset(
        csv_file=cfg.TEST_CSV,
        img_dir=cfg.TEST_IMG_DIR,
        label_dir=cfg.TEST_LABEL_DIR,
        transform=cfg.TEST_TRANSFORMS
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        shuffle=True,
        drop_last=False
    )

    pred_boxes, target_boxes = get_bboxes(test_loader, model, iou_threshold=cfg.IOU_THRESHOLD, threshold=cfg.THRESHOLD, box_format="midpoint")
    mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=cfg.IOU_THRESHOLD, box_format="midpoint", num_classes=1)
    
    print(f"Test mAP: {mean_avg_prec}")

if __name__ == "__main__":
    help = f'''./{argv[0]} [MODEL_PATH]
          [MODEL_PATH] Path to the trained model (Defaults to the one in the config file)
        '''
    if "-h" in argv or "--help" in argv:
        print(help)
        exit()

    def index_in_list(l, index):
        return index < len(l)
    
    model_path = argv[1] if index_in_list(argv, 1) else cfg.MODEL_FILE
    main(model_path)
