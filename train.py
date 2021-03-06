import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import config.config as cfg
from dataset import AniDataset
from loss import YOLOv1Loss
from model import YOLOv1
from utils import load_checkpoint, save_checkpoint, save_model_only


def train_func(train_loader, model, optimizer, loss_func):
    # For progress bar
    loop = tqdm(train_loader, leave=True)

    mean_loss = []

    for _, (x, y) in enumerate(loop):
        x, y = x.to(cfg.DEVICE), y.to(cfg.DEVICE)
        out = model(x)
        loss = loss_func(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update progress bar
        loop.set_postfix(loss = loss.item())
    
    return sum(mean_loss)/len(mean_loss)

def main(split_size, num_boxes, num_classes):
    model = YOLOv1(in_channels=3, split_size=split_size, num_boxes=num_boxes, num_classes=num_classes).to(cfg.DEVICE)

    # Original paper uses SGD as optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.LEARN_RATE,
        weight_decay=cfg.WEIGHT_DECAY
    )

    # We schedule to multiply by gamma every milestone epoch reached
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.MILESTONES, gamma=0.1, verbose=True)

    loss_func = YOLOv1Loss()
    
    start_epoch = 0
    if cfg.LOAD_MODEL:
        start_epoch = load_checkpoint(torch.load(cfg.MODEL_FILE, map_location=cfg.DEVICE), model, optimizer, scheduler)
        model.train()

    train_dataset = AniDataset(
        csv_file=cfg.TRAIN_CSV,
        img_dir=cfg.TRAIN_IMG_DIR,
        label_dir=cfg.TRAIN_LABEL_DIR,
        transform=cfg.TRAIN_TRANSFORMS
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        shuffle=True,
        drop_last=False
    )
    for e in range(start_epoch+1, cfg.EPOCHS):
        print(f"EPOCH: {e}")

        # On the whole dataset it takes ~5 minutes / epoch on colab
        mean_loss = train_func(train_loader, model, optimizer, loss_func)        
        scheduler.step()
        
        print(f"Mean loss was {mean_loss}")

        # Save the model every 10 epochs for retraining
        if e > 0 and e % 10 == 0:
            save_checkpoint(model, optimizer, scheduler, e, filename=f"{cfg.MODEL_FILE}_e{e}_checkpoint")


    # If all epochs are done, save the model and a checkpoint
    save_model_only(model, filename=f"{cfg.MODEL_FILE}")
    save_checkpoint(model, optimizer, scheduler, e, filename=f"{cfg.MODEL_FILE}_e{cfg.EPOCHS}_checkpoint")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single image through the network", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-s", "--split", metavar="SPLIT_SIZE", type=int, default=7, dest="split_size", help="Split size of the image")
    parser.add_argument("-b", "--boxes", metavar="NUM_BOXES", type=int, default=2, dest="num_boxes", help="Number of boxes per split cell")
    parser.add_argument("-c", "--classes", metavar="NUM_CLASSES", type=int, default=1, dest="num_classes", help="Number of classes")
    args = parser.parse_args()

    split_size = args.split_size
    num_boxes = args.num_boxes
    num_classes = args.num_classes
    
    main(split_size, num_boxes, num_classes)
