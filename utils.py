from collections import Counter

import torch
from PIL import Image, ImageDraw, ImageFont

import config.config as cfg

# Used for numerical stability later on
EPSILON = 1e-6

def intersection_over_union(boxes_preds, boxes_labels, box_format):
    """
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for when the boxes dont intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + EPSILON)


def non_max_suppression(bboxes, iou_threshold, threshold, box_format):
    """
    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU) 
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [box for box in bboxes if box[0] != chosen_box[0] or intersection_over_union(torch.tensor(chosen_box[2:]), torch.tensor(box[2:]), box_format=box_format) < iou_threshold]
        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def mean_average_precision(pred_boxes, true_boxes, iou_threshold, box_format, num_classes=1):
    """
    Calculates mean average precision 

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """

    # list storing all AP for respective classes
    average_precisions = []

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + EPSILON)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + EPSILON))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)

def _get_font_text(img, img_fraction, text):
    img_w, img_h = img.size
    # portion of image width you want text width to be
    fontsize = 1  # starting font size
    font = ImageFont.truetype(cfg.FONT_NAME, fontsize)
    breakpoint = min(img_w, img_h) * img_fraction
    if breakpoint <= 20:
        breakpoint = 30

    jumpsize = 75
    while True:
        if font.getsize(text)[0] < breakpoint:
            fontsize += jumpsize
        else:
            jumpsize = jumpsize // 2
            fontsize -= jumpsize
        font = ImageFont.truetype(cfg.FONT_NAME, fontsize)
        if jumpsize <= 1:
            break
    return font, fontsize

def _get_box_coords(bboxes, img):
    img_w, img_h = img.size

    coords = []
    for box in bboxes:
        if box != []:
            # Adapted from https://github.com/pjreddie/darknet/blob/810d7f797bdb2f021dbe65d2524c2ff6b8ab5c8b/src/image.c#L283-L291
            pred_class, class_prob, x, y, w, h = tuple(box)
            l = int((x - w / 2) * img_w)
            r = int((x + w / 2) * img_w)
            t = int((y - h / 2) * img_h)
            b = int((y + h / 2) * img_h)
            if l < 0:
                l = 0
            if r > img_w - 1:
                r = img_w - 1
            if t < 0:
                t = 0
            if b > img_h - 1:
                b = img_h - 1
            coords.append([pred_class, class_prob, l, r, t, b])

    return coords

def draw_pred_image(image_path, thickness, save_image, boxes):
    """
    Takes the predicted boxes and draws them on the image
    """
    img = Image.open(image_path).convert("RGB")
    boxes = _get_box_coords(boxes, img)

    if thickness == 0:
        img_w, img_h = img.size
        thickness_fraction = 0.005
        thickness = int(max(img_h, img_w) * thickness_fraction)
        if thickness <= 1:
            thickness = 2

    for box in boxes:
        _, prob, l, r, t, b = box
        # Format prob on image
        prob = box[1] * 100.0
        if prob >= 100:
            text = "100%"
        else:
            text = f"{(box[1] * 100):.2f}%"
        
        font, fontsize = _get_font_text(img, 0.08, text)


        # Draw rectangle
        ImageDraw.Draw(img).rectangle([(l, t), (r, b)], outline="red", width=thickness)
        ImageDraw.Draw(img).rectangle([(l, t), (r, b-(b-t)-fontsize)], fill="red")

        ImageDraw.Draw(img).text((l+thickness, t-fontsize), text=text, font=font, fill=(0,0,0))

    if save_image:
        from os.path import splitext
        file_no_ext = splitext(image_path)[0]
        new_file = file_no_ext + "_pred.png"
        img.save(new_file)
    else:
        img.show()

def get_bboxes(loader, model, iou_threshold, threshold, box_format):
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for _, (x, labels) in enumerate(loader):
        x = x.to(cfg.DEVICE)
        labels = labels.to(cfg.DEVICE)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes

def get_single_bboxes(image, model, iou_threshold, threshold, box_format):
    """
    Takes a (3, 448, 448) image tensor runs the mode on it
    Returns a list of nms bboxes
    """
    model.eval()
    
    image = image.to(cfg.DEVICE)
    image = image.reshape((1, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))

    with torch.no_grad():
        predictions = model(image)
    
    bboxes = cellboxes_to_boxes(predictions)
    nms_boxes = non_max_suppression(bboxes[0], iou_threshold=iou_threshold, threshold=threshold, box_format=box_format)

    model.train()

    return nms_boxes

def convert_cellboxes(predictions, S=7):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. Tried to do this
    vectorized, but this resulted in quite difficult to read
    code... Use as a black box? Or implement a more intuitive,
    using 2 for loops iterating range(S) and convert them one
    by one, resulting in a slower but more readable implementation.
    """

    predictions = predictions.to(cfg.DEVICE)
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, S, S, 11)
    bboxes1 = predictions[..., 2:6]
    bboxes2 = predictions[..., 7:11]
    scores = torch.cat((predictions[..., 1].unsqueeze(0), predictions[..., 6].unsqueeze(0)), dim=0)
    best_box = scores.argmax(0).unsqueeze(-1).to(cfg.DEVICE)
    best_boxes = (bboxes1 * (1 - best_box) + best_box * bboxes2).to(cfg.DEVICE)
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1).to(cfg.DEVICE)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :1].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 1], predictions[..., 6]).unsqueeze(-1)
    converted_preds = torch.cat((predicted_class, best_confidence, converted_bboxes), dim=-1)

    return converted_preds


def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes

def save_checkpoint(model, optimizer, scheduler, epoch, filename):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "last_epoch": epoch
    }

    torch.save(checkpoint, filename)

def save_model_only(model, filename):
    print("=> Saving model only")
    checkpoint = {
        "state_dict": model.state_dict()
    }
    torch.save(checkpoint, filename)

def load_model_only(checkpoint, model):
    print("=> Loading model only")
    model.load_state_dict(checkpoint["state_dict"])

def load_checkpoint(checkpoint, model, optimizer, scheduler):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint["last_epoch"]
