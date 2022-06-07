import torch
import torch.nn as nn

from utils import intersection_over_union

EPS = 1e-6  # For numerical stability in case we'll ever have sqrt(0)

class YOLOv1Loss(nn.Module):
    """
    S = Split size of image (Original: 7)
    B = Number of boxes/cell (Original: 2)
    C = Number of classes (Original: 20, ours: 1 (Can be extended))
    """
    def __init__(self, S=7, B=2, C=1):
        super(YOLOv1Loss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C+self.B*5)  # Reshape the predictions to (S, S, C+B*5)

        # C+1:C+5, C+1:C+5
        # iou_bbox1 = intersection_over_union(predictions[..., 2:6], target[..., 2:6], box_format="midpoint")
        iou_bbox1 = intersection_over_union(predictions[..., self.C+1:self.C+5], target[..., self.C+1:self.C+5], box_format="midpoint")
        # C+6:C+10, C+1:C+5
        # iou_bbox2 = intersection_over_union(predictions[..., 7:11], target[..., 2:6], box_format="midpoint")
        iou_bbox2 = intersection_over_union(predictions[..., self.C+6:self.C+10], target[..., self.C+1:self.C+5], box_format="midpoint")

        ious = torch.cat([iou_bbox1.unsqueeze(0), iou_bbox2.unsqueeze(0)], dim=0)

        _, bestbox = torch.max(ious, dim=0)

        # C
        # exists_box = target[..., 1].unsqueeze(3)
        exists_box = target[..., self.C].unsqueeze(3)  # Identity of object i from the paper (I_obj_i)

        ##### For box coords #####
        # C+6:C+10 ; C+1 C+5
        # box_predictions = exists_box * ((bestbox * predictions[..., 7:11] + (1 - bestbox) * predictions[..., 2:6]))
        box_predictions = exists_box * ((bestbox * predictions[..., self.C+6:self.C+10] + (1 - bestbox) * predictions[..., self.C+1:self.C+5]))
        # C+1:C+5
        # box_targets = exists_box * target[..., 2:6]
        box_targets = exists_box * target[..., self.C+1:self.C+5]

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4] + EPS))
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # (N, S, S, 4) -> (N*S*S, 4) using the flatten
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )

        ##### For obj loss #####
        # C+5:C+6 ; C:C+1
        # pred_box = (bestbox * predictions[..., 6:7] + (1-bestbox) * predictions[..., 1:2])
        pred_box = (bestbox * predictions[..., self.C+5:self.C+6] + (1-bestbox) * predictions[..., self.C:self.C+1])
        # (N*S*S, 1)
        # C:C+1
        obj_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            # torch.flatten(exists_box * target[..., 1:2])
            torch.flatten(exists_box * target[..., self.C:self.C+1])
        )

        ##### For no obj loss #####
        # (N, S, S, 1) -> (N, S*S)
        # C:C+1 ; C:C+1
        no_obj_loss = self.mse(
            # torch.flatten((1 - exists_box) * predictions[..., 1:2], start_dim=1),
            torch.flatten((1 - exists_box) * predictions[..., self.C:self.C+1], start_dim=1),
            # torch.flatten((1 - exists_box) * target[..., 1:2], start_dim=1)
            torch.flatten((1 - exists_box) * target[..., self.C:self.C+1], start_dim=1)
        )

        # C+5:C+6 ; C:C+1
        no_obj_loss += self.mse(
            # torch.flatten((1 - exists_box) * predictions[..., 6:7], start_dim=1),
            torch.flatten((1 - exists_box) * predictions[..., self.C+5:self.C+6], start_dim=1),
            # torch.flatten((1 - exists_box) * target[..., 1:2], start_dim=1)
            torch.flatten((1 - exists_box) * target[..., self.C:self.C+1], start_dim=1)
        )

        ##### For class loss #####
        # (N, S, S, 1) -> (N*S*S, 1)
        # :C ; :C
        class_loss = self.mse(
            # torch.flatten(exists_box * predictions[..., :1], end_dim=-2),
            torch.flatten(exists_box * predictions[..., :self.C], end_dim=-2),
            # torch.flatten(exists_box * target[..., :1], end_dim=-2)
            torch.flatten(exists_box * target[..., :self.C], end_dim=-2)
        )

        loss = self.lambda_coord * box_loss + obj_loss + self.lambda_noobj * no_obj_loss + class_loss

        return loss
