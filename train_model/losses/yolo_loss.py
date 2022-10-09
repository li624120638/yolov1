# author: lgx
# date: 2022/9/25 15:51
# description: Implementation for YOLO multi-part loss function
import torch
import torch.nn as nn


class YOLOv1Loss(nn.Module):
    def __init__(self, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5,
                 W=448, H=448, sigmoid=False, softmax=False):
        super(YOLOv1Loss, self).__init__()
        # The same notations in the YOLOv1 paper(https://arxiv.org/abs/1506.02640)
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

        self.W, self.H = W, H  # image size
        self.grid_w, self.grid_h = self.W / self.S, self.H / self.S  # grid(cell) size
        self.sigmoid = sigmoid  # whether do sigmoid for output
        self.softmax = softmax  # whether do softmax for output class probabilities
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.cross_loss = nn.CrossEntropyLoss()

    def forward(self, target, output):
        target = target.reshape(-1, self.S, self.S, self.B * 5 + self.C)
        output = output.reshape(-1, self.S, self.S, self.B * 5 + self.C)
        if self.sigmoid:
            output = torch.sigmoid(output)
        batch_size = output.shape[0]
        output = output.clone()  # must clone, they will be operated by square root function.
        target = target.clone()

        # Target boxes parameters
        target_coords = target[..., 0:4]  # (batch_size, S, S, 4)
        target_confs = target[..., 4:5]  # (batch_size, S, S, 1)
        target_classes = target[..., self.B * 5:]  # (batch_size, S, S, C)

        # Calculate output box indices which responsible for detection
        ious_grid = []
        for b in range(self.B):
            output_coord = output[..., b * 5:b * 5 + 4]
            iou_grid = self._calc_iou_gird(target_coords, output_coord).unsqueeze(-1)
            ious_grid.append(iou_grid)
        _, resp_box_idxs = torch.max(torch.concat(ious_grid, dim=-1), dim=-1, keepdim=True)  # (batch_size, S, S, 1)

        # Responsible boxes parameters
        output_coords = (resp_box_idxs == 0) * output[..., 0:4]  # (batch_size, S, S, 4)
        output_confs = (resp_box_idxs == 0) * output[..., 4:5]  # (batch_size, S, S, 1)
        output_classes = output[..., self.B * 5:]  # (batch_size, S, S, C)
        for b in range(1, self.B):
            output_coords += (resp_box_idxs == b) * output[..., b * 5:b * 5 + 4]
            output_confs += (resp_box_idxs == b) * output[..., b * 5 + 4:b * 5 + 5]

        # square root: w, h
        output_coords[..., 2:4] = torch.sign(output_coords[..., 2:4]) * torch.sqrt(torch.abs(output_coords[..., 2:4] +
                                                                                             1e-6))
        target_coords[..., 2:4] = torch.sqrt(target_coords[..., 2:4])

        obj_i = target_confs  # (batch_size, S, S, 1), whether the grid exisit object.

        # coordinate-part loss
        coord_loss = self.lambda_coord * self.mse_loss(
            torch.flatten(obj_i * output_coords, end_dim=-2),  # (batch_size * S * S, 4)
            torch.flatten(obj_i * target_coords, end_dim=-2),
        )

        # confidence-part loss
        conf_loss = self.mse_loss(
            torch.flatten(obj_i * output_confs, end_dim=-2),  # (batch_size * S * S, 1)
            torch.flatten(obj_i * target_confs, end_dim=-2)
        )

        # class-part loss
        cls_loss = self.mse_loss(
            torch.flatten(obj_i * output_classes, end_dim=-2),  # (batch_size * S * S, C)
            torch.flatten(obj_i * target_classes, end_dim=-2)
        )

        # no-obj-part loss
        noobj_loss = torch.zeros(1, requires_grad=True, device=output.device)
        for j in range(self.B):
            noobj_loss = noobj_loss + self.mse_loss(
                torch.flatten((obj_i == 0) * output[..., j * 5 + 4:j * 5 + 5], start_dim=1),  # (batch_size * S * S, 1)
                torch.flatten((obj_i == 0) * target[..., 4:5], start_dim=1)
            )
        # Times the proportion of cells that have no object to balance this part loss gradient.
        noobj_loss = noobj_loss * self.lambda_noobj * torch.sum(obj_i) / (self.S * self.S * batch_size)

        return (coord_loss + conf_loss + cls_loss + noobj_loss),\
            coord_loss, conf_loss, cls_loss,  noobj_loss

    def _calc_iou_gird(self, target_boxes_coord, output_box_coord):
        """
        Calculate IoU according to certain grid position in YOLO box format.
        :param target_boxes_coord: (batch_size, S, S, 4)
        :param output_box_coord: (batch_size, S, S, 4)
        :return: (batch_size, S, S)
        """
        batch_size = target_boxes_coord.shape[0]
        device = target_boxes_coord.device
        cell_idxs = torch.arange(self.S, device=device).repeat(batch_size, self.S, 1)

        target_boxes_cx = (target_boxes_coord[..., 0] + cell_idxs) * self.grid_w
        target_boxes_cy = (target_boxes_coord[..., 1] + cell_idxs.permute(0, 2, 1)) * self.grid_h
        target_boxes_w = target_boxes_coord[..., 2] * self.W
        target_boxes_h = target_boxes_coord[..., 3] * self.H
        target_boxes_x1 = target_boxes_cx - target_boxes_w / 2
        target_boxes_x2 = target_boxes_cx + target_boxes_w / 2
        target_boxes_y1 = target_boxes_cy - target_boxes_h / 2
        target_boxes_y2 = target_boxes_cy + target_boxes_h / 2

        output_box_cx = (output_box_coord[..., 0] + cell_idxs) * self.grid_w
        output_box_cy = (output_box_coord[..., 1] + cell_idxs.permute(0, 2, 1)) * self.grid_h
        output_box_w = output_box_coord[..., 2] * self.W
        output_box_h = output_box_coord[..., 3] * self.H
        output_box_x1 = output_box_cx - output_box_w / 2
        output_box_x2 = output_box_cx + output_box_w / 2
        output_box_y1 = output_box_cy - output_box_h / 2
        output_box_y2 = output_box_cy + output_box_h / 2

        x1 = torch.max(target_boxes_x1, output_box_x1)
        y1 = torch.max(target_boxes_y1, output_box_y1)
        x2 = torch.min(target_boxes_x2, output_box_x2)
        y2 = torch.min(target_boxes_y2, output_box_y2)

        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        target_area = target_boxes_w * target_boxes_h
        output_area = output_box_w * output_box_h

        return intersection / (target_area + output_area - intersection + 1e-6)

