# author: lgx
# date: 2022/9/26 10:09
# description: Calculate some metrics(e.g. mAP) for detection task.
import numpy as np
import torch
import torch.nn as nn
from torchvision.ops import batched_nms, box_iou
from collections import Counter


class YOLOv1Acc(nn.Module):
    def __init__(self, S=7, B=2, C=20, W=448, H=448, conf_thres=0.5, nms_thres=0.5, iou_thres=0.5,
                 sigmoid=True, softmax=False):
        super(YOLOv1Acc, self).__init__()
        # The same notations in the YOLOv1 paper(https://arxiv.org/abs/1506.02640)
        self.S = S
        self.B = B
        self.C = C

        self.W, self.H = W, H  # image size
        self.grid_w, self.grid_h = self.W / self.S, self.H / self.S  # grid(cell) size

        self.conf_thres = conf_thres  # Outputs with score greater than it will be considered as having an object
        self.nms_thres = nms_thres  # IoU threshold used to perform non-maximum suppression
        if not isinstance(iou_thres, list):
            self.iou_thres = [iou_thres]
        else:
            self.iou_thres = iou_thres  # Iou threshold for calculating mAP

        self.sigmoid = sigmoid  # whether do sigmoid for output
        self.softmax = softmax  # whether do softmax for output class probabilities

    def forward(self, target, output):
        # Convert to tensor
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)
        if isinstance(output, np.ndarray):
            output = torch.from_numpy(output)
        # Batched
        if target.ndim == 3:
            target = target.unsqueeze(0)
        if output.ndim == 3:
            output = target.unsqueeze(0)

        target = target.reshape(-1, self.S, self.S, self.B * 5 + self.C)
        output = output.reshape(-1, self.S, self.S, self.B * 5 + self.C)
        if self.sigmoid:
            output = torch.sigmoid(output)
        batch_size = output.shape[0]
        device = output.device

        idxs = torch.arange(self.S, device=device).repeat(batch_size, self.S, 1).unsqueeze(-1)  # (batch_size, S, S, 1)
        # Target box parameters
        target_confs = target[..., 4:5]  # (batch_size, S, S, 1)
        target_class_idxs = torch.argmax(target[..., self.B * 5:], dim=-1, keepdim=True)  # (batch_size, S, S, 1)
        target_coords = target[..., 0:4]  # (batch_size, S, S, 4)
        target_boxes_cx = (target_coords[..., 0:1] + idxs) * self.grid_w  # (batch_size, S, S, 1)
        target_boxes_cy = (target_coords[..., 1:2] + idxs.permute(0, 2, 1, 3)) * self.grid_h
        target_boxes_w = target_coords[..., 2:3] * self.W  # (batch_size, S, S, 1)
        target_boxes_h = target_coords[..., 3:4] * self.H  # (batch_size, S, S, 1)
        target_boxes_x1 = (target_boxes_cx - target_boxes_w / 2)  # (batch_size, S, S, 1)
        target_boxes_x2 = (target_boxes_cx + target_boxes_w / 2)  # (batch_size, S, S, 1)
        target_boxes_y1 = (target_boxes_cy - target_boxes_h / 2)  # (batch_size, S, S, 1)
        target_boxes_y2 = (target_boxes_cy + target_boxes_h / 2)  # (batch_size, S, S, 1)
        target_boxes = torch.cat([target_class_idxs, target_confs,
                                  target_boxes_x1, target_boxes_y1,
                                  target_boxes_x2, target_boxes_y2], dim=-1)  # (batch_size, S, S, 6)

        # Output box parameters
        confs = []
        for b in range(self.B):
            confs.append(output[..., b * 5 + 4:b * 5 + 5])

        output_confs, resp_idxs = torch.max(torch.concat(confs, dim=-1), dim=-1, keepdim=True)  # (batch_size, S, S, 1)
        output_classes = torch.argmax(output[..., self.B * 5:], dim=-1, keepdim=True)  # (batch_size, S, S, 1)
        output_coords = (resp_idxs == 0) * output[..., 0:4]  # (batch_size, S, S, 4)
        for b in range(1, self.B):
            output_coords += (resp_idxs == b) * output[..., b * 5:b * 5 + 4]
        output_box_cx = (output_coords[..., 0:1] + idxs) * self.grid_w  # (batch_size, S, S, 1)
        output_box_cy = (output_coords[..., 1:2] + idxs.permute(0, 2, 1, 3)) * self.grid_h  # (batch_size, S, S, 1)
        output_box_w = output_coords[..., 2:3] * self.W  # (batch_size, S, S, 1)
        output_box_h = output_coords[..., 3:4] * self.H  # (batch_size, S, S, 1)
        output_box_x1 = (output_box_cx - output_box_w / 2)  # (batch_size, S, S, 1)
        output_box_x2 = (output_box_cx + output_box_w / 2)  # (batch_size, S, S, 1)
        output_box_y1 = (output_box_cy - output_box_h / 2)  # (batch_size, S, S, 1)
        output_box_y2 = (output_box_cy + output_box_h / 2)  # (batch_size, S, S, 1)
        output_boxes = torch.cat([output_classes, output_confs,
                                  output_box_x1, output_box_y1,
                                  output_box_x2, output_box_y2], dim=-1)  # (batch_size, S, S, 6)

        """
            Count all ground truth and predicted bounding boxes. 
            And store them in a python list respectively.
            The format of elements of the list is [img_id, cls_idx, conf, x1, y1, x2, y2], a python list too.
        """
        pred_boxes = []
        gt_boxes = []
        for img_id in range(batch_size):
            target_box = torch.flatten(target_boxes[img_id], end_dim=-2)  # (S*S, 6)
            # (N, 7), N is the number of objects.
            gt_boxes.extend([[img_id] + [x.item() for x in box] for box in target_box if box[1] > self.conf_thres])

            output_box = torch.flatten(output_boxes[img_id], end_dim=-2)  # (S*S, 6)
            nms_res = batched_nms(output_box[:, 2:], output_box[:, 1], output_box[:, 0], self.nms_thres)
            pred_boxes.extend(
                [[img_id] + [x.item() for x in box] for box in output_box[nms_res] if box[1] > self.conf_thres])

        # Calculate mAP throughout given IoUs
        res = []
        for iou_thre in self.iou_thres:
            average_precisions = []
            # Calculate per-class AP
            for c in range(self.C):
                # boxes with same class
                same_cls_gts = []  # [[img_id, cls_idx, conf, x1, y1, x2, y2], ... ]
                same_cls_preds = []
                for box in gt_boxes:
                    if int(box[1]) == c:
                        same_cls_gts.append(box)
                for box in pred_boxes:
                    if int(box[1]) == c:
                        same_cls_preds.append(box)
                # ground truth boxes in certain image whether have be detected out
                amount_bboxes_per_img = Counter([box[0] for box in same_cls_gts])
                for k, v in amount_bboxes_per_img.items():
                    amount_bboxes_per_img[k] = torch.zeros(v)

                TP = torch.zeros(len(same_cls_preds))
                FP = torch.zeros(len(same_cls_preds))
                total_true_bboxes = len(same_cls_gts)
                if total_true_bboxes == 0:
                    continue

                for pred_id, pred in enumerate(same_cls_preds):
                    # ground truth boxes with same class and in same image(same to 'pred')
                    selected_gts = [
                        box for box in same_cls_gts if box[0] == pred[0]
                    ]

                    max_iou = 0
                    pred_tensor = torch.tensor(pred[3:]).unsqueeze(0)
                    for gt_idx, gt in enumerate(selected_gts):
                        iou = box_iou(pred_tensor, torch.tensor(gt[3:]).unsqueeze(0))[0].item()
                        if iou > max_iou:
                            max_iou = iou
                            best_idx = gt_idx

                    if max_iou > iou_thre:
                        """
                            The ground truth box which has max IoU with 'pred' has not been detected,
                            then 'pred' is treated as TP. Ohterwise, 'pred' is treated as FP.
                        """
                        if amount_bboxes_per_img[pred[0]][best_idx] == 0:
                            TP[pred_id] = 1
                            amount_bboxes_per_img[pred[0]][best_idx] = 1
                        else:
                            FP[pred_id] = 1
                    else:
                        # The max IoU no greater than IoU threshold, 'pred' is treated as FP.
                        FP[pred_id] = 1

                TP_cum = torch.cumsum(TP, dim=0)
                FP_cum = torch.cumsum(FP, dim=0)
                recall = TP_cum / (total_true_bboxes + 1e-6)
                precision = torch.div(TP_cum, TP_cum + FP_cum + 1e-6)
                # the first point of P-R curve is (0, 1)
                precision = torch.cat((torch.tensor([1]), precision))
                recall = torch.cat((torch.tensor([0]), recall))
                # per-class AP is area of region under P-R curve
                average_precisions.append(torch.trapz(precision, recall))

            res.append(sum(average_precisions) / len(average_precisions))

        return sum(res) / len(res)


def test_all():
    import sys
    sys.path.append('../')
    sys.path.append('./')
    from dataloaders import VOCDetectionDatasetv2
    from losses import YOLOv1Loss
    # from visualization import YOLOv1Vis
    voc = VOCDetectionDatasetv2()
    img1, target1 = voc[0]
    img2, target2 = voc[1]
    target1 = torch.from_numpy(target1)
    target2 = torch.from_numpy(target2)
    target = torch.concat([target1.unsqueeze(0), target2.unsqueeze(0)], dim=0)
    voc.visualize(img1, target1)
    voc.visualize(img2, target2)

    criterion = YOLOv1Loss(sigmoid=False)
    evalueator = YOLOv1Acc(sigmoid=False)
    print(criterion(target, target))
    print(evalueator(target, target))


if __name__ == '__main__':
    test_all()