# author: lgx
# date: 2022/9/26 11:55
# description: Visualization tool for detection output.
import torch
import torch.nn as nn
from torchvision.ops import batched_nms
from torchvision.utils import draw_bounding_boxes, make_grid
import cv2
from PIL import Image
import numpy as np
from math import log2


class YOLOv1Vis(nn.Module):
    def __init__(self, S=7, B=2, C=20, W=448, H=448, conf_thres=0.5, nms_thres=0.7,
                 sigmoid=False, softmax=False, denorm=True):
        super(YOLOv1Vis, self).__init__()
        # The same notations in the YOLOv1 paper(https://arxiv.org/abs/1506.02640)
        self.S = S
        self.B = B
        self.C = C

        self.W, self.H = W, H  # image size
        self.grid_w, self.grid_h = self.W / self.S, self.H / self.S  # grid(cell) size

        self.conf_thres = conf_thres  # Outputs with score greater than it will be considered as having an object
        self.nms_thres = nms_thres  # IoU threshold used to perform non-maximum suppression

        self.sigmoid = sigmoid  # whether do sigmoid for output
        self.softmax = softmax  # whether do softmax for output class probabilities
        self.denorm = denorm  # whether do de-normalize for images

        self.class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        self.class_colors = self.generate_colormap(C)

    def forward(self, images,  outputs, prefix):
        """
        :param images: the input of YOLOv1Net (batch_size, C, W, H)
        :param outputs: the output of YOLOv1Net (batch_size, S * S * (B*5+C))
        :param prefix: save path
        :return: None
        """
        # Convert to Tensor
        if isinstance(outputs, np.ndarray):
            outputs = torch.tensor(outputs)
        device = outputs.device
        if isinstance(images, list):
            images = torch.tensor(np.array([img.resize((self.W, self.H))for img in images]), device=device)
        if isinstance(images, Image.Image):
            img = torch.tensor(np.array(images.resize((self.W, self.H))), device=device)
        if isinstance(images, np.ndarray):
            if images.ndim == 3:
                images = torch.from_numpy(images).permute(1, 2, 0).to(device)
            elif images.ndim == 4:
                images = torch.from_numpy(images).permute(0, 2, 1, 3).to(device)

        # Batched
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if outputs.ndim == 1:
            outputs = outputs.unsqueeze(0)
        batch_size = outputs.shape[0]

        # De-normalized
        if self.denorm:
            images = (images * 255.).to(torch.uint8)
        if self.sigmoid:
            outputs = torch.sigmoid(outputs)

        outputs = outputs.reshape(batch_size, self.S, self.S, self.B * 5 + self.C)
        confs = torch.cat([outputs[..., b * 5 + 4:b * 5 + 5] for b in range(self.B)], dim=-1)  # (batch_size, S, S, B)
        output_confs, resp_box_idxs = torch.max(confs, dim=-1, keepdim=True)  # (batch_size, S, S, 1)
        output_coords = (resp_box_idxs == 0) * outputs[..., 0:4]  # (batch_size, S, S, 4)
        for b in range(1, self.B):
            output_coords += (resp_box_idxs == b) * outputs[..., b * 5:b * 5 + 4]
        idxs = torch.arange(self.S, device=device).repeat(batch_size, self.S, 1).unsqueeze(-1)  # (batch_size, S, S, 1)
        output_box_cx = (output_coords[..., 0:1] + idxs) * self.grid_w
        output_box_cy = (output_coords[..., 1:2] + idxs.permute(0, 2, 1, 3)) * self.grid_h
        output_box_w = output_coords[..., 2:3] * self.W
        output_box_h = output_coords[..., 3:4] * self.H
        output_box_x1 = output_box_cx - output_box_w / 2  # (batch_size, S, S, 1)
        output_box_x2 = output_box_cx + output_box_w / 2  # (batch_size, S, S, 1)
        output_box_y1 = output_box_cy - output_box_h / 2  # (batch_size, S, S, 1)
        output_box_y2 = output_box_cy + output_box_h / 2  # (batch_size, S, S, 1)
        output_boxes = torch.cat([output_box_x1, output_box_y1,
                                  output_box_x2, output_box_y2], dim=-1)  # (batch_size, S, S, 4)
        output_classes_probs, output_classes_idxs = \
            torch.max(outputs[..., self.B * 5:], dim=-1, keepdim=True)  # (batch_size, S, S, 1)

        preds = []
        for image_id in range(batch_size):
            image = images[image_id]
            output_conf = torch.flatten(output_confs[image_id])  # (S * S, )
            output_box = torch.flatten(output_boxes[image_id], end_dim=-2)  # (S * S, 4)
            output_classes_idx = torch.flatten(output_classes_idxs[image_id])  # (S * S, )
            output_classes_prob = torch.flatten(output_classes_probs[image_id])  # (S * S, )

            output_classes_idx = output_classes_idx[output_conf > self.conf_thres]
            output_classes_prob = output_classes_prob[output_conf > self.conf_thres]
            output_box = output_box[output_conf > self.conf_thres]
            output_conf = output_conf[output_conf > self.conf_thres]

            nms_res = batched_nms(output_box, output_conf, output_classes_idx, self.nms_thres)
            output_conf = output_conf[nms_res]
            output_box = output_box[nms_res]
            output_classes_idx = output_classes_idx[nms_res]
            output_classes_prob = output_classes_prob[nms_res]

            output_class_names = [
                self.class_names[cls_idx] +
                ' {}%'.format(round((output_conf[i] * output_classes_prob[i] * 100).item(), 1))
                for i, cls_idx in enumerate(output_classes_idx)
            ]
            output_box_colors = [self.class_colors[cls_idx] for cls_idx in output_classes_idx]
            boxed_nms_img = draw_bounding_boxes(image, output_box, output_class_names, output_box_colors, width=2)
            preds.append(boxed_nms_img)

        grid_img = make_grid(preds, nrow=8).permute(1, 2, 0).cpu().numpy()
        cv2.imwrite(prefix, grid_img[:, :, ::-1])

    @staticmethod
    def generate_colormap(nc=20):
        colors = []
        i = int(8 - log2(nc) / 3)
        step = 2 ** i
        carry = 2 ** (8 - i)
        for c in range(nc):
            r = step * int(c % carry)
            g = step * int(c / carry % carry)
            b = step * int(c / carry / carry)
            colors.append((b, g, r))
        return colors
