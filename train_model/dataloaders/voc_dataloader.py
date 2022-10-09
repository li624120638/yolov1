# author: lgx
# date: 2022/9/24 11:06
# description:
import math
import os
import cv2
import torch
from torchvision.transforms import functional
from torchvision.datasets import VOCDetection
import numpy as np
from PIL import Image


class VOCDetectionDataset(VOCDetection):
    def __init__(self, root='E:/datasets/VOC', year='2007', image_set='trainval', download=False,
                 visualization=False, work_dir=None,
                 phase='train', translate=(0.2, 0.2), scale=(0.8, 1.2), remain_thres=0.4, transform=None,
                 S=7, B=2, C=20, W=448, H=448):
        self.year = year
        if isinstance(self.year, int):
            self.year = str(year)
        super(VOCDetectionDataset, self).__init__(root, self.year, image_set, download)
        self.root = root
        self.image_set = image_set
        self.phase = phase

        self.visualization = visualization
        if visualization:
            assert work_dir is not None
            self.work_dir = work_dir
            self.vis_dir = os.path.join(self.work_dir, 'visulization')
            self.gt_vis_path = os.path.join(self.vis_dir, self.phase)
            self.pred_vis_path = os.path.join(self.vis_dir, '{}_predict'.format(self.phase))
            if not os.path.exists(self.gt_vis_path):
                os.makedirs(os.path.join(self.vis_dir, self.phase))
            if not os.path.exists(self.pred_vis_path):
                os.makedirs(os.path.join(self.vis_dir, '{}_predict'.format(self.phase)))

        self.translate = translate
        self.scale = scale
        self.remain_thres = remain_thres
        self.transform = transform

        # The same notations in the YOLOv1 paper(https://arxiv.org/abs/1506.02640)
        self.S = S
        self.B = B
        self.C = C
        self.l_tensor = B * 5 + C
        self.W, self.H = W, H  # image size
        self.grid_w, self.grid_h = self.W / self.S, self.H / self.S  # grid(cell) size

        self.class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        self.class_colors = self.generate_colormap(C)

        if visualization:
            pass

    def __getitem__(self, item):
        img, _target = super(VOCDetectionDataset, self).__getitem__(item)
        img_w, img_h = img.size

        dx, dy = 0, 0
        scale = 1
        if self.phase == 'train':
            # Saturation aug
            if np.random.uniform(0, 1) < 0.5:
                img = functional.adjust_saturation(img, 1.5)
            # Exposure aug
            if np.random.uniform(0, 1) < 1:
                img = functional.adjust_gamma(img, 1.5)
            # Translate aug
            if self.translate and np.random.uniform(0, 1) < 0.5:
                dx = np.random.uniform(-self.translate[0], self.translate[0]) * img_w
                dy = np.random.uniform(-self.translate[1], self.translate[1]) * img_h
            # Scale aug
            if self.scale and np.random.uniform(0, 1) < 0.5:
                scale = np.random.uniform(self.scale[0], self.scale[1])
            img = functional.affine(img, angle=0, translate=[dx, dy], scale=scale, shear=0)
        img = img.resize((self.W, self.H))

        target = self._generate_target(img_w, img_h, _target['annotation']['object'], dx, dy, scale)
        if self.transform:
            img = self.transform(img)

        return img, target

    def _generate_target(self, img_w, img_h, objects, dx, dy, scale):
        target = np.zeros(shape=(self.S, self.S, self.l_tensor), dtype=np.float32)
        for objid, obj in enumerate(objects):
            cls_idx = self.class_names.index(obj['name'])
            bndbox = obj['bndbox']
            xmin, ymin = float(bndbox['xmin']), float(bndbox['ymin'])
            xmax, ymax = float(bndbox['xmax']), float(bndbox['ymax'])

            if self.phase == 'train':
                # Scale: keep image center invariant
                img_cx, img_cy = img_w / 2, img_h / 2
                xmin = img_cx + (xmin - img_cx) * scale
                xmax = img_cx + (xmax - img_cx) * scale
                ymin = img_cy + (ymin - img_cy) * scale
                ymax = img_cy + (ymax - img_cy) * scale
                # Translate
                xmin += dx
                xmax += dx
                ymin += dy
                ymax += dy
                # Truncated
                xmin_remain = min(max(xmin, 0), img_w)
                xmax_remain = min(max(xmax, 0), img_w)
                ymin_remain = min(max(ymin, 0), img_h)
                ymax_remain = min(max(ymax, 0), img_h)
                # Discard truncated boxes whose remain area less than the value of original area times remain threshold.
                if (xmax_remain - xmin_remain) * (ymax_remain - ymin_remain) / (
                        (ymax - ymin) * (xmax - xmin)) < self.remain_thres:
                    continue
                xmin, ymin, xmax, ymax = xmin_remain, ymin_remain, xmax_remain, ymax_remain

            # Aligned to (W, H) e.g.(448, 448)
            xmin *= self.W / img_w
            xmax *= self.W / img_w
            ymin *= self.H / img_h
            ymax *= self.H / img_h
            # Convert xyxy format to yolov1 format
            grid_x, grid_y, cx, cy, w, h = self.xyxy2yolov1(xmin, ymin, xmax, ymax)
            target[grid_y, grid_x, 0:4] = cx, cy, w, h
            target[grid_y, grid_x, 4] = 1
            target[grid_y, grid_x, self.B * 5 + cls_idx] = 1
        return target

    @staticmethod
    def xyxy2cxcywh(x1, y1, x2, y2):
        assert x2 > x1 and y2 > y1
        cx, cy = (x2 + x1) / 2, (y2 + y1) / 2
        w, h = x2 - x1, y2 - y1
        return cx, cy, w, h

    def cxcywh2yolov1(self, cx, cy, w, h):
        grid_x, grid_y = cx // self.grid_w, cy // self.grid_h
        cx, cy = cx / self.grid_w - grid_x, cy / self.grid_h - grid_y
        w, h = w / self.W, h / self.H
        return int(grid_x), int(grid_y), cx, cy, w, h

    def xyxy2yolov1(self, x1, y1, x2, y2):
        return self.cxcywh2yolov1(*self.xyxy2cxcywh(x1, y1, x2, y2))

    @staticmethod
    def cxcywh2xyxy(cx, cy, w, h):
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2
        return x1, y1, x2, y2

    def yolov12xyxy(self, grid_x, grid_y, cx, cy, w, h):
        return self.cxcywh2xyxy(*self.yolov12cxcywh(grid_x, grid_y, cx, cy, w, h))

    def yolov12cxcywh(self, grid_x, grid_y, cx, cy, w, h):
        cx, cy = (grid_x + cx) * self.grid_w, (grid_y + cy) * self.grid_h
        w, h = w * self.W, h * self.H
        return cx, cy, w, h

    # used for debug
    def visualize(self, img, target, denorm=False, save_path=None):
        if denorm:
            img = img * 255.
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy().astype(np.uint8)
        if isinstance(target, torch.Tensor):
            target = target.numpy()
        if isinstance(img, Image.Image):
            img = np.array(img)

        if target.ndim == 1:
            target = target.reshape(self.S, self.S, self.l_tensor)

        exisit_box = target[..., 4] == 1
        best_box = np.argmax(target[..., 4:self.B*5:5], axis=-1)
        cell_indices = np.tile(np.arange(self.S), (self.S, 1))
        cx = self.grid_w * (target[..., 0] + cell_indices)
        cy = self.grid_h * (target[..., 1] + cell_indices.transpose(1, 0))
        w = self.W * ((1 - best_box) * target[..., 2] + best_box * target[..., 7])
        h = self.H * ((1 - best_box) * target[..., 3] + best_box * target[..., 8])
        cls_idxs = np.argmax(target[..., self.B*5:], axis=-1)[exisit_box]
        x1s = (cx - w / 2)[exisit_box]
        x2s = (cx + w / 2)[exisit_box]
        y1s = (cy - h / 2)[exisit_box]
        y2s = (cy + h / 2)[exisit_box]
        for i, cls_idx in enumerate(cls_idxs):
            img = self._draw_rectangle(img, x1s[i], y1s[i], x2s[i], y2s[i], cls_idx)

        if save_path:
            cv2.imwrite(save_path, img[:, :, ::-1])
        else:
            cv2.imshow("visualize", img[:, :, ::-1])
            cv2.waitKey()
            cv2.destroyWindow("visualize")

    @staticmethod
    def generate_colormap(nc=20):
        colors = []
        i = int(8 - math.log2(nc) / 3)
        step = 2 ** i
        carry = 2**(8 - i)
        for c in range(nc):
            r = step * int(c % carry)
            g = step * int(c / carry % carry)
            b = step * int(c / carry / carry)
            colors.append((b, g, r))
        return colors

    def _draw_rectangle(self, img, x1, y1, x2, y2, cls_idx):
        cls_name = self.class_names[cls_idx]
        img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color=self.class_colors[cls_idx], thickness=2)
        (text_w, text_h), _ = cv2.getTextSize(cls_name, 1, 1, 1)
        img = cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + text_w), int(y1 + text_h)),
                            color=self.class_colors[cls_idx], thickness=cv2.FILLED)
        img = cv2.putText(img, cls_name, (int(x1), int(y1 + text_h)), 1, 1, color=(255, 255, 255), thickness=1)
        return img
    

if __name__ == '__main__':
    voc2007 = VOCDetectionDataset(year='2007', image_set='trainval')
    voc20072 = VOCDetectionDataset(year='2007', image_set='trainval', phase='test')
    idx = 1236
    img, target = voc2007[idx]
    img2, target2 = voc20072[idx]
    voc2007.visualize(img, target, denorm=False)
    voc2007.visualize(img2, target2, denorm=False)