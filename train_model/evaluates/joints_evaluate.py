# author: lgx
# date: 2022/9/2 14:49
# description: calculate the accuracy of joints prediction
import torch
import torch.nn as nn
from utils.pts_transform import transform_preds


class JointsAcc(nn.Module):
    def __init__(self, dist_thres: float = 0.5, hm_type: str = 'gaussian'):
        super(JointsAcc, self).__init__()
        self.dist_thres = dist_thres
        self.hm_type = hm_type

    def forward(self, output, target):
        idx = list(range(output.shape[1]))
        norm = 1.0
        if self.hm_type == 'gaussian':
            pred, _ = self.get_max_preds(output)
            target, _ = self.get_max_preds(target)
            h = output.shape[2]
            w = output.shape[3]
            norm = torch.ones((pred.shape[0], 2), device=output.device) * torch.tensor([h, w], device=output.device)/10
        dists = self.calc_dists(pred, target, norm)

        acc = torch.zeros(len(idx), device=target.device)
        avg_acc = 0
        cnt = 0
        for i in range(len(idx)):
            acc[i] = self.dist_acc(dists[idx[i]])
            if acc[i] >= 0:
                avg_acc = avg_acc + acc[i]
                cnt += 1
        avg_acc = avg_acc / cnt if cnt != 0 else 0
        return acc, avg_acc, cnt, pred

    @staticmethod
    def get_max_preds(batch_heatmaps: torch.Tensor):
        assert isinstance(batch_heatmaps, torch.Tensor), 'batch_heatmaps should be torch.Tensor'
        assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

        batch_size, num_points, _, width = batch_heatmaps.shape
        heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_points, -1))
        indices = torch.argmax(heatmaps_reshaped, dim=2)
        maxvals = torch.amax(heatmaps_reshaped, dim=2)
        maxvals = maxvals.reshape((batch_size, num_points, 1))
        indices = indices.reshape((batch_size, num_points, 1))

        preds = torch.tile(indices, (1, 1, 2)).float()
        preds[:, :, 0] = (preds[:, :, 0]) % width
        preds[:, :, 1] = torch.floor((preds[:, :, 1]) / width)

        pred_mask = torch.tile(torch.greater(maxvals, 0.0), (1, 1, 2))
        pred_mask = pred_mask.float()

        preds *= pred_mask
        return preds, maxvals

    def get_final_preds(self, post_process, batch_heatmaps, center, scale):
        coords, maxvals = self.get_max_preds(batch_heatmaps)

        heatmap_height = batch_heatmaps.shape[2]
        heatmap_width = batch_heatmaps.shape[3]

        if post_process:
            for batch_idx in range(coords.shape[0]):
                for point_idx in range(coords.shape[1]):
                    heatmap = batch_heatmaps[batch_idx][point_idx]
                    point_x = int(torch.floor(coords[batch_idx][point_idx][0] + 0.5))
                    point_y = int(torch.floor(coords[batch_idx][point_idx][1] + 0.5))
                    if 1 < point_x < heatmap_width - 1 and 1 < point_y < heatmap_height - 1:
                        diff = torch.tensor([
                                heatmap[point_y][point_x + 1] - heatmap[point_y][point_x - 1],
                                heatmap[point_y + 1][point_x] - heatmap[point_y - 1][point_x]], device=heatmap.device)
                        coords += 0.25 * torch.sign(diff)

        preds = coords.cpu().clone().numpy()
        coords = coords.cpu()
        for i in range(coords.shape[0]):
            preds[i] = transform_preds(
                coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
            )
        return preds, maxvals

    @staticmethod
    def calc_dists(preds, target, normalize):
        dists = torch.zeros((preds.shape[1], preds.shape[0]), device=preds.device)
        for n in range(preds.shape[0]):
            for c in range(preds.shape[1]):
                if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                    normed_preds = preds[n, c, :] / normalize[n]
                    normed_targets = target[n, c, :] / normalize[n]
                    dists[c, n] = torch.linalg.norm(normed_preds - normed_targets)
                else:
                    dists[c, n] = -1
        return dists

    def dist_acc(self, dists):
        dist_cal = torch.not_equal(dists, -1)
        num_dist_cal = dist_cal.sum()
        if num_dist_cal > 0:
            return torch.less(dists[dist_cal], self.dist_thres).sum() * 1.0 / num_dist_cal
        else:
            return -1
