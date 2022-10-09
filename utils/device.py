# author: GeniusLgx
# developTime: 2022/7/8 17:29
import numpy as np
import torch
import torch.nn as nn


class GpuDataParallel(object):
    def __init__(self):
        self.gpu_list = []
        self.output_device = None

    def set_device(self, device):
        if device == 'cpu':
            self.output_device == 'cpu'
            return

        if device is not None and torch.cuda.device_count():
            self.gpu_list = list(range(torch.cuda.device_count()))
            output_device = self.gpu_list[0]
        self.output_device = output_device if len(self.gpu_list) > 0 else "cpu"

    def model_to_device(self, model):
        model = model.to(self.output_device)
        if len(self.gpu_list) > 1:
            print('DataParallel', self.gpu_list)
            model = nn.DataParallel(model, device_ids=self.gpu_list, output_device=self.output_device)
        return model

    def data_to_device(self, data):
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(self.output_device, detype=torch.float32)
        return data.to(self.output_device)

    def criterion_to_device(self, loss):
        return loss.to(self.output_device)

    @staticmethod
    def load_weights(model, weights_path, ignore_weights):
        print('Load weights from {}.'.format(weights_path))
        try:
            weights = torch.load(weights_path, map_location=torch.device('cpu'))
            for w in ignore_weights:
                if weights.pop(w, None) is not None:
                    print('Sucessfully Remove Weights: {}.'.format(w))
                else:
                    print('Can Not Remove Weights: {}.'.format(w))
            model.load_state_dict(weights, strict=True)
        except RuntimeError:
            weights = torch.load(weights_path, map_location=torch.device('cpu'))['model_state_dict']
            for w in ignore_weights:
                if weights.pop(w, None) is not None:
                    print('Sucessfully Remove Weights: {}.'.format(w))
                else:
                    print('Can Not Remove Weights: {}.'.format(w))
            model.load_state_dict(weights, strict=True)
        return model

