# author: GeniusLgx
# developTime: 2022/7/8 17:50
import argparse
import glob
import importlib

import scipy.ndimage.interpolation
import torch
import numpy as np
from scipy import interpolate

def import_class(name):
    components = name.rsplit('.', 1)
    mod = importlib.import_module(components[0])
    mod = getattr(mod, components[1])
    return mod


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def fun(airs: np.ndarray = [1],
        bubbles: [str] = ['a']) -> None:
    for a in airs:
        for b in bubbles:
            print(a, b)


if __name__ == '__main__':
    import torch.nn as nn
    import torch.nn.functional as f


    # lstm = nn.LSTM(input_size = 39, hidden_size = 128, num_layers = 2, batch_first=True)
    # inp = torch.randn(64, 17, 39)
    # out, (hn, cn) = lstm(inp)
    # print(out.shape)
    # print(hn.shape)
    # print(cn.shape)
    # torch.random.manual_seed(0)
    # #
    # a = np.random.randint(0, 10, size=(2, 4, 4)).astype(dtype=np.float32)
    # print(a)
    # b = scipy.ndimage.interpolation.zoom(a, [1, 0.5 ,1])
    # import pickle
    # print(pickle.dumps(a))
    #
    #
    # c = pickle.loads(pickle.dumps(a))
    # print(c)
    # from scipy.signal import medfilt
    # c = np.array([[1,2], [1,2,3]])
    # print(c)
    # import pickle
    #
    # save_root = 'E:/CMB/workplace/output_data_process/shrec/'
    # with open(save_root + 'train_y28.pkl', 'rb') as file:
    #     b =pickle.load(file)
    # print(max(b))

    # import yaml
    # with open('test.yaml', 'r') as file:
    #     a = yaml.load(file, yaml.FullLoader)
    #     print(type(a['test']))

    # import numpy as np
    # import collections
    # import sys
    # sys.path.append('../')
    # from data_process.labels_info import labels
    # root_dir = 'E:/datasets/gestures/HandGestureDataset_SHREC2017/'
    # train_txt_path = 'train_gestures.txt'
    # lines = np.loadtxt(root_dir+train_txt_path, dtype=np.int8)
    # mp = collections.defaultdict()
    # for line in lines:
    #     _1, _2, _, _, cls_14, cls_28, frame_num = line
    #     mp[(line[0], line[1])] = line[-2]
    # for k, v in mp.items():
    #     print(labels[k[0]-1], k[1], v )

    # a = torch.empty(3, dtype=torch.long).random_(5)
    # print(a)

    # tmp = np.array([1,2,3])
    # print((tmp[tmp==0]).shape)

    # from scipy.ndimage.interpolation import zoom
    # a = np.random.randn(29)
    # print(a.shape)
    # b = zoom(a, 32/29)
    # print(b.shape)

    # a = np.load('E:\CMB\workplace\output_data_process\cvlab_npy\data1_bye-onehand_subject_6_0.npy', )
    # print(a.reshape(-1, 2, 21, 3))

    # from tqdm import tqdm
    # a = [1, 2, 3, 4]
    # for i, d in enumerate(tqdm(a)):
    #     print(d)

    # print(torch.backends.cudnn.deterministic )
    # print(torch.backends.cudnn.enabled)
    # print(torch.backends.cudnn.benchmark)

    # dataset_dir = 'E:/datasets/human_pose/NTU RGB+D/'
    # patterns = 'nturgbd_rgb_s001/nturgb+d_rgb/*_rgb.ai'
    # print((glob.glob(dataset_dir + patterns)) is None)

    # a = [1, 2 ,3]
    # print(a[-4: ])
    a = None
    b = 2
