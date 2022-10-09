# author: lgx
# date: 2022/7/30 19:35
# description: preprocessed shrec17 datas' loader implementation
import torch
import torch.utils.data as data
import pickle


class Shrec17Loader(data.Dataset):
    def __init__(self, dataset_dir, file_list,
                 cls_num=14, pts_size=22, pts_dim=3, **kwargs):
        super(Shrec17Loader, self).__init__()
        with open(dataset_dir+file_list[0], 'rb') as file:
            self.xs = pickle.load(file)
        with open(dataset_dir+file_list[1], 'rb') as file:
            self.ys = pickle.load(file)
        self.len = len(self.ys)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]


if __name__ == '__main__':
    shrec17 = Shrec17Loader('E:/CMB/workplace/output_data_process/shrec/',
                            ['train_x.pkl', 'train_y28.pkl'])
    loader = data.DataLoader(shrec17, shuffle=False, batch_size=64)
    with open('E:/CMB/workplace/output_data_process/shrec/train_x.pkl', 'rb') as file:
        xs = pickle.load(file)
    with open('E:/CMB/workplace/output_data_process/shrec/train_y28.pkl', 'rb') as file:
        ys = pickle.load(file)
    for x, y in loader:
        break


    print(y)





