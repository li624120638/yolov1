# author: lgx
# date: 2022/7/7 21:52
import os


def check_and_create_dir(path):
    if not os.path.exists(path):
        path_tree = path.split('/')
        for i in range(len(path_tree)):
            cur_path = '/'.join(path_tree[:i+1])
            if not os.path.exists(cur_path):
                os.mkdir(cur_path)
