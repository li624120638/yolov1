

This repository implements YOLOv1 with Pytorch. [Paper Link](https://arxiv.org/abs/1506.02640)

I implemented it for studying purpose, and I used Pytorch functions as much as possible for simplicity, such as calculating IoU, non-maximum suppression operations. I didn't encapsulate the function for some same operation, so it may have some code overlap, but it is more clearer without many function calls.

What's more, I tried my best to reproduce the paper as much as possible(network architecture, loss function, data augmentation and so on). But I didn't pretrain on the Imagenet.

Finally, I hope this repsitory will help you to understand original YOLO algorithms. You can follow the step to train your own YOLOv1 model. I will publish my training results when my program finishes running.

>**Clone this repository**
>
>```
>git clone https://github.com/li624120638/yolov1
>```

> **[optional] Creating a Virtual Environment**
>
> ```
> conda create -n yolov1 python=3.8 -y
> conda activate yolov1
> ```

>**Install Pytorch**  [Link](https://pytorch.org/get-started/locally/)
>
>The version I installed is **pytorch-1.12.1, torchaudio-0.12.1, with cudatoolkit-11.6.0**
>
>```
>conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
>```

>**Install other package**
>
>```
>pip install -r requirements.txt
>```

>**Train model**
>
>If you haven't downloaded the VOC dataset yet, please set `download` to `true`, it will download VOC2007 trainval, test and VOC2012 trainval in `dataset_root`. VOC2007 test used as validate set, while other used as train set.
>
>```
>cd train_model
>python train_yolo.py --config ./configs/VOC2007_detection_yolov1.yaml
>```

>**Test model**
>
>Set `model_args['weights']` to be the path of model you want to test.
>
>```
>python train_yolo.py --config ./configs/VOC2007_detection_yolov1_vis.yaml
>```
>
>The results will be saved in `${work_dir}/visulization/test_predict`

