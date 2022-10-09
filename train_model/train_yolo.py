# author: lgx
# date: 2022-09-26 10:15:35
import os
import sys
import time
sys.path.append("../")
import shutil
import inspect
import argparse
from tqdm import tqdm
import yaml
import numpy as np
import torch
import torch.nn as nn
import tensorboard_logger
import torchvision.transforms as transforms
from utils.random_state import RandomState
from utils.device import GpuDataParallel
from utils.tools import import_class
from utils.optimizer import Optimizer
from utils.recorder import Recorder


ENABLE_TENSORBOARD = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True


def config():
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Framework for training keypoints detector')
    parser.add_argument('--config', default='./configs/VOC2007_detection_yolov1_vis.yaml',
                        help='path to the configuration file')
    parser.add_argument('--work-dir', default='./work_dir/temp', help='the work folder for storing results')
    parser.add_argument('--phase', default='train', help='must be train or test')
    parser.add_argument('--random-fix', type=bool, default=True, help='fix the random seed or not')
    parser.add_argument('--random-seed', type=int, default=42, help='the default value for random seed.')
    parser.add_argument('--device', type=str, default=0, help='the indexes of GPUs for training or testing')
    parser.add_argument('--print-log', type=bool, default=True, help='print logging or not')
    parser.add_argument('--log-interval', type=int, default=20, help='the interval for printing messages (#iteration)')
    parser.add_argument('--dataloader', default='dataloader.dataloader', help='data loader will be used')
    parser.add_argument('--dataset-root', default=None, help='dataset root path')
    parser.add_argument('--download', type=bool, default=False, help='whether download voc datasets')
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument('--batch-size', type=int, default=16, help='training batch size')

    parser.add_argument('--W', type=int, default=448, help='image width')
    parser.add_argument('--H', type=int, default=448, help='imamge height')
    # same notations in YOLOv1 paper
    parser.add_argument('--S', type=int, default=7, help='')
    parser.add_argument('--B', type=int, default=2, help='')
    parser.add_argument('--C', type=int, default=20, help='')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--nms_thres', type=float, default=0.5, help='iou threshold used for nms')
    parser.add_argument('--sigmoid', type=bool, default=True, help='do sigmoid for output')
    parser.add_argument('--softmax', type=bool, default=False, help='do softmax for output class proabilities')

    parser.add_argument('--dataset-args', type=dict, default=dict(), help='the arguments of dataset')
    parser.add_argument('--dataloader-args', type=dict, default=dict(), help='the arguments of dataloader')
    parser.add_argument('--model-args', type=dict, default=dict(), help='the arguments of model')
    parser.add_argument('--optimizer-args', type=dict, default=dict(),  help='the arguments of optimizer')
    parser.add_argument('--criterion-args', type=dict, default=dict(), help='the arguments of criterion')
    parser.add_argument('--evaluate-args', type=dict, default=dict(), help='the arguments of evalueta process')
    return parser


class Processor:
    def __init__(self, args):
        self.args = args
        self.save_arg()
        self.checkpoint_path = '{}/checkpoint.pt'.format(self.args.work_dir)
        print('work_dir', self.args.work_dir)
        if self.args.random_fix:
            self.rng = RandomState(seed=self.args.random_seed)

        self.device = GpuDataParallel()
        self.data_loader = {}
        self.topk = (1, 5)
        self.top_mAP = 0
        self.criterion_function = self.criterion()
        self.acc_function = self.accuracy()
        self.vis_function = self.visualization()
        self.model, self.optimizer = self.Loading()

        global ENABLE_TENSORBOARD
        if self.args.phase == 'test':
            ENABLE_TENSORBOARD = False

        self.recoder = Recorder(self.args.work_dir, True)
        if ENABLE_TENSORBOARD:
            tensorboard_logger.configure(self.args.work_dir)

    def Loading(self):
        self.device.set_device(self.args.device)
        self.load_data()

        print("Loading model")
        if self.args.model:
            model_class = import_class(self.args.model)
            shutil.copy2(inspect.getfile(model_class), self.args.work_dir)

            model = self.device.model_to_device(model_class(**self.args.model_args))
            if self.args.model_args['weights']:
                try:
                    print("Loading pretrained model...")
                    state_dict = torch.load(self.args.model_args['weights'])
                    for w in self.args.model_args['ignore_weights']:
                        if state_dict.pop(w, None) is not None:
                            print('Sucessfully Remove Weights: {}.'.format(w))
                        else:
                            print('Can Not Remove Weights: {}.'.format(w))
                    model.load_state_dict(state_dict, strict=True)
                    optimizer = Optimizer(model, self.args.optimizer_args)
                except RuntimeError:
                    print("Loading from checkpoint...")
                    state_dict = torch.load(self.args.model_args['weights'])
                    self.rng.set_rng_state(state_dict['rng_state'])
                    self.args.optimizer_args['start_epoch'] = state_dict["epoch"] + 1
                    print("Resuming from checkpoint: epoch {}".format(self.args.optimizer_args['start_epoch']))
                    model = self.device.load_weights(model, self.args.model_args['weights'], self.args.model_args['ignore_weights'])
                    optimizer = Optimizer(model, self.args.optimizer_args)
                    if self.args.optimizer_args['resume']:
                        optimizer.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
                        optimizer.optimizer.param_groups[0]['lr'] = self.args.optimizer_args['base_lr']
                        optimizer.optimizer.param_groups[0]['weight_decay'] = self.args.optimizer_args['weight_decay']
                        optimizer.scheduler.load_state_dict(state_dict["scheduler_state_dict"])

            else:
                optimizer = Optimizer(model, self.args.optimizer_args)
        else:
            raise ValueError("No Models.")
        print("Loading model finished.")
        return model, optimizer

    def load_data(self):
        print("Loading data")

        feeder = import_class(self.args.dataloader)
        shutil.copy2(inspect.getfile(feeder), self.args.work_dir)

        if self.device.gpu_list:
            self.args.batch_size *= len(self.device.gpu_list)

        self.data_loader = dict()

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_args = []
        valid_args = []
        test_args = []
        for k in self.args.dataset_args.keys():
            if k.startswith('train'):
                train_args.append(self.args.dataset_args[k])
            elif k.startswith('valid'):
                valid_args.append(self.args.dataset_args[k])
            else:
                test_args.append(self.args.dataset_args[k])
        if len(train_args) > 0 and train_args[0] != {} and self.args.phase == 'train':
            dataset = feeder(**train_args[0], transform=transform)
            for i in range(1, len(train_args)):
                if train_args[i] != {}:
                    dataset += feeder(**train_args[i], transform=transform)
            bat_s = self.args.dataloader_args['train']['batch_size']
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=bat_s if bat_s > 0 else len(dataset),
                shuffle=self.args.dataloader_args['train']['shuffle'],
                num_workers=self.args.dataloader_args['train']['num_workers'],
                pin_memory=self.args.dataloader_args['train']['pin_memory'],
                # collate_fn=collate_fn
            )

        if len(valid_args) > 0 and valid_args[0] != {}:
            dataset = feeder(**valid_args[0], transform=transform)
            for i in range(1, len(valid_args)):
                if valid_args[i] != {}:
                    dataset += feeder(**valid_args[i], transform=transform)
            bat_s = self.args.dataloader_args['valid']['batch_size']
            self.data_loader['valid'] = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=bat_s if bat_s > 0 else len(dataset),
                shuffle=self.args.dataloader_args['valid']['shuffle'],
                num_workers=self.args.dataloader_args['valid']['num_workers'],
                pin_memory=self.args.dataloader_args['valid']['pin_memory'],
                # collate_fn=collate_fn
            )

        if len(test_args) > 0 and test_args[0] != {}:
            dataset = feeder(**test_args[0], transform=transform)
            for i in range(1, len(test_args)):
                if test_args[i] != {}:
                    dataset += feeder(**test_args[i], transform=transform)
            bat_s = self.args.dataloader_args['test']['batch_size']
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=bat_s if bat_s > 0 else len(dataset),
                shuffle=self.args.dataloader_args['test']['shuffle'],
                num_workers=self.args.dataloader_args['test']['num_workers'],
                pin_memory=self.args.dataloader_args['test']['pin_memory'],
                # collate_fn=collate_fn
            )
        print("Loading data finished.")

    def criterion(self):
        if self.args.criterion_args['loss'] == 'CrossEntropyLoss':
            loss = nn.CrossEntropyLoss()
            return self.device.criterion_to_device(loss)
        else:
            criterion_class = import_class(self.args.criterion_args['loss'])
            return criterion_class(**self.args.criterion_args['loss_args'])

    def accuracy(self):
        criterion_class = import_class(self.args.criterion_args['accuracy'])
        acc_function = criterion_class(**self.args.criterion_args['accuracy_args'])
        return self.device.criterion_to_device(acc_function)

    def visualization(self):
        criterion_class = import_class(self.args.evaluate_args['visualize']['vis_tool'])
        vis_function = criterion_class(**self.args.evaluate_args['visualize']['vis_args'])
        return self.device.criterion_to_device(vis_function)

    def start(self):
        self.recoder.print_log('Parameters:\n{}\n'.format(str(vars(self.args))))

        if self.args.phase == 'train':
            for epoch in range(self.args.optimizer_args['start_epoch'], self.args.optimizer_args['num_epochs']+1):
                self.train(epoch)

                self.eval(epoch, loader_name=['valid'])

        elif self.args.phase == 'test':
            if self.args.model_args['weights'] is None:
                raise ValueError('Please appoint --weights.')
            self.recoder.print_log('Model:   {}.'.format(self.args.model))
            self.recoder.print_log('Weights: {}.'.format(self.args.model_args['weights']))

            self.eval(1, loader_name=['valid'])
            self.recoder.print_log('Evaluation Done.\n')

        self.recoder.print_log('Work_dir: {}'.format(self.args.work_dir))
        self.recoder.print_log('Top mAP: {}'.format(self.top_mAP))

    def train(self, epoch):
        self.model.train()

        # self.recoder.print_log('Training epoch: {}'.format(epoch))
        loader = self.data_loader['train']
        loss_values = []
        coord_loss_values, conf_loss_values, noobj_loss_values, cls_loss_values = [], [], [], []
        ap_values = []

        current_learning_rate = [group['lr'] for group in self.optimizer.optimizer.param_groups]
        if ENABLE_TENSORBOARD:
            tensorboard_logger.log_value('learning-rate', current_learning_rate[0], epoch)

        self.recoder.timer_reset()
        with tqdm(loader) as loop:
            for batch_idx, data in enumerate(loop):
                self.recoder.record_timer("load_data")
                img, target = data
                input = self.device.data_to_device(img)
                target = self.device.data_to_device(target)
                self.recoder.record_timer("device")
                output = self.model(input)
                self.recoder.record_timer("forward")
                loss, coord_loss, conf_loss, cls_loss, noobj_loss = self.criterion_function(target, output)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.recoder.record_timer("backward")
                loss_values.append(loss.item())
                coord_loss_values.append(coord_loss.item())
                conf_loss_values.append(conf_loss.item())
                noobj_loss_values.append(noobj_loss.item())
                cls_loss_values.append(cls_loss.item())
                with torch.no_grad():
                    ap = self.acc_function(target, output)
                ap_values.append(ap.item())
                loop.set_description("[ " + time.asctime() + ' ] ' + f'Train Epoch: {epoch}')
                info_dict = {'Loss': '{:.8f}'.format(loss_values[-1]), 'mAP': '{:.6f}'.format(ap_values[-1] * 100),
                             'lr': '{:.6f}'.format(current_learning_rate[0])}
                loop.set_postfix(info_dict)

        mAP = np.mean(ap_values)
        train_loss = np.mean(loss_values)
        coord_loss = np.mean(coord_loss_values)
        conf_loss = np.mean(conf_loss_values)
        noobj_loss = np.mean(noobj_loss_values)
        cls_loss = np.mean(cls_loss_values)
        self.optimizer.step_scheduler(train_loss)

        self.recoder.print_log('train loss: {:.5f}  mAP: {:.6f}.'.format(train_loss, mAP * 100))
        self.recoder.print_log('coord-loss: {:.5f}  conf-loss: {:.5f}  noobj-loss: {:.5f}  cls-loss: {:.5f}.'
                               .format(coord_loss, conf_loss, noobj_loss, cls_loss))
        self.recoder.print_time_statistics()
        if ENABLE_TENSORBOARD:
            tensorboard_logger.log_value('mAP', mAP, epoch)
            tensorboard_logger.log_value('train-loss', train_loss, epoch)
        self.save_state(epoch, self.checkpoint_path)

    def eval(self, epoch, loader_name):
        self.model.eval()
        for l_name in loader_name:
            loader = self.data_loader[l_name]

            loss_values = []
            coord_loss_values, conf_loss_values, noobj_loss_values, cls_loss_values = [], [], [], []
            ap_values = []
            with torch.no_grad():
                with tqdm(loader) as loop:
                    for batch_idx, data in enumerate(loop):
                        imgs, target = data
                        input = self.device.data_to_device(imgs)
                        target = self.device.data_to_device(target)
                        output = self.model(input)

                        loss, coord_loss, conf_loss, noobj_loss, cls_loss = self.criterion_function(target, output)
                        loss_values.append(loss.item())
                        coord_loss_values.append(coord_loss.item())
                        conf_loss_values.append(conf_loss.item())
                        noobj_loss_values.append(noobj_loss.item())
                        cls_loss_values.append(cls_loss.item())

                        ap = self.acc_function(target, output)
                        ap_values.append(ap.item())
                        if self.args.evaluate_args['visualize']['flag']:
                            prefix = os.path.join(loader.dataset.vis_dir, '{}_predict'.format(loader.dataset.phase),
                                                  'batch_{}.jpg'.format(batch_idx))
                            self.vis_function(imgs, output, prefix)
                        loop.set_description("[ " + time.asctime() + ' ] ' + f'Test Epoch: {epoch}')
                        info_dict = {'Loss': '{:.8f}'.format(loss_values[-1]), 'mAP': '{:.6f}'.format(ap_values[-1] * 100)}
                        loop.set_postfix(info_dict)

        mAP = np.mean(ap_values)
        val_loss = np.mean(loss_values)
        coord_loss = np.mean(coord_loss_values)
        conf_loss = np.mean(conf_loss_values)
        noobj_loss = np.mean(noobj_loss_values)
        cls_loss = np.mean(cls_loss_values)

        if mAP > self.top_mAP:
            self.top_mAP = mAP
            model_best_path = '{}/best_mAP.pt'.format(self.args.work_dir)
            self.save_state(epoch, model_best_path)

        self.recoder.print_log(
            '  val loss: {:.5f}    mAP: {:.6f}    top mAP: {:.6f}.'.format(val_loss, mAP * 100,  self.top_mAP * 100))
        self.recoder.print_log('coord-loss: {:.5f}  conf-loss: {:.5f}  noobj-loss: {:.5f}  cls-loss: {:.5f}.'
                               .format(coord_loss, conf_loss, noobj_loss, cls_loss))
        self.recoder.print_time_statistics()
        if ENABLE_TENSORBOARD:
            tensorboard_logger.log_value('val-loss', val_loss, epoch)
            tensorboard_logger.log_value('val-mAP', mAP, epoch)

    def save_state(self, epoch, path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.optimizer.state_dict(),
            'scheduler_state_dict': self.optimizer.scheduler.state_dict(),
            'rng_state': self.rng.get_rng_state()
        }, path)

    def save_arg(self):
        arg_dict = vars(self.args)
        if not os.path.exists(self.args.work_dir):
            os.makedirs(self.args.work_dir)
        with open('{}/config.yaml'.format(self.args.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)


if __name__ == '__main__':
    global parser, args
    parser = config()
    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, 'r') as f:
            try:
                yaml_arg = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError:
                yaml_arg = yaml.load(f)
        key = vars(args).keys()
        for k in yaml_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**yaml_arg)
    args = parser.parse_args()

    processor = Processor(args)
    processor.start()

