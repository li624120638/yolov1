# author: GeniusLgx
# developTime: 2022/7/8 17:57
import torch
import torch.optim as optim


class Optimizer(object):
    def __init__(self, model, optim_dict):
        self.optim_dict = optim_dict
        if self.optim_dict["optimizer"] == 'SGD':
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=self.optim_dict['base_lr'],
                momentum=0.9,
                nesterov=self.optim_dict['nesterov'],
                weight_decay=self.optim_dict['weight_decay']
            )
        elif self.optim_dict["optimizer"] == 'Adam':
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=self.optim_dict['base_lr'],
                weight_decay=self.optim_dict['weight_decay']
            )
        else:
            raise ValueError()
        self.scheduler = self.define_lr_scheduler(self.optimizer)

    @property
    def name(self):
        return self.optim_dict["optimizer"] + '.' + self.optim_dict["lr_scheduler"]

    def define_lr_scheduler(self, optimizer):
        if self.optim_dict["lr_scheduler"] == 'plateau':
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.1, patience=10, cooldown=0, min_lr=5e-6, verbose=True)
        elif self.optim_dict["lr_scheduler"] == 'MultiStepLR':
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.optim_dict['step'], gamma=0.1)
        return lr_scheduler

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def step_scheduler(self, loss=None):
        if self.optim_dict["lr_scheduler"] == 'plateau':
            self.scheduler.step(loss)
        else:
            self.scheduler.step()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)




