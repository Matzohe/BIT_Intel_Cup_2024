import torch
import math


class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, init_lr, all_steps, start_steps=0):
        self._optimizer = optimizer
        self.all_steps = all_steps
        self.n_current_steps = 0
        self.init_lr = init_lr
        self.start_steps = start_steps

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        if (self.n_current_steps - self.start_steps) < ( (self.all_steps - self.start_steps) / 1.2) and\
                self.n_current_steps >= self.start_steps:
            return 1 + math.cos((self.n_current_steps - self.start_steps) / (self.all_steps
                                                                             - self.start_steps) * math.pi)
        elif self.n_current_steps < self.start_steps:
            return 2
        else:
            return 1 + math.cos((1 / 1.2) * math.pi)

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
