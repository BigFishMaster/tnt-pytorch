from tnt.utils.logging import logger
from functools import partial
import math


def constant(optimizer, epoch=None, step=None):
    pass


def stepwise(optimizer, epoch, step, decay_scale, start_lr, step_epochs):
    lr = start_lr * (decay_scale ** (epoch // step_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def cosine(optimizer, epoch, step, warmup_epochs, warmup_lr, lr_range, num_epochs, steps_each_epoch):
    if epoch < warmup_epochs:
        start_lr, end_lr = warmup_lr
        scale_lr = (end_lr - start_lr) / (warmup_epochs * steps_each_epoch)
        lr = start_lr + step * scale_lr
    else:
        lr_max, lr_min = lr_range
        cur_step = step - warmup_epochs * steps_each_epoch
        all_steps = (num_epochs - warmup_epochs) * steps_each_epoch
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * cur_step / all_steps))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Please define learning rate strategies before this line.
strategies = {k: v for k, v in globals().items() if "__" not in k and callable(v)}


class LRStrategy:
    def __init__(self, config):
        st = config["name"]
        if st not in strategies:
            logger.exception("{} is not in defined strategy list: {}.".format(st, strategies))

        if st == "constant":
            self.fn = constant
        elif st == "stepwise":
            start_lr = config.get("start_lr", 0.01)
            step_epochs = config.get("step_epochs", 30)
            decay_scale = config.get("decay_scale", 0.0002)
            self.fn = partial(stepwise, decay_scale=decay_scale,
                              start_lr=start_lr, step_epochs=step_epochs)
        elif st == "cosine":
            # number of epochs to warmup
            warmup_epochs = config.get("warmup_epochs", 5)
            # warmup from 0.001 --> 0.01
            warmup_lr = config.get("warmup_lr", [0.001, 0.01])
            # cosine decrease from 0.01 to 0.00001
            lr_range = config.get("lr_range", [0.01, 0.00001])
            # number of epochs to train
            num_epochs = config.get("num_epochs", 120)
            # steps for each epoch
            steps_each_epoch = config.get("steps_each_epoch", 5000)
            self.fn = partial(cosine, warmup_epochs=warmup_epochs, warmup_lr=warmup_lr,
                              lr_range=lr_range, num_epochs=num_epochs,
                              steps_each_epoch=steps_each_epoch)
        else:
            logger.exception("Unexcepted strategy: {}".format(st))

    def set_lr(self, optimizer, epoch, step):
        self.fn(optimizer=optimizer, epoch=epoch, step=step)

    def get_lr(self, optimizer):
        lr = optimizer.param_groups[0]["lr"]
        return lr




