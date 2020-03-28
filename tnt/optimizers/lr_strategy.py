from tnt.utils.logging import logger
from functools import partial


def stepwise(optimizer, epoch, step, decay_scale, start_lr, step_epochs):
    lr = start_lr * (decay_scale ** (epoch // step_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def constant(optimizer, epoch=None, step=None):
    pass


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
            start_lr = config["start_lr"]
            step_epochs = config["step_epochs"]
            decay_scale = config["decay_scale"]
            self.fn = partial(stepwise, decay_scale=decay_scale,
                              start_lr=start_lr, step_epochs=step_epochs)
        else:
            logger.exception("Unexcepted strategy: {}".format(st))

    def set_lr(self, optimizer, epoch, step):
        self.fn(optimizer=optimizer, epoch=epoch, step=step)

    def get_lr(self, optimizer):
        lr = optimizer.param_groups[0]["lr"]
        return lr




