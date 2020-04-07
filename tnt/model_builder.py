import time
import sys
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import tnt.losses as losses
import tnt.optimizers as optimizers
from tnt.optimizers.lr_strategy import LRStrategy
from tnt.utils.logging import logger
from tnt.utils.io import load_model_from_file, save_checkpoint, load_checkpoint
from tnt.utils.statistics import Statistics
from tnt.metric import Metric
import tnt.pretrainedmodels as pretrainedmodels


class ModelImpl:
    def __init__(self, model_name_or_path, num_classes, pretrained=None, gpu=None):
        if os.path.exists(model_name_or_path):
            # TODO: test model_file mode.
            model_file = model_name_or_path
            model = load_model_from_file(model_file)
            if pretrained:
                state_dict = torch.load(pretrained)
                model.load_state_dict(state_dict)
        elif model_name_or_path in pretrainedmodels.model_names:
            model_name = model_name_or_path
            # TODO: if not pretrained, the below members will not be initialized:
            # input_space, input_size, input_range, mean, std
            model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
        else:
            logger.exception("'{}' is not available.".format(model_name_or_path))
            sys.exit()

        # TODO: fix the parameters.
        # e.g.
        # for param in model.parameters():
        #     param.requires_grad = False

        # TODO: initialize the last_linear layer
        # url: https://pytorch.org/docs/master/notes/autograd.html
        if hasattr(model, "last_linear"):
            in_features = model.last_linear.in_features
            out_features = model.last_linear.out_features
            if out_features != num_classes:
                model.last_linear = nn.Linear(in_features, num_classes)
        else:  #  model.fc
            in_features = model.fc.in_features
            out_features = model.fc.out_features
            if out_features != num_classes:
                model.fc = nn.Linear(in_features, num_classes)

        if gpu is not None:
            torch.cuda.set_device(gpu)
            model = model.cuda(gpu)
        else:
            if torch.cuda.is_available():
                model = torch.nn.DataParallel(model).cuda()

        self.model = model
        self.gpu = gpu

    @classmethod
    def from_config(cls, config):
        model_name = config["name"]
        pretrained = config["pretrained"]
        gpu = config["gpu"]
        num_classes = config["num_classes"]
        self = cls(model_name, num_classes, pretrained, gpu)
        return self.model


class LossImpl:
    def __init__(self, loss_name, gpu):
        loss = losses.__dict__[loss_name]()
        loss.cuda(gpu)
        self.loss = loss
        self.gpu = gpu

    @classmethod
    def from_config(cls, config):
        loss_name = config["name"]
        gpu = config["gpu"]
        return cls(loss_name, gpu).loss


class OptImpl:
    def __init__(self, model, config):
        optimizer_name = config["name"]
        optimizer = None
        # TODO: per-layer learning rates
        # url: https://pytorch.org/docs/stable/optim.html
        # discuss: https://discuss.pytorch.org/t/how-to-perform-finetuning-in-pytorch/419/7
        # e.g. 1:
        #    ignored_params = list(map(id, model.fc.parameters()))
        #    base_params = filter(lambda p: id(p) not in ignored_params,
        #                         model.parameters())

        #    optimizer = torch.optim.SGD([
        #        {'params': base_params},
        #        {'params': model.fc.parameters(), 'lr': opt.lr}
        #    ], lr=opt.lr * 0.1, momentum=0.9)
        # e.g. 2:
        #    optim.SGD([
        #        {'params': model.base.parameters()},
        #        {'params': model.classifier.parameters(), 'lr': 1e-3}
        #    ], lr=1e-2, momentum=0.9)
        if optimizer_name == "SGD":
            lr = config["lr"]
            momentum = config["momentum"]
            weight_decay = config["weight_decay"]
            optimizer = optimizers.__dict__[optimizer_name](
                model.parameters(), lr,
                momentum=momentum,
                weight_decay=weight_decay
            )

        self.optimizer = optimizer

    @classmethod
    def from_config(cls, model, config):
        return cls(model, config).optimizer


class ModelBuilder:
    def __init__(self, config):
        self.model = ModelImpl.from_config(config["model"])
        self.loss = LossImpl.from_config(config["loss"])
        self.optimizer = OptImpl.from_config(self.model, config["optimizer"])
        self.metric = Metric(config["metric"])
        self.lr_strategy = LRStrategy(config["lr_strategy"])
        self.init_global(config["global"])
        self.init_state()

        # used to initialize tranforms (mean, std, color space, image size, range etc.)
        # Note: the model is DataParallel.module
        config["data"]["opts"] = self.model.module if hasattr(self.model, "module") \
                                 else self.model
        if config["image_size"] and hasattr(config["data"]["opts"], "input_size"):
            config["data"]["opts"].input_size = [3, config["image_size"], config["image_size"]]

    def init_state(self):
        self.min_valid_loss = 1.0e+10
        self.train_steps = 0
        self.train_epochs = 0
        self.start_epoch = 0
        self.best_epoch = 0

        if self.resume:
            checkpoint = load_checkpoint(self.resume, self.gpu)
            if checkpoint is None:
                raise ValueError("checkpoint is not valid: {}".format(checkpoint))
            self.min_valid_loss = checkpoint.get("min_valid_loss", self.min_valid_loss)
            self.start_epoch = checkpoint.get("epoch", self.start_epoch)
            self.train_epochs = self.start_epoch
            self.best_epoch = checkpoint.get("best_epoch", self.best_epoch)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

    def init_global(self, config):
        self.resume = config["resume"]
        self.gpu = config["gpu"]
        self.writer = SummaryWriter(config["log_dir"])
        self.num_epochs = config["num_epochs"]
        self.save_epoch_steps = config["save_epoch_steps"]
        self.save_model_file = os.path.join(config["log_dir"], config["save_model_file"])
        self.save_checkpoint_file = os.path.join(config["log_dir"], config["save_checkpoint_file"])
        self.report_interval = config["report_interval"]

    def _run_epoch(self, data_iter, mode="test"):
        if mode == "train":
            self.model.train()
        else:
            self.model.eval()
        start = time.time()
        # TODO: 'topk' parameter should be initialized ?
        report_stats = Statistics()
        learning_rate = -1
        step = -1
        for step, batch in enumerate(data_iter):
            if mode != "test":
                input, target = batch
            else:
                # TODO: process the output when testing
                input, target = batch, None

            if self.gpu is not None:
                input = input.cuda(self.gpu, non_blocking=True)
            output = self.model(input)

            if mode != "test":
                if torch.cuda.is_available():
                    target = target.cuda(self.gpu, non_blocking=True)
                loss = self.loss(output, target)

                if mode == "train":
                    self.train_steps += 1
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                batch_stats = self.metric(output=output, target=target, loss=loss.item())
                report_stats.update(batch_stats)
                if (step+1) % self.report_interval == 0 and mode == "train":
                    learning_rate = self.lr_strategy.get_lr(self.optimizer)
                    current_step = self.train_steps
                    report_stats.print(mode, step+1, self.train_epochs, learning_rate, start)
                    report_stats.log(mode, self.writer, learning_rate, current_step)
                    start = time.time()
            else:
                # TODO: 'test' mode is not implemented. save prediction result ?
                pass
        report_stats.print(mode, step+1, self.train_epochs, learning_rate, start)
        report_stats.log("progress/"+mode, self.writer, learning_rate, self.train_epochs)
        return report_stats.avgloss()

    def run(self, train_iter=None, valid_iter=None, test_iter=None):
        if test_iter:
            self._run_epoch(test_iter, mode="test")
            return

        if valid_iter and not train_iter:
            self._run_epoch(valid_iter, mode="valid")
            return

        for epoch in range(self.start_epoch, self.num_epochs):
            # TODO: update lr for each step is not implemented.
            self.lr_strategy.set_lr(optimizer=self.optimizer, epoch=epoch, step=self.train_steps)
            self.train_epochs += 1
            self._run_epoch(train_iter, mode="train")
            if valid_iter:
                with torch.no_grad():
                    valid_loss = self._run_epoch(valid_iter, mode="valid")

                if valid_loss and self.min_valid_loss > valid_loss:
                    logger.info('validation loss reduced %.4f -> %.4f' % (self.min_valid_loss, valid_loss))
                    self.best_epoch = epoch + 1
                    self.min_valid_loss = valid_loss
                logger.info('the best model is epoch %d, with loss: %f.' % (self.best_epoch, self.min_valid_loss))
                if epoch % self.save_epoch_steps == 0:
                    save_checkpoint(
                        state={"epoch": epoch + 1,
                               "state_dict": self.model.state_dict(),
                               "min_valid_loss": self.min_valid_loss,
                               "best_epoch": self.best_epoch,
                               "optimizer": self.optimizer.state_dict()},
                        filename=self.save_model_file.format(epoch+1),
                        save_checkpoint_file=self.save_checkpoint_file
                    )

        self.writer.close()

