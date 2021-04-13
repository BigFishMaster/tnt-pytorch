import time
import os
import torch
import numpy as np
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from tnt.optimizers.lr_strategy import LRStrategy
from tnt.utils.logging import logger
from tnt.utils.io import save_checkpoint, load_checkpoint
from tnt.utils.statistics import Statistics
from tnt.utils.metric import Metric
from tnt.impls import *
from tnt.pretrainedmodels.models.facenet import face_model_names
from tnt.utils.accum_weights import AccumWeights


class ModelBuilder:
    def __init__(self, config):
        if config["model"]["name"] in face_model_names:
            input_size = config["image_size"] if config["image_size"] is not None else 112
            config["model"]["input_size"] = [input_size, input_size]
            self.model = FaceModelImpl.from_config(config["model"])
        else:
            self.model = ModelImpl.from_config(config["model"])
        self.multiple_pooling = config["model"]["multiple_pooling"]
        self.loss = LossImpl.from_config(config["loss"])
        if self.loss.name == "ClassBalancedLoss":
            for m in self.model.modules():
                if m.__class__.__name__ == "Linear":
                    torch.nn.init.constant_(m.bias, -np.log(config["loss"]["num_classes"] - 1))
        if self.loss.name in ["CosFaceLoss", "ArcFaceLoss", "MetricCELoss", "MultipleCosFaceLoss",
                              "CosFaceLossWithNeg"]:
            self.others = self.loss
        else:
            self.others = None
        self.optimizer = OptImpl.from_config(self.model, config["optimizer"], self.others)
        # keep weights of last layer
        self.keep_last_layer = config["keep_last_layer"]
        # set bn momentum: useful when accumulating steps
        bn_momentum = config["optimizer"].get("bn_momentum", None)
        if bn_momentum is not None and 0 <= bn_momentum <= 1:
            def set_bn_momentum(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.momentum = bn_momentum
            self.model.apply(set_bn_momentum)
            logger.info("BN layers use bn_momentum:{}".format(bn_momentum))
        # clip gradients
        self.clip_norm = config["optimizer"].get("clip_norm", None)
        if self.clip_norm is not None and self.clip_norm > 0:
            logger.info("gradients will be clipped to clip_norm:{}".format(self.clip_norm))
        # fix BatchNorm
        # it can be used when the performance is affected by batch size.
        self.fix_bn = config["optimizer"].get("fix_bn", False)
        self.fix_res = config["optimizer"].get("fix_res", False)
        self.fix_finetune = config["optimizer"].get("fix_finetune", False)
        self.use_accum_weights = config["optimizer"].get("use_accum_weights", False)
        # tensorboard logging
        self.tb_log = config["tb_log"]
        logger.info("fix batchnorm:{}".format(self.fix_bn))
        # accumulate steps
        self.accum_steps = config["optimizer"].get("accum_steps", 1)
        self.accum_steps = 1 if self.accum_steps is None else self.accum_steps
        if self.accum_steps is not None and self.accum_steps > 0:
            logger.info("gradients will be accumulated with steps:{}".format(self.accum_steps))
        # output test result to file
        out_cfg = config["data"].get("output")
        self.fout = None
        if out_cfg is None:
            self.out_mode = "top5"
            self.out_file = config["global"]["log_dir"] + "test.output"
        else:
            self.out_mode = out_cfg.get("mode", "top5")
            self.out_file = out_cfg.get("file")
            if self.out_file is None:
                self.out_file = config["global"]["log_dir"] + "test.output"

        self.metric = Metric(config["metric"])
        self.lr_strategy = LRStrategy(config["lr_strategy"])
        self.steps_each_epoch = config["lr_strategy"]["steps_each_epoch"]
        self.init_global(config["global"])
        self.init_state()

        # used to initialize tranforms (mean, std, color space, image size, range etc.)
        # Note: the model is DataParallel.module
        config["data"]["opts"] = self.model.module if hasattr(self.model, "module") \
                                 else self.model
        if config["image_size"] and hasattr(config["data"]["opts"], "input_size"):
            config["data"]["opts"].input_size = [3, config["image_size"], config["image_size"]]

        # multi-crops when testing.
        self.is_multicrop = config["five_crop"] or config["ten_crop"]
        config["data"]["opts"].five_crop = config["five_crop"]
        config["data"]["opts"].ten_crop = config["ten_crop"]
        # image transforms
        config["data"]["opts"].transform_type = config["transform_type"]
        config["data"]["opts"].image_scale = config["image_scale"]
        config["data"]["opts"].erase_count = config["erase_count"]
        config["data"]["opts"].preserve_aspect_ratio = config["preserve_aspect_ratio"] != 0
        config["data"]["opts"].random_erase = config["enable_random_erase"]
        config["data"]["opts"].random_resized_crop = config["enable_random_resized_crop"]
        config["data"]["opts"].random_crop = config["disable_random_crop"] is False
        config["data"]["opts"].random_hflip = config["disable_random_hflip"] is False
        config["data"]["opts"].box_extend = config["box_extend"]

        # hard sampling
        self.hard_sampling = config["data"]["sampler"].get("strategy") in ["knn_sampler"]
        self.disable_knn_build = config["disable_knn_build"]
        logger.info("whether hard sampling: {}".format(self.hard_sampling))

    def update(self, config):
        if self.loss.name == "ClassBalancedLoss":
            samples_per_class = config["data"]["samples_per_class"]
            self.loss.update(samples_per_class)

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
            self.train_steps = self.train_epochs * self.steps_each_epoch
            self.best_epoch = checkpoint.get("best_epoch", self.best_epoch)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            if self.others is not None:
                self.loss.load_state_dict(checkpoint["loss_state_dict"])
            logger.info("model resume: %s", self.resume)

        if self.weight:
            checkpoint = load_checkpoint(self.weight, self.gpu)
            if checkpoint is None:
                raise ValueError("weight can not be loaded from: {}".format(self.weight))
            state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
            logger.info("keep the weights of last layer:{}".format(self.keep_last_layer))
            if self.keep_last_layer:
                missing_keys, unexcepted_keys = self.model.load_state_dict(state_dict, strict=False)
                logger.info("In keep_last_layer, loading model weights, missing_keys:{}, unexcepted_keys:{}".format(
                    missing_keys, unexcepted_keys))
            else:
                last_layer_name = self.model.last_layer_name
                if list(state_dict.keys())[0].startswith("module"):
                    last_layer_name = "module." + last_layer_name
                ignore_keys = filter(lambda x:x.startswith(last_layer_name + "."), state_dict.keys())
                ignore_keys = list(ignore_keys)
                new_state_dict = OrderedDict()
                for key in state_dict.keys():
                    if key in ignore_keys:
                        continue
                    new_state_dict[key] = state_dict[key]
                missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
                logger.info("loading model weights, missing_keys:{}, unexcepted_keys:{}".format(
                    missing_keys, unexpected_keys))
            if self.others is not None and "loss_state_dict" in checkpoint:
                try:
                    self.loss.load_state_dict(checkpoint["loss_state_dict"])
                    logger.info("Load loss_state_dict success.")
                except:
                    logger.error("Load loss_state_dict failed.")
            logger.info("model weight: %s", self.weight)

    def init_global(self, config):
        self.resume = config["resume"]
        self.weight = config.get("weight", None)
        self.gpu = config["gpu"]
        self.writer = SummaryWriter(config["log_dir"]) if self.tb_log else None
        self.num_epochs = config["num_epochs"]
        self.save_epoch_steps = config["save_epoch_steps"]
        self.save_model_file = os.path.join(config["log_dir"], config["save_model_file"])
        self.save_checkpoint_file = os.path.join(config["log_dir"], config["save_checkpoint_file"])
        self.report_interval = config["report_interval"]

    def _out(self, output):
        with torch.no_grad():
            topk = int(self.out_mode[3:]) if self.out_mode != "raw" else -1
            if self.fout is not None and self.fout.closed is False:
                score = None
                if topk > 1:
                    maxk = min(topk, output.shape[1])
                    output = output.softmax(1)
                    score, pred = output.topk(maxk, 1, True, True)
                    score = score.cpu().numpy()
                    pred = pred.cpu().numpy()
                else:
                    pred = output
                    pred = pred.cpu().numpy()
                num = len(pred)
                for i in range(num):
                    if score is None:
                        out = " ".join([str(_) for _ in pred[i]])
                    else:
                        out = " ".join([str(p) + ":" + str(s) for p, s in zip(pred[i], score[i])])
                    self.fout.write(out+"\n")
                    self.fout.flush()
            else:
                return

    def _run_epoch(self, data_iter, mode="test"):
        if mode == "train":
            if self.use_accum_weights:
               accumer = AccumWeights(self.model)

            if self.others is not None:
                self.others.train()
            self.model.train()
            if self.fix_bn:
                def set_bn_eval(m):
                    classname = m.__class__.__name__
                    if classname.find('BatchNorm') != -1:
                        m.eval()
                self.model.apply(set_bn_eval)
            elif self.fix_res:
                # change the behavior of dropout and BN.
                self.model.eval()
                # fix parameters expect the last linear layer.
                # TODO: the linear layer must be a standalone module.
                last_bn = None
                last_ln = None
                for name, module in self.model.named_modules():
                    class_name = module.__class__.__name__
                    if class_name == "BatchNorm2d":
                        last_bn = module
                    if class_name == "Linear":
                        last_ln = module
                for param in self.model.parameters():
                    param.requires_grad = False
                for param in last_ln.parameters():
                    param.requires_grad = True
                # train the running mean and running var of the last BN.
                # the gradients of the scale and bias are not trained.
                last_bn.train()
            elif self.fix_finetune:
                self.model.eval()
                last_ln = None
                for name, module in self.model.named.modules():
                    class_name = module.__class__.__name__
                    if class_name == "Linear":
                        last_ln = module
                for param in self.model.parameters():
                    param.requires_grad = False
                for param in last_ln.parameters():
                    param.requires_grad = True
                if self.others is not None:
                    self.others.eval()
        else:
            if self.others is not None:
                self.others.eval()
            self.model.eval()
        start = time.time()
        report_stats = Statistics()
        learning_rate = -1
        step = -1
        self.optimizer.zero_grad()
        for step, batch in enumerate(data_iter):
            if mode != "test":
                if len(batch) == 2:
                    input, target = batch
                else:  # > 2:
                    input, target = batch[0], batch[1:]
            else:
                input, target = batch[0], None

            if self.gpu is not None:
                input = input.cuda(self.gpu, non_blocking=True)
            if self.is_multicrop:
                bs, ncrops, c, h, w = input.size()
                output = self.model(input.view(-1, c, h, w))
                output = output.view(bs, ncrops, -1).mean(1)
            else:
                output = self.model(input)
                # multiple pooling support:
                if self.multiple_pooling and target is not None:
                    target = target.reshape(-1, 1).repeat(1, 16).reshape(-1)

            if mode != "test":
                if torch.cuda.is_available():
                    if isinstance(target, list):
                        target = [t.cuda(self.gpu, non_blocking=True) for t in target]
                    else:
                        target = target.cuda(self.gpu, non_blocking=True)
                loss = self.loss(output, target)
                metric_output = None
                if isinstance(loss, tuple) and len(loss) == 2:
                    loss, metric_output = loss

                if mode == "train":
                    if self.hard_sampling:
                        with torch.no_grad():
                            data_iter.sampler.update(output, target)
                    loss.backward()
                    if (step+1) % self.accum_steps == 0:
                        self.lr_strategy.set_lr(optimizer=self.optimizer,
                                                epoch=self.train_epochs, step=self.train_steps)
                        self.train_steps += 1
                        if self.clip_norm is not None and self.clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        if self.use_accum_weights:
                            accumer.update_avg(self.model)

                if hasattr(self.loss, "loss1") and hasattr(self.loss, "loss2"):
                    out_loss = [loss.item()]
                    out_loss.append(self.loss.loss1)
                    out_loss.append(self.loss.loss2)
                else:
                    if len(loss.size()) > 0:
                        out_loss = [loss.sum().item()]
                    else:
                        out_loss = [loss.item()]
                if metric_output is not None:
                    batch_stats = self.metric(output=metric_output, target=target[-1] if isinstance(target, list) else target,
                                              loss=out_loss)
                else:
                    if self.loss.name == "PseudoLabelLoss":
                        target_output = target[0]
                    elif isinstance(target, list):
                        target_output = target[-1]
                    elif isinstance(target, tuple):
                        target_output = target[0]
                    else:
                        target_output = target
                    batch_stats = self.metric(output=output, target=target_output,
                                              loss=out_loss)
                report_stats.update(batch_stats)
                if (step+1) % self.report_interval == 0:
                    learning_rate = self.lr_strategy.get_lr(self.optimizer)
                    report_stats.print(mode, step+1, self.train_epochs+1, learning_rate, start)
                    if self.tb_log:
                        report_stats.log(mode, self.writer, learning_rate, self.train_steps)
                    start = time.time()
            else:
                logger.info("(%s) step %s; batch size: %s" % (mode, step+1, output.shape[0]))
                self._out(output)
                pass

        if mode == "train" and self.use_accum_weights:
            accumer.update_cur(self.model)

        report_stats.print(mode, step+1, self.train_epochs+1, learning_rate, start)
        if self.tb_log:
            report_stats.log("progress/"+mode, self.writer, learning_rate, self.train_epochs+1)
        return report_stats.avgloss()

    def run(self, train_iter=None, valid_iter=None, test_iter=None):
        if test_iter:
            self.fout = open(self.out_file, "w", encoding="utf8")
            with torch.no_grad():
                self._run_epoch(test_iter, mode="test")
                self.fout.close()
            return

        if valid_iter and not train_iter:
            with torch.no_grad():
                self._run_epoch(valid_iter, mode="valid")
            return

        if self.hard_sampling and (not self.disable_knn_build):
            train_iter.sampler.build(self.model)

        for epoch in range(self.start_epoch, self.num_epochs):
            self._run_epoch(train_iter, mode="train")
            if valid_iter:
                with torch.no_grad():
                    valid_loss = self._run_epoch(valid_iter, mode="valid")

                if valid_loss and self.min_valid_loss > valid_loss:
                    logger.info('validation loss reduced %.4f -> %.4f' % (self.min_valid_loss, valid_loss))
                    self.best_epoch = epoch + 1
                    self.min_valid_loss = valid_loss
                logger.info('the best model is epoch %d, with loss: %f.' % (self.best_epoch, self.min_valid_loss))
            if (epoch+1) % self.save_epoch_steps == 0:
                # state_dict from model and optimizer
                state={"epoch": epoch + 1,
                       "state_dict": self.model.state_dict(),
                       "min_valid_loss": self.min_valid_loss,
                       "best_epoch": self.best_epoch,
                       "optimizer": self.optimizer.state_dict()}
                # check the state_dict from loss
                if self.others is not None:
                    state.update({"loss_state_dict": self.loss.state_dict()})
                save_checkpoint(
                    state=state,
                    filename=self.save_model_file.format(epoch+1),
                    save_checkpoint_file=self.save_checkpoint_file)
            self.train_epochs += 1

        if self.tb_log:
            self.writer.close()

