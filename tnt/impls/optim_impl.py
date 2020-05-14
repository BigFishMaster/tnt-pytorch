import tnt.optimizers as optimizers


class OptImpl:
    def __init__(self, model, config, others=None):
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
        if others is None:
            all_parameters = model.parameters()
        else:
            all_parameters = [{"params": model.parameters()}, {"params": others.parameters()}]
        if optimizer_name == "SGD":
            lr = config["lr"]
            momentum = config["momentum"]
            weight_decay = config["weight_decay"]
            optimizer = optimizers.__dict__[optimizer_name](
                all_parameters, lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        elif optimizer_name == "RMSprop":
            lr = config["lr"]
            momentum = config["momentum"]
            weight_decay = config["weight_decay"]
            optimizer = optimizers.__dict__[optimizer_name](
                all_parameters, lr,
                momentum=momentum,
                weight_decay=weight_decay
            )

        self.optimizer = optimizer

    @classmethod
    def from_config(cls, model, config, others=None):
        return cls(model, config, others).optimizer
