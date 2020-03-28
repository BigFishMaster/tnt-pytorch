import models_example as models
from torch.nn.modules import Module

model_names = sorted(name for name in models.__dict__
    if not name.startswith("__")
    and callable(models.__dict__[name]))
print("models:", model_names)


class Model(Module):
    def __init__(self):
        super(Model, self).__init__()

m = Model()
m = m.cuda("cpu")
print("model:", m)