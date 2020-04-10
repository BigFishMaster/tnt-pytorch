import sys
import time
from tnt.dataloaders.data_loader import GeneralDataLoader as GDLoader
from tnt.runner import create_config
from tnt.model_builder import ModelBuilder

# TEST the samples from one batch is in order.
argv = sys.argv
argv += [
    "--config",
    "../config/test_config.yaml"
]
config = create_config()
builder = ModelBuilder(config)
test_iter = GDLoader.from_config(cfg=config["data"], mode="test")
start = time.time()
for step, batch in enumerate(test_iter):
    print("batch:", batch[1])
    pass
print("time:", time.time() - start)
