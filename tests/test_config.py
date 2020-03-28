import yaml
import argparse
from functools import partial

config = yaml.load(open("data/training_example.yml", "r"), Loader=yaml.FullLoader)
#print("config:", config)
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument("--params", type=partial(yaml.load, Loader=yaml.FullLoader), nargs='+',
          help="initialize parameters from yaml file.")

fin = open("data/training_example.yml", "r")
#default_opts = ["--params", "data/training_example.yml"]
default_opts = ["--params", fin]
opt = parser.parse_args(default_opts)
print("opt:", opt)
