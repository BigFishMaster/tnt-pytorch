## Installation

### Requirements

- Linux, Windows or MacOS
- Python 3.6+
- PyTorch 1.1+
- CUDA 9.1 or 10.1


### Install tnt-pytorch

a. Create a conda virtual environment and activate it.

```shell
conda create -n tnt-pytorch python=3.6 -y
conda activate tnt-pytorch
```


b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch torchvision -c pytorch
```

Note: Make sure that your compilation CUDA version and runtime CUDA version match.
You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

`E.g.1` If you have CUDA 10.1 installed under `/usr/local/cuda` and would like to install
PyTorch 1.4, you need to install the prebuilt PyTorch with CUDA 10.1.

```python
conda install pytorch=1.4.0 cudatoolkit=10.1 torchvision=0.5.0 -c pytorch
```

`E.g. 2` If you have CUDA 9.1 installed under `/usr/local/cuda` and would like to install
PyTorch 1.1.0, you need to install the prebuilt PyTorch with CUDA 9.1.

```python
conda install pytorch=1.1.0 cudatoolkit=9.1 torchvision=0.3.0 -c pytorch
```

If you build PyTorch from source instead of installing the prebuilt pacakge,
you can use more CUDA versions such as 9.0.

c. Clone the tnt-pytorch repository.

```shell
git clone https://github.com/BigFishMaster/tnt-pytorch.git
cd tnt-pytorch
```

d. Install python requirements and then install tnt-pytorch.

```shell
pip install -r requirements.txt
python setup.py install
```

### Install with CPU only
The tnt-pytorch can be installed for CPU only environment (where CUDA isn't available). Please create CPU enviroment
 instead:

```shell
conda install pytorch torchvision cpuonly -c pytorch
```

### Optional: Docker Image

We provide a [Dockerfile](https://github.com/BigFishMaster/tnt-pytorch/blob/master/Dockerfile) to build an image.

```shell
# build an image with PyTorch 1.4, CUDA 10.1
docker build -t tnt:latest .
```

Run it with

```shell
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/tnt-pytorch/data bash
```

### A from-scratch setup script

Here is a shell script for setting up tnt-pytorch with conda.

```shell
conda create -n tnt-pytorch python=3.6 -y
conda activate tnt-pytorch

# install latest pytorch prebuilt with the default prebuilt CUDA version (usually the latest)
conda install -c pytorch pytorch torchvision -y
git clone https://github.com/BigFishMaster/tnt-pytorch.git
cd tnt-pytorch
pip install -r requirements.txt
python setup.py install
```
