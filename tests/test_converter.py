from PIL import Image
import numpy as np
import tensorflow as tf
from torchvision.transforms import functional as F
from torchvision import transforms


def test_resize():
    scale = 255.0
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tf_mean = tf.constant(mean, dtype=tf.float32, shape=(1, 1, 3))
    tf_std = tf.constant(std, dtype=tf.float32, shape=(1, 1, 3))
    filename = "data/0027b63df5e33cbe.jpg"
    image = Image.open(filename).convert("RGB")
    size = 256
    # Pytorch
    resized = F.resize(image, size)
    w, h = resized.size
    print("resized:", h, w)
    data1 = np.array(resized)
    tfs = [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]
    transformer = transforms.Compose(tfs)
    result1 = transformer(resized)
    print("result1:", result1.shape)

    # TF
    np_data = np.array(image)
    print("np_data:", np_data.shape)
    tf_data = tf.convert_to_tensor(np_data)
    print("tf_data:", tf_data.shape, tf_data.dtype)
    #tf_data = tf.cast(tf_data, tf.float32)
    print("tf_data:", tf_data.shape, tf_data.dtype)
    tf_resized = tf.image.resize(tf_data, [h, w], antialias=True)
    tf_resized = tf.round(tf_resized)
    print("tf_resized:", tf_resized.shape)
    data2 = tf_resized.numpy()
    tf_resized = (tf_resized/scale - tf_mean) / tf_std
    result2 = tf_resized.numpy().transpose(2, 0, 1)
    print("result2:", result2.shape)
    diff = np.abs(data1 - data2).sum()
    print("diff-sum:", diff)
    diff = np.abs(data1 - data2).mean()
    print("diff-mean:", diff)

    diff = np.abs(result1 - result2).sum()
    print("diff-sum:", diff)
    diff = np.abs(result1 - result2).mean()
    print("diff-mean:", diff)
    print("ok")


if __name__ == "__main__":
    test_resize()
