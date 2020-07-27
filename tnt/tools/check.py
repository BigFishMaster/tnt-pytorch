import tensorflow as tf
import numpy as np
from PIL import Image

tf.autograph.set_verbosity(0)
tf.compat.v1.logging.set_verbosity(0)
imported = tf.saved_model.load("../result/saved_model/")
REQUIRED_SIGNATURE = "serving_default"
REQUIRED_OUTPUT = "global_descriptor"
found_signatures = list(imported.signatures.keys())
if REQUIRED_SIGNATURE in found_signatures:
    print("checking the signatures is ok.")

outputs = imported.signatures[REQUIRED_SIGNATURE].structured_outputs
if REQUIRED_OUTPUT in outputs:
    print("checking the output name is ok.")

embedding_fn = imported.signatures[REQUIRED_SIGNATURE]

image_path = "../../tests/data/0027b63df5e33cbe.jpg"
image_data = np.array(Image.open(str(image_path)).convert('RGB'))
image_tensor = tf.convert_to_tensor(image_data)
output = embedding_fn(image_tensor)[REQUIRED_OUTPUT].numpy()
print("output:", output)
