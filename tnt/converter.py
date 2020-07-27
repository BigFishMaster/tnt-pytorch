import os
import numpy as np
import onnx
from onnx_tf.backend import prepare
import torch
import tensorflow as tf
import tnt.pretrainedmodels as models
import onnxruntime as ort

import argparse
import tnt.opts as opts
from tnt.utils.logging import init_logger, logger, beautify_info
from collections import OrderedDict
from PIL import Image
from torchvision import transforms

parser = argparse.ArgumentParser(description="Converter for python to tensorflow transfering.")


def wrap_frozen_graph(graph_def, inputs, outputs):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")
    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph
    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))


class CustomModule(tf.Module):

    def __init__(self, model_pb_file, l2norm=True, scale=255.0, target=None,
                 size=None, preserve_aspect_ratio=False, mean=None, std=None):
        """
        Args:
            model_pb_file: tensorflow graph file.
            l2norm: whether use l2-norm or not. Default: True
            scale: the scale to divide from raw pixels. Default: 224
            size: size vector for model input.
            preserve_aspect_ratio: preserve the aspect ratio or not. Defautl: False
            mean: mean vector.
            std: std vector.
        """
        super(CustomModule, self).__init__()
        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(open(model_pb_file, "rb").read())
        self.model_func = wrap_frozen_graph(graph_def, inputs="input1:0", outputs="output1:0")
        self.l2norm = l2norm
        self.scale = scale
        self.target = target
        self.size = size
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.mean = mean
        self.std = std

    @tf.function(input_signature=[tf.TensorSpec([None, None, 3], tf.uint8, name="input_image")])
    def extract_feature(self, image):
        #TODO: check the order of h and w.
        h = tf.shape(image)[0]
        w = tf.shape(image)[1]
        image_data = tf.cast(image, tf.float32)

        def resize_and_extract(size, target):
            if self.preserve_aspect_ratio:
                if (w <= h and w == size) or (h <= w and h == size):
                    resized_image = tf.image.resize_with_crop_or_pad(image_data, target, target)
                elif w < h:
                    ow = size
                    oh = tf.cast(size * h / w, tf.int32)
                    scaled_image = tf.image.resize(image_data, [oh, ow], antialias=True)
                    scaled_image = tf.round(scaled_image)
                    resized_image = tf.image.resize_with_crop_or_pad(scaled_image, target, target)
                else:
                    oh = size
                    ow = tf.cast(size * w / h, tf.int32)
                    scaled_image = tf.image.resize(image_data, [oh, ow], antialias=True)
                    scaled_image = tf.round(scaled_image)
                    resized_image = tf.image.resize_with_crop_or_pad(scaled_image, target, target)
            else:
                scaled_image = tf.image.resize(image_data, [size, size], antialias=True)
                scaled_image = tf.round(scaled_image)
                resized_image = tf.image.resize_with_crop_or_pad(scaled_image, target, target)
            preprocess = resized_image / self.scale
            preprocess = (preprocess - self.mean) / self.std
            # shape: 3 x 224 x 224
            preprocess = tf.transpose(preprocess, perm=[2, 0, 1])
            # shape: 1 x 3 x 224 x 224
            preprocess = tf.expand_dims(preprocess, axis=0)
            # shape: 1 x 2048
            output = self.model_func(preprocess)
            return output

        output_global = resize_and_extract(self.size[0], self.target[0])
        num_size = tf.shape(self.size)[0]
        for ind in tf.range(1, num_size):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(output_global, tf.TensorShape([None, None]))])
            global_descriptor = resize_and_extract(self.size[ind], self.target[ind])
            output_global = tf.concat([output_global, global_descriptor], 0)

        #tf.print("output_global:", output_global.shape)
        # average multi-scale features
        output_global = tf.reduce_mean(output_global, axis=0, keepdims=False, name="global_mean")

        # l2-normalize the feature.
        if self.l2norm:
            output_global = tf.nn.l2_normalize(output_global, axis=0, name="global_l2norm")

        # output the feature with a key.
        named_output_tensor = {"global_descriptor": tf.identity(
            output_global, name="global_descriptor")}
        return named_output_tensor


def test_pytorch_model(model, image, target):
    mean = model.mean
    std = model.std
    size = int(target/0.875)
    tfs = [
        transforms.Resize((size, size)),  # 256 x 256
        transforms.CenterCrop(target),  # 224 x 224
        transforms.ToTensor(),  # [0, 1]
        transforms.Normalize(mean=mean, std=std)  # normalize
    ]
    transformer = transforms.Compose(tfs)
    torch_input = transformer(image)
    torch_input = torch_input.unsqueeze(0)
    #print("torch_input:", torch_input.shape)
    model.eval()
    with torch.no_grad():
        torch_out = model(torch_input)
    logger.info("pytorch model output:{}, shape:{}".format(torch_out[0, :10], torch_out.shape))

    return torch_input, torch_out


def convert(config):
    torch.random.manual_seed(0)
    model_name = config["model_name"]
    weight_path = config["weight"]
    image_target_list = [int(sz) for sz in config["image_size"].split(",")]
    image_size_list = [int(sz / 0.875) for sz in image_target_list]
    image_size = image_target_list[0]
    output_dir = config["output_dir"]
    image_path = config["image_path"]
    preserve_aspect_ratio = config["preserve_aspect_ratio"]
    l2norm = config["l2norm"]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    extract_feature = config["extract_feature"]
    if extract_feature:
        kwargs = {"extract_feature": extract_feature}
        model = models.__dict__[model_name](pretrained=None, **kwargs)
    else:
        model = models.__dict__[model_name](pretrained=None)

    logger.info("loading model name {} is ok.".format(model_name))

    if os.path.exists(weight_path):
        checkpoint = torch.load(weight_path, map_location="cpu")
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        new_state_dict = OrderedDict()
        if list(state_dict.keys())[0].startswith("module"):
            for key in state_dict.keys():
                new_key = key.replace("module.", "")
                new_state_dict[new_key] = state_dict[key]
        else:
            new_state_dict = state_dict

        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        logger.info("loading model weights, missing_keys:{}, unexcepted_keys:{}"
                    .format(missing_keys, unexpected_keys))
    else:
        raise FileNotFoundError("the path of model weights does not exist.")

    output_onnx_name = os.path.join(output_dir, model_name + ".onnx")

    image = Image.open(image_path).convert("RGB")
    rgb_data = np.array(image)
    logger.info("The image shape: {}".format(rgb_data.shape))

    resized_image = image.resize((image_size, image_size))
    resized_rgb_data = np.array(resized_image)
    logger.info("The resized image shape: {}".format(resized_rgb_data.shape))
    transposed_rgb_data = resized_rgb_data.transpose(2, 0, 1)

    input = torch.Tensor(transposed_rgb_data).unsqueeze(0)
    logger.info("The input tensor shape: {}".format(input.shape))

    input_names = ["input1"]
    output_names = ["output1"]
    torch.onnx.export(model, (input,), output_onnx_name, verbose=True, opset_version=11,
                      input_names=input_names, output_names=output_names)

    logger.info("exporting model to {} is ok.".format(output_onnx_name))
    onnx_model = onnx.load(output_onnx_name)
    onnx.checker.check_model(onnx_model)
    print_graph = onnx.helper.printable_graph(onnx_model.graph)
    logger.info("onnx graph: {}".format(print_graph))
    logger.info("check the onnx model is ok.")
    logger.info("checking the difference between pytorch and onnx models.")

    torch_input, torch_out = test_pytorch_model(model, image, image_size)

    ort_session = ort.InferenceSession(output_onnx_name)
    onnx_input = torch_input.numpy()
    outputs = ort_session.run(None, {"input1": onnx_input})
    logger.info("onnx model output:{}, shape:{}".format(outputs[0][0, :10], outputs[0].shape))
    diff = (torch_out.numpy() - outputs[0]).mean()
    logger.info("The diff between pytorch and onnx models: {}".format(diff))

    """Convert onnx to tf-graph"""
    onnx_model = onnx.load(output_onnx_name)
    tf_rep = prepare(onnx_model)
    logger.info("tf_rep inputs:{}, outputs:{}".format(tf_rep.inputs, tf_rep.outputs))

    output_tfgraph_name = os.path.join(output_dir, model_name + ".pb")
    tf_rep.export_graph(output_tfgraph_name)

    """Convert tf-graph to saved_model."""
    scale = 255.0 if max(model.input_range) == 1.0 else 1.0
    target = tf.constant(image_target_list, dtype=tf.int32, name="image_target_list")
    size = tf.constant(image_size_list, dtype=tf.int32, name="image_size_list")
    mean = tf.constant(model.mean, dtype=tf.float32, shape=(1,1,3), name="input_mean")
    std = tf.constant(model.std, dtype=tf.float32, shape=(1,1,3), name="input_std")
    print("tf mean:", mean, "tf std:", std)
    module = CustomModule(output_tfgraph_name, l2norm=l2norm, scale=scale,
                          preserve_aspect_ratio=preserve_aspect_ratio,
                          target=target, size=size, mean=mean, std=std)
    output_savedmodel_name = os.path.join(output_dir, "saved_model/")
    signatures = {
        'serving_default': module.extract_feature,
    }
    tf.saved_model.save(module, output_savedmodel_name, signatures=signatures)
    logger.info("saving tf-2.2 model to {} is ok.".format(output_savedmodel_name))
    imported = tf.saved_model.load(output_savedmodel_name)
    REQUIRED_SIGNATURE = "serving_default"
    REQUIRED_OUTPUT = "global_descriptor"
    found_signatures = list(imported.signatures.keys())
    if REQUIRED_SIGNATURE in found_signatures:
        logger.info("checking the signatures is ok.")

    outputs = imported.signatures[REQUIRED_SIGNATURE].structured_outputs
    if REQUIRED_OUTPUT in outputs:
        logger.info("checking the output name is ok.")

    embedding_fn = imported.signatures[REQUIRED_SIGNATURE]

    torch.random.manual_seed(0)
    tf_input = tf.convert_to_tensor(rgb_data, dtype=tf.uint8)
    print("tf_input:", tf_input.shape)
    tmp = module(tf_input)
    output = embedding_fn(tf_input)[REQUIRED_OUTPUT].numpy()
    logger.info("tf-saved_model output: {}".format(output[:10]))


def main():
    opts.convert_opts(parser)
    args = parser.parse_args()
    init_logger()
    config = args.__dict__
    convert(config)


if __name__ == "__main__":
    main()
