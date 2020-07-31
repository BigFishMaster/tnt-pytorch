import os
import numpy as np
import onnx
from onnx_tf.backend import prepare
import torch
import tensorflow as tf
from tnt.impls import ModelImpl
import onnxruntime as ort

import argparse
import tnt.opts as opts
from tnt.utils.logging import init_logger, logger, beautify_info
from collections import OrderedDict
from PIL import Image
from torchvision import transforms
from tensorflow.python.eager import wrap_function

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

    #def __init__(self, model_pb_file, l2norm=True, scale=255.0, target=None,
    #             size=None, preserve_aspect_ratio=False, mean=None, std=None):
    def __init__(self, g1, scale1, target1, size1, mean1, std1, p1,
                       g2, scale2, target2, size2, mean2, std2, p2, l2norm=True):
        super(CustomModule, self).__init__()
        graph_def1 = tf.compat.v1.GraphDef()
        graph_def1.ParseFromString(open(g1, "rb").read())
        graph_def2 = tf.compat.v1.GraphDef()
        graph_def2.ParseFromString(open(g2, "rb").read())
        self.model_func1 = wrap_function.function_from_graph_def(graph_def1, inputs="input1:0", outputs="output1:0")
        self.model_func2 = wrap_function.function_from_graph_def(graph_def2, inputs="input1:0", outputs="output1:0")
        self.l2norm = l2norm
        self.scale1 = scale1
        self.target1 = target1
        self.size1 = size1
        self.mean1 = mean1
        self.std1 = std1
        self.p1 = p1
        self.scale2 = scale2
        self.target2 = target2
        self.size2 = size2
        self.mean2 = mean2
        self.std2 = std2
        self.p2 = p2

    @tf.function(input_signature=[tf.TensorSpec([None, None, 3], tf.uint8, name="input_image")])
    def extract_feature(self, image):
        #TODO: check the order of h and w.
        h = tf.shape(image)[0]
        w = tf.shape(image)[1]
        image_data = tf.cast(image, tf.float32)

        def resize_and_extract(scale, target, size, mean_value, std_value, preserve_aspect_ratio, model_func):
            if preserve_aspect_ratio:
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
            preprocess = resized_image / scale
            preprocess = (preprocess - mean_value) / std_value
            # shape: 3 x 224 x 224
            preprocess = tf.transpose(preprocess, perm=[2, 0, 1])
            # shape: 1 x 3 x 224 x 224
            preprocess = tf.expand_dims(preprocess, axis=0)
            # shape: 1 x 2048
            output = model_func(preprocess)
            return output

        output_global1 = resize_and_extract(self.scale1, self.target1[0], self.size1[0],
                                            self.mean1, self.std1, self.p1, self.model_func1)
        output_global2 = resize_and_extract(self.scale2, self.target2[0], self.size2[0],
                                            self.mean2, self.std2, self.p2, self.model_func2)
        if self.l2norm:
            output_global1 = tf.nn.l2_normalize(output_global1, axis=1, name="global1_l2norm")
            output_global2 = tf.nn.l2_normalize(output_global2, axis=1, name="global2_l2norm")

        output_global = tf.concat([output_global1[0], output_global2[0]], axis=0)

        # output the feature with a key.
        named_output_tensor = {"global_descriptor": tf.identity(
            output_global, name="global_descriptor")}
        return named_output_tensor


def test_pytorch_model(model, image, target, image_scale):
    mean = model.mean
    std = model.std
    size = int(target/image_scale)
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


def Int2Bool(x):
    if x == 1:
        return True
    else:
        return False


def comb_convert(config):
    model_name = config["model_name"].split(",")
    weight_path = config["weight"].split(",")
    image_scale = [float(s) for s in config["image_scale"].split(",")]
    image_target_list = [[int(sz)] for sz in config["image_size"].split(",")]
    image_size_list = [[int(sz[0] / image_scale[i])] for i, sz in enumerate(image_target_list)]
    output_dir = config["output_dir"]
    output_dir = [output_dir, output_dir]
    image_path = config["image_path"]
    image_path = [image_path, image_path]
    num_classes = [int(n) for n in config["num_classes"].split(",")]
    p = [Int2Bool(int(p)) for p in config["preserve_aspect_ratio"].split(",")]
    l2norm = config["l2norm"]
    extract_feature = [Int2Bool(int(e)) for e in config["extract_feature"].split(",")]

    graphs = []
    models = []
    for index, param in enumerate(zip(model_name, weight_path, image_scale, image_target_list,
                                      output_dir, image_path, num_classes, extract_feature)):
        tf_graph_name, pytorch_model = convert_tf_graph(*param, index=index)
        graphs.append(tf_graph_name)
        models.append(pytorch_model)
        print("convert tf-graph {} is ok.".format(index))
    convert_saved_model(graphs[0], models[0], p[0], graphs[1], models[1], p[1], image_size_list, image_target_list,
                        l2norm, output_dir[0], image_path[0])



def convert_tf_graph(model_name, weight_path, image_scale, image_target_list,
            output_dir, image_path, num_classes, extract_feature, index=0):
    torch.random.manual_seed(0)

    image_size = image_target_list[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = ModelImpl(model_name, num_classes, extract_feature=extract_feature).model
    model_name = model_name + "_" + str(index)
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
    torch.onnx.export(model, input, output_onnx_name, verbose=True, opset_version=11,
                      input_names=input_names, output_names=output_names)

    logger.info("exporting model to {} is ok.".format(output_onnx_name))
    onnx_model = onnx.load(output_onnx_name)
    onnx.checker.check_model(onnx_model)
    print_graph = onnx.helper.printable_graph(onnx_model.graph)
    logger.info("onnx graph: {}".format(print_graph))
    logger.info("check the onnx model is ok.")
    logger.info("checking the difference between pytorch and onnx models.")

    torch_input, torch_out = test_pytorch_model(model, image, image_size, image_scale)

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
    return output_tfgraph_name, model


def convert_saved_model(g1, m1, p1, g2, m2, p2, size_lists, target_lists, l2norm, output_dir, image_path):
    """Convert tf-graph to saved_model."""
    scale1 = 255.0 if max(m1.input_range) == 1.0 else 1.0
    target1 = tf.constant(target_lists[0], dtype=tf.int32, name="image_target_list1")
    size1 = tf.constant(size_lists[0], dtype=tf.int32, name="image_size_list1")
    mean1 = tf.constant(m1.mean, dtype=tf.float32, shape=(1,1,3), name="input_mean1")
    std1 = tf.constant(m1.std, dtype=tf.float32, shape=(1,1,3), name="input_std1")
    print("tf mean1:", mean1, "tf std1:", std1)
    scale2 = 255.0 if max(m2.input_range) == 1.0 else 1.0
    target2 = tf.constant(target_lists[1], dtype=tf.int32, name="image_target_list2")
    size2 = tf.constant(size_lists[1], dtype=tf.int32, name="image_size_list2")
    mean2 = tf.constant(m2.mean, dtype=tf.float32, shape=(1,1,3), name="input_mean2")
    std2 = tf.constant(m2.std, dtype=tf.float32, shape=(1,1,3), name="input_std2")
    print("tf mean2:", mean2, "tf std2:", std2)
    module = CustomModule(g1, scale1, target1, size1, mean1, std1, p1,
                          g2, scale2, target2, size2, mean2, std2, p2,
                          l2norm=l2norm)
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
    image = Image.open(image_path).convert("RGB")
    rgb_data = np.array(image)
    tf_input = tf.convert_to_tensor(rgb_data, dtype=tf.uint8)
    print("tf_input:", tf_input.shape)
    output = embedding_fn(tf_input)[REQUIRED_OUTPUT].numpy()
    logger.info("tf-saved_model output1: {}, output2:{}, shape: {}, l2-sum: {}".format(
        output[:10], output[512:522], output.shape, tf.reduce_sum(output*output)))


def main():
    opts.comb_convert_opts(parser)
    args = parser.parse_args()
    init_logger()
    config = args.__dict__
    comb_convert(config)


if __name__ == "__main__":
    main()
