import os
import onnx
from onnx_tf.backend import prepare
import numpy as np
import torch
from torch.onnx import utils
import tensorflow as tf
import tnt.pretrainedmodels as models
import onnxruntime as ort


def check():
    onnx_model = onnx.load("resnet50.onnx")
    onnx.checker.check_model(onnx_model)
    print_graph = onnx.helper.printable_graph(onnx_model.graph)
    print("onnx graph:", print_graph)
    model = models.__dict__["resnet50"](pretrained="imagenet")
    model.eval()
    torch.random.manual_seed(0)
    #input = torch.rand(1, 3, 224, 224)
    input = torch.randint(0, 1, size=(224, 224, 3)).float()
    input = input.permute(2, 0, 1).contiguous()
    print("input contiguous:", input.is_contiguous())

    #input = input.permute(2, 0, 1).contiguous()
    input = input.unsqueeze(0)
    with torch.no_grad():
        torch_out = model(input)
    print("torch output:", torch_out[0,:10], torch_out.shape)
    ort_session = ort.InferenceSession('resnet50.onnx')
    print("start onnx running")
    onnx_input = input.numpy()
    print("onnx_input:", onnx_input.shape)
    outputs = ort_session.run(None, {"input1": onnx_input})
    print("onnx output:", outputs[0][0, :10])
    diff = (torch_out.numpy() - outputs[0]).mean()
    print("diff:", diff)


def load_checkpoint(model, filename):
    pass


def convert():
    torch.random.manual_seed(0)
    model = models.__dict__["resnest200"](pretrained=None)
    input = torch.rand(1, 3, 224, 224)
    print("input:", input.shape)
    input_names = ["input1"] + ["learned_%d" % i for i in range(10)]
    output_names = ["output1"]
    torch.onnx.export(model, (input,), "resnet50.onnx", verbose=True, opset_version=11,
                      input_names=input_names, output_names=output_names)
    print("export ok")
    onnx_model = onnx.load("resnet50.onnx")
    onnx.checker.check_model(onnx_model)
    print_graph = onnx.helper.printable_graph(onnx_model.graph)
    print("onnx graph:", print_graph)
    model.eval()
    with torch.no_grad():
        torch_out = model(input)
    print("torch output:", torch_out[0,:10], torch_out.shape)
    ort_session = ort.InferenceSession('resnet50.onnx')
    print("start onnx running")
    onnx_input = input.numpy()
    print("onnx_input:", onnx_input.shape)
    outputs = ort_session.run(None, {"input1": onnx_input})
    print("onnx output:", outputs[0][0, :10])
    diff = (torch_out.numpy() - outputs[0]).mean()
    print("diff:", diff)

def test():
    onnx_model = onnx.load("resnet50.onnx")
    torch.random.manual_seed(0)
    input = torch.rand(1, 3, 224, 224).numpy()
    output = prepare(onnx_model).run(input)
    print("output:", output)

    tf_rep = prepare(onnx_model)
    print("tf_rep.inputs:", tf_rep.inputs)
    print("tf_rep.outputs:", tf_rep.outputs)
    tf_rep.export_graph("./resnet50.graph")


def wrap_frozen_graph(graph_def, inputs, outputs):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")
    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph
    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))


@tf.function
def preprocess(x):
    return x - 1


class CustomModule(tf.Module):

    def __init__(self, model_pb_file):
        super(CustomModule, self).__init__()
        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(open(model_pb_file, "rb").read())
        self.model_func = wrap_frozen_graph(graph_def, inputs="input1:0", outputs="output1:0")
        self.mean = tf.Variable(tf.zeros(shape=(1, 1, 3)))
        print("Initialize ok.")

    @tf.function(input_signature=[tf.TensorSpec([None, None, 3], tf.uint8, name="input_image")])
    def __call__(self, x):
        output1 = self.preprocess(x)
        # shape: [3, None, None]
        output2 = tf.transpose(output1, perm=[2, 0, 1])
        # shape: [1, 3, None, None]
        output3 = tf.expand_dims(output2, axis=0)
        output = self.model_func(output3)
        output = tf.squeeze(output, axis=0)
        named_output_tensor = {"global_descriptor": tf.identity(
            output, name="global_descriptor")}
        return named_output_tensor

    @tf.function(input_signature=[tf.TensorSpec([None, None, 3], tf.uint8)])
    def preprocess(self, x):
        x = tf.cast(x, dtype=tf.float32)
        return x - self.mean


def test_convert_pb_to_tf_function():
    torch.random.manual_seed(0)
    input = torch.rand(1, 3, 224, 224).numpy()
    input = tf.convert_to_tensor(input, dtype=tf.float32)

    graph_def = tf.compat.v1.GraphDef()
    loaded = graph_def.ParseFromString(open("resnet50.pb", "rb").read())
    model_func = wrap_frozen_graph(graph_def, inputs="input1:0", outputs="output1:0")
    print("model_func:", model_func)
    preprocess_func = preprocess.get_concrete_function(x=tf.TensorSpec([None, 3, 224, 224], tf.float32))
    input1 = preprocess_func(input)
    output = model_func(input1)
    print("output:", output, output.shape)


def test_pb_to_saved_model():
    module = CustomModule("resnet50.pb")
    module_no_signatures_path = os.path.join("./", "module_no_signatures")
    signatures = {
        'serving_default': module.__call__.get_concrete_function(),
    }
    tf.saved_model.save(module, module_no_signatures_path, signatures=signatures)
    print("saved_model ok")
    imported = tf.saved_model.load(module_no_signatures_path)
    REQUIRED_SIGNATURE = "serving_default"
    REQUIRED_OUTPUT = "global_descriptor"
    found_signatures = list(imported.signatures.keys())
    if REQUIRED_SIGNATURE in found_signatures:
        print("signatures ok")

    outputs = imported.signatures[REQUIRED_SIGNATURE].structured_outputs
    if REQUIRED_OUTPUT in outputs:
        print("output name ok")

    embedding_fn = imported.signatures[REQUIRED_SIGNATURE]

    torch.random.manual_seed(0)
    input = torch.randint(0, 255, size=(224, 224, 3)).numpy()
    input = tf.convert_to_tensor(input, dtype=tf.uint8)
    tmp = embedding_fn(input)
    output = embedding_fn(input)[REQUIRED_OUTPUT].numpy()
    print("output:", output[:10])


if __name__ == "__main__":
    convert()
    #check()
    #test()
    #test_pb_to_saved_model()
