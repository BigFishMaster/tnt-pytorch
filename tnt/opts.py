
def basic_opts(parser):
    parser.add_argument("--gpu", default=None, type=int,
                        help="GPU id to use.")
    parser.add_argument("--num_classes", default=None, type=int,
                        help="number of classes to classify.")
    parser.add_argument("--image_size", default=None, type=int,
                        help="the input image size: [3, image_size, image_size].")
    parser.add_argument("--batch_size", default=None, type=int,
                        help="the batch size.")
    parser.add_argument("--pretrained", default=None, type=str,
                        help="use pre-trained model.")
    parser.add_argument("--test", default=None, type=str,
                        help="test file.")
    parser.add_argument("--out_file", default=None, type=str,
                        help="output file for test.")
    parser.add_argument("--out_mode", default=None, type=str,
                        help="output mode, e.g.: topk or raw")
    parser.add_argument("--model_name", default=None, type=str,
                        help="mode name, e.g.: resnet50")
    parser.add_argument("--mode", default=None, type=str,
                        help="set a mode, e.g.: train,val,test")
    parser.add_argument("--resume", default=None, type=str,
                        help="path to last checkpoint.")
    parser.add_argument("--weight", default=None, type=str,
                        help="model weight to load.")
    parser.add_argument("--fix_bn", default=False, action="store_true",
                        help="fix batchnorm layer.")
    parser.add_argument("--transform_type", default=None, type=str,
                        help="image transform type: raw, original, originalr, v0, v0r or None.")
    parser.add_argument("--five_crop", default=False, action="store_true",
                        help="five crops when testing.")
    parser.add_argument("--ten_crop", default=False, action="store_true",
                        help="ten crops when testing.")
    parser.add_argument("--num_epochs", default=None, type=int,
                        help="number of epochs to train.")
