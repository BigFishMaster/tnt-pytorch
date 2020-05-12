
def basic_opts(parser):
    parser.add_argument("--gpu", default=None, type=int,
                        help="GPU id to use.")
    parser.add_argument("--num_classes", default=None, type=int,
                        help="number of classes to classify.")
    parser.add_argument("--num_features", default=None, type=int,
                        help="number of features for each sample.")
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
    parser.add_argument("--disable_pin_memory", default=False, action="store_true",
                        help="disable the pin memory in dataloader.")
    parser.add_argument("--fix_res", default=False, action="store_true",
                        help="fix resolution by fine-tune last BN and linear layer.")
    parser.add_argument("--keep_last_layer", default=False, action="store_true",
                        help="keep weights of last layer when weighting model.")
    parser.add_argument("--tb_log", default=1, type=int,
                        help="tensorboard logging.")
    parser.add_argument("--transform_type", default=None, type=str,
                        help="image transform type: raw, original, originalr, v0, v0r or None.")
    parser.add_argument("--five_crop", default=False, action="store_true",
                        help="five crops when testing.")
    parser.add_argument("--ten_crop", default=False, action="store_true",
                        help="ten crops when testing.")
    parser.add_argument("--num_epochs", default=None, type=int,
                        help="number of epochs to train.")
    parser.add_argument("--image_scale", default=0.875, type=float,
                        help="image to be scaled before cropping.")
    parser.add_argument("--preserve_aspect_ratio", default=1, type=int,
                        help="whether to preserve image aspect ratio when scaling.")
    parser.add_argument("--relativelabelloss_gamma", default=0.2, type=float,
                        help="gamma of RelativeLabelLoss.")
    parser.add_argument("--use_first_label", default=False, action="store_true",
                        help="use first label for sampling weight.")
    parser.add_argument("--enable_random_erase", default=False, action="store_true",
                        help="disable the random erasing transform mode.")
