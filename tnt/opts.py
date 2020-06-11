
def basic_opts(parser):
    global_opts(parser)
    data_opts(parser)
    transform_opts(parser)
    model_opts(parser)
    loss_opts(parser)
    test_opts(parser)
    # TODO: add the below opts.
    #optim_opts(parser)
    #metric_opts(parser)
    #lrate_opts(parser)


def global_opts(parser):
    parser.add_argument("--num_epochs", default=None, type=int,
                        help="number of epochs to train.")
    parser.add_argument("--gpu", default=None, type=int,
                        help="GPU id to use. default: use all gpus.")
    parser.add_argument("--resume", default=None, type=str,
                        help="path to last checkpoint or last_checkpoint.txt.")
    parser.add_argument("--weight", default=None, type=str,
                        help="model weight to load.")
    parser.add_argument("--keep_last_layer", default=False, action="store_true",
                        help="keep weights of last layer when loading model weights.")
    parser.add_argument("--tb_log", default=1, type=int,
                        help="whether to output tensorboard log.")


def data_opts(parser):
    parser.add_argument("--num_classes", default=None, type=int,
                        help="number of classes in classification task.")
    parser.add_argument("--num_features", default=None, type=int,
                        help="number of features of last linear layer.")
    parser.add_argument("--image_size", default=None, type=int,
                        help="the input image size: [3, image_size, image_size].")
    parser.add_argument("--batch_size", default=None, type=int,
                        help="the batch size.")
    parser.add_argument("--mode", default=None, type=str,
                        help="set a mode, e.g.: train,val,test")
    parser.add_argument("--disable_pin_memory", default=False, action="store_true",
                        help="disable the pin memory in dataloader.")
    parser.add_argument("--use_first_label", default=False, action="store_true",
                        help="use first label for data sampling.")
    parser.add_argument("--box_extend", default=None, type=str,
                        help="extend ratio of the annotated box for cropping image, e.g.: 0.1,0.1,0.1,0.1")


def transform_opts(parser):
    parser.add_argument("--transform_type", default=None, type=str,
                        help="image transform type: raw, original, originalr, v0, v0r or None.")
    parser.add_argument("--disable_random_crop", default=False, action="store_true",
                        help="disable the random crop transform mode.")
    parser.add_argument("--disable_random_hflip", default=False, action="store_true",
                        help="disable the random horizontal flip transform mode.")
    parser.add_argument("--image_scale", default=0.875, type=float,
                        help="image will be scaled by it before cropping.")
    parser.add_argument("--five_crop", default=False, action="store_true",
                        help="five crops when testing.")
    parser.add_argument("--ten_crop", default=False, action="store_true",
                        help="ten crops when testing.")
    parser.add_argument("--preserve_aspect_ratio", default=1, type=int,
                        help="whether to preserve image aspect ratio when scaling.")
    parser.add_argument("--enable_random_erase", default=False, action="store_true",
                        help="enable the random erasing transform mode.")


def model_opts(parser):
    parser.add_argument("--pretrained", default=None, type=str,
                        help="use pre-trained model.")
    parser.add_argument("--model_name", default=None, type=str,
                        help="mode name, e.g.: resnet50")
    parser.add_argument("--fix_bn", default=False, action="store_true",
                        help="fix batchnorm layer.")
    parser.add_argument("--fix_res", default=False, action="store_true",
                        help="fix resolution by fine-tune last BN and linear layer.")


def test_opts(parser):
    parser.add_argument("--test", default=None, type=str,
                        help="test file to load.")
    parser.add_argument("--out_file", default=None, type=str,
                        help="output file for test.")
    parser.add_argument("--out_mode", default=None, type=str,
                        help="output mode, e.g.: topk or raw")


def loss_opts(parser):
    parser.add_argument("--relativelabelloss_gamma", default=0.2, type=float,
                        help="gamma of RelativeLabelLoss.")
    parser.add_argument("--arcface_scale", default=30, type=float,
                        help="scale parameter for cosface and arcface.")
    parser.add_argument("--arcface_margin", default=0.5, type=float,
                        help="margin parameter for cosface and arcface.")
    parser.add_argument("--hc_margin", default=1.0, type=float,
                        help="margin for HCLoss.")
    parser.add_argument("--hc_sample_type", default="ratio", type=str,
                        help="sample type for HCLoss.")
    parser.add_argument("--hc_each_class", default=5, type=int,
                        help="samples per class for HCLoss.")
    parser.add_argument("--hc_beta", default=10000.0, type=float,
                        help="beta for HCLoss.")
    parser.add_argument("--hc_pos_nn", default=1.0, type=float,
                        help="pos_nn for HCLoss.")
