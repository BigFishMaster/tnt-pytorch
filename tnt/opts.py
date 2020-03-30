
def basic_opts(parser):
    parser.add_argument("--gpu", default=None, type=int,
                        help="GPU id to use.")
    parser.add_argument("--num_classes", default=2, type=int,
                        help="number of classes to classify.")
    parser.add_argument("--image_size", default=None, type=int,
                        help="the input image size: [3, image_size, image_size].")
    parser.add_argument("--pretrained", default="", type=str,
                        help="use pre-trained model.")
    parser.add_argument("--resume", default="", type=str,
                        help="path to last checkpoint.")
    parser.add_argument("--num-epochs", default=10, type=int,
                        help="number of epochs to train.")
