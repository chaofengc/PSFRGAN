from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--src_dir', type=str, default='', help='source directory containing test images')
        parser.add_argument('--test_img_path', type=str, default='', help='path for single image test')
        parser.add_argument('--test_upscale', type=float, default=1, help='upsample scale for single image test')
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        parser.add_argument('--parse_net_weight', type=str, default='./pretrain_models/parse_multi_iter_90000.pth', help='parse model path')
        parser.add_argument('--psfr_net_weight', type=str, default='./pretrain_models/psfrgan_latest_net_G.pth', help='parse model path')
        # rewrite devalue values
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
