from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--visual_freq', type=int, default=400, help='frequency of show training images in tensorboard')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        # network saving and loading parameters
        parser.add_argument('--save_iter_freq', type=int, default=5000, help='frequency of saving the models')
        parser.add_argument('--save_latest_freq', type=int, default=500, help='save latest freq')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--no_strict_load', action='store_true', help='set strict load to false')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--resume_epoch', type=int, default=0, help='training resume epoch')
        parser.add_argument('--resume_iter', type=int, default=0, help='training resume iter')
        parser.add_argument('--total_epochs', type=int, default=50, help='# of epochs to train')
        parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--g_lr', type=float, default=0.0001, help='generator learning rate')
        parser.add_argument('--d_lr', type=float, default=0.0004, help='discriminator learning rate')
        parser.add_argument('--gan_mode', type=str, default='hinge', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--lr_decay_gamma', type=float, default=1, help='multiply by a gamma every lr_decay_iters iterations')

        self.isTrain = True
 
        return parser
