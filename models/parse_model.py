import torch
from .base_model import BaseModel
from . import networks
from utils import utils

class ParseModel(BaseModel):
    def modify_commandline_options(parser, is_train):
        if is_train:
            parser.add_argument('--parse_map', type=float, default=1.0, help='weight for parsing map')
            parser.add_argument('--parse_sr', type=float, default=1.0, help='weight for sr')
        return parser
    
    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        self.loss_names = ['P', 'SR']
        self.visual_names = ['img_LR', 'img_HR', 'gt_Parse', 'img_SR', 'pred_Parse']

        self.model_names = ['P']
        self.netP = networks.define_P(opt)

        if self.isTrain:  # only defined during training time
            self.criterionParse = torch.nn.CrossEntropyLoss()
            self.criterionSR = torch.nn.L1Loss()
            self.optimizer = torch.optim.Adam(self.netP.parameters(), lr=opt.lr, betas=(0.9, 0.999))
            self.optimizers = [self.optimizer]

    def set_input(self, input, cur_iters=None):
        self.img_LR = input['LR'].to(self.opt.device)
        self.img_HR = input['HR'].to(self.opt.device)
        self.gt_Parse = input['Mask'].to(self.opt.device)
        if self.opt.debug:
            print('ParseNet input shape:', self.img_LR.shape, self.img_HR.shape, self.gt_Parse.shape)

    def load_pretrain_models(self,):
        self.netP.eval()
        print('Loading pretrained LQ face parsing network from', self.opt.parse_net_weight)
        self.netP.load_state_dict(torch.load(self.opt.parse_net_weight))

    def forward(self):
        self.pred_Parse, self.img_SR = self.netP(self.img_LR)
        if self.opt.debug:
            print('ParseNet output shape', self.pred_Parse.shape, self.img_SR.shape)

    def backward(self):
        self.loss_P = self.criterionParse(self.pred_Parse, self.gt_Parse) * self.opt.parse_map
        self.loss_SR = self.criterionSR(self.img_SR, self.img_HR) * self.opt.parse_sr

        loss = self.loss_P + self.loss_SR
        loss.backward()      

    def optimize_parameters(self):
        self.optimizer.zero_grad()   # clear network G's existing gradients
        self.backward()              # calculate gradients for network G
        self.optimizer.step()

    def get_current_visuals(self, size=512):
        out = []
        visual_imgs = []
        out.append(utils.tensor_to_numpy(self.img_LR))
        out.append(utils.tensor_to_numpy(self.img_SR))
        out.append(utils.tensor_to_numpy(self.img_HR))
        out_imgs = [utils.batch_numpy_to_image(x, size) for x in out]

        visual_imgs.append(out_imgs[0])
        visual_imgs.append(out_imgs[1])
        visual_imgs.append(utils.color_parse_map(self.pred_Parse))
        visual_imgs.append(utils.color_parse_map(self.gt_Parse.unsqueeze(1)))
        visual_imgs.append(out_imgs[2])

        return visual_imgs
        
