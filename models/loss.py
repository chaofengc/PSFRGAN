import torch
from torchvision import models
from utils import utils
from torch import nn


def tv_loss(x):
    """
    Total Variation Loss.
    """
    return torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
            ) + torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))


class PCPFeat(torch.nn.Module):
    """
    Features used to calculate Perceptual Loss based on ResNet50 features.
    Input: (B, C, H, W), RGB, [0, 1]
    """
    def __init__(self, weight_path, model='vgg'):
        super(PCPFeat, self).__init__()
        if model == 'vgg':
            self.model = models.vgg19(pretrained=False)
            self.build_vgg_layers()
        elif model == 'resnet':
            self.model = models.resnet50(pretrained=False)
            self.build_resnet_layers()

        self.model.load_state_dict(torch.load(weight_path))
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def build_resnet_layers(self):
        self.layer1 = torch.nn.Sequential(
                    self.model.conv1,
                    self.model.bn1,
                    self.model.relu,
                    self.model.maxpool,
                    self.model.layer1
                    )
        self.layer2 = self.model.layer2
        self.layer3 = self.model.layer3
        self.layer4 = self.model.layer4
        self.features = torch.nn.ModuleList(
                [self.layer1, self.layer2, self.layer3, self.layer4]
                )
    
    def build_vgg_layers(self):
        vgg_pretrained_features = self.model.features
        self.features = []
        feature_layers = [0, 3, 8, 17, 26, 35]
        for i in range(len(feature_layers)-1): 
            module_layers = torch.nn.Sequential() 
            for j in range(feature_layers[i], feature_layers[i+1]):
                module_layers.add_module(str(j), vgg_pretrained_features[j])
            self.features.append(module_layers)
        self.features = torch.nn.ModuleList(self.features)

    def preprocess(self, x):
        x = (x + 1) / 2
        mean = torch.Tensor([0.485, 0.456, 0.406]).to(x)
        std  = torch.Tensor([0.229, 0.224, 0.225]).to(x)
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
        x = (x - mean) / std
        if x.shape[3] < 224:
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return x

    def forward(self, x):
        x = self.preprocess(x)
        
        features = []
        for m in self.features:
            x = m(x)
            features.append(x)
        return features 


class PCPLoss(torch.nn.Module):
    """Perceptual Loss.
    """
    def __init__(self, 
            opt, 
            layer=5,
            model='vgg',
            ):
        super(PCPLoss, self).__init__()

        self.mse = torch.nn.MSELoss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x_feats, y_feats):
        loss = 0
        for xf, yf, w in zip(x_feats, y_feats, self.weights): 
            loss = loss + self.mse(xf, yf.detach()) * w
        return loss 


class FMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, x_feats, y_feats):
        loss = 0
        for xf, yf in zip(x_feats, y_feats):
            loss = loss + self.mse(xf, yf.detach()) 
        return loss


class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'hinge':
            pass
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real, for_discriminator=True):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    loss = nn.ReLU()(1 - prediction).mean()
                else:
                    loss = nn.ReLU()(1 + prediction).mean() 
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss  = - prediction.mean()
            return loss

        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class RegionStyleLoss(nn.Module):
    def __init__(self, reg_num=19, eps=1e-8):
        super().__init__()
        self.reg_num = reg_num
        self.eps = eps 
        self.mse = nn.MSELoss()

    def __masked_gram_matrix(self, x, m):
        b, c, h, w = x.shape
        m = m.view(b, -1, h*w)
        x = x.view(b, -1, h*w)
        total_elements = m.sum(2) + self.eps

        x = x * m
        G = torch.bmm(x, x.transpose(1, 2))
        return G / (c * total_elements.view(b, 1, 1)) 

    def __layer_gram_matrix(self, x, mask):
        b, c, h, w = x.shape
        all_gm = []
        for i in range(self.reg_num):
            sub_mask = mask[:, i].unsqueeze(1) 
            gram_matrix = self.__masked_gram_matrix(x, sub_mask)
            all_gm.append(gram_matrix)
        return torch.stack(all_gm, dim=1)

    def forward(self, x_feats, y_feats, mask):
        loss = 0
        for xf, yf in zip(x_feats[2:], y_feats[2:]):
            tmp_mask = torch.nn.functional.interpolate(mask, xf.shape[2:])
            xf_gm = self.__layer_gram_matrix(xf, tmp_mask)
            yf_gm = self.__layer_gram_matrix(yf, tmp_mask)
            tmp_loss = self.mse(xf_gm, yf_gm.detach())
            loss = loss + tmp_loss
        return loss

            
if __name__ == '__main__':
    x = [
            torch.randn(2, 64, 512, 512), 
            torch.randn(2, 128, 256, 256), 
            torch.randn(2, 256, 128, 128), 
            torch.randn(2, 512, 64, 64), 
            torch.randn(2, 512, 32, 32), 
            ]

    y = torch.randint(10, (2, 19, 512, 512)).float()
    loss = RegionStyleLoss()
    print(loss(x, x, y))

