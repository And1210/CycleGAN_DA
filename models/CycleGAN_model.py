import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel
import numpy as np
import matplotlib.pyplot as plt
import random

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16):
        super(GeneratorResNet, self).__init__()

        # First layer
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, 256, kernel_size=9, stride=1, padding=4), nn.MaxPool2d(2), nn.PReLU())
        self.conv1_2 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=9, stride=1, padding=4), nn.MaxPool2d(2), nn.PReLU())
        self.conv1_3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=9, stride=1, padding=4), nn.PReLU())

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64, 0.8))

        # Upsampling layers
        upsampling = []
        for out_features in range(2):
            upsampling += [
                # nn.Upsample(scale_factor=2),
                nn.Conv2d(64, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Sequential(nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4), nn.Tanh())

    def forward(self, x):
        out1 = self.conv1_1(x)
        out1 = self.conv1_2(out1)
        out1 = self.conv1_3(out1)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)

#PyTorch defined model
class CycleGAN(nn.Module):
    """basenet for fer2013"""
    def __init__(self, device, img_size=512, latent_dim=128, channels=3):
        super(CycleGAN, self).__init__()

        self.device = device
        self.latent_dim = latent_dim

        self.generator_trans = GeneratorResNet()
        self.generator_recon = GeneratorResNet()

    def forward(self, x, x_trg):
        #Forward cycle
        x_trans = self.generator_trans(x)
        x_recon = self.generator_recon(x_trans)

        #Backward cycle
        x_recon_trg = self.generator_recon(x_trg)
        x_trans_trg = self.generator_trans(x_recon_trg)

        #Identity
        x_trans_identity = self.generator_trans(x_trg)
        x_recon_identity = self.generator_recon(x)

        return x_trans, x_recon, x_recon_trg, x_trans_trg, x_trans_identity, x_recon_identity

#The abstract model class, uses above defined class and is used in the train script
class CycleGANmodel(BaseModel):
    """basenet for fer2013"""

    def __init__(self, configuration):
        super().__init__(configuration)

        self.latent_dim = configuration['latent_dim']
        self.img_size = configuration['img_size']

        #Initialize model defined above
        self.model = CycleGAN(device=self.device, img_size=self.img_size, latent_dim=self.latent_dim)
        self.model.cuda()
        self.discriminator_trans = Discriminator((3, self.img_size, self.img_size))
        self.discriminator_trans.cuda()
        self.discriminator_recon = Discriminator((3, self.img_size, self.img_size))
        self.discriminator_recon.cuda()

        #Define loss function
        self.criterion_loss = nn.BCELoss().cuda()
        self.l1_loss = nn.L1Loss().cuda()

        #Define optimizer
        self.optimizer_g = torch.optim.Adam(
            self.model.parameters(),
            lr=configuration['lr'],
            betas=(configuration['momentum'], 0.999),
            weight_decay=configuration['weight_decay']
        )
        self.optimizer_d = torch.optim.Adam(
            list(self.discriminator_trans.parameters())+list(self.discriminator_recon.parameters()),
            lr=configuration['lr'],
            betas=(configuration['momentum'], 0.999),
            weight_decay=configuration['weight_decay']
        )

        #Need to include these arrays with the optimizers and names of loss functions and models
        #Will be used by other functions for saving/loading
        # self.optimizers = [self.optimizers[i] for i in range(4)]
        self.optimizers = [self.optimizer_g, self.optimizer_d]
        self.loss_names = ['g', 'd']
        self.network_names = ['model']

        self.val_images = []
        self.val_predictions = []
        self.val_labels = []

    #Calls the models forwards function
    def forward(self):
        x = self.input
        x_trg = self.target
        self.output_trans, self.output_recon, self.output_recon_trg, self.output_trans_trg, self.output_trans_identity, self.output_recon_identity = self.model.forward(x, x_trg)
        # print(self.output_trans.shape)
        # print(self.output_recon.shape)
        return self.output_trans

    #Computes the loss with the specified name (in this case 'total')
    def compute_loss(self):
        valid = torch.ones((self.output_trans.shape[0], *self.discriminator_trans.output_shape), requires_grad=False).to(self.device)
        fake = torch.zeros((self.output_trans.shape[0], *self.discriminator_trans.output_shape), requires_grad=False).to(self.device)

        #Adversarial loss
        self.target_logits = self.discriminator_trans(self.target)
        self.trans_logits = self.discriminator_trans(self.output_trans)
        self.input_logits = self.discriminator_recon(self.input)
        self.recon_logits = self.discriminator_recon(self.output_recon)

        self.loss_g_trans = self.criterion_loss(self.trans_logits, valid)
        self.loss_g_recon = self.criterion_loss(self.recon_logits, valid)

        self.target_logits = self.discriminator_trans(self.target)
        self.trans_logits = self.discriminator_trans(self.output_trans.detach())
        self.input_logits = self.discriminator_recon(self.input)
        self.recon_logits = self.discriminator_recon(self.output_recon.detach())

        self.loss_d_trans = (self.criterion_loss(self.trans_logits, fake) + self.criterion_loss(self.target_logits, valid))/2
        self.loss_d_recon = (self.criterion_loss(self.recon_logits, fake) + self.criterion_loss(self.input_logits, valid))/2

        #Consistency loss
        self.loss_consistency = self.l1_loss(self.output_recon, self.input)
        self.loss_consistency_trg = self.l1_loss(self.output_trans_trg, self.target)

        #Identity loss
        self.loss_identity = self.l1_loss(self.output_recon_identity, self.input)
        self.loss_identity_trg = self.l1_loss(self.output_trans_identity, self.target)

        self.loss_g = self.loss_g_trans + self.loss_g_recon + self.loss_consistency + self.loss_consistency_trg + self.loss_identity + self.loss_identity_trg
        self.loss_d = self.loss_d_trans + self.loss_d_recon


    #Compute backpropogation for the model
    def optimize_parameters(self):
        self.loss_g.backward()
        self.optimizer_g.step()
        self.optimizer_g.zero_grad()
        self.loss_d.backward()
        self.optimizer_d.step()
        self.optimizer_d.zero_grad()
        torch.cuda.empty_cache()

    #Test function for the model
    def test(self):
        super().test() # run the forward pass

        # save predictions and labels as flat tensors
        # self.val_images.append(self.input)
        # self.val_predictions.append(self.output)
        # self.val_labels.append(self.input)

    #Should be run after each epoch, outputs accuracy
    def post_epoch_callback(self, epoch, visualizer):
        self.val_predictions = torch.cat(self.val_predictions, dim=0)
        predictions = torch.argmax(self.val_predictions, dim=1)
        predictions = torch.flatten(predictions).cpu()

        self.val_labels = torch.cat(self.val_labels, dim=0)
        labels = torch.flatten(self.val_labels).cpu()

        self.val_images = torch.squeeze(torch.cat(self.val_images, dim=0)).cpu()

        # Calculate and show accuracy
        val_accuracy = accuracy_score(labels, predictions)

        metrics = OrderedDict()
        metrics['Accuracy'] = val_accuracy

        if (visualizer != None):
            visualizer.plot_current_validation_metrics(epoch, metrics)
        print('Validation accuracy: {0:.3f}'.format(val_accuracy))

        # Here you may do something else with the validation data such as
        # displaying the validation images or calculating the ROC curve

        self.val_images = []
        self.val_predictions = []
        self.val_labels = []


if __name__ == "__main__":
    net = TEMPLATEmodel().cuda()
    from torchsummary import summary

    print(summary(net, input_size=(1, 48, 48)))
