import argparse
from datasets import create_dataset
from utils import parse_configuration
import math
from models import create_model
import time
from utils.visualizer import Visualizer
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch
import matplotlib.pyplot as plt

"""Performs training of a specified model.

Input params:
    config_file: Either a string with the path to the JSON
        system-specific config file or a dictionary containing
        the system-specific, dataset-specific and
        model-specific settings.
    export: Whether to export the final model (default=True).
"""
def train(config_file, export=True):
    print('Reading config file...')
    configuration = parse_configuration(config_file)

    print('Initializing dataset...')
    train_dataset = create_dataset(configuration['train_dataset_params'])
    train_dataset_size = len(train_dataset)
    print('The number of training samples = {0}'.format(train_dataset_size))

    val_dataset = create_dataset(configuration['val_dataset_params'])
    val_dataset_size = len(val_dataset)
    print('The number of validation samples = {0}'.format(val_dataset_size))

    print('Initializing model...')
    model = create_model(configuration['model_params'])
    model.setup()

    print('Initializing visualization...')
    visualizer = Visualizer(configuration['visualization_params'])   # create a visualizer that displays images and plots

    if (type(configuration['model_params']['load_checkpoint']) == str):
        starting_epoch = configuration['model_params']['scheduler_epoch'] + 1
    else:
        starting_epoch = configuration['model_params']['load_checkpoint'] + 1
    num_epochs = configuration['model_params']['max_epochs']

    best_loss = 1000000000

    #Loops through all epochs
    for epoch in range(starting_epoch, num_epochs):
        epoch_start_time = time.time()  # timer for entire epoch
        train_dataset.dataset.pre_epoch_callback(epoch)
        model.pre_epoch_callback(epoch)

        train_iterations = len(train_dataset)
        train_batch_size = configuration['train_dataset_params']['loader_params']['batch_size']
        input_size = configuration['train_dataset_params']['input_size']

        g_loss = 0
        d_loss = 0

        model.train()
        #On every epoch, loop through all data in train_dataset
        for i, data in enumerate(train_dataset):  # inner loop within one epoch
            # if (i > 0):
            #     break
            visualizer.reset()
            cur_data = data

            model.set_input(cur_data)         # unpack data from dataset and apply preprocessing

            output = model.forward()
            model.compute_loss()

            g_loss += model.loss_g.item()
            d_loss += model.loss_d.item()

            if i % configuration['model_update_freq'] == 0:
                model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if i % configuration['printout_freq'] == 0:
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, num_epochs, i, math.floor(train_iterations / train_batch_size), losses)
                visualizer.plot_current_losses(epoch, float(i) / math.floor(train_iterations / train_batch_size), losses)

        model.eval()
        for i, data in enumerate(val_dataset):
            if (i > 0):
                break

            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.test()

            img = data[0][0].permute(1, 2, 0).cpu().detach().numpy()
            out_trans = model.output_trans[0].permute(1, 2, 0).cpu().detach().numpy()
            out_recon = model.output_recon[0].permute(1, 2, 0).cpu().detach().numpy()

            fig, axs = plt.subplots(1, 3)
            axs[0].imshow(img)
            axs[0].set_title('Original')
            axs[1].imshow(out_trans)
            axs[1].set_title('Generated')
            axs[2].imshow(out_recon)
            axs[2].set_title('Reconstructed')
            plt.savefig("./plots/epoch_{}.png".format(epoch))
            plt.close()

        # model.post_epoch_callback(epoch, visualizer)
        train_dataset.dataset.post_epoch_callback(epoch)

        if (g_loss+d_loss < best_loss):
            best_loss = g_loss+d_loss
            print('Saving new best model at the end of epoch {0}'.format(epoch))
            model.save_networks("best")
            model.save_optimizers("best")

        print('Saving latest model at the end of epoch {0}'.format(epoch))
        model.save_networks("last")
        model.save_optimizers("last")
        # model.update_prev_losses()

        data = OrderedDict()
        data['G Loss'] = g_loss
        data['D Loss'] = d_loss
        visualizer.plot_current_epoch_loss(epoch, data)

        print('End of epoch {0} / {1} \t Time Taken: {2} sec'.format(epoch, num_epochs, time.time() - epoch_start_time))
        print('Generator Loss: {:.4f}, Discriminator Loss: {:.4f}'.format(g_loss, d_loss))

        model.update_learning_rate() # update learning rates every epoch

    if export:
        print('Exporting model')
        model.eval()
        custom_configuration = configuration['train_dataset_params']
        custom_configuration['loader_params']['batch_size'] = 1 # set batch size to 1 for tracing
        dl = train_dataset.get_custom_dataloader(custom_configuration)
        sample_input = next(iter(dl)) # sample input from the training dataset
        model.set_input(sample_input)
        model.export()

    return model.get_hyperparam_result()

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', True)

    parser = argparse.ArgumentParser(description='Perform model training.')
    parser.add_argument('configfile', default="./config_fer.json", help='path to the configfile')

    args = parser.parse_args()
    train(args.configfile)
