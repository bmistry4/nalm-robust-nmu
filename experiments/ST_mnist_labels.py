"""
Static MNIST Product; Colour-channel concatenated Version
"""
import math
import os
os.environ['MPLCONFIGDIR'] = '/tmp'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}  # Ignore tf info logging # TODO - remove for server runs
import argparse
import ast
import os.path as path
import random
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

from stable_nalu.layer.two_digit_mnist import *
import stable_nalu.writer as writer
from stable_nalu.abstract import ExtendedTorchModule
import torch.optim.lr_scheduler as lr_scheduler
from stable_nalu.layer import ReRegualizedLinearMNACLayer
from stable_nalu.layer import ReRegualizedLinearNACLayer
from torchvision.utils import save_image

parser = argparse.ArgumentParser(description='Runs the simple function static task')
parser.add_argument('--id',
                    action='store',
                    default=-1,
                    type=int,
                    help='Unique id to identify experiment')
parser.add_argument('--seed',
                    action='store',
                    default=0,
                    type=int,
                    help='Specify the seed to use')
parser.add_argument('--data-path',
                    action='store',
                    default='../stable_nalu/dataset/data/two_digit_mnist',
                    type=str,
                    help='Where the (mnist) data should be stored')
parser.add_argument('--operation',
                    action='store',
                    default='add',
                    choices=['add', 'mul'],
                    type=str,
                    help='Specify the operation to use, e.g. add, mul')

parser.add_argument('--max-epochs',
                    action='store',
                    default=100,
                    type=int,
                    help='Specify the max number of epochs to use')
parser.add_argument('--samples-per-permutation',
                    action='store',
                    default=1000,
                    type=int,
                    help='Number of samples per permutation (e.g. there are 90 permutations in the train set so 1000 * 90).')
parser.add_argument('--num-folds',
                    action='store',
                    default=10,
                    type=int,
                    help='Number of folds for cross-val')
parser.add_argument('--batch-size',
                    action='store',
                    default=128,
                    type=int,
                    help='Specify the batch-size to be used for training')
parser.add_argument('--learning-rate',
                    action='store',
                    default=1e-3,
                    type=float,
                    help='Specify the learning-rate')

parser.add_argument('--no-cuda',
                    action='store_true',
                    default=False,
                    help=f'Force no CUDA (cuda usage is detected automatically as {torch.cuda.is_available()})')
parser.add_argument('--name-prefix',
                    action='store',
                    default='two_digit_mnist',
                    type=str,
                    help='Where the data should be stored')
parser.add_argument('--remove-existing-data',
                    action='store_true',
                    default=False,
                    help='Should old results be removed')
parser.add_argument('--verbose',
                    action='store_true',
                    default=False,
                    help='Should network measures (e.g. gates) and gradients be shown')

parser.add_argument('--log-interval',
                    action='store',
                    default=1,
                    type=int,
                    help='Log to tensorboard every X epochs.')
parser.add_argument('--mb-log-interval',
                    action='store',
                    default=100,
                    type=int,
                    help='Log to tensorboard every X minibatches.')
parser.add_argument('--dataset-workers',
                    action='store',
                    default=0,
                    type=int,
                    help='Number of workers for multi-process data loading')

parser.add_argument('--use-nalm',
                    action='store_true',
                    default=False,
                    help=f'')
parser.add_argument('--learn-labels2out',
                    action='store_true',
                    default=False,
                    help=f'If the last layer should have learnable params (True) or just apply the correct operation (False)')

parser.add_argument('--regualizer-scaling-start',
                    action='store',
                    default=30,
                    type=int,
                    help='Start linear scaling at this global step.')
parser.add_argument('--regualizer-scaling-end',
                    action='store',
                    default=40,
                    type=int,
                    help='Stop linear scaling at this global step.')
parser.add_argument('--regualizer',
                    action='store',
                    default=100,
                    type=float,
                    help='Specify the regualization lambda to be used')

parser.add_argument('--beta-nau',
                    action='store_true',
                    default=False,
                    help='Have nau weights initialised using a beta distribution B(7,7)')
parser.add_argument('--nau-noise',
                    action='store_true',
                    default=False,
                    help='Applies/ unapplies additive noise from a ~U[1,5] during training.')
parser.add_argument('--nmu-noise',
                    action='store_true',
                    default=False,
                    help='Applies/ unapplies multiplicative noise from a ~U[1,5] during training. Aids with failure ranges on a vinilla NMU.')
parser.add_argument('--noise-range',
                    action='store',
                    default=[1, 5],
                    type=ast.literal_eval,
                    help='Range at which the noise for applying stochasticity is taken from. (Originally for sNMU.)')

parser.add_argument('--no-save',
                    action='store_true',
                    default=False,
                    help='Do not save model at the end of training')
parser.add_argument('--load-checkpoint',
                    action='store_true',
                    default=False,
                    help='Loads a saved checkpoint and resumes training')

parser.add_argument('--rgb',
                    action='store_true',
                    default=False,
                    help='If images are in colour (rgb). Used so we know the number of colour channels.')
parser.add_argument('--val-split',
                    action='store',
                    default=0.15,
                    type=float,
                    help='Validation split proportion from training data.')
parser.add_argument('--scheduler-step-size',
                    action='store',
                    default=30,
                    type=int,
                    help='Step size for the StepLR scheduler.')

parser.add_argument('--img2label-model',
                    action='store',
                    default='concat',
                    type=str,
                    help='Specify the type of img2label network you want')
parser.add_argument('--optimizer',
                    action='store',
                    default='sgd',
                    choices=['adam', 'sgd'],
                    type=str,
                    help='The optimization algorithm to use, Adam or SGD')
parser.add_argument('--no-scheduler',
                    action='store_true',
                    default=False,
                    help='Switches off scheduler.')
parser.add_argument('--image-concat-dim',
                    action='store',
                    default='colour',
                    choices=['colour', 'width'],
                    type=str,
                    help='The dimension which the different images should be concatenated on.')
parser.add_argument('--clip-grad-norm',
                        action='store',
                        default=None,
                        type=float,
                        help='Norm clip value for gradients.')
parser.add_argument('--tps-unbounded-stn',
                    action='store_true',
                    default=False,
                    help='Set TPS LocNet to be unbounded i.e. not be scaled between [-1.1].')

def create_tb_writer(args, fold_idx, use_dummy=False):
    if use_dummy:
        return writer.DummySummaryWriter()
    else:
        return writer.SummaryWriter(
            name=
            f'{args.name_prefix}/{args.id}'
            f'_f{fold_idx}'
            f'_op-{args.operation}'
            f'_nalm{str(args.use_nalm)[0]}'
            f'_learnL{str(args.learn_labels2out)[0]}'
            f'_s{args.seed}',
            remove_existing_data=args.remove_existing_data
        )


########################## HOOKS ###################################################################
def fhook_channel_grid(module, input, output):
    # show batch of imgs (as grid) containing the input and the spatially transformed imgs f.e. img channel
    show_channel_conat_batch(input[0], title="input")
    show_channel_conat_batch(output[1], title="st1_out")
    show_channel_conat_batch(output[2], title="st2_out")


########################## IMAGE PLOTTING FUNCTIONS ###################################################################
def get_image_grid(image_grid):
    # generate image grid to plot a batch of images
    # unnormalize the images
    image_grid = image_grid / 2 + 0.5
    image_grid = image_grid.numpy()
    # transpose to make channels last
    image_grid = np.transpose(image_grid, (1, 2, 0))
    return image_grid


def show_image(image, cmap=None, title=""):
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.show()


def show_image_per_channel(image, title=""):
    # given a single image,plot the img f.e. colour channel
    for c in range(image.shape[0]):
        show_image(image.cpu()[c].unsqueeze(0).permute(1, 2, 0), cmap='gray', title=f'{title} (ch. {c})')


def plot_example(X, y, z=None):
    """Plot the first 5 images and their labels in a row."""
    if z is None:
        for i, (img, y) in enumerate(zip(X[:5].reshape(5, 28, -1), y[:5])):
            plt.subplot(151 + i)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
            plt.title(y)
    else:
        for i, (img, y, z) in enumerate(zip(X[:5].reshape(5, 28, -1), y[:5], z[:5])):
            plt.subplot(151 + i)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
            plt.title(f"({z[0]},{z[1]}); {y}")
    plt.show()


def show_channel_conat_batch(data, title=""):
    """
    Show images in batch
    Args:
        data: [B, C=2, H, W]. Assumes the 2 digits are concatenated in the colour channel dim
    """
    data = data.cpu()
    for i in range(data.shape[1]):
        image_grid = torchvision.utils.make_grid(data[:, i:i+1, :, :])
        grid = get_image_grid(image_grid)
        show_image(grid, title=f"{title} (ch.{i})")
        # import sys
        # sys.exit()


def confusion_matrix(model, data_loader, device, operation='add'):
    # from https://stackoverflow.com/questions/53290306/confusion-matrix-and-test-accuracy-for-pytorch-transfer-learning-tutorial
    if operation == 'add':
        nb_classes = 19
        class_names = list(range(nb_classes))
    elif operation == 'mul':
        nb_classes = 37
        class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36,
                       40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81]

    confusion_matrix = np.zeros((nb_classes, nb_classes))
    with torch.no_grad():
        for i, (inputs, classes, _) in enumerate(data_loader):
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs, _ = model(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    # print(confusion_matrix)
    print(confusion_matrix.diagonal() / confusion_matrix.sum(1))    # per-class accuracy

    plt.figure(figsize=(15, 10))
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names).astype(int)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def digit_confusion_matrix(model, data_loader, device, digit_idx=None, round=False, old_model=False, save_data={}):
    """
    Plots the confusion matrix for an output which predicts a digit label (between 0-9).
    Predicted labels will be floats not ints so require casting to get the predicted label.
    Args:
        model: net which predicts the digit labels
        data_loader:
        device:
        digit_idx: represents which digit we want the confusion matrix of. use 0 for img1, 1 for img 2, or None to do both
        device: if to round the digit before evaluating
        old_model: if an earlier network is being used. Such a network only returns 1 output for the img2label net.
        save_data: save information dict for saving confusion matrix and class accuracies.
            Includes: save_dir, model_name. Use {} to not save.
    Returns: None

    """
    # from https://stackoverflow.com/questions/53290306/confusion-matrix-and-test-accuracy-for-pytorch-transfer-learning-tutorial
    nb_classes = 10
    confusion_matrix = np.zeros((nb_classes, nb_classes))
    with torch.no_grad():
        for i, (inputs, _, digit_labels) in enumerate(data_loader):
            inputs = inputs.to(device)
            digit_labels = digit_labels.to(device)
            # [B,2] -> assumes the predicted values (not the logits)
            if old_model:
                digit_preds = model(inputs)
            else:
                digit_preds, _, _ = model(inputs)

            if round:
                digit_preds = digit_preds.round()

            if digit_labels is not None:
                # get relevant digit (0=1st digit, 1=2nd digit)
                digit_labels = digit_labels[:, digit_idx]
                digit_preds = digit_preds[:, digit_idx]
            for t, p in zip(digit_labels.view(-1), digit_preds.view(-1)):
                # cast to int as pred is a float. Can't round otherwise 9 can become 10.
                confusion_matrix[t.long(), int(p)] += 1

    print("confusion_matrix.diagonal() / confusion_matrix.sum(1) # per-class accuracy")
    class_accs = confusion_matrix.diagonal() / confusion_matrix.sum(1)
    print(class_accs)    # per-class accuracy

    plt.figure(figsize=(15, 10))
    class_names = list(range(nb_classes))
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names).astype(int)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=15)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=15)
    plt.ylabel(f'True label (digit_idx {"0 and 1" if digit_idx is None else digit_idx})')
    plt.xlabel(f'Predicted label (digit_idx {"0 and 1" if digit_idx is None else digit_idx})')

    if save_data:
        assert set(save_data.keys()) == set(['save_dir', 'model_name']), "save_data dict is missing a key"
        base_save_path = path.join(save_data['save_dir'], save_data['model_name'])

        # save class accuracies to a csv
        class_accs_df = pd.DataFrame(class_accs, columns=["accuracy"])
        class_accs_df.insert(0, "model", save_data['model_name'], False)
        class_accs_df.insert(1, "digit", class_names, False)
        class_accs_filepath = base_save_path + '_class_accs.csv'
        class_accs_df.to_csv(class_accs_filepath, index=False)

        # save confusion matrix plot
        confusion_matrix_filepath = base_save_path + '_confusion_matrix.pdf'
        plt.savefig(confusion_matrix_filepath, dpi=300)

        print("Class accuracy csv and confusion matrix saved")
    # if not saving then show the confusion matrix
    else:
        plt.show()
#######################################################################################################################

########################### DATASET RELATED FUNCS/CLASSES ###########################################################
class NoneTransform(object):
    ''' Does nothing to the image. To be used instead of None '''
    def __call__(self, image):
        return image


class RandomPadding(object):
    """
    Adds random amount of padding to image so new shape of image goes from [C,H,W] to [C, max_height, max_width]
    """

    def __init__(self, max_height, max_width):
        self.max_h = max_height
        self.max_w = max_width

    def __call__(self, img):
        c, h, w = img.shape
        # sample random x,y position to place img
        offset_h = np.random.randint(0, self.max_h - h)
        offset_w = np.random.randint(0, self.max_w - w)
        new_img = torch.zeros(c, self.max_h, self.max_w)
        # place image onto new 'padded' image at the random location
        new_img[:, offset_h: offset_h + h, offset_w: offset_w + w] = img
        return new_img

    def __repr__(self):
        return self.__class__.__name__ + '()'


class TwoDigitMNISTDataset(Dataset):
    """
    Generate dataset from numpy arrays.
    Transforms will be applied in __getitem__
    """
    def __init__(self, X, y, z=None, transform=None, img_concat_dim=0):
        self.X = X  # [B,H,W,2]
        self.y = torch.tensor(y).to(torch.float32)  # convert to a float tensor; [B]
        self.z = torch.tensor(z).to(torch.float32)  # [B,2]
        self.transform = transform
        self.img_concat_dim = img_concat_dim

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Similar to the MNIST dataset class
        img, target, intermediate_targets = self.X[idx], self.y[idx], self.z[idx]
        if self.transform is not None:
            # X[idx] = [H, W, C]
            img1 = self.transform(Image.fromarray(self.X[idx, :, :, 0]))    # affine transforms require PIL imgs
            img2 = self.transform(Image.fromarray(self.X[idx, :, :, 1]))
            # [C,H,W] where C is the colour channel
            img = torch.cat((img1, img2), dim=self.img_concat_dim)  # concat on 0 for colour and 2 for width
        return img, target, intermediate_targets


def check_digit_data_split_and_recombine(data):
    """
    Sanity check if the splitting of the 2 digits to single digits (stored across the batch dim) works as expected.
    Also check the recombining of the individual digits back to their two digit single img format.
    """
    print(data.shape)  # [B,C,H,W]
    # splits digits and collapses indep digits into batch dim
    split_digits = torch.cat(data.split(data.shape[-1] // 2, -1), 0)  # [2*B, C=1, 28, 28]
    # recombine the original digits back to the 2-digit img
    recombined_digits = torch.cat(split_digits.split(split_digits.shape[0] // 2, 0), -1)  # [B, C=1, 28, 56]

    # select random img from batch
    img_idx = np.random.randint(0, data.shape[0])

    plt.imshow(data[img_idx].permute(1, 2, 0))
    plt.show()
    plt.imshow(split_digits[img_idx].permute(1, 2, 0))
    plt.show()
    plt.imshow(split_digits[data.shape[0] + img_idx].permute(1, 2, 0))
    plt.show()
    plt.imshow(recombined_digits[img_idx].permute(1, 2, 0))
    plt.show()


def check_both_channels_have_all_digits(pairings, state):
    """
    Checks if all left/right digits contain at least 1 instance of the digits 0-9
    """
    left_digits = set(map(lambda x: int(x[0]), pairings))
    right_digits = set(map(lambda x: int(x[1]), pairings))
    assert (left_digits - set(range(10))) == set(), f"{state}: Left digits do not contain all possible digits 0-9"
    assert (right_digits - set(range(10))) == set(), f"{state}: Right digits do not contain all possible digits 0-9"
    print(f"Passed left and right digit checks for {state}")

#######################################################################################################################

########################### Networks ##################################################################################
class Labels2Out(ExtendedTorchModule):
    def __init__(self, in_features, out_features, learn_labels2out, operation, **kwargs):
        super(Labels2Out, self).__init__('last_layer', **kwargs)

        self.learn_labels2out = learn_labels2out  # if to learn the last layer of to just apply the correct operation
        self.operation = operation

        if learn_labels2out:
            if self.operation == 'add':
                self.fc = torch.nn.Linear(in_features, out_features, bias=False)
            elif self.operation == 'mul':
                self.fc = torch.nn.Parameter(torch.Tensor(out_features, in_features))

    def reset_parameters(self):
        # same init as NMU
        if self.operation == 'mul' and self.learn_labels2out:
            std = math.sqrt(0.25)
            r = min(0.25, math.sqrt(3.0) * std)
            torch.nn.init.uniform_(self.fc, 0.5 - r, 0.5 + r)

    def forward(self, x):
        if not self.learn_labels2out:
            if self.operation == 'add':
                out = x.sum(1)
            elif self.operation == 'mul':
                out = x.prod(1)
        else:
            # x = [B, I=2]
            if self.operation == 'add':
                out = self.fc(x)
            # uses product aggregator
            elif self.operation == 'mul':
                out_size, in_size = self.fc.size()
                x = x.view(x.size()[0], in_size, 1)
                W = self.fc.t().view(1, in_size, out_size)
                # [B,O] = prod([B,I,1] * [1,I,O])
                out = torch.prod(x * W, -2)
        # [B, 1]
        return out

    def get_weights(self):
        if self.operation == 'add':
            return self.fc.weight
        elif self.operation == 'mul':
            return self.fc


class Net(ExtendedTorchModule):
    def __init__(self, img2label_model='concat', img2label_in=2, img2label_out=1, use_nalm=False,
                 learn_labels2out=False, operation=None, writer=None, device=None, **kwags):
        super(Net, self).__init__('network', writer=writer, **kwags)

        if img2label_model == 'concat':
            self.img2label = Img2LabelsSpatialTransformerConcat(device=device)
        elif img2label_model == 'no-concat-linear':
            self.img2label = Img2LabelsSpatialTransformerLinearNoConcat(device=device)
        elif img2label_model == 'no-concat-conv':
            self.img2label = Img2LabelsSpatialTransformerConvNoConcat(device=device)
        elif img2label_model == 'concat-conv':
            self.img2label = Img2LabelsSpatialTransformerConvConcat(device=device)
        elif img2label_model == 'no-concat-indep-conv':
            self.img2label = Img2LabelsSpatialTransformerIndepConvNoConcat(device=device)
        elif img2label_model == 'width-no-concat-conv':
            self.img2label = Img2LabelsWidthSpatialTransformerConvNoConcat(device=device)
        elif img2label_model == 'TPS-no-concat-conv':
            self.img2label = Img2LabelsSpatialTransformerTPSConvNoConcat(device=device,
                 image_height=kwags['image_height'], image_width=kwags['image_width'],
                 span_range_height=kwags['span_range_height'], span_range_width=kwags['span_range_width'],
                 grid_height_points=kwags['grid_height_points'], grid_width_points=kwags['grid_width_points'],
                 tps_unbounded_stn=kwags['tps_unbounded_stn']
            )
        else:
            raise KeyError('invalid img2label_model name given')

        self.operation = operation
        self.learn_labels2out = learn_labels2out
        if use_nalm:
            if operation == 'add':
                self.labels2Out = ReRegualizedLinearNACLayer(img2label_in, img2label_out, nac_oob='clip',
                                                             regualizer_shape='linear', writer=self.writer, **kwags)
            else:
                self.labels2Out = ReRegualizedLinearMNACLayer(img2label_in, img2label_out, nac_oob='clip',
                                                              regualizer_shape='linear', writer=self.writer, **kwags)
        else:
            self.labels2Out = Labels2Out(img2label_in, img2label_out, self.learn_labels2out, self.operation,
                                         writer=self.writer, **kwags)

    def forward(self, x):
        digit_preds, st1_out, st2_out = self.img2label(x)
        out = self.labels2Out(digit_preds)
        out = out.squeeze()
        return out, digit_preds

#######################################################################################################################

class TwoDigitExperiment:
    def __init__(self):
        self._global_step = 0
        self.main()

    def get_label2out_weight(self, model, idx, is_nalm, learn_last_layer):
        if is_nalm:
            return model.labels2Out.W.view(-1)[idx].item()
        elif learn_last_layer:
            return model.labels2Out.get_weights().view(-1)[idx].item()

    def calc_reg_loss(self, model, w_scale, args):
        regualizers = model.regualizer()
        reg_loss = regualizers['W'] * w_scale * args.regualizer
        return reg_loss

    def generate_data_samples(self, set_pairs, dataset, x, y, z, args):
        """
        Appends to lists representing the input data and labels.
        Args:
            set_pairs: contains array of strings represetning input digits which are allowed to occur
                e.g. '04' means img 1 = 0 and img 2 = 4
            dataset: mnist dataset (either train or test)
            x: empty list to fill with the input images
            y: empty list to fill with the target labels
            z: empty list to fill with the intermediary image labels e.g. a input with the number 56 will give labels
                '5' and '6'
            args: parser args

        Returns: None. The x,y and z passed in are object references so nothing requires to be returned.

        """
        for train_set_pair in set_pairs:
            for _ in range(args.samples_per_permutation):
                rand_i = np.random.choice(np.where(dataset.targets == int(train_set_pair[0]))[0])
                rand_j = np.random.choice(np.where(dataset.targets == int(train_set_pair[1]))[0])

                temp_image = np.concatenate((dataset.data[rand_i].unsqueeze(2), dataset.data[rand_j].unsqueeze(2)), axis=2)
                x.append(temp_image)
                target_zi = dataset.targets[rand_i]
                target_zj = dataset.targets[rand_j]
                z.append([target_zi, target_zj])

                if args.operation == 'add':
                    y.append(target_zi + target_zj)
                elif args.operation == 'mul':
                    y.append(target_zi * target_zj)
                else:
                    raise KeyError(f'Invalid operation ({args.operation}) given.')

    def eval_dataloader(self, model, device, dataloader, summary_writer, state):
        """
        Prints and logs stats for a given dataloader.
        Model is run in eval mode so weights are fixed.
        """
        model.eval()
        loss = 0
        correct_output = 0
        correct_output_rounded = 0

        img_label_losses = torch.zeros(2).to(device)
        correct_img_labels = torch.zeros(2).to(device)
        correct_img_labels_rounded = torch.zeros(2).to(device)

        with torch.no_grad():
            for data, target, img_label_targets in dataloader:
                data, target, img_label_targets = data.to(device), target.to(device), img_label_targets.to(device)
                ###############################################################################################
                # if loss == 0:
                #     show_channel_conat_batch(data)
                ###############################################################################################
                output, img_labels_output = model(data)
                # loss += F.cross_entropy(output, target.long(), reduction='sum').item()
                # output_pred = output.max(1, keepdim=True)[1]        # get the index of the max log-probability
                loss += F.mse_loss(output, target, reduction='sum').item()
                correct_output += output.eq(target.view_as(output)).sum().item()
                output_pred_rounded = output.round()       # get the index of the max log-probability
                correct_output_rounded += output_pred_rounded.eq(target.view_as(output_pred_rounded)).sum().item()

                img_label_losses += F.mse_loss(img_labels_output, img_label_targets, reduction='none').sum(dim=0)
                correct_img_labels += img_labels_output.eq(img_label_targets).sum(dim=0)
                correct_img_labels_rounded += img_labels_output.round().eq(img_label_targets).sum(dim=0)

        loss /= len(dataloader.dataset)
        img_label_losses /= len(dataloader.dataset)

        acc_output = 100. * correct_output / len(dataloader.dataset)
        acc_label_1 = 100. * correct_img_labels[0].item() / len(dataloader.dataset)
        acc_label_2 = 100. * correct_img_labels[1].item() / len(dataloader.dataset)

        acc_output_rounded = 100. * correct_output_rounded / len(dataloader.dataset)
        acc_label_1_rounded = 100. * correct_img_labels_rounded[0].item() / len(dataloader.dataset)
        acc_label_2_rounded = 100. * correct_img_labels_rounded[1].item() / len(dataloader.dataset)

        # prints the average: epoch loss, accuracy of the final output and intermediate label losses.
        print('{}: {:.5f}, acc: {}/{} ({:.2f}%), img1: {:.5f} ({:.2f}%), img2: {:.5f} ({:.2f}%)\t '
              'Rounded: acc: {}/{} ({:.2f}%), img1:({:.2f}%), img2: ({:.2f}%)'.format(
            state, loss, correct_output, len(dataloader.dataset),
            acc_output,
            img_label_losses[0],
            acc_label_1,
            img_label_losses[1],
            acc_label_2,
            correct_output_rounded, len(dataloader.dataset),
            acc_output_rounded,
            acc_label_1_rounded,
            acc_label_2_rounded
        ))

        # log stats to tensorboard
        # log stats to tensorboard
        summary_writer.add_scalar(f'metric/{state}/output/loss', loss)
        summary_writer.add_scalar(f'metric/{state}/label1/loss', img_label_losses[0])
        summary_writer.add_scalar(f'metric/{state}/label2/loss', img_label_losses[1])

        summary_writer.add_scalar(f'metric/{state}/output/acc', acc_output)
        summary_writer.add_scalar(f'metric/{state}/label1/acc', acc_label_1)
        summary_writer.add_scalar(f'metric/{state}/label2/acc', acc_label_2)

        summary_writer.add_scalar(f'metric/{state}/output_rounded/acc', acc_output_rounded)
        summary_writer.add_scalar(f'metric/{state}/label1_rounded/acc', acc_label_1_rounded)
        summary_writer.add_scalar(f'metric/{state}/label2_rounded/acc', acc_label_2_rounded)

        model.train()

    def epoch_step(self, model, train_loader, args, optimizer, epoch, w_scale, summary_writer, test_loader, valid_loader):
        """
        Train and test the model for a single epoch. Logging occurs at the start of the epoch before any optimisation
        meaning epoch 0 will log the model stats before any param updates.
        """
        model.train()
        for batch_idx, (data, target, img_label_targets) in enumerate(train_loader):
            data, target, img_label_targets = data.to(args.device), target.to(args.device), img_label_targets.to(args.device)

            ###########################################################################################################
            # sanity check print out imgs (after transformations have been applied)
            # show_channel_conat_batch(data)
            ###########################################################################################################

            self._global_step += 1
            summary_writer.set_iteration(self._global_step)
            summary_writer.add_scalar('epoch', epoch)

            # log to tensorboard. Metrics include stats for training, testing (over the entire dataloader) and the
            # weights or the label2out layer (if any exists).
            if epoch % args.log_interval == 0 and batch_idx == 0:
                self.eval_dataloader(model, args.device, train_loader, summary_writer, 'train')
                self.eval_dataloader(model, args.device, valid_loader, summary_writer, 'valid')
                self.eval_dataloader(model, args.device, test_loader, summary_writer, 'test')
                if args.learn_labels2out:
                    # plot weight values of final layer
                    summary_writer.add_scalar('label2out/weights/w0',
                                              self.get_label2out_weight(model, 0, args.use_nalm, args.learn_labels2out))
                    summary_writer.add_scalar('label2out/weights/w1',
                                              self.get_label2out_weight(model, 1, args.use_nalm, args.learn_labels2out))
            optimizer.zero_grad()

            with summary_writer.force_logging(epoch % args.log_interval == 0 and batch_idx == 0):
                # check_digit_data_split_and_recombine(data)
                output, img_labels_output = model(data)

            # loss = F.cross_entropy(output, target.long(), reduction='mean')
            loss = F.mse_loss(output, target, reduction='mean')
            total_loss = loss

            if args.use_nalm:
                reg_loss = self.calc_reg_loss(model, w_scale, args)
                total_loss += reg_loss

            total_loss.backward()
            if args.clip_grad_norm != None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            model.optimize(loss)

    def reset_weights(self, m):
        """
          Reset model weights to avoid weight leakage.
        """
        if hasattr(m, 'reset_parameters'):
            # print(f'Reset trainable parameters of layer = {m}')
            m.reset_parameters()

    def main(self):
        ####################################################################################################################
        args = parser.parse_args()
        args.device = torch.device("cpu" if (args.no_cuda or not torch.cuda.is_available()) else "cuda")

        ####################################################################################################################
        def set_reproducability_flags(seed):
            # Set reproducability flags - see https://pytorch.org/docs/stable/notes/randomness.html
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np_rng = np.random.RandomState(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.set_default_dtype(torch.float32)
            return np_rng

        fold_idx = args.seed
        set_reproducability_flags(fold_idx)
        assert fold_idx < args.num_folds, "Seed must be less than number of folds. (Seed is analogous to fold idx.)"
        ####################################################################################################################

        # TODO - only for dev - remove when real run.
        use_dummy_writer = False  # dummywriter won't save tensorboard files

        # set the dimension for concatanating the two images
        if args.image_concat_dim == 'colour':
            concat_dim = 0
        elif args.image_concat_dim == 'width':
            concat_dim = 2

        ################################################################################
        # check unsupported edge case
        assert (not args.use_nalm or args.learn_labels2out), "NALM with fixed weights is not supported"

        # print parser args
        print(' '.join(f'{k}: {v}\n' for k, v in vars(args).items()))

        ################################################################################
        transform = transforms.Compose([
            # affine transforms require PIL imgs so must be first
            transforms.RandomAffine(
                degrees=(-45, 45),
                scale=(0.7, 1.2),
            ),
            transforms.ToTensor(),
            # https://discuss.pytorch.org/t/grayscale-to-rgb-transform/18315/9
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)) if args.rgb else NoneTransform(),
            RandomPadding(42, 42),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        # will be creating a new dataset which joins 2 mnist images together. As the __get__item applies the Transform
        # cannot use Transform here as it won't get called.
        train_dataset = datasets.MNIST(args.data_path, train=True, download=True)
        test_dataset = datasets.MNIST(args.data_path, train=False, download=False)
        ################################################################################

        unique_pairs = [str(x) + str(y) for x in range(10) for y in range(10)]
        kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=np.random.RandomState(0))
        unique_pairs_np = np.asarray(unique_pairs)

        # only use if run only does 1 fold
        fold_unique_pairs_list = list(kf.split(unique_pairs))
        train_index, test_index = fold_unique_pairs_list[fold_idx]

        ################################################################################
        # use loop if want to do all folds in sequence
        # for fold_idx, (train_index, test_index) in enumerate(kf.split(unique_pairs)):
        # set seed for each new fold. Makes reproducability simpler if rereun/checkpointing of a single fold is required.
        np_rng = set_reproducability_flags(fold_idx)
        self._global_step = 0   # reset iteration step count every new fold

        print('=====================================================')
        print(f"Fold {fold_idx}")
        summary_writer = create_tb_writer(args, fold_idx, use_dummy=use_dummy_writer)
        print("Writer name: ", summary_writer.name)
        print('-----------------------------------------------------')
        # get the digit pair that are used for the current fold
        test_set_pairs = unique_pairs_np[test_index]
        train_set_pairs = unique_pairs_np[train_index]

        # Sanity checks
        print(f"Train pairs: {len(train_set_pairs)}, test pairs: {len(test_set_pairs)}\n")
        # assert (len(test_set_pairs) == 10)
        # assert (len(train_set_pairs) == 90)
        for test_set_pair in test_set_pairs:
            assert (test_set_pair not in train_set_pairs)

        ################################################################################
        X_train = []
        y_train = []
        z_train = []    # img labels
        self.generate_data_samples(train_set_pairs, train_dataset, X_train, y_train, z_train, args)

        X_test = []
        y_test = []
        z_test = []
        self.generate_data_samples(test_set_pairs, test_dataset, X_test, y_test, z_test, args)

        ################################################################################

        X_train_shuffled, y_train_shuffled, z_train_shuffled = \
            shuffle(X_train, y_train, z_train, random_state=np_rng)
        X_test_shuffled, y_test_shuffled, z_test_shuffled = \
            shuffle(X_test, y_test, z_test, random_state=np_rng)

        # can't convert to a tensor yet otherwise the Dataset Transforms won't work
        two_digit_train_X = np.asarray(X_train_shuffled)
        two_digit_train_y = np.asarray(y_train_shuffled)
        two_digit_train_z = np.asarray(z_train_shuffled)

        # split train set to get validation data
        two_digit_train_X, two_digit_valid_X, \
        two_digit_train_y, two_digit_valid_y, \
        two_digit_train_z, two_digit_valid_z = train_test_split(two_digit_train_X, two_digit_train_y, two_digit_train_z,
                                                            test_size=args.val_split, random_state=0)

        # print(len(two_digit_train_X), len(two_digit_val_X))
        # print(len(two_digit_train_y), len(two_digit_val_y))
        # print(len(two_digit_train_z), len(two_digit_val_z))

        two_digit_test_X = np.asarray(X_test_shuffled)
        two_digit_test_y = np.asarray(y_test_shuffled)
        two_digit_test_z = np.asarray(z_test_shuffled)

        # Sanity checks
        # check_both_channels_have_all_digits(two_digit_train_z, 'train')
        # check_both_channels_have_all_digits(two_digit_valid_z, 'valid')
        # check_both_channels_have_all_digits(two_digit_test_z, 'test')

        processed_train_dataset = TwoDigitMNISTDataset(two_digit_train_X, two_digit_train_y, two_digit_train_z,
                                                       transform, concat_dim)
        processed_valid_dataset = TwoDigitMNISTDataset(two_digit_valid_X, two_digit_valid_y, two_digit_valid_z,
                                                       transform, concat_dim)
        processed_test_dataset = TwoDigitMNISTDataset(two_digit_test_X, two_digit_test_y, two_digit_test_z,
                                                      transform, concat_dim)

        train_dataloader = DataLoader(processed_train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.dataset_workers,
                                      worker_init_fn=None, pin_memory=True)
        valid_dataloader = DataLoader(processed_valid_dataset, batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.dataset_workers,
                                      worker_init_fn=None, pin_memory=True)
        test_dataloader = DataLoader(processed_test_dataset, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.dataset_workers,
                                     worker_init_fn=None, pin_memory=True)

        ############################################################################################################
        # Sanity check the generated dataset, displaying the first 5 samples for both train and test datasets
        # self.plot_example(two_digit_train_X, two_digit_train_y, two_digit_train_z)
        # self.plot_example(two_digit_test_X, two_digit_test_y, two_digit_test_z)
        ############################################################################################################

        # create model and optimizer
        model = Net(
                    img2label_model=args.img2label_model,
                    writer=summary_writer.every(args.log_interval).verbose(args.verbose),
                    img2label_in=2, img2label_out=1,
                    use_nalm=args.use_nalm,
                    learn_labels2out=args.learn_labels2out,
                    operation=args.operation,
                    beta_nau=args.beta_nau,
                    nau_noise=args.nau_noise,
                    nmu_noise=args.nmu_noise,
                    noise_range=args.noise_range,
                    device=args.device,
                    image_height=42, image_width=42,
                    span_range_height=0.9, span_range_width=0.9,
                    grid_height_points=4, grid_width_points=4,
                    tps_unbounded_stn=args.tps_unbounded_stn
                ).to(args.device)
        model.apply(self.reset_weights)

        ###############################################################################################################
        # PRETRAINED MODEL VISUALISATION CODE
        # # quick and dirty loading of pretrained model (no setting random states/ opt)
        # # load pretrained model
        # load_filename = '60_f2_op-mul_nalmF_learnLF_s2'     # mul
        # # load_filename = '61_f2_op-mul_nalmT_learnLT_s2'   # snmu U[1,5]
        # # load_filename = '62_f2_op-mul_nalmT_learnLT_s2'   # nmu
        # # load_filename = '63_f2_op-mul_nalmF_learnLT_s2'   # fc
        # # load_filename = '69_f2_op-mul_nalmT_learnLT_s2'   # batch-snmu
        # checkpoint = torch.load(f'../save/{load_filename}.pth')
        # # args = checkpoint['args']
        # model.load_state_dict(checkpoint['model_state_dict'])
        # print('Pretrained model loaded')
        # model.to(args.device)
        # model.eval()
        #
        # # digit_confusion_matrix(model.img2label, train_dataloader, args.device, digit_idx=0)
        # # digit_confusion_matrix(model.img2label, train_dataloader, args.device, digit_idx=1)
        # digit_confusion_matrix(model.img2label, test_dataloader, args.device, digit_idx=None, round=True, old_model=False,
        #                        save_data={"model_name": load_filename,
        #                                   "save_dir": "../save/two_digit_mnist_plots/"})
        #
        # # model.img2label.register_forward_hook(fhook_channel_grid)
        # # self.eval_dataloader(model, args.device, test_dataloader, summary_writer, 'test')
        # # self.eval_dataloader(model, args.device, valid_dataloader, summary_writer, 'valid')
        #
        # import sys
        # sys.exit()
        ###############################################################################################################

        if fold_idx == 0:
            print(model)
        print(f"Param count (all): {sum(p.numel() for p in model.parameters())}")
        print(f"Param count (trainable): {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")

        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
        else:
            raise ValueError(f'{args.optimizer} is not a valid optimizer algorithm')

        if not args.no_scheduler:
            scheduler = lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=0.1)

        ###########################################################################################################
        # Train/test loop
        for epoch in range(args.max_epochs + 1):
            print(f"Epoch {epoch}")
            w_scale = max(0, min(1, (
                    (epoch - args.regualizer_scaling_start) /
                    (args.regualizer_scaling_end - args.regualizer_scaling_start)
            )))
            self.epoch_step(model, train_dataloader, args, optimizer, epoch, w_scale, summary_writer, test_dataloader, valid_dataloader)

            if not args.no_scheduler:
                current_learning_rate = float(scheduler.get_last_lr()[-1])
                print('lr:', current_learning_rate)
                scheduler.step()
        ###########################################################################################################

        if not use_dummy_writer:
            summary_writer._root.close()

        if not args.no_save:
            writer.save_model_checkpoint(summary_writer.name, epoch + 1, model, optimizer,
                                         {'torch': torch.get_rng_state(), 'numpy': np.random.get_state()},
                                         args=args)
            print(f'model saved for fold {fold_idx}')


if __name__ == '__main__':
    TwoDigitExperiment()

