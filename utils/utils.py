import numpy as np
import pandas as pd
import scipy
import copy
import torch.cuda
import torch.nn as nn
import skimage
import torchvision
from skimage import io
from torchvision import *
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict


# IMAGE DATASET CLASS TO LOAD IMAGES DIRECTLY FROM FOLDER READY FOR TRAINING

class ImageDataset(Dataset):
    """
    Custom dataset class compatible with Pytorch DataLoader
    """

    def __init__(self, folder, set_type, transform, output="FFR", tasks=None):
        """
        :param folder: folder containing the dataset. It should contain the folders train, validation and test
        :param set_type: "train", "validation" or "test"
        :param transform: pipeline of operations to be done on the image after it is received from the folder
        :param output: quantity for inference, should be among all_tasks defined at line 31
        :param tasks: tasks used in multitask learning, should be any combination of the strings at line 31
        """

        all_tasks = ["MI", "FFR", "A", "stenosis", "multitask"]

        assert output in all_tasks

        self.output = output
        self.csv = pd.read_csv(folder + str(set_type) + "/" + str(output) + "_labels.csv")
        self.transform = transform

        if tasks is not None:
            assert output == "multitask"
            self.tasks = tasks
            self.column_indices = []
            if "MI" in self.tasks:
                self.column_indices.append(0)
            if "FFR" in self.tasks:
                self.column_indices.append(1)
                self.column_indices.append(2)
            if "A" in self.tasks:
                self.column_indices.append(3)
            if "stenosis" in self.tasks:
                self.column_indices.append(4)

        self.normal_img_folder = folder + str(set_type) + "/normal/"
        self.rotated_img_folder = folder + str(set_type) + "/rotated/"
        self.labels = self.csv.loc[:, self.csv.columns != "Unnamed: 0"].to_numpy()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        image1 = skimage.io.imread(self.normal_img_folder + "noisy_snap_" + str(index) + '.png')
        image2 = skimage.io.imread(self.rotated_img_folder + "noisy_snap_" + str(index) + '.png')

        image1 = (image1 - np.mean(image1)) / np.std(image1)
        image2 = (image2 - np.mean(image2)) / np.std(image2)

        image = np.zeros((image1.shape[0], image1.shape[1], 2))
        image[:, :, 0] = image1
        image[:, :, 1] = image2

        image = self.transform(image)

        if self.output == "multitask":
            target = self.labels[index, self.column_indices]
        else:
            target = self.labels[index]

        return image, target


# LOAD DATASET ALREADY SPLIT IN TRAIN, VALIDATION AND TEST

def load_data(batch_size, transform, output="FFR", tasks=None, smaller_dataset=False):
    """
    Returns data loaders for train, validation and test set
    :param batch_size: batch size for training
    :param transform: pipeline of operations to be done on the image after it is received from the folder
    :param output: quantity for inference, should be among ["MI", "FFR", "A", "stenosis", "multitask"]
    :param tasks: tasks used in multitask learning, should be any combination of the possible strings for output
    :param smaller_dataset: True to use the reduced dataset to reproduce clinical scenarios
    :return
        - DataLoader for training, validation and test sets
    """
    if not smaller_dataset:
        folder = "data/"
    else:
        folder = "data/MI_transfer/"

    train_dataset = ImageDataset(folder, "train", transform, output, tasks)

    validation_dataset = ImageDataset(folder, "validation", transform, output, tasks)

    test_dataset = ImageDataset(folder, "test", transform, output, tasks)

    BATCH_SIZE = batch_size

    trainLoader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=2
    )

    validationLoader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=2
    )

    testLoader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=2
    )

    return trainLoader, validationLoader, testLoader


# DEFINE CUSTOM MODEL STARTING FROM RESNET18

def define_model(device=None, pretrained_model=None, freezed_layers=None, output="FFR",
                 tasks=None, common_layers=None, specific_layers=None, weighting_strategy="OL_AUX"):
    """
    Define ResNet18 custom model with modifications if needed
    :param device: device, "cpu" or "cuda:0"
    :param pretrained_model: pretrained model to use to perform transfer learning, should be the directory of the .pth file
        to be used
    :param freezed_layers: layers to be possibly freezed when doing transfer learning from a pretrained model
    :param output: quantity for inference, should be among ["MI", "FFR", "A", "stenosis", "multitask"]
    :param tasks: tasks used in multitask learning, should be any combination of the possible strings for output
    :param common_layers: layers to be shared in MTL between the tasks, should be any subset (respecting the order required
        by nn.Sequential) of ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc"]
    :param specific_layers: task-specific layers for MTL, should be the complementary of the layers chosen for common_layers
        in the set ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc"]
    :param weighting_strategy: weighting strategy for MTL, should be among the child classes of AbstractMTL
    :return
        - Model ready for training
        - Loss criterion
    """
    # check
    assert output in ["FFR", "MI", "A", "multitask", "stenosis"]

    # check
    if pretrained_model is not None and output != "MI":
        raise ValueError("Transfer learning from FFR should be done on MI")

    # ResNet18
    init_model = models.resnet18(weights=None)

    # make network compatible with the input
    init_model.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # reset fully connected layer
    init_model.fc = nn.Identity()

    # define new model
    if output == "FFR":
        output_layer = multioutput_layer()
        model = nn.Sequential(init_model, output_layer).to(device=device)
        loss_criterion = multiple_MSE(reduction="mean").to(device=device)
    elif output == "MI" or output == "A":
        output_layer = singleoutput_layer()
        model = nn.Sequential(init_model, output_layer).to(device=device)
        loss_criterion = nn.MSELoss(reduction="mean").to(device=device)

        if pretrained_model is not None:  # load weights and biases from .pth file

            FFR_weights = torch.load(pretrained_model)
            FFR_weights.pop("1.branch_a1.weight")
            FFR_weights.pop("1.branch_a1.bias")

            # load state dictionary from file .pth
            model.load_state_dict(FFR_weights)

            # freeze all layers given in input if any
            if freezed_layers is not None:
                for layer in freezed_layers:
                    to_freeze = eval("model[0]." + str(layer))
                    to_freeze.training = False

    elif output == "stenosis":
        output_layer = classification_layer()
        model = nn.Sequential(init_model, output_layer).to(device=device)
        loss_criterion = nn.CrossEntropyLoss().to(device=device)

    else:  # output is necessarily multitask
        model = MultiTask_Model(common_layers, specific_layers, tasks).to(device=device)
        loss_criterion = eval(weighting_strategy)(tasks).to(device=device)

    return model, loss_criterion


def design_multitask_network(common_layers, specific_layers):
    """
    Design multitask network for MI
    :param common_layers: layers to be shared in MTL between the tasks, should be any subset (respecting the order required
        by nn.Sequential) of ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc"]
    :param specific_layers: task-specific layers for MTL, should be the complementary of the layers chosen for common_layers
        in the set ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc"]
    """
    base_model = torchvision.models.resnet18(weights=None)

    # make network compatible with the input
    base_model.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    common_layers_dict = dict.fromkeys(list(common_layers), 0)
    for key in common_layers_dict.keys():
        common_layers_dict[key] = eval("base_model." + str(key))

    specific_layers_dict = dict.fromkeys(list(specific_layers), 0)
    for key in specific_layers_dict.keys():
        specific_layers_dict[key] = eval("base_model." + str(key))

    common_model = nn.Sequential(OrderedDict(common_layers_dict))
    specific_model = nn.Sequential(OrderedDict(specific_layers_dict))

    return common_model, specific_model


# CLASSES FOR THE SECOND MODEL ON TOP OF THE RESNET18 TO HANDLE DIFFERENT NUMBER OF OUTPUTS

class singleoutput_layer(nn.Module):
    def __init__(self,):
        super(singleoutput_layer, self).__init__()

        self.branch_a2 = nn.Linear(512, 1)

    def forward(self, x):
        out = torch.sigmoid(self.branch_a2(x))

        return out


class multioutput_layer(nn.Module):
    def __init__(self):
        super(multioutput_layer, self).__init__()

        self.branch_a1 = nn.Linear(512, 1)
        self.branch_a2 = nn.Linear(512, 1)

    def forward(self, x):

        out1 = torch.tensor(0.6) + torch.tensor(0.8) * torch.sigmoid(self.branch_a1(x))
        out2 = torch.tensor(0.6) + torch.tensor(0.8) * torch.sigmoid(self.branch_a2(x))

        return out1, out2


class classification_layer(nn.Module):
    def __init__(self):
        super(classification_layer, self).__init__()

        self.branch_a2 = nn.Linear(512, 4)

    def forward(self, x):
        out = self.branch_a2(x)

        return out


class MultiTask_Model(nn.Module):
    def __init__(self, common_layers, specific_layers, tasks):
        """
        Multitask model class
        :param common_layers: layers to be shared in MTL between the tasks, should be any subset (respecting the order required
            by nn.Sequential) of ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc"]
        :param specific_layers: task-specific layers for MTL, should be the complementary of the layers chosen for common_layers
            in the set ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc"]
        :param tasks: tasks used in multitask learning, should be any combination of ["MI", "FFR", "A", "stenosis"]
        """
        super(MultiTask_Model, self).__init__()

        # set common and specific layers
        self.common_layers = common_layers
        self.specific_layers = specific_layers

        # get common and specific architecture
        common_model, specific_model = design_multitask_network(common_layers, specific_layers)
        self.common_model = common_model

        self.tasks = tasks

        self.task_parameters = {}

        # adapt the network to the number of tasks with specific types of output layers
        if "MI" in self.tasks:

            self.MI_layers = copy.deepcopy(specific_model)
            self.MI_layers.fc = nn.Linear(512, 1)

            self.task_parameters["MI"] = list(self.MI_layers.parameters())

        if "FFR" in self.tasks:

            self.FFR_layers = copy.deepcopy(specific_model)
            self.FFR_layers.fc = multioutput_layer()

            self.task_parameters["FFR"] = list(self.FFR_layers.parameters())

        if "A" in self.tasks:

            self.A_layers = copy.deepcopy(specific_model)
            self.A_layers.fc = nn.Linear(512, 1)

            self.task_parameters["A"] = list(self.A_layers.parameters())

        if "stenosis" in self.tasks:

            self.stenosis_layers = copy.deepcopy(specific_model)
            self.stenosis_layers.fc = classification_layer()

            self.task_parameters["stenosis"] = list(self.stenosis_layers.parameters())

    def get_tasks(self):
        return self.tasks

    def forward(self, x):
        x = self.common_model(x)

        outputs = []

        if "MI" in self.tasks:
            MI_out = eval("self.MI_layers." + str(self.specific_layers[0]))(x)
            for layer in self.specific_layers[1:-1]:
                MI_out = eval("self.MI_layers." + str(layer))(MI_out)
            MI_out = torch.flatten(MI_out, 1)  # this is needed after the avgpool layer
            MI_out = self.MI_layers.fc(MI_out)
            MI_out = torch.sigmoid(MI_out)
            outputs.append(MI_out)

        if "FFR" in self.tasks:
            FFR_out = eval("self.FFR_layers." + str(self.specific_layers[0]))(x)
            for layer in self.specific_layers[1:-1]:
                FFR_out = eval("self.FFR_layers." + str(layer))(FFR_out)
            FFR_out = torch.flatten(FFR_out, 1)  # this is needed after the avgpool layer
            FFR_out = self.FFR_layers.fc(FFR_out)
            outputs.append(FFR_out[0])
            outputs.append(FFR_out[1])

        if "A" in self.tasks:
            A_out = eval("self.A_layers." + str(self.specific_layers[0]))(x)
            for layer in self.specific_layers[1:-1]:
                A_out = eval("self.A_layers." + str(layer))(A_out)
            A_out = torch.flatten(A_out, 1)  # this is needed after the avgpool layer
            A_out = self.A_layers.fc(A_out)
            A_out = 0.8 * torch.sigmoid(A_out)
            outputs.append(A_out)

        if "stenosis" in self.tasks:
            stenosis_out = eval("self.stenosis_layers." + str(self.specific_layers[0]))(x)
            for layer in self.specific_layers[1:-1]:
                stenosis_out = eval("self.stenosis_layers." + str(layer))(stenosis_out)
            stenosis_out = torch.flatten(stenosis_out, 1)  # this is needed after the avgpool layer
            stenosis_out = self.stenosis_layers.fc(stenosis_out)
            outputs.append(stenosis_out)

        return tuple(outputs)


class AbstractMTL(nn.Module):
    """
    Common class for MTL losses, contains common methods independently of the weighting strategy. These are
    eventually overwritten in the child classes in case of some particular discrepancy.
    """
    def __init__(self, tasks):
        super(AbstractMTL, self).__init__()
        
        self.tasks = tasks

        self.K = len(self.tasks)
        
        self.criterions = {"MI": nn.MSELoss(reduction="mean"),
                           "FFR": multiple_MSE(reduction="mean"),
                           "A": nn.MSELoss(reduction="mean"),
                           "stenosis": nn.CrossEntropyLoss()}

        self.weights = {}
        for key in self.tasks:
            self.weights[key] = torch.tensor(1/self.K).float().to(device="cuda:0")

        self.current_losses = {}
        for key in self.tasks:
            self.current_losses[key] = torch.tensor(0).float().to(device="cuda:0")

        self.regularization = torch.tensor(0).float().to(device="cuda:0")

    def get_weights(self):
        return self.weights

    def get_single_losses(self):
        return self.current_losses

    def get_regularization(self):
        return self.regularization

    def forward(self, output, target):
        losses = []
        count_outputs = 0
        if "MI" in self.tasks:
            MI_loss = self.criterions["MI"](output[count_outputs].float(), target[:, count_outputs].float().unsqueeze(1))
            losses.append(MI_loss)
            self.current_losses["MI"] = MI_loss.item()
            count_outputs += 1
        if "FFR" in self.tasks:
            FFR_loss = self.criterions["FFR"](output[count_outputs:(count_outputs + 2)], target[:, count_outputs:(count_outputs + 2)])
            count_outputs += 2
            self.current_losses["FFR"] = FFR_loss.item()
            losses.append(FFR_loss)
        if "A" in self.tasks:
            A_loss = self.criterions["A"](output[count_outputs].float(), target[:, count_outputs].float().unsqueeze(1))
            count_outputs += 1
            self.current_losses["A"] = A_loss.item()
            losses.append(A_loss)
        if "stenosis" in self.tasks:
            stenosis_loss = self.criterions["stenosis"](output[count_outputs], target[:, count_outputs].type(torch.LongTensor).to(device="cuda:0"))
            self.current_losses["stenosis"] = stenosis_loss.item()
            losses.append(stenosis_loss)

        return tuple(losses)


class UniformWeightedMSE(AbstractMTL):
    """
    All weights are kept constant at 1/K, where K is the number of tasks
    """
    def __init__(self, tasks):
        super(UniformWeightedMSE, self).__init__(tasks=tasks)


class AdaptiveMSE(AbstractMTL):
    """
    The weights are added to the parameters to be optimized with a regularization term in order to not make them vanish.
    For a reference, see https://arxiv.org/abs/1705.07115
    """
    def __init__(self, tasks):
        super(AdaptiveMSE, self).__init__(tasks=tasks)

        self.sigmas = {}
        for key in self.tasks:
            self.sigmas[key] = nn.Parameter(torch.ones(1), requires_grad=True).to(device="cuda:0")

        self.weights = {key: torch.exp(-self.sigmas[key]).float().to(device="cuda:0") for key in self.tasks}

        self.regularization = sum([torch.log(1 + self.sigmas[key] ** 2) for key in self.tasks])


class WeightedDynamicalAverage(AbstractMTL):
    """
    The weights are adjusted in order to give more importance to the tasks which have the slower rate of descent.
    For a reference, see https://arxiv.org/pdf/1803.10704.pdf
    """
    def __init__(self, tasks):
        super(WeightedDynamicalAverage, self).__init__(tasks=tasks)

        self.T = torch.tensor(2.0).float().to(device="cuda:0")  # Boltzmann temperature
        
        self.lambdas = {}
        for key in self.tasks:
            self.lambdas[key] = torch.tensor(1)

    def compute_MI_lambda(self, loss_1, loss_2):
        self.lambdas["MI"] = loss_1 / loss_2

    def compute_FFR_lambda(self, loss_1, loss_2):
        self.lambdas["FFR"] = loss_1 / loss_2

    def compute_A_lambda(self, loss_1, loss_2):
        self.lambdas["A"] = loss_1 / loss_2

    def compute_stenosis_lambda(self, loss_1, loss_2):
        self.lambdas["stenosis"] = loss_1 / loss_2

    def update_weights(self):
        normalization = sum([torch.exp(self.lambdas[key]/self.T) for key in self.tasks])
        for key in self.tasks:
            self.weights[key] = torch.exp(self.lambdas[key] / self.T) / normalization

    def get_lambdas(self):
        return self.lambdas


class OL_AUX(AbstractMTL):
    """
    Implements Adaptive Auxiliary Tasks as described in https://papers.nips.cc/paper/2019/file/0e900ad84f63618452210ab8baae0218-Paper.pdf
    """

    def __init__(self, tasks, beta=1e-1):
        super(OL_AUX, self).__init__(tasks=tasks)

        self.beta = beta  # learning rate for weights
        
        self.gradients = {}
        for key in self.tasks:
            self.gradients[key] = 0

        self.lambdas = {}
        for key in self.tasks:
            self.lambdas[key] = torch.tensor(1).float().to(device="cuda:0")

    def compute_MI_lambda(self, loss_1, loss_2):
        self.lambdas["MI"] = torch.tensor(loss_1 / loss_2).float().to(device="cuda:0")

    def compute_FFR_lambda(self, loss_1, loss_2):
        self.lambdas["FFR"] = torch.tensor(loss_1 / loss_2).float().to(device="cuda:0")

    def compute_A_lambda(self, loss_1, loss_2):
        self.lambdas["A"] = torch.tensor(loss_1 / loss_2).float().to(device="cuda:0")

    def compute_stenosis_lambda(self, loss_1, loss_2):
        self.lambdas["stenosis"] = torch.tensor(loss_1 / loss_2).float().to(device="cuda:0")

    def update_weights(self, gradients):
        for key in set(gradients.keys() - {"MI"}):
            self.gradients[key] = gradients[key]
            self.weights[key] -= self.beta * self.gradients[key]
        normalization = sum([torch.exp(self.weights[key]) for key in self.tasks])
        for key in self.tasks:
            self.weights[key] = torch.exp(self.weights[key]) / normalization

    def get_gradients(self):
        return self.gradients

    def get_lambdas(self):
        return self.lambdas


def concat_generators(*args):
    for gen in args:
        yield from gen


def compute_weights_gradients(model, weight, loss, key, weighting_strategy):
    """
    Computes gradients of both common and task specific layers in MTL for all tasks, then sets them for the optimizer in the task specific layers
    """
    if weighting_strategy == "FWS":
        weight = torch.tensor(1.0).float().to(device="cuda:0")
    if weighting_strategy != "OL_AUX":
        gradients = torch.autograd.grad(loss, list(model.common_model.parameters()), retain_graph=True), torch.autograd.grad(loss, list(model.task_parameters[key]), retain_graph=True)
    else:
        gradients = torch.autograd.grad(torch.log(loss), list(model.common_model.parameters()), retain_graph=True), \
                    torch.autograd.grad(torch.log(loss), list(model.task_parameters[key]), retain_graph=True)
    
    for idx, p in enumerate(model.task_parameters[key]):
        if p.requires_grad:
            p.grad = gradients[1][idx]
    
    return gradients


def set_common_gradients(model, weights, gradients, weighting_strategy, cosine_similarities):
    """
    Set common gradients properly weighted in case of FWS
    """
    if weighting_strategy == "FWS":
        K = len(model.get_tasks())
        M = np.zeros((K, K))

        for i, key_row in enumerate(gradients.keys()):
            for j, key_column in enumerate(gradients.keys()):
                M[i][j] = sum([torch.dot(torch.flatten(gradients[key_row][0][item]), torch.flatten(gradients[key_column][0][item])) for item in range(len(gradients[key_row][0]))])
        alpha = Franke_Wolfe_Solver(model, M, 10)
    else:
        alpha = weights
    
    for key in gradients.keys():
        if weighting_strategy in ["UniformWeightedMSE", "WeightedDynamicalAverage"]:
            for idx, p in enumerate(model.common_model.parameters()):
                if p.requires_grad:
                    if p.grad is None:
                        p.grad = alpha[key] * gradients[key][0][idx]
                    else:
                        p.grad += alpha[key] * gradients[key][0][idx]
        elif weighting_strategy == "OL_AUX":
            cosine_similarity = sum([torch.dot(torch.flatten(gradients["MI"][0][item]), torch.flatten(gradients[key][0][item])) for item in range(len(gradients["MI"][0]))])
            norm = sum([torch.linalg.norm(torch.flatten(gradients["MI"][0][item])) * torch.linalg.norm(torch.flatten(gradients[key][0][item])) for item in range(len(gradients["MI"][0]))])
            cosine_similarities[key].append(cosine_similarity.item() / norm.item())
            for idx, p in enumerate(model.common_model.parameters()):
                if p.requires_grad:
                    if p.grad is None:
                        p.grad = alpha[key] * gradients[key][0][idx]
                    else:
                        p.grad += alpha[key] * gradients[key][0][idx]


def compute_delta_w_i(prev_gradients, gradients_main_loss, gradients_aux_losses, lr, momentum=0.0):
    """
    Computes Delta w_i for all auxiliary tasks, mainly used for the Adaptive Auxiliary Tasks algorithm
    """
    gradients = dict.fromkeys(gradients_aux_losses, 0)
    for key in gradients_aux_losses.keys():
        gradients[key] = momentum * prev_gradients[key] - lr * gradients_aux_losses[key]
    return gradients


def Franke_Wolfe_Solver(model, M, num_iter):
    """
    Solves the minimum norm problem, returns the coefficients to be used for the update of shared parameters.
    For a reference, see https://arxiv.org/pdf/1810.04650.pdf
    """
    K = M.shape[0]
    alpha = 1/K * np.ones(K)
    for itr in range(num_iter):
        t_hat_versor = np.array([0 if i != np.argmin([M @ alpha]) else 1 for i in range(K)])
        gamma_hat = line_search(M, alpha, t_hat_versor)
        alpha = (1 - gamma_hat) * alpha + gamma_hat * t_hat_versor
    alpha = {key: alpha[ind] for ind, key in enumerate(model.get_tasks())}
    return alpha


def line_search(M, alpha, t_hat_versor):
    """
    Solves for the optimal learning rate to be used in the Franke Wolfe multi-objective algorithm above.
    For a reference, see Algorithm 1 in https://arxiv.org/pdf/1810.04650.pdf
    """
    theta = scipy.linalg.sqrtm(M) @ alpha
    theta_bar = scipy.linalg.sqrtm(M) @ t_hat_versor
    if np.dot(theta, theta_bar) >= np.dot(theta, theta):
        gamma = 0
    elif np.dot(theta, theta_bar) >= np.dot(theta_bar, theta_bar):
        gamma = 1
    else:
        gamma = np.dot(theta_bar - theta, theta_bar) / np.dot(theta - theta_bar, theta - theta_bar)
    return gamma


class multiple_MSE(nn.Module):
    """
    Simple sum of the single component losses for FFR inference on the bifurcation geometry
    """
    def __init__(self, reduction="mean"):
        super(multiple_MSE, self).__init__()
        self.reduction = reduction

    def forward(self, output, target):
        criterion = nn.MSELoss(reduction=self.reduction)
        loss = criterion(output[0].float(), target[:, 0].float().unsqueeze(1)) + \
            criterion(output[1].float(), target[:, 1].float().unsqueeze(1))
        return loss


# CUSTOM CLASSES TO COMPUTE AVERAGES OF LOSSES AND ACCURACIES FOR STENOSIS POSITION PREDICTIONS

def categorical_accuracy(y_true, output, topk=1):
    """
    Computes the precision@k for the specified values of k
    :param y_true: target
    :param output: output of the current model
    :param topk: topk percentage for the accuracy
    :return:
        - the topk accuracy computed by comparing output and y_true
    """
    prediction = output.topk(topk, dim=1, largest=True, sorted=False).indices
    n_labels = float(len(y_true))
    return prediction.eq(y_true).sum().item() / n_labels


class RunningAverage(object):
    """Tracks the running average of n numbers"""

    def __init__(self, n):
        self.n = n
        self.reset()

    def reset(self):
        self.sum = 0
        self.avg = 0
        self.entries = []

    def result(self):
        return self.avg

    def get_count(self):
        return len(self.entries)

    def is_complete(self):
        return len(self.entries) == self.n

    def __call__(self, val):
        if len(self.entries) == self.n:
            l = self.entries.pop(0)
            self.sum -= l
        self.entries.append(val)
        self.sum += val
        self.avg = self.sum / len(self.entries)

    def __str__(self):
        return str(self.avg)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0
        self.avg = 0

    def result(self):
        return self.avg

    def __call__(self, val, n=1):
        """val is an average over n samples. To compute the overall average, add val*n to sum and increase count by n"""
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return str(self.avg)
