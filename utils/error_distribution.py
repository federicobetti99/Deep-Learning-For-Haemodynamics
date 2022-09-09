from barbar import Bar
from utils import *
from preprocessing_helper import *
import matplotlib.pyplot as plt
from torchvision.transforms import *
import sys
import seaborn as sns

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
infer_quantity = "MI"
batch_size = 150
common_layers = ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3"]
specific_layers = ["layer4", "avgpool", "fc"]
weighting_strategy = "OL_AUX"
alpha = 0.05

transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomRotation(degrees=(-45, 45))])
trainLoader, validationLoader, testLoader = load_data(batch_size, transform, infer_quantity)

model, loss_criterion = define_model(device=device, output=infer_quantity, common_layers=common_layers, specific_layers=specific_layers, weighting_strategy="UniformWeightedMSE")
if infer_quantity == "multitask":
    model.load_state_dict(torch.load("models/" + str(infer_quantity) + "/" + str(weighting_strategy) + "/trained_model_weights.pth"))
else:
    model.load_state_dict(torch.load("models/" + str(infer_quantity) + "/trained_model_weights.pth"))

# initialize necessary metrics objects
train_loss = AverageMeter()
validation_loss = AverageMeter()
test_loss = AverageMeter()


# function to reset metrics
def reset_metrics():
    train_loss.reset()
    test_loss.reset()
    validation_loss.reset()


if infer_quantity == "MI" or infer_quantity == "A":
    train_losses = []
    validation_losses = []
    train_targets = []
    validation_targets = []
elif infer_quantity == "FFR":
    FFR_2_train_losses = []
    FFR_2_validation_losses = []
    FFR_3_train_losses = []
    FFR_3_validation_losses = []
    FFR_2_train_targets = []
    FFR_2_validation_targets = []
    FFR_3_train_targets = []
    FFR_3_validation_targets = []
elif infer_quantity == "multitask":
    MI_train_losses = []
    MI_validation_losses = []
    MI_train_targets = []
    MI_validation_targets = []
    FFR_2_train_losses = []
    FFR_2_validation_losses = []
    FFR_3_train_losses = []
    FFR_3_validation_losses = []
    FFR_2_train_targets = []
    FFR_2_validation_targets = []
    FFR_3_train_targets = []
    FFR_3_validation_targets = []
    A_train_losses = []
    A_validation_losses = []
    A_train_targets = []
    A_validation_targets = []


def evaluate_model(data="train", infer=infer_quantity):
    if data == "train":
        loader = trainLoader
        mean_loss = train_loss
        if infer == "MI" or infer == "A":
            losses = train_losses
            targets = train_targets
        if infer == "FFR":
            FFR_2_losses = FFR_2_train_losses
            FFR_3_losses = FFR_3_train_losses
            FFR_2_targets = FFR_2_train_targets
            FFR_3_targets = FFR_3_train_targets
        if infer == "multitask":
            MI_losses = MI_train_losses
            MI_targets = MI_train_targets
            FFR_2_losses = FFR_2_train_losses
            FFR_3_losses = FFR_3_train_losses
            FFR_2_targets = FFR_2_train_targets
            FFR_3_targets = FFR_3_train_targets
            A_losses = A_train_losses
            A_targets = A_train_targets
    elif data == "test":
        loader = testLoader
        mean_loss = test_loss
    elif data == "validation":
        loader = validationLoader
        mean_loss = validation_loss
        if infer == "MI" or infer == "A":
            losses = validation_losses
            targets = validation_targets
        if infer == "FFR":
            FFR_2_losses = FFR_2_validation_losses
            FFR_3_losses = FFR_3_validation_losses
            FFR_2_targets = FFR_2_validation_targets
            FFR_3_targets = FFR_3_validation_targets
        if infer == "multitask":
            MI_losses = MI_validation_losses
            MI_targets = MI_validation_targets
            FFR_2_losses = FFR_2_validation_losses
            FFR_3_losses = FFR_3_validation_losses
            FFR_2_targets = FFR_2_validation_targets
            FFR_3_targets = FFR_3_validation_targets
            A_losses = A_validation_losses
            A_targets = A_validation_targets

    sys.stdout.write(f"Evaluation of {data} data:\n")

    # iteration over the dataset
    for x_input, y_target in Bar(loader):
        x_input, y_target = x_input.to(device=device), y_target.to(device=device)  # move to GPU
        output = model.eval()(x_input.float())
        if infer == "MI" or infer == "A":
            loss = loss_criterion(output.float(), y_target.float())  # compute the loss
            losses.append(torch.abs(output - y_target.unsqueeze(1)).cpu().detach().squeeze(1))
            targets.append(y_target.cpu().detach())
        elif infer == "FFR":
            loss = loss_criterion(output, y_target)
            FFR_2_losses.append(torch.abs(output[0] - y_target[:, 0].unsqueeze(1)).cpu().detach().squeeze(1))
            FFR_2_targets.append(y_target[:, 0].cpu().detach())
            FFR_3_losses.append(torch.abs(output[1] - y_target[:, 1].unsqueeze(1)).cpu().detach().squeeze(1))
            FFR_3_targets.append(y_target[:, 1].cpu().detach())
        elif infer == "multitask":
            weights = loss_criterion.get_weights()
            MI_loss, FFR_loss, A_loss = loss_criterion(output, y_target)
            loss = weights["MI"] * MI_loss + weights["FFR"] * FFR_loss + weights["A"] * A_loss + loss_criterion.get_regularization()
            MI_losses.append(torch.abs(output[0]-y_target[:, 0].unsqueeze(1)).cpu().detach().squeeze(1))
            MI_targets.append(y_target[:, 0].cpu().detach())
            FFR_2_losses.append(torch.abs(output[1] - y_target[:, 1].unsqueeze(1)).cpu().detach().squeeze(1))
            FFR_2_targets.append(y_target[:, 1].cpu().detach())
            FFR_3_losses.append(torch.abs(output[2] - y_target[:, 2].unsqueeze(1)).cpu().detach().squeeze(1))
            FFR_3_targets.append(y_target[:, 2].cpu().detach())
            A_losses.append(torch.abs(output[3] - y_target[:, 3].unsqueeze(1)).cpu().detach().squeeze(1))
            A_targets.append(y_target[:, 3].cpu().detach())

        # update metrics
        mean_loss(loss.item(), len(y_target))


evaluate_model("train")
evaluate_model("validation")

if infer_quantity == "MI" or infer_quantity == "A":
    train_targets = torch.hstack(train_targets).numpy()
    train_losses = torch.hstack(train_losses).numpy()
    validation_targets = torch.hstack(validation_targets).numpy()
    validation_losses = torch.hstack(validation_losses).numpy()

if infer_quantity == "FFR":
    FFR_2_train_losses = torch.hstack(FFR_2_train_losses).numpy() 
    FFR_2_validation_losses = torch.hstack(FFR_2_validation_losses).numpy()
    FFR_3_train_losses = torch.hstack(FFR_3_train_losses).numpy()
    FFR_3_validation_losses = torch.hstack(FFR_3_validation_losses).numpy()
    FFR_2_train_targets = torch.hstack(FFR_2_train_targets)
    FFR_2_validation_targets = torch.hstack(FFR_2_validation_targets).numpy()
    FFR_3_train_targets = torch.hstack(FFR_3_train_targets).numpy()
    FFR_3_validation_targets = torch.hstack(FFR_3_validation_targets).numpy()

if infer_quantity == "multitask":
    MI_train_losses = torch.hstack(MI_train_losses).numpy()
    MI_validation_losses = torch.hstack(MI_validation_losses).numpy()
    MI_train_targets = torch.hstack(MI_train_targets).numpy()
    MI_validation_targets = torch.hstack(MI_validation_targets).numpy()
    FFR_2_train_losses = torch.hstack(FFR_2_train_losses).numpy()
    FFR_2_validation_losses = torch.hstack(FFR_2_validation_losses).numpy()
    FFR_3_train_losses = torch.hstack(FFR_3_train_losses).numpy()
    FFR_3_validation_losses = torch.hstack(FFR_3_validation_losses).numpy()
    FFR_2_train_targets = torch.hstack(FFR_2_train_targets).numpy()
    FFR_2_validation_targets = torch.hstack(FFR_2_validation_targets).numpy()
    FFR_3_train_targets = torch.hstack(FFR_3_train_targets).numpy()
    FFR_3_validation_targets = torch.hstack(FFR_3_validation_targets).numpy()
    A_train_losses = torch.hstack(A_train_losses).numpy()
    A_validation_losses = torch.hstack(A_validation_losses).numpy()
    A_train_targets = torch.hstack(A_train_targets).numpy()
    A_validation_targets = torch.hstack(A_validation_targets).numpy()

if infer_quantity == "MI":
    ranges_of_values = [np.linspace(0.0, 1.0, 11)]
    intervals_list = [['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']]
elif infer_quantity == "A":
    ranges_of_values = [np.linspace(0.1, 0.6, 6)]
    intervals_list = [['0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6']]
elif infer_quantity == "FFR":
    ranges_of_values = [np.linspace(0.95, 1.05, 3), np.linspace(0.7, 1.4, 8)]
    intervals_list = [['0.95-1.0', '1.0-1.05'],
                 ['0.7-0.8', '0.8-0.9', '0.9-1.0', '1.0-1.1', '1.1-1.2', '1.2-1.3', '1.3-1.4']]
elif infer_quantity == "multitask":
    ranges_of_values = [np.linspace(0.0, 1.0, 11), np.linspace(0.95, 1.05, 3), np.linspace(0.7, 1.4, 8), np.linspace(0.1, 0.6, 6)]
    intervals_list = [['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0'],
                      ['0.95-1.0', '1.0-1.05'],
                      ['0.7-0.8', '0.8-0.9', '0.9-1.0', '1.0-1.1', '1.1-1.2', '1.2-1.3', '1.3-1.4'],
                      ['0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6']]

if infer_quantity == "MI" or infer_quantity == "A":
    train_targets = [train_targets]
    train_losses = [train_losses]
    validation_targets = [validation_targets]
    validation_losses = [validation_losses]
    mae_ranges = [[]]
    std_ranges = [[]]
elif infer_quantity == "FFR":
    train_targets = [FFR_2_train_targets, FFR_3_train_targets]
    train_losses = [FFR_2_train_losses, FFR_3_train_losses]
    validation_targets = [FFR_2_validation_targets, FFR_3_validation_targets]
    validation_losses = [FFR_2_validation_losses, FFR_3_validation_losses]
    mae_ranges = [[], []]
    std_ranges = [[], []]
elif infer_quantity == "multitask":
    train_targets = [MI_train_targets, FFR_2_train_targets, FFR_3_train_targets, A_train_targets]
    train_losses = [MI_train_losses, FFR_2_train_losses, FFR_3_train_losses, A_train_losses]
    validation_targets = [MI_validation_targets, FFR_2_validation_targets, FFR_3_validation_targets, A_validation_targets]
    validation_losses = [MI_validation_losses, FFR_2_validation_losses, FFR_3_validation_losses, A_validation_losses]
    mae_ranges = [[], [], [], []]
    std_ranges = [[], [], [], []]

if infer_quantity == "MI":
    keys = ["MI"]
if infer_quantity == "A":
    keys = ["A"]
if infer_quantity == "FFR":
    keys = ["FFR_2", "FFR_3"]
if infer_quantity == "multitask":
    keys = ["MI", "FFR_2", "FFR_3", "A"]

for ind, train_target in enumerate(train_targets):
    for j in range(len(ranges_of_values[ind])-1):
        range_indices = np.where(np.logical_and(train_targets[ind] >= ranges_of_values[ind][j], train_targets[ind] <= ranges_of_values[ind][j+1]))[0]
        if len(range_indices) > 0:
            mae_ranges[ind].append(np.mean(train_losses[ind][range_indices]))
            std_ranges[ind].append(1/np.sqrt(alpha) * np.sqrt(np.var(train_losses[ind][range_indices], ddof=1)/len(range_indices)))
        else:
            mae_ranges[ind].append(0)
            std_ranges[ind].append(0)

sns.set(rc={'figure.figsize': (11.7, 8.27)})
for ind, key in enumerate(keys):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(intervals_list[ind])), np.array(mae_ranges[ind]), yerr=np.array(std_ranges[ind]), align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_xticks(np.arange(len(intervals_list[ind])))
    ax.set_xticklabels(intervals_list[ind])
    ax.set_yticks(np.linspace(0.0, 0.05, 11))
    ax.set_title(f"Error distribution")
    ax.set_ylabel("Mean absolute error")
    ax.set_xlabel("Intervals")
    if infer_quantity == "multitask":
        plt.savefig("figures/error_distributions/" + str(infer_quantity) + "/" + str(weighting_strategy) + "_" + str(key) + "_train_error_distribution.png")
    else:
        plt.savefig("figures/error_distributions/" + str(infer_quantity) + "/" + str(key) + "_train_error_distribution.png")


if infer_quantity == "MI" or infer_quantity == "A":
    train_targets = train_targets
    train_losses = train_losses
    validation_targets = validation_targets
    validation_losses = validation_losses
    mae_ranges = [[]]
    std_ranges = [[]]
elif infer_quantity == "FFR":
    train_targets = [FFR_2_train_targets, FFR_3_train_targets]
    train_losses = [FFR_2_train_losses, FFR_3_train_losses]
    validation_targets = [FFR_2_validation_targets, FFR_3_validation_targets]
    validation_losses = [FFR_2_validation_losses, FFR_3_validation_losses]
    mae_ranges = [[], []]
    std_ranges = [[], []]
elif infer_quantity == "multitask":
    train_targets = [MI_train_targets, FFR_2_train_targets, FFR_3_train_targets, A_train_targets]
    train_losses = [MI_train_losses, FFR_2_train_losses, FFR_3_train_losses, A_train_losses]
    validation_targets = [MI_validation_targets, FFR_2_validation_targets, FFR_3_validation_targets, A_validation_targets]
    validation_losses = [MI_validation_losses, FFR_2_validation_losses, FFR_3_validation_losses, A_validation_losses]
    mae_ranges = [[], [], [], []]
    std_ranges = [[], [], [], []]

for ind, validation_target in enumerate(validation_targets):
    for j in range(len(ranges_of_values[ind])-1):
        range_indices = np.where(np.logical_and(validation_targets[ind] >= ranges_of_values[ind][j], validation_targets[ind] <= ranges_of_values[ind][j+1]))[0]
        if len(range_indices) > 0:
            mae_ranges[ind].append(np.mean(validation_losses[ind][range_indices]))
            std_ranges[ind].append(1/np.sqrt(alpha) * np.sqrt(np.var(train_losses[ind][range_indices], ddof=1)/len(range_indices)))
        else:
            mae_ranges[ind].append(0)
            std_ranges[ind].append(0)

sns.set(rc={'figure.figsize': (11.7, 8.27)})
for ind, key in enumerate(keys):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(intervals_list[ind])), np.array(mae_ranges[ind]), yerr=np.array(std_ranges[ind]), align="center", alpha=0.5, ecolor='black', capsize=10)
    ax.set_xticks(np.arange(len(intervals_list[ind])))
    ax.set_xticklabels(intervals_list[ind])
    ax.set_yticks(np.linspace(0.0, 0.05, 11))
    ax.set_title("Error distribution")
    ax.set_ylabel("Mean absolute error")
    ax.set_xlabel("Intervals")           
    if infer_quantity == "multitask":
        plt.savefig("figures/error_distributions/" + str(infer_quantity) + "/" + str(weighting_strategy) + "_" + str(key) + "_validation_error_distribution.png")
    else:
        plt.savefig("figures/error_distributions/" + str(infer_quantity) + "/" + str(key) + "_validation_error_distribution.png")
