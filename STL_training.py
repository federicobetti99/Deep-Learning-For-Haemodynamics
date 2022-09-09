from barbar import Bar
from utils.utils import *
import matplotlib.pyplot as plt
from torchvision.transforms import *
import pickle
import time
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# @title Train-test split

# batch size
batch_size = 128  # @param{type: "integer"}

# choice of regression and loss function
infer_quantity = "stenosis"
lower_scale = False  # @param {type: "boolean"}
lr = 5e-4  # @param {type: "number"}

# data augmentation and rotations
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.RandomRotation(degrees=(-45, 45)),
                                transforms.Normalize((0,), (1,))])

# define model
freezed_layers = ["conv1", "bn1", "relu", "maxpool", "layer1"]
smaller_dataset = False
add_dropout = False
inject_FFR_bias = False
simplify_model = False

# data loaders
trainLoader, validationLoader, testLoader = load_data(batch_size, transform, infer_quantity, None, smaller_dataset)

# model
model, loss_criterion = define_model(device=device, pretrained_model=None, freezed_layers=None, output=infer_quantity)

# set optimizer
if lower_scale:
    optimizer = torch.optim.Adam(params={
        {"params": model[0].conv1.parameters(), "lr": 1e-6},
        {"params": model[0].bn1.parameters(), "lr": 5e-5},
        {"params": model[0].relu.parameters(), "lr": 5e-5},
        {"params": model[0].maxpool.parameters(), "lr": 2e-5},
        {"params": model[0].layer1.parameters(), "lr": 1e-5},
        {"params": model[0].layer2.parameters(), "lr": lr},
        {"params": model[0].layer3.parameters(), "lr": lr},
        {"params": model[0].layer4.parameters(), "lr": lr},
        {"params": model[1].parameters(), "lr": lr},
        }, lr=lr)
else:
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

nepochs = 50  # @param{type:"integer"}
train_losses = []
validation_losses = []

# initialize necessary metrics objects
train_loss = AverageMeter()
validation_loss = AverageMeter()
test_loss = AverageMeter()
if infer_quantity == "stenosis":
    train_accuracy = AverageMeter()
    validation_accuracy = AverageMeter()
    test_accuracy = AverageMeter()


# function to reset metrics
def reset_metrics():
    train_loss.reset()
    validation_loss.reset()
    test_loss.reset()
    if infer_quantity == "stenosis":
        train_accuracy.reset()
        validation_accuracy.reset()
        test_accuracy.reset()


def evaluate_model(data="train", infer=infer_quantity):
    if data == "train":
        loader = trainLoader
        mean_loss = train_loss
        if infer == "stenosis":
            mean_accuracy = train_accuracy
    elif data == "test":
        loader = testLoader
        mean_loss = test_loss
        if infer == "stenosis":
            mean_accuracy = test_accuracy
    elif data == "validation":
        loader = validationLoader
        mean_loss = validation_loss
        if infer == "stenosis":
            mean_accuracy = validation_accuracy

    sys.stdout.write(f"Evaluation of {data} data:\n")

    # iteration over the dataset
    for x_input, y_target in Bar(loader):
        x_input, y_target = x_input.to(device=device), y_target.to(device=device)  # move to GPU
        output = model.eval()(x_input.float())
        if infer == "MI" or infer == "A":
            loss = loss_criterion(output.float(), y_target.float())  # compute the loss
        elif infer == "FFR":
            loss = loss_criterion(output, y_target)
        elif infer == "stenosis":
            loss = loss_criterion(output, y_target.squeeze(1))
        # update metrics
        mean_loss(loss.item(), len(y_target))
        if infer == "stenosis":
            mean_accuracy(categorical_accuracy(y_true=y_target, output=output), len(y_target))


for epoch in torch.arange(0, nepochs + 1):

    start = time.time()  # start to time
    reset_metrics()  # reset the metrics from the previous epoch

    sys.stdout.write(f"\n\nEpoch {epoch}/{nepochs}\n")

    if epoch == 0:
        evaluate_model(data="train")  # first pass through the network
    else:
        sys.stdout.write(f"Training:\n")

        for x_input, y_target in Bar(trainLoader):
            x_input, y_target = x_input.to(device=device), y_target.to(device=device)  # move to GPU
            optimizer.zero_grad()  # Zero the gradient buffers
            output = model.train()(x_input.float())  # compute the output
            if infer_quantity == "MI" or infer_quantity == "A":
                loss = loss_criterion(output.float(), y_target.float())  # compute the loss
            elif infer_quantity == "FFR":
                loss = loss_criterion(output, y_target)  # compute the loss
            elif infer_quantity == "stenosis":
                loss = loss_criterion(output, y_target.squeeze(1))  # compute the loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update all weights requiring grad

            train_loss(loss.item(), len(y_target))
            if infer_quantity == "stenosis":
                train_accuracy(categorical_accuracy(y_true=y_target, output=output), len(y_target))
        scheduler.step()

    # evaluate the model on the validation set and print statistics for the current epoch
    evaluate_model(data="validation")
    sys.stdout.write(f"\n Finished epoch {epoch}/{nepochs}: Train Loss {train_loss.result()} | Validation Loss {validation_loss.result()} \n")
    if infer_quantity == "stenosis":
        sys.stdout.write(f"\n Finished epoch {epoch}/{nepochs}: Train Accuracy {train_accuracy.result()} | Validation Accuracy {validation_accuracy.result()} \n")

    # collect training statistics of the current epoch
    train_losses.append(train_loss.result())
    validation_losses.append(validation_loss.result())

stats = {
    "train_losses": train_losses,
    "validation_losses": validation_losses
}

if not smaller_dataset:
    torch.save(model.state_dict(), "models/" + str(infer_quantity) + "/trained_model_weights.pth")
else:
    if not inject_FFR_bias:
        torch.save(model.state_dict(), "models/smaller_dataset/" + str(infer_quantity) + "/trained_model_weights.pth")
    else:
        torch.save(model.state_dict(), "models/smaller_dataset/" + str(infer_quantity) + "/biased_trained_model_weights.pth")

plt.semilogy(np.arange(nepochs + 1), stats["train_losses"], label="Train")
plt.semilogy(np.arange(nepochs + 1), stats["validation_losses"], label="Validation")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend(loc="best")
if not smaller_dataset:
    plt.savefig("figures/training_trends/" + str(infer_quantity) + "/training_trend.png")
else:
    if not inject_FFR_bias:
        plt.savefig("figures/smaller_dataset/training_trends/" + str(infer_quantity) + "/training_trend.png")
    else:
        plt.savefig("figures/smaller_dataset/training_trends/" + str(infer_quantity) + "/biased_training_trend.png")
