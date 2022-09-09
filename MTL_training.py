from barbar import Bar
from utils.utils import *
from torchvision.transforms import *
import time
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# @title Train-test split

# batch size
batch_size = 128  # @param{type: "integer"}

# choice of regression and loss function
infer_quantity = "multitask"
weighting_strategy = "OL_AUX"
lr = 5e-4  # @param {type: "number"}
FWS = False   # @param {type: "boolean"}

if FWS:
    assert weighting_strategy != "OL_AUX"

# data augmentation and rotations
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.RandomRotation(degrees=(-45, 45)),
                                transforms.Normalize((0,), (1,))])

# define model
common_layers = ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2"]
specific_layers = ["layer3", "layer4", "avgpool", "fc"]
tasks = ["MI", "FFR", "A", "stenosis"]
smaller_dataset = True
N = 1

# data loaders
trainLoader, validationLoader, testLoader = load_data(batch_size, transform, infer_quantity, tasks, smaller_dataset)

model, loss_criterion = define_model(device=device, output=infer_quantity, tasks=tasks,
                                     common_layers=common_layers, specific_layers=specific_layers,
                                     weighting_strategy=weighting_strategy)

evaluate_criterion = UniformWeightedMSE(tasks=tasks).to(device=device)

if weighting_strategy == "UniformWeightedMSE" and FWS:
    weighting_strategy = "FWS"

# set optimizer
if weighting_strategy == "AdaptiveMSE":
    optimizer = torch.optim.Adam(params=list(model.parameters()) + list(loss_criterion.parameters()), lr=lr)
else:
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

nepochs = 100  # @param{type:"integer"}

train_losses = {}
for key in model.get_tasks():
    train_losses[key] = []
train_losses["all"] = []

validation_losses = {}
for key in model.get_tasks():
    validation_losses[key] = []
validation_losses["all"] = []

test_losses = {}
for key in model.get_tasks():
    test_losses[key] = []
test_losses["all"] = []

weights = {}
for key in model.get_tasks():
    weights[key] = []

cosine_similarities = {}
std_cosines = {}
for key in model.get_tasks():
    cosine_similarities[key] = []
    cosine_similarities[key].append(0)
    std_cosines[key] = []
    std_cosines[key].append(0)

# initialize necessary metrics objects
train_meters = {}
for key in model.get_tasks():
    train_meters[key] = AverageMeter()
train_meters["all"] = AverageMeter()

validation_meters = {}
for key in model.get_tasks():
    validation_meters[key] = AverageMeter()
validation_meters["all"] = AverageMeter()

test_meters = {}
for key in model.get_tasks():
    test_meters[key] = AverageMeter()
test_meters["all"] = AverageMeter()


# function to reset metrics
def reset_metrics():
    for key in train_meters.keys():
        train_meters[key].reset()
    for key in validation_meters.keys():
        validation_meters[key].reset()
   

def evaluate_model(data="train"):
    mean_loss = {}
    if data == "train":
        loader = trainLoader
        for key in train_meters.keys():
            mean_loss[key] = train_meters[key]
    elif data == "test":
        loader = testLoader
        for key in test_meters.keys():
            mean_loss[key] = test_meters[key]
    elif data == "validation":
        loader = validationLoader
        for key in validation_meters.keys():
            mean_loss[key] = validation_meters[key]

    sys.stdout.write(f"Evaluation of {data} data:\n")

    # iteration over the dataset
    for x_input, y_target in Bar(loader):
        x_input, y_target = x_input.to(device=device), y_target.to(device=device)  # move to GPU
        output = model.eval()(x_input.float())
        current_weights = evaluate_criterion.get_weights()
        losses = evaluate_criterion(output, y_target)
        losses = {key: losses[ind] for ind, key in enumerate(model.get_tasks())}
        loss = sum([current_weights[key] * losses[key] for key in model.get_tasks()]) + evaluate_criterion.get_regularization()
        # update metrics
        mean_loss["all"](loss.item(), len(y_target))
        for key in model.get_tasks():
            mean_loss[key](losses[key].item(), len(y_target))


for epoch in torch.arange(0, nepochs + 1):

    start = time.time()  # start to time
    reset_metrics()  # reset the metrics from the previous epoch

    sys.stdout.write(f"\n\nEpoch {epoch}/{nepochs}\n")

    if epoch == 0:
        evaluate_model(data="train")  # first pass through the network
    else:
        sys.stdout.write(f"Training:\n")
        
        count = 0
        
        current_cosines = {}
        for key in model.get_tasks():
            current_cosines[key] = []

        for x_input, y_target in Bar(trainLoader):
            x_input, y_target = x_input.to(device=device), y_target.to(device=device)  # move to GPU
            optimizer.zero_grad()  # Zero the gradient buffers
            output = model.train()(x_input.float())  # compute the output
            
            if epoch >= 1 and weighting_strategy == "WeightedDynamicalAverage":
                loss_criterion.update_weights()
            
            current_weights = loss_criterion.get_weights()
            losses = loss_criterion(output, y_target)  # get single losses
            losses = {key: losses[ind] for ind, key in enumerate(model.get_tasks())}
            
            count += 1

            gradients = {}

            for key in model.get_tasks():
                gradients[key] = compute_weights_gradients(model, current_weights[key], losses[key], key, weighting_strategy)
            
            set_common_gradients(model, current_weights, gradients, weighting_strategy, current_cosines)

            optimizer.step()  # update weights with cumulative gradients of chosen tasks
            
            loss = sum([current_weights[key] * losses[key] for key in model.get_tasks()]) + loss_criterion.get_regularization()
            train_meters["all"](loss.item(), len(y_target))

            for key in model.get_tasks():
                train_meters[key](losses[key].item(), len(y_target))

            if weighting_strategy == "OL_AUX" and count % N == 0:
                loss_criterion.update_weights(compute_delta_w_i(loss_criterion.get_gradients(),
                                                         np.sum(current_cosines["MI"]), {k: np.sum(cosine_similarities[k]) for k in set(list(current_cosines.keys())) - {"MI"}},
                                                         lr, momentum=0.0))  # update weights

        scheduler.step()

        for key in model.get_tasks():
                cosine_similarities[key].append(np.mean(current_cosines[key]))
                std_cosines[key].append(np.std(current_cosines[key], ddof=1) / np.sqrt(len(current_cosines[key])))


    # evaluate the model on the validation set and print statistics for the current epoch
    evaluate_model(data="validation")

    train_loss = train_meters["all"].result()
    validation_loss = validation_meters["all"].result()
    train_results_string = f"\n Finished epoch {epoch}/{nepochs}: Train Loss {train_loss} | Validation Loss {validation_loss} \n"
    sys.stdout.write(train_results_string)
    
    specific_tasks_train_string = f"\n Finished epoch {epoch}/{nepochs}: "
    for key in model.get_tasks():
        train_loss = train_meters[key].result()
        specific_tasks_train_string += f" {key} Train Loss {train_loss} |"
    specific_tasks_train_string += "\n"
    sys.stdout.write(specific_tasks_train_string)
    
    specific_tasks_validation_string = f"\n Finished epoch {epoch}/{nepochs}: "
    for key in model.get_tasks():
        validation_loss = validation_meters[key].result()
        specific_tasks_validation_string += f" {key} Validation Loss {validation_loss} |"
    specific_tasks_validation_string += "\n"
    sys.stdout.write(specific_tasks_validation_string)
    
    specific_tasks_weights_string = f"\n Finished epoch {epoch}/{nepochs}: "
    for key in model.get_tasks():
        weight = loss_criterion.get_weights()[key].cpu()
        specific_tasks_weights_string += f" {key} weight {weight} | "
    specific_tasks_weights_string += "\n"
    sys.stdout.write(specific_tasks_weights_string)

    # collect training statistics of the current epoch
    for key in train_meters.keys():
        train_losses[key].append(train_meters[key].result())
    for key in validation_meters.keys():
        validation_losses[key].append(validation_meters[key].result())
    for key in loss_criterion.get_weights().keys():
        weights[key].append(loss_criterion.get_weights()[key].cpu().item())

    if epoch >= 1 and weighting_strategy == "WeightedDynamicalAverage":
        for key in model.get_tasks():
            eval("loss_criterion.compute_" + str(key) + "_lambda")(train_losses[key][-1], train_losses[key][-2])

stats = {
     "train_losses": train_losses,
     "validation_losses": validation_losses,
     "weights": weights
}

print(cosine_similarities)
print(std_cosines)

delim = "_"
tasks = list(map(str, tasks))
tasks = delim.join(tasks)

if weighting_strategy == "OL_AUX":
    weighting_strategy += "_"
    weighting_strategy += str(N)

# if not smaller_dataset:
#    torch.save(model.state_dict(), "models/" + str(infer_quantity) + "/" + str(tasks) + "/" + str(weighting_strategy) + "/trained_model_weights.pth")
# else:
#    torch.save(model.state_dict(), "models/smaller_dataset/" + str(infer_quantity) + "/" + str(tasks) + "/" + str(weighting_strategy) + "/trained_model_weights.pth")
