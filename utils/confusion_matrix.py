from sklearn.metrics import confusion_matrix
from barbar import Bar
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *
from preprocessing_helper import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

infer_quantity = "stenosis"
smaller_dataset = False
batch_size = 150

transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomRotation(degrees=(-45, 45))])

trainLoader, validationLoader, testLoader = load_data(batch_size, transform, infer_quantity, smaller_dataset)

model, loss_criterion = define_model(device=device, output=infer_quantity)

model.load_state_dict(torch.load("models/" + str(infer_quantity) + "/trained_model_weights.pth"))


def create_confusion_matrix(data="train"):
    if data == "train":
        loader = trainLoader
    elif data == "validation":
        loader = validationLoader
    
    y_pred = []
    y_true = []
    topk = 1

    for x_input, y_target in Bar(loader):
        x_input, y_target = x_input.to(device="cuda:0"), y_target.to(device="cuda:0")
        output = model.eval()(x_input.float())  # Feed Network
        output = output.topk(topk, dim=1, largest=True, sorted=False).indices.cpu().numpy()
        y_pred.extend(output)  # Save Prediction
        y_true.extend(y_target.cpu().detach().numpy())  # Save Truth

    # constant for classes
    classes = ('Inlet', 'Bifurcation', 'Big branch', 'Small branch')

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred, normalize="true")
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes], columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sns.heatmap(df_cm, cbar=False, annot=True, cmap="rocket_r",  vmin=0, vmax=2)
    plt.savefig("figures/error_distributions/stenosis/" + str(data) + "_confusion_matrix.png")


create_confusion_matrix("train")
create_confusion_matrix("validation")
