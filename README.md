# DNNForHemo

### Dataset generation and post-processing

This repository contains the implementation of deep learning experiments to predict fractional flow reserve and myocardial infraction values from snapshots of steady haemodynamical simulations. The [code](https://github.com/lucapegolotti/RedMA/tree/18022022_SemesterProject/redma) used for the generation of the dataset is available, while the [dataset](https://drive.switch.ch/index.php/apps/files/?dir=/correct_data&fileid=5083459784) is only accessible to the repository owners.


## Structure
* `utils` - Training and visualization utilities.
  * `utils.py` - Utilities for training and architectures definition.
  * `preprocessing_helper.py` - Utilities for preprocessing of the data.
  * `paraview_utils.py` - Utilities for snapshots taking in Paraview.
  * `error_distribution.py` - Utilities to plot distributions of errors with a pretrained model.
  * `MTL.py` - Utilities for custom loss functions for multitask learning.
* `labels` - Folder containing all labels .csv.
* `figures` - Training trends figures.
* `report.pdf` - Report pdf file.
*  `STL_training.py` - Executable to perform single task training.
*  `MTL_training.py` - Executable to perform multitask training.
* `requirements.txt` - Requirements text file.

## Installation
To clone the following repository, please run:
```
git clone --recursive https://github.com/federicobetti99/Deep-Learning-For-Haemodynamics.git
```

## Requirements
Requirements for the needed packages are available in `requirements.txt`. To install the needed packages, please run:
```
pip install -r requirements.txt
```

## Instructions and example of usage
This section is just meant to give the necessary instructions to perform training with the developed code.
For an exhaustive description of the code, please refer to the documentation of the code itself and
references thereby.

To perform training and reproduce the obtained result, one should download [here](https://drive.switch.ch/index.php/apps/files/?dir=/correct_data&fileid=5083459784) the dataset, assuming you have been given access.
For the correct working of the code, save the folder in the directory where the repository is cloned and make sure that 
it contains the folders `train`, `validation`, `test` and `MI_transfer`. The first three
folders correspond to the train-validation-test split of the whole dataset, while
the last one contains a random 20% of the original dataset used for transfer and multitask
learning experiments to simulate clinically meaningful scenarios in which MI labels are
way less available than FFR ones.

To perform training on the entire dataset, make sure you set
```
smaller_dataset = False
```
in the training files. Otherwise, set
```
smaller_dataset = True
```
### Single task training
In order to start training, you need to further precise the target quantity you want
to infer. In particular, you should set `output` to be among the following
ones: `"MI"`, `"FFR"`, `"A"` (diameter stenosis), `"stenosis"` (stenosis position). 
To use a pretrained model instead of random initialization of weights and biases,
you should set `pretrained_model` to be the desired .pth file. You can then choose to freeze
an arbitrary set of layers by setting for example
```
freezed_layers = ["conv1", "bn1", "relu", "maxpool", "layer1"]
```

### Multitask learning
In this case, you must set
```
output = "multitask"
```
instead. Before launching training, it is necessary to specify the layers of the  `ResNet18` model
that are going to be shared between the tasks in  `common_layers` and the ones
that are going to be task specific  in `specific_layers`. These variables expect a list
of strings coherently with their names in the baseline network documentation, e.g.
```
common_layers = ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2"]
specific_layers = ["layer3", "layer4", "avgpool", "fc"]
```
Note that the order of the layers passed here above should be compatible with
`nn.Sequential` and thus respect the natural structure of the `ResNet18` architecture.
Additionally, it is required to choose the tasks by setting for example
```
tasks = ["MI", "FFR"]  # any combination among ["MI", "FFR", "A", "stenosis"]
```
and the weighting strategy to be used. For the latter, the string passed should be
coherent with one of the child classes of `AbstractMTL` defined in `utils/utils.py`, e.g.
```
weighting_strategy = "WeightedDynamicalAverage"  # or OL_AUX for Adaptive Auxiliary Tasks
```

## Results
A complete report will be added soon. For the main results,
please refer to the temporary slides available in the main folder of the repository.
