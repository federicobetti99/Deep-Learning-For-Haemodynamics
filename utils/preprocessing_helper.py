import numpy as np
import os
import pickle
from h5py import *
from skimage import io
from skimage.io import imsave
from skimage.util import random_noise
import pandas as pd
import random


# ADD NOISE TO SNAPSHOTS

def add_noise_to_snapshots(snapshots_folder, normal_img_folder, rotated_img_folder):
    """
    Add Poisson noises to all snapshots, this function requires the two folders "augmented_normal_snapshots" and
    "augmented_rotated_snapshots" to be in the directory containing the repository. Additionally, it translates the images
    from RGB to BW and saves them onto the two different folders for successive use
    :param snapshots_folder: folder containing the Paraview snapshots for the whole dataset
    :param normal_img_folder: folder where to store normal images
    :param rotated_img_folder: folder where to store rotated images
    :return:
    """
    count = 0
    for i in range(12):
        folder = snapshots_folder + "/snapshots_" + str(i)
        list_0 = os.listdir(folder)
        for j in range(int(len(list_0)/2)):
            img1 = io.imread(folder + "/snap_" + str(2*j) + ".png")
            img1 = random_noise(img1, mode="poisson")  # add noise
            rgb_weights = [0.2989, 0.5870, 0.1140]
            img1 = np.dot(img1[..., :3], rgb_weights)  # BW
            imsave(normal_img_folder + "/noisy_snap_" + str(count) + ".png", img1)  # save figure

            img2 = io.imread(folder + "/snap_" + str(2*j+1) + ".png")
            img2 = random_noise(img2, mode="poisson")  # add noise
            rgb_weights = [0.2989, 0.5870, 0.1140]
            img2 = np.dot(img2[..., :3], rgb_weights)  # BW
            imsave(rotated_img_folder + "/noisy_snap_" + str(count) + ".png", img2)  # save figure
            count += 1


# READING H5 FILE FOR GNN PREPROCESSING

def readAll(file_name):
    """
    Returns a dict with all the data from a h5 file in input
    :param file_name: should be in h5 format
    :return: a dictionary with
        - "MeshPoints": mesh point coordinates
        - "Pressure": a dictionary with key timestep and value the FE pressure solution
        - "Velocity": a dictionary with key timestep and value the FE velocity solution
        - "Displacement": a dictionary with key timestep and value the FE displacement solution
        - "WSS": a dictionary with key timestep and value the FE WSS solution
    """

    data = {}
    pressureData = {}
    displacementData = {}
    WSSData = {}
    velocityData = {}

    with File(file_name, "r") as f:
        # extract keys
        keys = f.keys()

        # extract mesh point coordinates
        mesh_x_points = np.array(f[list(keys)[1]]["Values"])
        mesh_y_points = np.array(f[list(keys)[2]]["Values"])
        mesh_z_points = np.array(f[list(keys)[3]]["Values"])

        numPoints = mesh_z_points.shape[0]
        meshPoints = np.zeros((numPoints, 3))
        meshPoints[:, 0] = mesh_x_points.ravel()
        meshPoints[:, 1] = mesh_y_points.ravel()
        meshPoints[:, 2] = mesh_z_points.ravel()

        # export also pressure, velocity, displacement and WSS for each timestep
        numTotalSimulations = int((len(keys) - 4) / 4)
        for timestep in range(numTotalSimulations):
            WSSData.update({timestep:
                            np.asarray(f[list(keys)[4 + 0 * numTotalSimulations + timestep]]["Values"])})
            displacementData.update({timestep:
                                    np.asarray(f[list(keys)[4 + 1 * numTotalSimulations + timestep]]["Values"])})
            pressureData.update({timestep:
                                np.asarray(f[list(keys)[4 + 2 * numTotalSimulations + timestep]]["Values"])})
            velocityData.update({timestep:
                                np.asarray(f[list(keys)[4 + 3 * numTotalSimulations + timestep]]["Values"])})

    # save everything
    data.update({"MeshPoints": meshPoints})
    data.update({"Pressure": pressureData})
    data.update({"Displacement": displacementData})
    data.update({"WSS": WSSData})
    data.update({"Velocity": velocityData})

    return data, numTotalSimulations


# READ FFR VALUES FROM PICKLE

def read_FFR_values(dataset_folder, branch):
    """
    Read FFR values and save with pickle
    :param dataset_folder: folder storing the whole dataset
    :param branch: simulation node
    :return:
    """
    folder = dataset_folder + "/snapshots_" + str(branch)
    file = open(folder + "/FFR_" + str(branch) + ".txt", "r")
    content = file.read()
    content = content.replace("\n", "")
    content = content.split(", ")
    del content[-1]
    FFR_dict = {}
    for i in range(int(len(content)/3)):
        FFR_dict.update({i: [float(content[3 * i][4:12]), float(content[3 * i + 1][4:12])]})
    file.close()

    output_folder = os.path.join(os.getcwd(), 'FFR_values')
    filename = output_folder + '/FFR_' + str(branch) + '.pkl'
    with open(filename, 'wb') as handle:
        pickle.dump(FFR_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def compute_WSS(dataset_folder, branch, activeStenosis, output_folder):
    """
    Computes the artificial myocardial infraction as a user-defined function of the point-wise WSS values. For the
    moment, it is just an average of the WSS values on some points below the stenosis
    :param dataset_folder: folder where the whole dataset is stored
    :param branch: simulation node
    :param activeStenosis: active stenosis for the simulations in file_name
    :param output_folder: output folder
    """
    file_name = dataset_folder + "/snapshots_" + str(branch) + "/block0.h5"
    data, numTotalSimulations = readAll(file_name)
    indices = []

    if activeStenosis == 0:
        indices = [2201, 1875, 2221, 730, 1872, 1565, 1260, 1840, 2151, 2037, 2038, 1839, 1838, 363, 2150,
                   2068, 1160, 295, 247, 248, 361, 362, 57, 1951, 2220, 2389, 2219, 2285, 2053, 2055, 2148]
    elif activeStenosis == 1:
        indices = [339, 337, 1512, 901, 863, 1566, 1381, 445, 444, 899, 900, 1401, 1462, 1489, 1491, 77, 153,
                   533, 655, 656, 279, 395, 549, 291, 44, 45, 1377, 1181, 2343, 548]
    elif activeStenosis == 2:
        indices = [513, 262, 303, 816, 817, 2007, 1285, 1283, 304, 471, 472, 419, 1067, 553, 1802,
                   2045, 743, 1025, 699, 744, 687, 688, 2064, 898, 700, 422, 864, 421, 425, 426]
    elif activeStenosis == 3:
        indices = [384, 406, 407, 416, 544, 383, 487, 515, 728, 301, 654, 367, 201, 202, 203, 127, 157,
                   350, 385, 2250, 1282, 2287, 857, 941, 71, 72, 64, 1277, 1159]

    MI = np.zeros(numTotalSimulations)

    for i in range(numTotalSimulations):
        MI[i] = np.sqrt(np.mean(np.linalg.norm(data["WSS"][i][indices], ord=2, axis=1) ** 2))

    filename = output_folder + '/MI_' + str(branch) + '.pkl'
    with open(filename, 'wb') as handle:
        pickle.dump(MI, handle, protocol=pickle.HIGHEST_PROTOCOL)


def train_test_manual_split(normal_img_folder, rotated_img_folder, output_folder, validation_size=0.1, test_size=0.2):
    """
    Split manually train and test set
    :param normal_img_folder: folder where the normal snapshots for the whole dataset are stored
    :param rotated_img_folder: folder where the rotated snapshots for the whole dataset are stored
    :param output_folder: output folder for the dataset
    :param validation_size: validation percentage
    :param test_size: test percentage
    """

    df = pd.read_csv("labels/FFR_labels.csv")
    FFR_labels = df.loc[:, df.columns != "Unnamed: 0"].to_numpy()

    df = pd.read_csv("labels/MI_labels.csv")
    MI_labels = df.loc[:, df.columns != "Unnamed: 0"].to_numpy()

    df = pd.read_csv("labels/parameters.csv")
    A_labels = df.loc[:, df.columns == "Stenosis amplitude"].to_numpy()

    df = pd.read_csv("labels/stenosis_labels.csv")
    stenosis_labels = df.loc[:, df.columns == "stenosis"].to_numpy()

    num_elements = int(len(os.listdir(normal_img_folder))/5)

    train_indices = random.sample(list(np.arange(0, num_elements)),
                                  int((1 - test_size - validation_size) * num_elements))
    mask = np.zeros(num_elements, dtype=bool)
    mask[train_indices] = True
    validation_test_indices = np.arange(0, num_elements)[np.logical_not(mask)]
    validation_indices = random.sample(list(validation_test_indices),
                                       int(validation_size / (validation_size + test_size)
                                           * len(validation_test_indices)))
    mask[validation_indices] = True
    test_indices = list(np.arange(0, num_elements)[np.logical_not(mask)])

    train_FFR_labels = []
    train_MI_labels = []
    train_A_labels = []
    train_stenosis_labels = []

    validation_FFR_labels = []
    validation_MI_labels = []
    validation_A_labels = []
    validation_stenosis_labels = []

    test_FFR_labels = []
    test_MI_labels = []
    test_A_labels = []
    test_stenosis_labels = []

    count = 0

    for index in train_indices:
        for j in range(5):
            img = io.imread(normal_img_folder + "/noisy_snap_" + str(5 * index + j) + ".png")
            imsave(output_folder + "/train/normal/noisy_snap_" + str(count) + ".png", img)

            img = io.imread(rotated_img_folder + "/noisy_snap_" + str(5 * index + j) + ".png")
            imsave(output_folder + "/train/rotated/noisy_snap_" + str(count) + ".png", img)

            train_FFR_labels.append(FFR_labels[index])
            train_MI_labels.append(MI_labels[index])
            train_A_labels.append(A_labels[index])
            train_stenosis_labels.append(stenosis_labels[index])

            count += 1

    count = 0

    for index in validation_indices:
        for j in range(5):
            img = io.imread(normal_img_folder + "/noisy_snap_" + str(5 * index + j) + ".png")
            imsave(output_folder + "/validation/normal/noisy_snap_" + str(count) + ".png", img)

            img = io.imread(rotated_img_folder + "/noisy_snap_" + str(5 * index + j) + ".png")
            imsave(output_folder + "/validation/rotated/noisy_snap_" + str(count) + ".png", img)

            validation_FFR_labels.append(FFR_labels[index])
            validation_MI_labels.append(MI_labels[index])
            validation_A_labels.append(A_labels[index])
            validation_stenosis_labels.append(stenosis_labels[index])

            count += 1

    count = 0

    for index in test_indices:
        for j in range(5):
            img = io.imread(normal_img_folder + "/noisy_snap_" + str(5 * index + j) + ".png")
            imsave(output_folder + "/test/normal/noisy_snap_" + str(count) + ".png", img)

            img = io.imread(rotated_img_folder + "/noisy_snap_" + str(5 * index + j) + ".png")
            imsave(output_folder + "/test/rotated/noisy_snap_" + str(count) + ".png", img)

            test_FFR_labels.append(FFR_labels[index])
            test_MI_labels.append(MI_labels[index])
            test_A_labels.append(A_labels[index])
            test_stenosis_labels.append(stenosis_labels[index])

            count += 1

    pd.DataFrame(train_FFR_labels).to_csv("data/train/FFR_labels.csv", header=["FFR_2", "FFR_3"])
    pd.DataFrame(train_MI_labels).to_csv("data/train/MI_labels.csv", header=["MI"])
    pd.DataFrame(train_A_labels).to_csv("data/train/A_labels.csv", header=["A"])
    pd.DataFrame(train_stenosis_labels).to_csv("data/train/stenosis_labels.csv", header=["stenosis"])

    pd.DataFrame(validation_FFR_labels).to_csv("data/validation/FFR_labels.csv", header=["FFR_2", "FFR_3"])
    pd.DataFrame(validation_MI_labels).to_csv("data/validation/MI_labels.csv", header=["MI"])
    pd.DataFrame(validation_A_labels).to_csv("data/validation/A_labels.csv", header=["A"])
    pd.DataFrame(validation_stenosis_labels).to_csv("data/validation/stenosis_labels.csv", header=["stenosis"])

    pd.DataFrame(test_FFR_labels).to_csv("data/test/FFR_labels.csv", header=["FFR_2", "FFR_3"])
    pd.DataFrame(test_MI_labels).to_csv("data/test/MI_labels.csv", header=["MI"])
    pd.DataFrame(test_A_labels).to_csv("data/test/A_labels.csv", header=["A"])
    pd.DataFrame(test_stenosis_labels).to_csv("data/test/stenosis_labels.csv", header=["stenosis"])

    for str_ in ["train", "validation", "test"]:
        df = pd.read_csv(output_folder + "/" + str(str_) + "/FFR_labels.csv")
        FFR_labels = df.loc[:, df.columns != "Unnamed: 0"].to_numpy()
        df = pd.read_csv(output_folder + "/" + str(str_) + "/MI_labels.csv")
        MI_labels = df.loc[:, df.columns == "MI"].to_numpy()
        df = pd.read_csv(output_folder + "/" + str(str_) + "/A_labels.csv")
        A_labels = df.loc[:, df.columns == "A"].to_numpy()
        df = pd.read_csv(output_folder + "/" + str(str_) + "/stenosis_labels.csv")
        stenosis_labels = df.loc[:, df.columns == "stenosis"].to_numpy()
        multitask_labels = np.concatenate([MI_labels, FFR_labels, A_labels, stenosis_labels], axis=1)
        pd.DataFrame(multitask_labels).to_csv(output_folder + "/" + str(str_) + "/multitask_labels.csv",
                                          header=["MI", "FFR_2", "FFR_3", "A", "stenosis"])

    indices = {
        "train": train_indices,
        "validation": validation_indices,
        "test": test_indices
    }

    filename = output_folder + '/indices.pkl'
    with open(filename, 'wb') as handle:
        pickle.dump(indices, handle, protocol=pickle.HIGHEST_PROTOCOL)


def compute_MI_normalization(file_name, output_pkl):
    """
    Compute normalization constant to make MI stenosis independent. On the same indices the norm of the WSS is averaged
    using a simulation with no stenosis
    :param file_name: directory of the .h5 file from which the WSS average norm should be computed
    :param output_pkl: pickle file where to save the obtained normalization constant
    """
    indices_0 = [2201, 1875, 2221, 730, 1872, 1565, 1260, 1840, 2151, 2037, 2038, 1839, 1838, 363, 2150,
                 2068, 1160, 295, 247, 248, 361, 362, 57, 1951, 2220, 2389, 2219, 2285, 2053, 2055, 2148]
    indices_1 = [339, 337, 1512, 901, 863, 1566, 1381, 445, 444, 899, 900, 1401, 1462, 1489, 1491, 77, 153,
                 533, 655, 656, 279, 395, 549, 291, 44, 45, 1377, 1181, 2343, 548]
    indices_2 = [513, 262, 303, 816, 817, 2007, 1285, 1283, 304, 471, 472, 419, 1067, 553, 1802,
                 2045, 743, 1025, 699, 744, 687, 688, 2064, 898, 700, 422, 864, 421, 425, 426]
    indices_3 = [384, 406, 407, 416, 544, 383, 487, 515, 728, 301, 654, 367, 201, 202, 203, 127, 157,
                 350, 385, 2250, 1282, 2287, 857, 941, 71, 72, 64, 1277, 1159]

    data, numTotalSimulations = readAll(file_name)

    normalization = [np.sqrt(np.mean(np.linalg.norm(data["WSS"][0][indices_0], ord=2, axis=1) ** 2)),
                     np.sqrt(np.mean(np.linalg.norm(data["WSS"][0][indices_1], ord=2, axis=1) ** 2)),
                     np.sqrt(np.mean(np.linalg.norm(data["WSS"][0][indices_2], ord=2, axis=1) ** 2)),
                     np.sqrt(np.mean(np.linalg.norm(data["WSS"][0][indices_3], ord=2, axis=1) ** 2))]

    with open(output_pkl, 'wb') as handle:
        pickle.dump(normalization, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_MI_function():
    """
    Creates an artificial MI function as explained in the report linked in the repository.
    For a reference on how the MI function is computed, see https://www.atherosclerosis-journal.com/article/S0021-9150(21)01437-4/fulltext
    or the slides in the repository for further explanation
    """
    df = pd.read_csv("labels/parameters.csv")
    A = df.loc[:, df.columns == "Stenosis amplitude"].to_numpy()
    df = pd.read_csv("labels/WSS_labels.csv")
    WSS = df.loc[:, df.columns == "MI"].to_numpy()
    MI = np.sqrt(0.5 * (1 + WSS ** 2)) * np.exp(A)
    weights = [np.exp(-2*0.123), np.exp(-2*0.443), np.exp(-2*0.752), np.exp(-2*0.921)]
    for i in range(4):
        MI[2025 * i:2025 * (i + 1)] *= weights[i]
    MI = np.tanh(MI)
    pd.DataFrame(MI).to_csv("labels/MI_labels.csv", header=["MI"])


def create_MI_dataset(normal_img_folder, rotated_img_folder, output_folder):
    """
    Creates a reduced MI dataset for transfer learning and clinical purposes from the overall dataset. In particular,
    it takes a random 20% of train, validation and test sets as datasets on which a MI model and an MI with FFR bias
    models are going to be trained.
    """
    filename = "data/indices.pkl"
    with open(filename, "rb") as handle:
        indices = pickle.load(handle)

    df = pd.read_csv("labels/FFR_labels.csv")
    FFR_labels = df.loc[:, df.columns != "Unnamed: 0"].to_numpy()

    df = pd.read_csv("labels/MI_labels.csv")
    MI_labels = df.loc[:, df.columns != "Unnamed: 0"].to_numpy()

    df = pd.read_csv("labels/parameters.csv")
    A = df.loc[:, df.columns == "Stenosis amplitude"].to_numpy()

    df = pd.read_csv("labels/stenosis_labels.csv")
    stenosis = df.loc[:, df.columns == "stenosis"].to_numpy()

    train_indices = indices["train"]
    train_indices = np.random.choice(train_indices, size=int(0.2*len(train_indices)))

    validation_indices = indices["validation"]
    validation_indices = np.random.choice(validation_indices, size=int(0.2*len(validation_indices)))

    test_indices = indices["test"]
    test_indices = np.random.choice(test_indices, size=int(0.2*len(test_indices)))

    train_FFR_labels = []
    train_MI_labels = []
    train_A_labels = []
    train_stenosis_labels = []

    validation_FFR_labels = []
    validation_MI_labels = []
    validation_A_labels = []
    validation_stenosis_labels = []

    test_FFR_labels = []
    test_MI_labels = []
    test_A_labels = []
    test_stenosis_labels = []

    count = 0

    for index in train_indices:
        for j in range(5):
            img = io.imread(normal_img_folder + "/noisy_snap_" + str(5 * index + j) + ".png")
            imsave(output_folder + "/train/normal/noisy_snap_" + str(count) + ".png", img)

            img = io.imread(rotated_img_folder + "/noisy_snap_" + str(5 * index + j) + ".png")
            imsave(output_folder + "/train/rotated/noisy_snap_" + str(count) + ".png", img)

            train_FFR_labels.append(FFR_labels[index])
            train_MI_labels.append(MI_labels[index])
            train_A_labels.append(A[index])
            train_stenosis_labels.append(stenosis[index])

            count += 1

    count = 0

    for index in validation_indices:
        for j in range(5):
            img = io.imread(normal_img_folder + "/noisy_snap_" + str(5 * index + j) + ".png")
            imsave(output_folder + "/validation/normal/noisy_snap_" + str(count) + ".png", img)

            img = io.imread(rotated_img_folder + "/noisy_snap_" + str(5 * index + j) + ".png")
            imsave(output_folder + "/validation/rotated/noisy_snap_" + str(count) + ".png", img)

            validation_FFR_labels.append(FFR_labels[index])
            validation_MI_labels.append(MI_labels[index])
            validation_A_labels.append(A[index])
            validation_stenosis_labels.append(stenosis[index])

            count += 1

    count = 0

    for index in test_indices:
        for j in range(5):
            img = io.imread(normal_img_folder + "/noisy_snap_" + str(5 * index + j) + ".png")
            imsave(output_folder + "/test/normal/noisy_snap_" + str(count) + ".png", img)

            img = io.imread(rotated_img_folder + "/noisy_snap_" + str(5 * index + j) + ".png")
            imsave(output_folder + "/test/rotated/noisy_snap_" + str(count) + ".png", img)

            test_FFR_labels.append(FFR_labels[index])
            test_MI_labels.append(MI_labels[index])
            test_A_labels.append(A[index])
            test_stenosis_labels.append(stenosis[index])

            count += 1

    pd.DataFrame(train_FFR_labels).to_csv(output_folder + "/train/FFR_labels.csv", header=["FFR_2", "FFR_3"])
    pd.DataFrame(train_MI_labels).to_csv(output_folder + "/train/MI_labels.csv", header=["MI"])
    pd.DataFrame(train_A_labels).to_csv(output_folder + "/train/A_labels.csv", header=["A"])
    pd.DataFrame(train_stenosis_labels).to_csv(output_folder + "/train/stenosis_labels.csv", header=["stenosis"])

    pd.DataFrame(validation_FFR_labels).to_csv(output_folder + "/validation/FFR_labels.csv", header=["FFR_2", "FFR_3"])
    pd.DataFrame(validation_MI_labels).to_csv(output_folder + "/validation/MI_labels.csv", header=["MI"])
    pd.DataFrame(validation_A_labels).to_csv(output_folder + "/validation/A_labels.csv", header=["A"])
    pd.DataFrame(validation_stenosis_labels).to_csv(output_folder + "/validation/stenosis_labels.csv", header=["stenosis"])

    pd.DataFrame(test_FFR_labels).to_csv(output_folder + "/test/FFR_labels.csv", header=["FFR_2", "FFR_3"])
    pd.DataFrame(test_MI_labels).to_csv(output_folder + "/test/MI_labels.csv", header=["MI"])
    pd.DataFrame(test_A_labels).to_csv(output_folder + "/test/A_labels.csv", header=["A"])
    pd.DataFrame(test_stenosis_labels).to_csv(output_folder + "/test/stenosis_labels.csv", header=["stenosis"])

    for str_ in ["train", "validation", "test"]:
        df = pd.read_csv(output_folder + "/" + str(str_) + "/FFR_labels.csv")
        FFR_labels = df.loc[:, df.columns != "Unnamed: 0"].to_numpy()
        df = pd.read_csv(output_folder + "/" + str(str_) + "/MI_labels.csv")
        MI_labels = df.loc[:, df.columns == "MI"].to_numpy()
        df = pd.read_csv(output_folder + "/" + str(str_) + "/A_labels.csv")
        A_labels = df.loc[:, df.columns == "A"].to_numpy()
        df = pd.read_csv(output_folder + "/" + str(str_) + "/stenosis_labels.csv")
        stenosis_labels = df.loc[:, df.columns == "stenosis"].to_numpy()
        multitask_labels = np.concatenate([MI_labels, FFR_labels, A_labels, stenosis_labels], axis=1)
        pd.DataFrame(multitask_labels).to_csv(output_folder + "/" + str(str_) + "/multitask_labels.csv",
                                              header=["MI", "FFR_2", "FFR_3", "A", "stenosis"])

    MI_indices = {
        "train": train_indices,
        "validation": validation_indices,
        "test": test_indices
    }

    filename = output_folder + '/indices.pkl'
    with open(filename, 'wb') as handle:
        pickle.dump(MI_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
