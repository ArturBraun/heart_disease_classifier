import torch.nn as nn
import torch.optim as optim
import torch
from collections import OrderedDict
import os
import pandas as pd
import numpy as np
import math
import json
import matplotlib.pyplot as plt


def prepare_data(file_path, training_percentage, training_data_path, test_data_path, normalization_parameters_path):
    # Loads data and divides it into two files: training dataset and testing dataset
    np_array = pd.read_csv(file_path).to_numpy()
    np_array = np_array[1:, :]  # removes 1 row, which is columns names
    np_array = np_array.astype(float)
    np_array, normalization_parameters = normalize(np_array)

    with open(normalization_parameters_path, 'w') as f:
        json.dump(normalization_parameters, f)

    np.random.shuffle(np_array)
    training_data_size = math.ceil(np.size(np_array,0) * training_percentage)
    np.savetxt(training_data_path, np_array[:training_data_size, :], delimiter=",")
    np.savetxt(test_data_path, np_array[training_data_size:, :], delimiter=",")


def load_data(training_data_path, test_data_path):
    return pd.read_csv(training_data_path).to_numpy().astype(float), pd.read_csv(test_data_path).to_numpy().astype(float)


def normalize(data):
    normalization_parameters = []

    for i in range(np.size(data, 1)):
        normalization_parameters.append({
            "min": data[:, i].min(),
            "max": data[:, i].max()
        })
        data[:, i] = (data[:, i] - normalization_parameters[i]["min"]) / (normalization_parameters[i]["max"] - normalization_parameters[i]["min"])

    return data, normalization_parameters


def draw_analysis_plots(training_data, test_data):
    names = ["healthy", "heart disease"]
    training_data_values = [np.count_nonzero(training_data[:, -1] == 0),
                            np.count_nonzero(training_data[:, -1] == 1)]
    test_data_values = [np.count_nonzero(test_data[:, -1] == 0),
                        np.count_nonzero(test_data[:, -1] == 1)]

    fig1, axs1 = plt.subplots(1, 2)
    fig1.suptitle('Categories plot')
    axs1[0].bar(names, training_data_values)
    axs1[0].set_title("training data")
    axs1[1].bar(names, test_data_values)
    axs1[1].set_title("test data")
    plt.show()


def create_net(net_structure):
    net_sequential = OrderedDict()
    index = 0

    for i in range(len(net_structure) - 1):
        index += 1
        net_sequential[str(index)] = nn.Linear(net_structure[i], net_structure[i+1])
        index += 1
        net_sequential[str(index)] = nn.Sigmoid()

    return nn.Sequential(net_sequential)


def train_net(net, epochs, lr, training_data, test_data):
    input_data = training_data[:, :-1]
    output_data = training_data[:, -1]
    output_data_size = np.size(output_data, 0)
    output_data = output_data.reshape((output_data_size, 1))

    train_dataset = torch.utils.data.TensorDataset(torch.Tensor(input_data), torch.Tensor(output_data))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    learning_parameters = {
        "training_data_correctness": [],
        "test_data_correctness": []
    }
    best_net_parameters = {
        "error": float("inf"),
        "weights": None,
        "correct_classification": -1.0
    }
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    for epoch in range(epochs):
        running_loss = 0.0
        running_sum = 0
        for input_data, desired_output in train_loader:
            optimizer.zero_grad()
            output = net(input_data)
            loss = criterion(output, desired_output)

            result = output.data.numpy()
            target = desired_output.data.numpy()
            if round(result[0][0]) == target[0][0]:
                running_sum += 1

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        correct_classification = 100 * running_sum / len(train_loader.dataset)
        learning_parameters["training_data_correctness"].append(correct_classification)
        print("In {0}. epoch error was {1:.5f}. Correct classification was {2:.5f}%.".format(epoch, running_loss, correct_classification))

        if correct_classification > best_net_parameters["correct_classification"]:
            best_net_parameters["error"] = running_loss
            best_net_parameters["state_dict"] = net.state_dict().copy()
            best_net_parameters["correct_classification"] = correct_classification

        with torch.no_grad():
            testing_sum = 0
            for input_data, desired_output in train_loader:
                output = net(input_data)
                result = output.data.numpy()
                target = desired_output.data.numpy()

                if round(result[0][0]) == target[0][0]:
                    testing_sum += 1

            learning_parameters["test_data_correctness"].append(100 * testing_sum / len(train_loader.dataset))

    print("\nBest correct classification was {1:.5f}%. Error was {0:.5f}.\n".format(best_net_parameters["error"], best_net_parameters["correct_classification"]))

    return net, learning_parameters


def draw_error_plots(learning_parameters, epochs_number):
    epochs_vec = np.arange(epochs_number)
    fig, axs = plt.subplots()
    fig.suptitle("Correct classification chart")
    axs.plot(epochs_vec, learning_parameters["training_data_correctness"], label="training data")
    axs.plot(epochs_vec, learning_parameters["test_data_correctness"], label="test data")
    axs.set_xlabel('epoch')
    axs.set_ylabel('classification correctness [%]')
    axs.legend()
    plt.show()


def main():
    net_structure = [13, 20, 20, 1]
    epochs = 500
    lr = 0.001
    data_file_path = "./heart_diseases.csv"
    training_data_path = "./training_data.csv"
    test_data_path = "./test_data.csv"
    normalization_parameters_path = "./normalization_parameters.json"

    if (not os.path.isfile(training_data_path)) or (not os.path.isfile(test_data_path)):
        prepare_data(data_file_path, 0.8, training_data_path, test_data_path, normalization_parameters_path)
    training_data, test_data = load_data(training_data_path, test_data_path)

    draw_analysis_plots(training_data, test_data)

    net = create_net(net_structure)
    net, learning_parameters = train_net(net, epochs, lr, training_data, test_data)
    draw_error_plots(learning_parameters, epochs)


if __name__ == '__main__':
    main()






