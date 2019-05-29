import numpy as np
from random import *
import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import math
import ast


class Network():
    def __init__(self, features_data, labeled_data, inputs_number, outputs_number, batches_number, batches_size,
                 hidden_layers=[4], lr=0.1, features_valid=None, labeled_valid=None):
        self.features_data = features_data
        self.labeled_data = labeled_data
        self.inputs_number = inputs_number
        self.outputs_number = outputs_number
        self.hidden_layers_number = len(hidden_layers)
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.batches_number = batches_number
        self.batches_size = batches_size
        self.features_valid = features_valid
        self.labeled_valid = labeled_valid
        self.run = True

        self.sizes = [inputs_number]
        for layer in self.hidden_layers:
            self.sizes.append(layer)
        self.sizes.append(outputs_number)

        self.weights = self.create_weights()

        self.biases = []
        for i in range(self.hidden_layers_number):
            self.biases.append(
                np.array([np.random.normal(0, math.sqrt(2 / (self.sizes[i] + self.sizes[i + 1]))) for _ in
                          range(self.hidden_layers[i])]))
        self.biases.append(np.array(
            [np.random.normal(0, math.sqrt(2 / (self.sizes[-2] + self.sizes[-1]))) for _ in
             range(self.outputs_number)]))

        self.learn()

    def create_weights(self):
        # creates weights with xavier initialization
        weights = []
        weights.append(
            np.array(
                [[np.random.normal(0, math.sqrt(2 / (self.sizes[0] + self.sizes[1]))) for _ in
                  range(self.inputs_number)]
                 for _ in range(self.hidden_layers[0])]))
        for i in range(1, self.hidden_layers_number):
            weights.append(np.array(
                [[np.random.normal(0, math.sqrt(2 / (self.sizes[i] + self.sizes[i + 1]))) for _ in
                  range(self.hidden_layers[i - 1])] for _ in
                 range(self.hidden_layers[i])]))
        weights.append(
            np.array(
                [[np.random.normal(0, math.sqrt(2 / (self.sizes[-1] + self.sizes[-2]))) for _ in
                  range(self.hidden_layers[self.hidden_layers_number - 1])] for _ in
                 range(self.outputs_number)]))
        return weights

    def backprop(self, i):
        # backpropagation on i sample from training data
        a = [self.features_data[i]]
        act = self.features_data[i]
        z = []
        for weights, biases in zip(self.weights, self.biases):
            zs = np.dot(weights, act) + biases
            z.append(zs)
            act = self.sigmoid(zs)
            a.append(act)

        weights_ret = [np.zeros(weight.shape) for weight in self.weights]
        biases_ret = [np.zeros(bias.shape) for bias in self.biases]
        delta = self.cost_derivative(a[-1], self.labeled_data[i] * self.sigmoid_derivative(z[-1]))[None].transpose()
        biases_ret[-1] = delta
        weights_ret[-1] = np.dot(delta, a[-2][None])

        for i in range(2, self.hidden_layers_number + 2):
            sigmoid_derivative = self.sigmoid_derivative(z[-i][None]).transpose()
            delta = np.multiply(np.dot(self.weights[-i + 1].transpose(), delta), sigmoid_derivative)
            weights_ret[-i] = np.dot(delta, a[-i - 1][None])
            biases_ret[-i] = delta
        biases_ret = [bias.reshape(bias.shape[0]) for bias in biases_ret]
        return weights_ret, biases_ret

    def cost_derivative(self, x, y):
        return 2 * (x - y)

    def learn(self):
        #  learning with batches
        for _ in range(self.batches_number):
            weights_add = [np.zeros(weights.shape) for weights in self.weights]
            biases_add = [np.zeros(biases.shape) for biases in self.biases]
            for i in range(self.batches_size):
                ind = randint(0, self.labeled_data.shape[0] - 1)
                weight, bias = self.backprop(ind)
                for j in range(len(weights_add)):
                    weights_add[j] += weight[j]
                    biases_add[j] += bias[j]
            ind = 0
            for weights, biases in zip(self.weights, self.biases):
                weightas = weights - (self.lr / self.batches_size) * weights_add[ind]
                biases = biases - (self.lr / self.batches_size) * biases_add[ind]
                self.weights[ind] = weightas
                self.biases[ind] = biases
                ind += 1
        if self.features_valid is not None:
            self.validate()

    def predict(self, data, labels):
        # function return accuracy on given data and labels
        result = 0
        for dana, label in zip(data, labels):
            layer = dana
            j = 0
            for weights, biases in zip(self.weights, self.biases):
                j += 1
                layer = np.dot(weights, layer)
                layer = layer + biases
                layer = self.sigmoid(layer)

            layer = layer.tolist()
            label = label.tolist()

            if layer.index(max(layer)) == label.index(max(label)):
                result += 1

        print(result / len(data))
        return result / len(data)

    def validate(self):
        # printing how good network is on validation data
        result = 0
        for i in range(0, len(self.features_valid)):
            layer = self.features_valid[i]
            j = 0
            for weights, biases in zip(self.weights, self.biases):
                j += 1
                layer = np.dot(weights, layer)
                layer = layer + biases
                layer = self.sigmoid(layer)
            layer = layer.tolist()
            if self.labeled_valid[i].tolist()[layer.index(max(layer))] == 1.0:
                result += 1
        print('Validation data accuracy: ' + str(result / len(self.features_valid)))
        return False

    def lossFunction(self, value, true_value):
        return np.sum((value - true_value) ** 2)

    def softmax(self, x):
        y = np.sum(np.exp(x))
        tab = np.array([(np.exp(x[i]) / y) for i in range(self.outputs_number)])
        return tab

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return np.exp(-x) / ((1 + np.exp(-x)) ** 2)

