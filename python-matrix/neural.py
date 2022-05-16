import math
import numpy as np
import random

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

class NeuralNetwork:
    def __init__(self, layers, activation='sigmoid', training='random'):
        self.activation = activation
        self.training = training
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(np.array([ [ random.uniform(-1, 1) for x in range(layers[i + 1]) ] for y in range(layers[i]) ]))

    def run(self, input):
        result = np.array(input)
        for layer in self.layers:
            result = result.dot(layer)
            if self.activation == 'sigmoid':
                result = sigmoid(result)
            if self.activation == 'relu':
                result = relu(result)
        return result

    def likely(self, input):
        result = list(self.run(input))
        maxIndex = result.index(max(result))
        for (symbol, outputs) in self.symbols.items():
            if outputs[maxIndex] == 1:
                return symbol

    def __error(self, trainingItems):
        errorSum = 0
        for item in trainingItems:
            result = self.run(item[0])
            correct = self.symbols[item[1]]
            errorSum += sum([ (result[i] - correct[i]) ** 2 for i in range(len(self.symbols)) ])
        return errorSum / len(trainingItems)

    def train(self, trainingItems, maxError):
        # Setup symbols
        self.symbols = {}
        for item in trainingItems:
            self.symbols[item[1]] = []

        symbols = list(self.symbols.keys())
        for i in range(len(symbols)):
            for j in range(len(symbols)):
                self.symbols[symbols[i]].append(i == j and 1 or 0)

        # Train layers
        trainingCycles = 0
        error = self.__error(trainingItems)
        while error > maxError:
            if self.training == 'random':
                changes = []
                for layer in self.layers:
                    x = random.randint(0, len(layer[0]) - 1)
                    y = random.randint(0, len(layer) - 1)
                    changes.append({ 'layer': layer, 'x': x, 'y': y, 'weight': layer[y][x] })
                    layer[y][x] += random.uniform(-1, 1) / 100

                newError = self.__error(trainingItems)
                if newError < error:
                    error = newError
                else:
                    for change in changes:
                        change['layer'][change['y']][change['x']] = change['weight']

            # I don't know what I'm doing, for more info:
            # http://iamtrask.github.io/2015/07/12/basic-python-network/
            if self.training == 'math':
                inputs = np.array([ item[0] for item in trainingItems ])
                outputs = np.array([ self.symbols[item[1]] for item in trainingItems ])

                results = [ inputs ]
                for layer in self.layers:
                    results.append(sigmoid(results[-1].dot(layer)))

                lastError = None
                for i, layer in reversed(list(enumerate(self.layers))):
                    if i == len(self.layers) - 1:
                        lastError = outputs - results[i + 1]
                        error = np.mean(np.abs(lastError))
                    else:
                        lastError = lastError.dot(self.layers[i + 1].T)

                    layerDelta = lastError * sigmoid_derivative(results[i + 1])
                    layer += results[i].T.dot(layerDelta)

            trainingCycles += 1
        return trainingCycles
