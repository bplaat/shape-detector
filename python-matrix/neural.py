import math
import numpy as np
import random

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

class NeuralNetwork:
    def __init__(self, layers, activation='sigmoid', training='random'):
        self.activation = activation
        self.training = training
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append({
                'weights': np.array([ [ random.uniform(-1, 1) for x in range(layers[i + 1]) ] for y in range(layers[i]) ]),
                'biases': np.array([ random.uniform(-1, 1) for y in range(layers[i + 1]) ])
            })

    def run(self, input):
        result = np.array(input)
        for layer in self.layers:
            result = result.dot(layer['weights']) + layer['biases']
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
            errorSum += sum([ (result[i] - correct[i]) ** 2 for i in range(len(result)) ])
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
                    if random.randint(1, 10) != 1:
                        x = random.randint(0, len(layer['weights'][0]) - 1)
                        y = random.randint(0, len(layer['weights']) - 1)
                        changes.append({ 'layer': layer, 'type': 'weight', 'x': x, 'y': y, 'oldWeight': layer['weights'][y][x] })
                        layer['weights'][y][x] += random.uniform(-1, 1) / 100
                    else:
                        y = random.randint(0, len(layer['biases']) - 1)
                        changes.append({ 'layer': layer, 'type': 'bias', 'y': y, 'oldBias': layer['biases'][y] })
                        layer['biases'][y] += random.uniform(-1, 1) / 100

                newError = self.__error(trainingItems)
                if newError < error:
                    error = newError
                else:
                    for change in changes:
                        if change['type'] == 'weight':
                            change['layer']['weights'][change['y']][change['x']] = change['oldWeight']
                        if change['type'] == 'bias':
                            change['layer']['biases'][change['y']] = change['oldBias']

            # I don't know what I'm doing, for more info:
            # http://iamtrask.github.io/2015/07/12/basic-python-network/
            if self.training == 'math' and self.activation == 'sigmoid':
                inputs = np.array([ item[0] for item in trainingItems ])
                outputs = np.array([ self.symbols[item[1]] for item in trainingItems ])

                results = [ inputs ]
                for layer in self.layers:
                    results.append(sigmoid(results[-1].dot(layer['weights'])))

                lastError = None
                for i, layer in reversed(list(enumerate(self.layers))):
                    if i == len(self.layers) - 1:
                        lastError = outputs - results[i + 1]
                        error = np.mean(np.abs(lastError))
                    else:
                        lastError = lastError.dot(self.layers[i + 1]['weights'].T)

                    layerDelta = lastError * sigmoid_derivative(results[i + 1])
                    layer['weights'] += results[i].T.dot(layerDelta)

            trainingCycles += 1
        return trainingCycles
