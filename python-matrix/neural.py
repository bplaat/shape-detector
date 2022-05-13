import numpy as np
import random

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(np.array([ [ random.uniform(0, 1) for x in range(layers[i]) ] for y in range(layers[i + 1]) ]))

    def run(self, inputs):
        result = np.array(inputs)
        for layer in self.layers:
            result = sigmoid(layer.dot(result))
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
        error = None
        while error == None or error > maxError:
            oldError = self.__error(trainingItems)

            changes = []
            for layer in self.layers:
                y = random.randint(0, len(layer) - 1)
                x = random.randint(0, len(layer[0]) - 1)
                changes.append({ 'layer': layer, 'x': x, 'y': y, 'weight': layer[y][x] })
                layer[y][x] += random.uniform(-1, 1) / 100

            error = self.__error(trainingItems)
            if error > oldError:
                for change in changes:
                    change['layer'][change['y']][change['x']] = change['weight']

            trainingCycles += 1
        return trainingCycles
