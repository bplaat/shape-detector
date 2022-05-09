import data
import numpy as np
import random

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Layer:
    def __init__(self, size):
        self.size = size

    def project(self, nextLayer):
        self.weights = np.array([ [ random.uniform(0, 1) for x in range(self.size) ] for y in range(nextLayer.size) ])

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def run(self, inputs):
        result = np.array(inputs)
        for layer in self.layers[:-1]:
            result = sigmoid(layer.weights.dot(result))
        return result

    def __error(self, trainingItems, symbols):
        errorSum = 0
        for item in trainingItems:
            result = self.run(item[0])
            correct = symbols[item[1]]
            errorSum += sum([ (result[i] - correct[i]) ** 2 for i in range(len(symbols)) ])
        return errorSum / len(trainingItems)

    def train(self, trainingItems, symbols, maxError):
        trainingCycles = 0
        error = None
        while error == None or error > maxError:
            oldError = self.__error(trainingItems, symbols)

            changes = []
            for layer in self.layers[:-1]:
                y = random.randint(0, len(layer.weights) - 1)
                x = random.randint(0, len(layer.weights[0]) - 1)
                changes.append({ 'layer': layer, 'x': x, 'y': y, 'weight': layer.weights[y][x] })
                layer.weights[y][x] += random.uniform(-1, 1) / 100

            error = self.__error(trainingItems, symbols)
            if error > oldError:
                for change in changes:
                    change['layer'].weights[change['y']][change['x']] = change['weight']

            trainingCycles += 1
        return trainingCycles

if __name__ == '__main__':
    inputLayer = Layer(9)
    hiddenLayer = Layer(4)
    outputLayer = Layer(2)

    inputLayer.project(hiddenLayer)
    hiddenLayer.project(outputLayer)

    network = NeuralNetwork([ inputLayer, hiddenLayer, outputLayer ])

    print('Training...')
    trainingCycles = network.train(data.trainingSet, data.outputDict, data.maxError)
    print('Training done in', trainingCycles, 'cycles!\n')

    for item in data.testSet:
        result = network.run(item[0])
        print(item[0][0:3], result)
        print(item[0][3:6], 'Guess:', result[0] > result[1] and 'O' or 'X', '| Correct:', item[1])
        correct = data.outputDict[item[1]]
        print(item[0][6:9], (result[0] > result[1]) == (correct[0] > correct[1]) and 'PASSED' or 'FAILED', '\n')
