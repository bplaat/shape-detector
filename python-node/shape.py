import data, math, random

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class ValueHolder:
    def getValue(self):
        raise NotImplementedError()

class Link(ValueHolder):
    def __init__(self, inputNode):
        self.weight = random.uniform(-1, 1)
        self.inputNode = inputNode

    def getValue(self):
        return self.weight * self.inputNode.getValue()

class Node(ValueHolder):
    pass

class InputNode(Node):
    def getValue(self):
        return self.value

class OutputNode(Node):
    def __init__(self, links):
        self.links = links

    def getValue(self):
        return sigmoid(sum([ link.getValue() for link in self.links ]))

class NeuralNetwork:
    def __init__(self, inputs, outputs):
        self.inputNodes = []
        for i in range(inputs):
            self.inputNodes.append(InputNode())

        self.outputNodes = []
        for i in range(outputs):
            self.outputNodes.append(OutputNode([ Link(inputNode) for inputNode in self.inputNodes ]))

    def run(self, input):
        for i, value in enumerate(input):
            self.inputNodes[i].value = value
        return [ outputNode.getValue() for outputNode in self.outputNodes ]

    def __error(self, trainingItems, symbols):
        errorSum = 0
        for item in trainingItems:
            result = self.run(item[0])
            correct = symbols[item[1]]
            errorSum += sum([ (result[i] - correct[i]) ** 2 for i in range(len(symbols)) ])
        return errorSum / len(trainingItems)

    def train(self, trainingItems, symbols, maxError):
        trainingCycles = 0
        error = self.__error(trainingItems, symbols)
        while error > maxError:
            changes = []
            for outputNode in self.outputNodes:
                index = random.randint(0, len(outputNode.links) - 1)
                link = outputNode.links[index]
                changes.append({ 'link': link, 'weight': link.weight })
                link.weight += random.uniform(-1, 1) / 100

            newError = self.__error(trainingItems, symbols)
            if newError < error:
                error = newError
            else:
                for change in changes:
                    change['link'].weight = change['weight']

            trainingCycles += 1

        return trainingCycles

# Create neural network train and test
if __name__ == '__main__':
    network = NeuralNetwork(data.inputDim, data.outputDim)

    print('Training...')
    trainingCycles = network.train(data.trainingSet, data.outputDict, data.maxError)
    print('Training done in', trainingCycles, 'cycles!')

    for item in data.testSet:
        result = network.run(item[0])
        correct = data.outputDict[item[1]]
        print(result, correct, (result[0] > result[1]) == (correct[0] > correct[1]) and 'PASSED' or 'FAILED')
