import data, math, random

# Our nice sigmoid function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# All the node classes
class ValueHolder:
    def getValue(self):
        raise NotImplementedError()

class Node(ValueHolder):
    pass

class InputNode(Node):
    def getValue(self):
        return self.value

class Link(ValueHolder):
    def __init__(self, inputNode):
        self.weight = random.uniform(-1, 1)
        self.inputNode = inputNode

    def getValue(self):
        return self.weight * self.inputNode.getValue()

class OutputNode(Node):
    def __init__(self, links):
        self.links = links

    def getValue(self):
        return sigmoid(sum([ link.getValue() for link in self.links ]))

# Our neural network class
class NeuralNetwork:
    # Create input and outputs on init
    def __init__(self, inputNodesCount, outputNodesCount):
        self.inputNodes = [ InputNode() for i in range(inputNodesCount) ]
        self.outputNodes = [ OutputNode([ Link(inputNode) for inputNode in self.inputNodes ]) for i in range(outputNodesCount) ]

    # To run the network update all the input value nodes then get value from output nodes
    def run(self, input):
        for i, value in enumerate(input):
            self.inputNodes[i].value = value
        return [ outputNode.getValue() for outputNode in self.outputNodes ]

    # To calculate the error subtract given value from wanted value and square that for each symbol for each training item
    def __error(self, trainingItems, symbols):
        errorSum = 0
        for item in trainingItems:
            result = self.run(item[0])
            correct = symbols[item[1]]
            errorSum += sum([ (result[i] - correct[i]) ** 2 for i in range(len(symbols)) ])
        return errorSum / len(trainingItems)

    # Adjust the weights random until the error value is less then the max error argument
    def train(self, trainingItems, symbols, maxError):
        trainingCycles = 0
        error = self.__error(trainingItems, symbols)
        while error > maxError:
            changes = []
            for outputNode in self.outputNodes:
                randomLink = outputNode.links[random.randint(0, len(outputNode.links) - 1)]
                changes.append({ 'link': randomLink, 'oldWeight': randomLink.weight })
                randomLink.weight += random.uniform(-1, 1) / 100

            newError = self.__error(trainingItems, symbols)
            if newError < error:
                error = newError
            else:
                for change in changes:
                    change['link'].weight = change['oldWeight']

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
