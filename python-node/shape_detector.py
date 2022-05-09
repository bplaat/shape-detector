import data, math, random

# Globals I know it is ugly
dataSource = None
dataIndex = None

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class ValueHolder:
    def getValue(self):
        pass

class Link(ValueHolder):
    def __init__(self, inputNode):
        self.weight = random.uniform(0, 1)
        self.inputNode = inputNode

    def getValue(self):
        return self.weight * self.inputNode.getValue()


class Node(ValueHolder):
    pass


class InputNode(Node):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getValue(self):
        return dataSource[dataIndex][0][self.y][self.x]

class OutputNode(Node):
    def __init__(self, links):
        self.links = links

    def getValue(self):
        return sigmoid(sum([ link.getValue() for link in self.links ]))


class NeuralNetwork:
    def __init__(self):
        self.inputNodes = []
        for y in range(3):
            for x in range(3):
                self.inputNodes.append(InputNode(x, y))

        self.circleOutputNode = OutputNode([ Link(inputNode) for inputNode in self.inputNodes ])
        self.crossOutputNode = OutputNode([ Link(inputNode) for inputNode in self.inputNodes ])

    def learn(self):
        global dataSource, dataIndex
        dataSource = data.trainingSet

        previousErrorAvg = None
        previousCircleLinkIndex = None
        previousCircleLinkWeight = None
        previousCrossLinkIndex = None
        previousCrossLinkWeight = None
        learningCycles = 0
        while previousErrorAvg == None or previousErrorAvg > data.maxError:
            learningCycles += 1

            errorSum = 0
            for i in range(len(data.trainingSet)):
                dataIndex = i
                correct = data.outputDict[data.trainingSet[i][1]]
                errorSum += (
                    (self.circleOutputNode.getValue() - correct[0]) ** 2 +
                    (self.crossOutputNode.getValue() - correct[1]) ** 2
                )
            errorAvg = errorSum / len(data.trainingSet)
            if learningCycles % 50 == 0:
                print('*' * round(100 * errorAvg))

            if previousErrorAvg != None and errorAvg > previousErrorAvg:
                # print('Worse the previous')
                errorAvg = previousErrorAvg
                self.circleOutputNode.links[previousCircleLinkIndex].weight = previousCircleLinkWeight
                self.crossOutputNode.links[previousCrossLinkIndex].weight = previousCrossLinkWeight

            previousErrorAvg = errorAvg

            previousCircleLinkIndex = random.randint(0, len(self.circleOutputNode.links) - 1)
            previousCircleLinkWeight = self.circleOutputNode.links[previousCircleLinkIndex].weight
            self.circleOutputNode.links[previousCircleLinkIndex].weight += random.uniform(-1, 1) / 100

            previousCrossLinkIndex = random.randint(0, len(self.crossOutputNode.links) - 1)
            previousCrossLinkWeight = self.crossOutputNode.links[previousCrossLinkIndex].weight
            self.crossOutputNode.links[previousCrossLinkIndex].weight += random.uniform(-1, 1) / 100

        print('Learning done in', learningCycles, 'cycles\n')

    def test(self):
        global dataSource, dataIndex
        dataSource = data.testSet
        for i in range(len(data.testSet)):
            dataIndex = i
            print(data.testSet[dataIndex])
            print([ self.circleOutputNode.getValue(), self.crossOutputNode.getValue() ])
            print(self.circleOutputNode.getValue() > self.crossOutputNode.getValue() and 'Circle' or 'Cross')
            print()

# Create neural network learn and test
if __name__ == '__main__':
    network = NeuralNetwork()
    network.learn()
    network.test()
