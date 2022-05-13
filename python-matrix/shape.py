import data
from neural import NeuralNetwork

if __name__ == '__main__':
    network = NeuralNetwork([ data.inputDim, 6, data.outputDim ])

    print('Training...')
    trainingCycles = network.train(data.trainingSet, data.maxError)
    print('Training done in', trainingCycles, 'cycles!')
    print(network.symbols)

    for item in data.testSet:
        result = network.likely(item[0])
        print(result, item[1], result == item[1] and 'PASSED' or 'FAILED')
