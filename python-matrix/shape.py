import data
from neural import NeuralNetwork

if __name__ == '__main__':
    # Random network
    randomNetwork = NeuralNetwork(
        activation='sigmoid',
        training='random',
        layers=[ data.inputDim, 6, 6, data.outputDim ]
    )

    print('Random network training...')
    trainingCycles = randomNetwork.train(data.trainingSet, data.maxError)
    print('Training done in', trainingCycles, 'cycles!')
    print(randomNetwork.symbols)

    for item in data.testSet:
        result = randomNetwork.likely(item[0])
        print(result, item[1], result == item[1] and 'PASSED' or 'FAILED')

    # Math network
    mathNetwork = NeuralNetwork(
        activation='sigmoid',
        training='math',
        layers=[ data.inputDim, 6, 6, data.outputDim ]
    )

    print('\nMath network training...')
    trainingCycles = mathNetwork.train(data.trainingSet, data.maxError)
    print('Training done in', trainingCycles, 'cycles!')
    print(mathNetwork.symbols)

    for item in data.testSet:
        result = mathNetwork.likely(item[0])
        print(result, item[1], result == item[1] and 'PASSED' or 'FAILED')
