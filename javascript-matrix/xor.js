import { NeuralNetwork } from './neural.js';

const network = new NeuralNetwork({
    activation: 'sigmoid',
    layers: [ 2, 2, 2, 2 ]
});

// Training
console.log('Training...')
const trainingCycles = network.train([
    {
        input: [
            0, 0
        ],
        output: 0
    },
    {
        input: [
            0, 1
        ],
        output: 1
    },
    {
        input: [
            1, 0
        ],
        output: 1
    },
    {
        input: [
            1, 1
        ],
        output: 0
    }
], 0.005);
console.log(`Training done in ${trainingCycles} cycles`);
console.log(network.symbols);

// Testing
const testingData = [
    {
        input: [
            0, 1
        ],
        output: 1
    },
    {
        input: [
            0, 0
        ],
        output: 0
    },
    {
        input: [
            1, 0
        ],
        output: 1
    },
    {
        input: [
            1, 1
        ],
        output: 0
    }
];
for (const item of testingData) {
    const result = network.likely(item.input);
    console.log(result, item.output, result == item.output ? 'PASSED' : 'FAILED');
}
