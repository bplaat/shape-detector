import { NeuralNetwork } from './neural.js';

const network = new NeuralNetwork({
    activation: 'sigmoid',
    layers: [ 9, 6, 3 ]
});

// Training
console.log('Training...')
const trainingCycles = network.train([
    {
        input: [
            1, 1, 1,
            1, 0, 1,
            1, 1, 1
        ],
        output: 'O'
    },
    {
        input: [
            0, 1, 0,
            1, 0, 1,
            0, 1, 0
        ],
        output: 'O'
    },
    {
        input: [
            0, 1, 0,
            1, 1, 1,
            0, 1, 0
        ],
        output: 'X'
    },
    {
        input: [
            1, 0, 1,
            0, 1, 0,
            1, 0, 1
        ],
        output: 'X'
    },
    {
        input: [
            1, 0, 0,
            0, 0, 0,
            0, 0, 0
        ],
        output: '*'
    },
    {
        input: [
            0, 1, 0,
            0, 0, 0,
            0, 0, 0
        ],
        output: '*'
    },
    {
        input: [
            0, 0, 1,
            0, 0, 0,
            0, 0, 0
        ],
        output: '*'
    }
], 0.01);
console.log(`Training done in ${trainingCycles} cycles`);
console.log(network.symbols);

// Testing
const testingData = [
    {
        input: [
            0, 1, 1,
            1, 0, 1,
            1, 1, 0
        ],
        output: 'O'
    },
    {
        input: [
            1, 0, 1,
            1, 0, 1,
            1, 1, 0
        ],
        output: 'O'
    },
    {
        input: [
            1, 0, 0,
            1, 1, 1,
            0, 0, 1
        ],
        output: 'X'
    },
    {
        input: [
            0, 0, 1,
            1, 1, 1,
            1, 0, 0
        ],
        output: 'X'
    },
    {
        input: [
            1, 0, 1,
            0, 0, 0,
            1, 0, 1
        ],
        output: 'O'
    },
    {
        input: [
            0, 1, 0,
            1, 0, 1,
            0, 1, 0
        ],
        output: 'O'
    },
    {
        input: [
            0, 0, 0,
            1, 1, 1,
            0, 0, 0
        ],
        output: 'X'
    },
    {
        input: [
            1, 0, 0,
            1, 1, 0,
            1, 0, 0
        ],
        output: 'X'
    },
    {
        input: [
            1, 0, 0,
            0, 0, 0,
            0, 0, 0
        ],
        output: '*'
    },
    {
        input: [
            0, 1, 1,
            0, 0, 0,
            0, 0, 0
        ],
        output: '*'
    }
];
for (const item of testingData) {
    const result = network.likely(item.input);
    console.log(result, item.output, result == item.output ? 'PASSED' : 'FAILED');
}
