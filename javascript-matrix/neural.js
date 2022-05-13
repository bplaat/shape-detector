export function randint(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

export function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

export function relu(x) {
    return Math.max(0, x);
}

export class Matrix {
    constructor(columns, rows) {
        this.columns = columns;
        this.rows = rows;
        this.elements = new Float32Array(rows * columns);
    }

    get(x, y) {
        return this.elements[y * this.columns + x];
    }

    set(x, y, value) {
        this.elements[y * this.columns + x] = value;
    }

    mul(vector) {
        const result = new Float32Array(this.rows);
        for (let y = 0; y < this.rows; y++) {
            result[y] = 0;
            for (let x = 0; x < this.columns; x++) {
                result[y] += this.elements[y * this.columns + x] * vector[x];
            }
        }
        return result;
    }
}

export class NeuralNetwork {
    constructor({ activation = 'sigmoid', layers }) {
        this.activation = activation;
        this.layers = [];
        for (let i = 0; i < layers.length - 1; i++) {
            const layer = new Matrix(layers[i], layers[i + 1]);
            for (let y = 0; y < layer.rows; y++) {
                for (let x = 0; x < layer.columns; x++) {
                    layer.set(x, y, Math.random() * 2 - 1);
                }
            }
            this.layers.push(layer);
        }
    }

    run(input) {
        let result = new Float32Array(input);
        for (const layer of this.layers) {
            result = layer.mul(result);
            if (this.activation == 'sigmoid') result = result.map(x => sigmoid(x));
            if (this.activation == 'relu') result = result.map(x => relu(x));
        }
        return result;
    }

    likely(input) {
        const result = this.run(input);
        const maxIndex = result.indexOf(Math.max(...result));
        for (const symbol in this.symbols) {
            if (this.symbols[symbol][maxIndex] == 1) {
                return symbol;
            }
        }
    }

    _error(trainingItems) {
        let errorSum = 0;
        for (const item of trainingItems) {
            const result = this.run(item.input);
            const correct = this.symbols[item.output];
            for (let i = 0; i < correct.length; i++) {
                errorSum += (result[i] - correct[i]) ** 2;
            }
        }
        return errorSum / trainingItems.length;
    }

    train(trainingItems, maxError) {
        // Setup symbols
        this.symbols = {};
        for (const item of trainingItems) {
            this.symbols[item.output] = [];
        }
        const symbols = Object.keys(this.symbols);
        for (let i = 0; i < symbols.length; i++) {
            for (let j = 0; j < symbols.length; j++) {
                this.symbols[symbols[i]].push(i == j ? 1 : 0);
            }
        }

        // Train weights
        let trainingCycles = 0;
        let error = this._error(trainingItems);
        while (error > maxError) {
            const changes = [];
            for (const layer of this.layers) {
                const y = randint(0, layer.rows - 1);
                const x = randint(0, layer.columns - 1);
                const weight = layer.get(x, y);
                changes.push({ layer, x, y, weight });
                layer.set(x, y, weight + (Math.random() * 2 - 1) / 100);
            }

            const newError = this._error(trainingItems);
            if (newError < error) {
                error = newError;
            } else {
                for (const change of changes) {
                    change.layer.set(change.x, change.y, change.weight);
                }
            }
            trainingCycles++;
        }
        return trainingCycles;
    }
}
