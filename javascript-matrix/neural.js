export function randint(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

export function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

export function relu(x) {
    return x > 0 ? x : 0;
}

export class Matrix {
    constructor(rows, columns) {
        this.rows = rows;
        this.columns = columns;
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

export class Layer {
    constructor(size) {
        this.size = size;
    }

    project(nextLayer) {
        this.weights = new Matrix(nextLayer.size, this.size);
        for (let y = 0; y < this.weights.rows; y++) {
            for (let x = 0; x < this.weights.columns; x++) {
                this.weights.set(x, y, Math.random());
            }
        }
    }
}

export class NeuralNetwork {
    constructor({ activation = 'sigmoid', layers }) {
        this.activation = activation;
        this.layers = layers.map(layerSize => new Layer(layerSize));
        for (let i = 0; i < this.layers.length - 1; i++) {
            this.layers[i].project(this.layers[i + 1]);
        }
    }

    run(input) {
        let result = input.slice();
        for (let i = 0; i < this.layers.length - 1; i++) {
            const layer = this.layers[i];
            result = layer.weights.mul(result);
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
                this.symbols[symbols[i]][j] = i == j ? 1 : 0;
            }
        }

        // Train weights
        let trainingCycles = 0;
        let error = null;
        while (error == null || error > maxError) {
            const oldError = this._error(trainingItems);

            const changes = [];
            for (const layer of this.layers.slice(0, -1)) {
                const y = randint(0, layer.weights.rows - 1);
                const x = randint(0, layer.weights.columns - 1);
                const weight = layer.weights.get(x, y);
                changes.push({ layer, x, y, weight });
                layer.weights.set(x, y, weight + (Math.random() * 2 - 1) / 100);
            }

            error = this._error(trainingItems);
            if (error > oldError) {
                for (const change of changes) {
                    change.layer.weights.set(change.x, change.y, change.weight);
                }
            }
            trainingCycles++;
        }
        return trainingCycles;
    }
}
