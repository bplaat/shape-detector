// gcc shape.c -o shape && ./shape

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

// Random
size_t seed = 1;
double random_random(void) {
    double x = sin(seed++) * 10000;
    return x - floor(x);
}

int32_t random_randint(int32_t min, int32_t max) {
    return floor(random_random() * (max - min + 1)) + min;
}

// Sigmoid
float sigmoid(float x) {
    return (1 / (1 + powf(2.71828182846, -x)));
}

// List
typedef struct List {
    void **items;
    size_t capacity;
    size_t size;
} List;

#define list_get(list, index) ((list)->items[index])

List *list_new(size_t capacity) {
    List *list = malloc(sizeof(List));
    list->items = malloc(sizeof(void *) * capacity);
    list->capacity = capacity;
    list->size = 0;
    return list;
}

void list_add(List *list, void *item) {
    if (list->size == list->capacity) {
        list->capacity <<= 1;
        list->items = realloc(list->items, sizeof(void *) * list->capacity);
    }
    list->items[list->size++] = item;
}

void list_free(List *list, void (*free_function)(void *item)) {
    if (free_function != NULL) {
        for (size_t i = 0; i < list->size; i++) {
            free_function(list->items[i]);
        }
    }
    free(list->items);
    free(list);
}

// Matrix
typedef struct Matrix {
    size_t rows;
    size_t columns;
    float *elements;
} Matrix;

#define matrix_get(matrix, row, column) ((matrix)->elements[(row) * (matrix)->columns + (column)])
#define matrix_set(matrix, row, column, value) ((matrix)->elements[(row) * (matrix)->columns + (column)] = (value))

Matrix *matrix_new(size_t rows, size_t columns) {
    Matrix *matrix = malloc(sizeof(Matrix));
    matrix->rows = rows;
    matrix->columns = columns;
    matrix->elements = malloc(sizeof(float) * rows * columns);
    return matrix;
}

void matrix_mul_vector(Matrix *matrix, float *vector, float *result) {
    for (size_t y = 0; y < matrix->rows; y++) {
        result[y] = 0;
        for (size_t x = 0; x < matrix->columns; x++) {
            result[y] += matrix->elements[y * matrix->columns + x] * vector[x];
        }
    }
}

void matrix_free(Matrix *matrix) {
    free(matrix->elements);
    free(matrix);
}

// Layer
typedef struct Layer {
    size_t size;
    Matrix *weights;
} Layer;

Layer *layer_new(size_t size) {
    Layer *layer = malloc(sizeof(Layer));
    layer->size = size;
    layer->weights = NULL;
    return layer;
}

void layer_project(Layer *layer, Layer *next_layer) {
    layer->weights = matrix_new(next_layer->size, layer->size);
    for (size_t y = 0; y < layer->weights->rows; y++) {
        for (size_t x = 0; x < layer->weights->columns; x++) {
            matrix_set(layer->weights, y, x, random_random());
        }
    }
}

void layer_free(Layer *layer) {
    if (layer->weights != NULL) {
        matrix_free(layer->weights);
    }
    free(layer);
}

// Layer change
typedef struct LayerChange {
    Layer *layer;
    size_t x;
    size_t y;
    float weight;
} LayerChange;

LayerChange *layer_change_new(Layer *layer, size_t x, size_t y, float weight) {
    LayerChange *change = malloc(sizeof(LayerChange));
    change->layer = layer;
    change->x = x;
    change->y = y;
    change->weight = weight;
    return change;
}

void layer_change_free(LayerChange *change) {
    free(change);
}

// NeuralNetwork
typedef struct NeuralNetwork {
    List *layers;
} NeuralNetwork;

NeuralNetwork *neural_network_new(void) {
    NeuralNetwork *network = malloc(sizeof(NeuralNetwork));
    network->layers = list_new(4);
    return network;
}

void neural_network_add_layer(NeuralNetwork *network, Layer *layer) {
    list_add(network->layers, layer);
}

void neural_network_run(NeuralNetwork *network, float *inputs, float *results) {
    float temp1[32]; // Ugly
    Layer *first_layer = (Layer *)list_get(network->layers, 0);
    memcpy(temp1, inputs, sizeof(float) * first_layer->size);

    for (size_t i = 0; i < network->layers->size - 1; i++) {
        Layer *layer = (Layer *)list_get(network->layers, i);

        float temp2[32]; // Ugly
        matrix_mul_vector(layer->weights, temp1, temp2);
        for (size_t j = 0; j < layer->size; j++) {
            temp2[j] = sigmoid(temp2[j]);
        }
        memcpy(temp1, temp2, sizeof(float) * 32);
    }

    Layer *output_layer = (Layer *)list_get(network->layers, network->layers->size - 1);
    memcpy(results, temp1, sizeof(float) * output_layer->size);
}

float neural_network_error(NeuralNetwork *network, float *trainItems, size_t trainSize) {
    Layer *first_layer = (Layer *)list_get(network->layers, 0);
    Layer *output_layer = (Layer *)list_get(network->layers, network->layers->size - 1);
    float errorSum = 0;
    for (size_t i = 0; i < trainSize; i++) {
        float *trainItemInputs = &trainItems[i * (first_layer->size + output_layer->size)];
        float *trainItemResults = trainItemInputs + first_layer->size;

        float results[output_layer->size];
        neural_network_run(network, trainItemInputs, results);

        for (size_t j = 0; j < output_layer->size; j++) {
            errorSum += (trainItemResults[j] - results[j]) * (trainItemResults[j] - results[j]);
        }
    }
    return errorSum / trainSize;
}

size_t neural_network_train(NeuralNetwork *network, float *trainItems, size_t trainSize, float maxError) {
    size_t trainingCycles = 0;
    float error = -1;
    while (error == -1 || error > maxError) {
        float oldError = neural_network_error(network, trainItems, trainSize);

        List *changes = list_new(network->layers->capacity);
        for (size_t i = 0; i < network->layers->size - 1; i++) {
            Layer *layer = (Layer *)list_get(network->layers, i);
            size_t y = random_randint(0, layer->weights->rows - 1);
            size_t x = random_randint(0, layer->weights->columns - 1);
            list_add(changes, layer_change_new(layer, x, y, matrix_get(layer->weights, y, x)));
            matrix_set(layer->weights, y, x, matrix_get(layer->weights, y, x) + (random_random() * 2 - 1) / 100);
        }

        error = neural_network_error(network, trainItems, trainSize);
        if (error > oldError) {
            for (size_t i = 0; i < changes->size; i++) {
                LayerChange *change = (LayerChange *)list_get(changes, i);
                matrix_set(change->layer->weights, change->y, change->x, change->weight);
            }
        }
        list_free(changes, (void (*)(void *))layer_change_free);
        trainingCycles++;
    }
    return trainingCycles;
}

void neural_network_free(NeuralNetwork *network) {
    list_free(network->layers, (void (*)(void *))layer_free);
    free(network);
}

// Main
typedef struct DataItem {
    float inputs[9];
    float results[2];
} DataItem;

int main(void) {
    // Create network
    Layer *input_layer = layer_new(9);
    Layer *hidden_layer = layer_new(4);
    Layer *output_layer = layer_new(2);

    layer_project(input_layer, hidden_layer);
    layer_project(hidden_layer, output_layer);

    NeuralNetwork *network = neural_network_new();
    neural_network_add_layer(network, input_layer);
    neural_network_add_layer(network, hidden_layer);
    neural_network_add_layer(network, output_layer);

    // Train network
    DataItem trainItems[] = {
        {{
            1, 1, 1,
            1, 0, 1,
            1, 1, 1
        }, {1, 0}},
        {{
            0, 1, 0,
            1, 0, 1,
            0, 1, 0
        }, {1, 0}},
        {{
            0, 1, 0,
            1, 1, 1,
            0, 1, 0
        }, {0, 1}},
        {{
            1, 0, 1,
            0, 1, 0,
            1, 0, 1
        }, {0, 1}}
    };
    printf("Start training...\n");
    size_t trainingCycles = neural_network_train(network, (float *)trainItems, sizeof(trainItems) / sizeof(DataItem), 0.005);
    printf("Training done in %zu cycles!\n", trainingCycles);

    // Test network
    DataItem testItems[] = {
        {{
            0, 1, 1,
            1, 0, 1,
            1, 1, 0
        }, {1, 0}},
        {{
            1, 0, 1,
            1, 0, 1,
            1, 1, 0
        }, {1, 0}},
        {{
            1, 0, 0,
            1, 1, 1,
            0, 0, 1
        }, {0, 1}},
        {{
            0, 0, 1,
            1, 1, 1,
            1, 0, 0
        }, {0, 1}},
        {{
            1, 0, 1,
            0, 0, 0,
            1, 0, 1
        }, {1, 0}},
        {{
            0, 1, 0,
            1, 0, 1,
            0, 1, 0
        }, {1, 0}},
        {{
            0, 0, 1,
            1, 1, 1,
            1, 0, 0
        }, {0, 1}},
        {{
            0, 0, 0,
            1, 1, 1,
            0, 0, 0
        }, {0, 1}},
        {{
            1, 0, 0,
            1, 1, 0,
            1, 0, 0
        }, {0, 1}}
    };
    for (size_t i = 0; i < (sizeof(testItems) / sizeof(DataItem)); i++) {
        DataItem *testItem = &testItems[i];
        float result[2];
        neural_network_run(network, testItem->inputs, result);
        printf("%f %f | %f %f | %s\n", result[0], result[1], testItem->results[0], testItem->results[1],
            result[0] > result[1] == testItem->results[0] > testItem->results[1] ? "PASSED" : "FAILED");
    }

    // Clean up
    neural_network_free(network);
    return EXIT_SUCCESS;
}
