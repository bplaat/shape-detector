#include "neural.h"
#include <string.h>
#include <math.h>
#include <time.h>

// Math
size_t random_seed = 1;

void random_init(void) {
    random_seed = (size_t)time(NULL);
}

double random_random(void) {
    double x = sin(random_seed++) * 10000;
    return x - floor(x);
}

int32_t random_randint(int32_t min, int32_t max) {
    return floor(random_random() * (max - min + 1)) + min;
}

float sigmoid(float x) {
    return 1 / (1 + powf(2.71828182846, -x));
}

float relu(float x) {
    return x > 0 ? x : 0;
}

// List
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
Matrix *matrix_new(size_t columns, size_t rows) {
    Matrix *matrix = malloc(sizeof(Matrix));
    matrix->columns = columns;
    matrix->rows = rows;
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

// Layer change
LayerChange *layer_change_new(Matrix *layer, size_t x, size_t y, float weight) {
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
NeuralNetwork *neural_network_new(Actication activation, size_t *layers, size_t layersSize) {
    NeuralNetwork *network = malloc(sizeof(NeuralNetwork));
    network->activation = activation;
    network->layers = list_new(layersSize);
    for (size_t i = 0; i < layersSize - 1; i++) {
        Matrix *layer = matrix_new(layers[i], layers[i + 1]);
        for (size_t y = 0; y < layer->rows; y++) {
            for (size_t x = 0; x < layer->columns; x++) {
                matrix_set(layer, y, x, random_random() * 2 - 1);
            }
        }
        list_add(network->layers, layer);
    }
    return network;
}

void neural_network_run(NeuralNetwork *network, float *input, float *output) {
    size_t first_layer_size = ((Matrix *)list_get(network->layers, 0))->columns;
    float result[first_layer_size];
    memcpy(result, input, sizeof(float) * first_layer_size);

    for (size_t i = 0; i < network->layers->size; i++) {
        Matrix *layer = (Matrix *)list_get(network->layers, i);

        float result_tmp[layer->rows];
        matrix_mul_vector(layer, result, result_tmp);
        memcpy(result, result_tmp, sizeof(float) * layer->rows);

        for (size_t j = 0; j < layer->rows; j++) {
            if (network->activation == ACTIVATION_SIGMOID) {
                result[j] = sigmoid(result[j]);
            }
            if (network->activation == ACTIVATION_RELU) {
                result[j] = relu(result[j]);
            }
        }
    }

    size_t output_layer_size = ((Matrix *)list_get(network->layers, network->layers->size - 1))->rows;
    memcpy(output, result, sizeof(float) * output_layer_size);
}

float neural_network_error(NeuralNetwork *network, float *trainItems, size_t trainSize) {
    size_t first_layer_size = ((Matrix *)list_get(network->layers, 0))->columns;
    size_t output_layer_size = ((Matrix *)list_get(network->layers, network->layers->size - 1))->rows;
    float errorSum = 0;
    for (size_t i = 0; i < trainSize; i++) {
        float *trainItemInputs = &trainItems[i * (first_layer_size + output_layer_size)];
        float *trainItemResults = trainItemInputs + first_layer_size;

        float results[output_layer_size];
        neural_network_run(network, trainItemInputs, results);

        for (size_t j = 0; j < output_layer_size; j++) {
            errorSum += (trainItemResults[j] - results[j]) * (trainItemResults[j] - results[j]);
        }
    }
    return errorSum / trainSize;
}

size_t neural_network_train(NeuralNetwork *network, float *trainItems, size_t trainSize, float maxError) {
    size_t trainingCycles = 0;
    float error = neural_network_error(network, trainItems, trainSize);
    while (error == -1 || error > maxError) {
        List *changes = list_new(network->layers->capacity);
        for (size_t i = 0; i < network->layers->size; i++) {
            Matrix *layer = (Matrix *)list_get(network->layers, i);
            size_t x = random_randint(0, layer->columns - 1);
            size_t y = random_randint(0, layer->rows - 1);
            float weight = matrix_get(layer, y, x);
            list_add(changes, layer_change_new(layer, x, y, weight));
            matrix_set(layer, y, x, weight + (random_random() * 2 - 1) / 100);
        }

        float newError = neural_network_error(network, trainItems, trainSize);
        if (newError < error) {
            error = newError;
        } else {
            for (size_t i = 0; i < changes->size; i++) {
                LayerChange *change = (LayerChange *)list_get(changes, i);
                matrix_set(change->layer, change->y, change->x, change->weight);
            }
        }
        list_free(changes, (void (*)(void *))layer_change_free);
        trainingCycles++;
    }
    return trainingCycles;
}

void neural_network_free(NeuralNetwork *network) {
    list_free(network->layers, (void (*)(void *))matrix_free);
    free(network);
}
