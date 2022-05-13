#ifndef NEURAL_H
#define NEURAL_H

#include <stdint.h>
#include <stdlib.h>

// Math
extern size_t random_seed;

void random_init(void);

double random_random(void);

int32_t random_randint(int32_t min, int32_t max);

float sigmoid(float x);

float relu(float x);


// List
typedef struct List {
    void **items;
    size_t capacity;
    size_t size;
} List;

#define list_get(list, index) ((list)->items[index])

List *list_new(size_t capacity);

void list_add(List *list, void *item);

void list_free(List *list, void (*free_function)(void *item));


// Matrix
typedef struct Matrix {
    size_t rows;
    size_t columns;
    float *elements;
} Matrix;

#define matrix_get(matrix, row, column) ((matrix)->elements[(row) * (matrix)->columns + (column)])

#define matrix_set(matrix, row, column, value) ((matrix)->elements[(row) * (matrix)->columns + (column)] = (value))

Matrix *matrix_new(size_t rows, size_t columns);

void matrix_mul_vector(Matrix *matrix, float *vector, float *result);

void matrix_free(Matrix *matrix);


// Layer change
typedef struct LayerChange {
    Matrix *layer;
    size_t x;
    size_t y;
    float weight;
} LayerChange;

LayerChange *layer_change_new(Matrix *layer, size_t x, size_t y, float weight);

void layer_change_free(LayerChange *change);


// Neural Network
typedef enum Actication {
    ACTIVATION_SIGMOID,
    ACTIVATION_RELU
} Actication;

typedef struct NeuralNetwork {
    Actication activation;
    List *layers;
} NeuralNetwork;

NeuralNetwork *neural_network_new(Actication activation, size_t *layers, size_t layersSize);

void neural_network_run(NeuralNetwork *network, float *input, float *output);

float neural_network_error(NeuralNetwork *network, float *trainItems, size_t trainSize);

size_t neural_network_train(NeuralNetwork *network, float *trainItems, size_t trainSize, float maxError);

void neural_network_free(NeuralNetwork *network);

#endif
