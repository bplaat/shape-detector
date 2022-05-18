#include <stdio.h>
#include <stdlib.h>
#include "neural.h"

typedef struct DataItem {
    float input[9];
    float output[2];
} DataItem;

int main(void) {
    random_init();

    size_t networkLayers[] = { 9, 6, 2 };
    NeuralNetwork *network = neural_network_new(
        ACTIVATION_SIGMOID,
        networkLayers, sizeof(networkLayers) / sizeof(size_t)
    );

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
    printf("Training...\n");
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
        float output[2];
        neural_network_run(network, testItem->input, output);
        printf("%f %f | %f %f | %s\n", output[0], output[1], testItem->output[0], testItem->output[1],
            output[0] > output[1] == testItem->output[0] > testItem->output[1] ? "PASSED" : "FAILED");
    }

    // Clean up
    neural_network_free(network);
    return EXIT_SUCCESS;
}
