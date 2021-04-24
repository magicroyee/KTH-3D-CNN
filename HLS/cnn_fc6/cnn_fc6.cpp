#include "cnn_fc6.h"
#include "cnn_fc7.h"
#include "cnn_fc8.h"

#define LINEAR6CHANNELSOUT 1024
#define LINEAR6CHANNELSIN 256 * 9
#define LINEAR7CHANNELSOUT 1024
#define LINEAR7CHANNELSIN 1024
#define LINEAR8CHANNELSOUT 10
#define LINEAR8CHANNELSIN 1024

#define max(a, b) ((a) > (b) ? (a) : (b))

void cnn_fc6()
{
    static float pool5_layer[256 * 9];
    static float linear6_layer[1024];
    static float dropout6_layer[1024];
    static float linear7_layer[1024];
    static float dropout7_layer[1024];
    static float linear8_layer[10];

    /**
     * pool5_layer赋值
     */

zeros:
    for (int co = 0; co < LINEAR6CHANNELSOUT; co++) {
        linear6_layer[co] = 0;
        if (co < LINEAR7CHANNELSOUT) {
            linear7_layer[co] = 0;
            if (co < LINEAR8CHANNELSOUT) {
                linear8_layer[co] = 0;
            }
        }
    }
linear6:
    for (int co = 0; co < LINEAR6CHANNELSOUT; co++) {
        for (int ci = 0; ci < LINEAR6CHANNELSIN; ci++) {
            linear6_layer[co] += fc6_filter[co][ci] * pool5_layer[ci];
        }
        linear6_layer[co] = max(0, linear6_layer[co]);
    }

    /**
     * dropout6
     */

linear7:
    for (int co = 0; co < LINEAR7CHANNELSOUT; co++) {
        for (int ci = 0; ci < LINEAR7CHANNELSIN; ci++) {
            linear7_layer[co] += fc7_filter[co][ci] * dropout6_layer[ci];
        }
        linear7_layer[co] = max(0, linear7_layer[co]);
    }

    /**
     * dropout8
     */

linear8:
    for (int co = 0; co < LINEAR8CHANNELSOUT; co++) {
        for (int ci = 0; ci < LINEAR8CHANNELSIN; ci++) {
            linear8_layer[co] += fc8_filter[co][ci] * dropout7_layer[ci];
        }
    }

    /**
     * log_softmax
     */
}