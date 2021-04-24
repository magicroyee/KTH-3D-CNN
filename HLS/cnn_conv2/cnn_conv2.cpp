#include "cnn_conv2.h"  // conv2_filter[32][16][3][3][3]

#define max(a, b) ((a) > (b) ? (a) : (b))

void cnn_conv2()
{
    static float pool1_layer[16][16 + 2][48 + 2][48 + 2];
    static float conv2_layer[16][48][48];
    static float pool2_layer[32][8][24][24];

input:
    /**
     * 给pool1_layer赋值
     */

    for (int co = 0; co < 32; co++) {
    zero:
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 48; j++) {
                for (int k = 0; k < 48; k++) {
                    conv2_layer[i][j][k] = 0;
                }
            }
        }
    conv2:
        for (int ci = 0; ci < 16; ci++) {
            for (int p = 0; p < 3; p++) {
                for (int q = 0; q < 3; q++) {
                    for (int r = 0; r < 3; r++) {
                        for (int i = 0; i < 16; i++) {
                            for (int j = 0; j < 48; j++) {
                                for (int k = 0; k < 48; k++) {
                                    conv2_layer[i][j][k] +=
                                        conv2_filter[co][ci][p][q][r] * pool1_layer[ci][i + p][j + q][k + r];
                                }
                            }
                        }
                    }
                }
            }
        }
    pool2:
        for (int i = 0, p = 0; i < 8; i++, p += 2) {
            for (int j = 0, q = 0; j < 24; j++, q += 2) {
                for (int k = 0, r = 0; k < 24; k++, r += 2) {
                    pool2_layer[co][i][j][k] =
                        max(0, max(max(max(conv2_layer[p][q][r], conv2_layer[p][q][r + 1]),
                                       max(conv2_layer[p][q + 1][r], conv2_layer[p][q + 1][r + 1])),
                                   max(max(conv2_layer[p + 1][q][r], conv2_layer[p + 1][q][r + 1]),
                                       max(conv2_layer[p + 1][q + 1][r], conv2_layer[p + 1][q + 1][r + 1]))));
                }
            }
        }
    }
output:
    /**
     * 将pool值导出
     */
    {
    }
}
