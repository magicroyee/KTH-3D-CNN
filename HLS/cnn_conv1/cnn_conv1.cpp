#include "cnn_conv1.h"  // conv1_filter[16][3][3][3]

#define max(a, b) ((a) > (b) ? (a) : (b))

void cnn_conv1()
{
    static char rowLayer[16 + 2][96 + 2][48 + 2];
    static float conv1_layer[16][96][48];
    static float pool1_layer[16][16][48][48];

input:
    /**
     * 给rowLayer赋值
     */

    for (int c_o = 0; c_o < 16; c_o++) {
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 96; j++) {
                for (int k = 0; k < 48; k++) {
                    conv1_layer[i][j][k] = 0;
                }
            }
        }
    conv1:
        for (int p = 0; p < 3; p++) {
            for (int q = 0; q < 3; q++) {
                for (int r = 0; r < 3; r++) {
                    for (int i = 0; i < 16; i++) {
                        for (int j = 0; j < 96; j++) {
                            for (int k = 0; k < 48; k++) {
                                conv1_layer[i][j][k] += conv1_filter[c_o][p][q][r] * rowLayer[i + p][j + q][k + r];
                            }
                        }
                    }
                }
            }
        }
    pool1:
        for (int i = 0, p = 0; i < 16; i++, p++) {
            for (int j = 0, q = 0; j < 48; j++, q += 2) {
                for (int k = 0, r = 0; k < 48; k++, r++) {
                    pool1_layer[c_o][i][j][k] = max(0, max(conv1_layer[p][q][r], conv1_layer[p][q + 1][r]));
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
