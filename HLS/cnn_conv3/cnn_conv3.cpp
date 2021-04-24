#include "cnn_conv3.h"  // conv2_filter[32][16][3][3][3]

#define max(a, b) ((a) > (b) ? (a) : (b))

void cnn_conv3()
{
    static float pool2_layer[32][8 + 2][24 + 2][24 + 2];
    static float conv3_layer[8][24][24];
    static float conv3a_layer[64][8 + 2][24 + 2][24 + 2];
    static float conv3b_layer[64][8][24][24];
    static float pool3_layer[64][4][12][12];

input:
    /**
     * 给pool2_layer赋值
     */

    // conv3a
    for (int co = 0; co < 64; co++) {
    zero:
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 24; j++) {
                for (int k = 0; k < 24; k++) {
                    conv3_layer[i][j][k] = 0;
                }
            }
        }
    conv3a_conv:
        for (int ci = 0; ci < 32; ci++) {
            for (int p = 0; p < 3; p++) {
                for (int q = 0; q < 3; q++) {
                    for (int r = 0; r < 3; r++) {
                        for (int i = 0; i < 8; i++) {
                            for (int j = 0; j < 24; j++) {
                                for (int k = 0; k < 24; k++) {
                                    conv3_layer[i][j][k] +=
                                        conv3a_filter[co][ci][p][q][r] * pool2_layer[ci][i + p][j + q][k + r];
                                }
                            }
                        }
                    }
                }
            }
        }
    conv3a:
        for (int i = 0; i < 8 + 2; i++) {
            for (int j = 0; j < 24 + 2; j++) {
                for (int k = 0; k < 24 + 2; k++) {
                    if (i == 0 && i == 9 && j == 0 && j == 25 && k == 0 && k == 25) {
                        conv3a_layer[co][i][j][k] = 0;
                    } else {
                        conv3a_layer[co][i][j][k] = conv3_layer[i - 1][j - 1][k - 1];
                    }
                }
            }
        }
    }

    // conv3b & pooling
    for (int co = 0; co < 64; co++) {
    zero:
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 24; j++) {
                for (int k = 0; k < 24; k++) {
                    conv3_layer[i][j][k] = 0;
                }
            }
        }
    conv3a_conv:
        for (int ci = 0; ci < 64; ci++) {
            for (int p = 0; p < 3; p++) {
                for (int q = 0; q < 3; q++) {
                    for (int r = 0; r < 3; r++) {
                        for (int i = 0; i < 8; i++) {
                            for (int j = 0; j < 24; j++) {
                                for (int k = 0; k < 24; k++) {
                                    conv3_layer[i][j][k] +=
                                        conv3b_filter[co][ci][p][q][r] * conv3a_layer[ci][i + p][j + q][k + r];
                                }
                            }
                        }
                    }
                }
            }
        }
    pool3:
        for (int i = 0, p = 0; i < 4; i++, p += 2) {
            for (int j = 0, q = 0; j < 12; j++, q += 2) {
                for (int k = 0, r = 0; k < 12; k++, r += 2) {
                    pool3_layer[co][i][j][k] =
                        max(0, max(max(max(conv3_layer[p][q][r], conv3_layer[p][q][r + 1]),
                                       max(conv3_layer[p][q + 1][r], conv3_layer[p][q + 1][r + 1])),
                                   max(max(conv3_layer[p + 1][q][r], conv3_layer[p + 1][q][r + 1]),
                                       max(conv3_layer[p + 1][q + 1][r], conv3_layer[p + 1][q + 1][r + 1]))));
                }
            }
        }
    }
}
