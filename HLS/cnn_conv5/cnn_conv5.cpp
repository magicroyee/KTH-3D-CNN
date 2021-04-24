#include "cnn_conv5.h"  // conv5a_filter[256][128][3][3][3], conv5b_filter[256][256][3][3][3]

#define max(a, b) ((a) > (b) ? (a) : (b))

void cnn_conv5()
{
    static float pool4_layer[128][2 + 2][6 + 2][6 + 2];
    static float conv5_layer[2][6][6];
    static float conv5a_layer[256][2 + 2][6 + 2][6 + 2];
    static float conv5b_layer[256][2][6][6];
    static float pool5_layer[256][1][3][3];

input:
    /**
     * 给pool4_layer赋值
     */

    // conv5a
    for (int co = 0; co < 256; co++) {
    zero:
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 6; j++) {
                for (int k = 0; k < 6; k++) {
                    conv5_layer[i][j][k] = 0;
                }
            }
        }
    conv4a_conv:
        for (int ci = 0; ci < 128; ci++) {
            for (int p = 0; p < 3; p++) {
                for (int q = 0; q < 3; q++) {
                    for (int r = 0; r < 3; r++) {
                        for (int i = 0; i < 2; i++) {
                            for (int j = 0; j < 6; j++) {
                                for (int k = 0; k < 6; k++) {
                                    conv5_layer[i][j][k] +=
                                        conv5a_filter[co][ci][p][q][r] * pool4_layer[ci][i + p][j + q][k + r];
                                }
                            }
                        }
                    }
                }
            }
        }
    conv5a:
        for (int i = 0; i < 2 + 2; i++) {
            for (int j = 0; j < 6 + 2; j++) {
                for (int k = 0; k < 6 + 2; k++) {
                    if (i == 0 && i == 3 && j == 0 && j == 7 && k == 0 && k == 7) {
                        conv5a_layer[co][i][j][k] = 0;
                    } else {
                        conv5a_layer[co][i][j][k] = conv5_layer[i - 1][j - 1][k - 1];
                    }
                }
            }
        }
    }

    // conv5b & pooling
    for (int co = 0; co < 256; co++) {
    zero:
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 6; j++) {
                for (int k = 0; k < 6; k++) {
                    conv5_layer[i][j][k] = 0;
                }
            }
        }
    conv5b_conv:
        for (int ci = 0; ci < 256; ci++) {
            for (int p = 0; p < 3; p++) {
                for (int q = 0; q < 3; q++) {
                    for (int r = 0; r < 3; r++) {
                        for (int i = 0; i < 2; i++) {
                            for (int j = 0; j < 6; j++) {
                                for (int k = 0; k < 6; k++) {
                                    conv5_layer[i][j][k] +=
                                        conv5b_filter[co][ci][p][q][r] * conv5a_layer[ci][i + p][j + q][k + r];
                                }
                            }
                        }
                    }
                }
            }
        }
    pool5:
        for (int i = 0, p = 0; i < 1; i++, p += 2) {
            for (int j = 0, q = 0; j < 3; j++, q += 2) {
                for (int k = 0, r = 0; k < 3; k++, r += 2) {
                    pool4_layer[co][i][j][k] =
                        max(0, max(max(max(conv5_layer[p][q][r], conv5_layer[p][q][r + 1]),
                                       max(conv5_layer[p][q + 1][r], conv5_layer[p][q + 1][r + 1])),
                                   max(max(conv5_layer[p + 1][q][r], conv5_layer[p + 1][q][r + 1]),
                                       max(conv5_layer[p + 1][q + 1][r], conv5_layer[p + 1][q + 1][r + 1]))));
                }
            }
        }
    }
}
