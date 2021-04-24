#include "cnn_conv4.h"  // conv4a_filter[128][64][3][3][3], conv4b_filter[128][128][3][3][3]

#define max(a, b) ((a) > (b) ? (a) : (b))

void cnn_conv4()
{
    static float pool3_layer[64][4 + 2][12 + 2][12 + 2];
    static float conv4_layer[4][12][12];
    static float conv4a_layer[128][4 + 2][12 + 2][12 + 2];
    static float conv4b_layer[128][4][12][12];
    static float pool4_layer[128][2][6][6];

input:
    /**
     * 给pool3_layer赋值
     */

    // conv4a
    for (int co = 0; co < 128; co++) {
    zero:
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 12; j++) {
                for (int k = 0; k < 12; k++) {
                    conv4_layer[i][j][k] = 0;
                }
            }
        }
    conv4a_conv:
        for (int ci = 0; ci < 64; ci++) {
            for (int p = 0; p < 3; p++) {
                for (int q = 0; q < 3; q++) {
                    for (int r = 0; r < 3; r++) {
                        for (int i = 0; i < 4; i++) {
                            for (int j = 0; j < 12; j++) {
                                for (int k = 0; k < 12; k++) {
                                    conv4_layer[i][j][k] +=
                                        conv4a_filter[co][ci][p][q][r] * pool3_layer[ci][i + p][j + q][k + r];
                                }
                            }
                        }
                    }
                }
            }
        }
    conv4a:
        for (int i = 0; i < 4 + 2; i++) {
            for (int j = 0; j < 12 + 2; j++) {
                for (int k = 0; k < 12 + 2; k++) {
                    if (i == 0 && i == 5 && j == 0 && j == 13 && k == 0 && k == 13) {
                        conv4a_layer[co][i][j][k] = 0;
                    } else {
                        conv4a_layer[co][i][j][k] = conv4_layer[i - 1][j - 1][k - 1];
                    }
                }
            }
        }
    }

    // conv4b & pooling
    for (int co = 0; co < 128; co++) {
    zero:
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 12; j++) {
                for (int k = 0; k < 12; k++) {
                    conv4_layer[i][j][k] = 0;
                }
            }
        }
    conv4a_conv:
        for (int ci = 0; ci < 128; ci++) {
            for (int p = 0; p < 3; p++) {
                for (int q = 0; q < 3; q++) {
                    for (int r = 0; r < 3; r++) {
                        for (int i = 0; i < 4; i++) {
                            for (int j = 0; j < 12; j++) {
                                for (int k = 0; k < 12; k++) {
                                    conv4_layer[i][j][k] +=
                                        conv4b_filter[co][ci][p][q][r] * conv4a_layer[ci][i + p][j + q][k + r];
                                }
                            }
                        }
                    }
                }
            }
        }
    pool4:
        for (int i = 0, p = 0; i < 2; i++, p += 2) {
            for (int j = 0, q = 0; j < 6; j++, q += 2) {
                for (int k = 0, r = 0; k < 6; k++, r += 2) {
                    pool4_layer[co][i][j][k] =
                        max(0, max(max(max(conv4_layer[p][q][r], conv4_layer[p][q][r + 1]),
                                       max(conv4_layer[p][q + 1][r], conv4_layer[p][q + 1][r + 1])),
                                   max(max(conv4_layer[p + 1][q][r], conv4_layer[p + 1][q][r + 1]),
                                       max(conv4_layer[p + 1][q + 1][r], conv4_layer[p + 1][q + 1][r + 1]))));
                }
            }
        }
    }
}
