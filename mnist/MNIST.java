/*
 Some helpful utility functions for displaying MNIST images
*/

package mnist;

import javatorch.*;

public class MNIST {
    // image data properties
    public static final int rows = 28;
    public static final int cols = 28;
    public static final int size = rows * cols;
    // value scale for displaying ascii images
    public static String valueScale = " .:1S#";

    public static void showImage(byte[] buffer) {
        // loop through rows and columns
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // print out the actual numerical value
                System.out.printf("%4d", buffer[i * cols + j] & 0xFF);
            }
            System.out.println();
        }
    }

    public static void showImageAscii(byte[] buffer) {
        // loop through rows and columns
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // use normalized value to index value scale, print out result with spacing
                System.out.printf("%2s",
                        valueScale.charAt((int) ((((buffer[i * 28 + j] & 0xFF)) / 256.) * valueScale.length())));
            }
            System.out.println();
        }
    }

    public static void showImageMatrix(Matrix buffer) {
        // loop through rows and columns
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // use normalized value to index value scale, print out result with spacing
                System.out.printf("%4.1f", buffer.data[0].data[i * 28 + j]);
            }
            System.out.println();
        }
    }

    public static void showImageMatrixAscii(Matrix buffer) {
        // loop through rows and columns
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // use normalized value to index value scale, print out result with spacing
                System.out.printf("%2s",
                        valueScale.charAt((int) ((buffer.data[0].data[i * 28 + j]) * (valueScale.length() - 1))));
            }
            System.out.println();
        }
    }

    public static Matrix cropMatrix(Matrix buffer) {
        assert buffer.shape[0] == 1 && buffer.shape[1] == size;
        Matrix out = buffer.clone();
        int farLeft = -1;
        int farRight = -1;
        int top = -1;
        int bottom = -1;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double val = out.data[0].data[i * cols + j];
                top = (top == -1 && val > 0) ? i : top;
                farLeft = (farLeft == -1 && val > 0) ? j : farLeft;
                farRight = (farRight == -1 && val > 0) ? j : farRight;
                bottom = (bottom == -1 && val > 0) ? i : bottom;

                farLeft = (j < farLeft && val > 0) ? j : farLeft;
                farRight = (j > farRight && val > 0) ? j : farRight;
                bottom = (i > bottom && val > 0) ? i : bottom;
            }
        }
        int rightShift = (int) Math.round(((cols - farRight) - farLeft) / 2.);
        int bottomShift = (int) Math.round(((rows - bottom) - top) / 2.);

        if (rightShift > 0) {
            for (int i = 0; i < rows; i++) {
                for (int j = cols - 1; j > rightShift; j--) {
                    out.data[0].data[i * cols + j] = out.data[0].data[i * cols + j - rightShift];
                }
                for (int k = 0; k <= rightShift; k++) {
                    out.data[0].data[i * cols + k] = 0.;
                }
            }
        } else {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols - Math.abs(rightShift) - 1; j++) {
                    out.data[0].data[i * cols + j] = out.data[0].data[i * cols + j + Math.abs(rightShift)];
                }
                for (int k = cols - 1; k >= cols - Math.abs(rightShift) - 1; k--) {
                    out.data[0].data[i * cols + k] = 0.;
                }
            }
        }

        if (bottomShift > 0) {
            for (int i = rows - 1; i > bottomShift; i--) {
                for (int j = 0; j < cols; j++) {
                    out.data[0].data[i * cols + j] = out.data[0].data[(i - bottomShift) * cols + j];
                }
            }
            for (int i = 0; i <= bottomShift; i++) {
                for (int j = 0; j < cols; j++) {
                    out.data[0].data[i * cols + j] = 0.;
                }
            }
        } else {
            for (int i = 0; i < cols - Math.abs(bottomShift); i++) {
                for (int j = 0; j < cols; j++) {
                    out.data[0].data[i * cols + j] = out.data[0].data[(i + Math.abs(bottomShift)) * cols + j];
                }
            }
            for (int i = cols - 1; i >= cols - Math.abs(bottomShift); i--) {
                for (int j = 0; j < cols; j++) {
                    out.data[0].data[i * cols + j] = 0.;
                }
            }
        }
        return out;
    }
}