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

}