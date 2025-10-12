package mnist;
import javatorch.*;
import java.io.*;
import java.nio.ByteBuffer;

public class MNIST {
    public static String pathToMNIST = "data/train-images.idx3-ubyte";
    public static String pathToMNISTLabels = "data/train-labels.idx1-ubyte";
    public static FileInputStream mnistFile;
    public static FileInputStream mnistLabelFile;
    public static int rows = 28;
    public static int cols = 28;
    public static int size = rows*cols;
    public static int numClasses = 10;
    public static void main(String[] args) throws IOException {
        init();
        for (int num = 0; num < 5; num++) {
            byte[] buffer = new byte[size];
            readNextImageTo(buffer);
            showImage(buffer);
            System.out.println(nextLabel());
        }

        closeFiles();
    }

    public static void showImage(byte[] buffer) {
        for (int i = 0; i < 28; i++) {
                
            for (int j = 0; j < 28; j++) {
                
                System.out.printf("%4d", buffer[i * 28 + j] & 0xFF);
            }
            System.out.println();
        }
    }

    public static void init() throws IOException {
        mnistFile = new FileInputStream(pathToMNIST);
        mnistLabelFile = new FileInputStream(pathToMNISTLabels);
        mnistFile.read(new byte[16]); // read metadata
        mnistLabelFile.read(new byte[8]);
    }

    public static void readNextImageTo(byte[] buffer) throws IOException {
        if (mnistFile.available() >= buffer.length) {
            mnistFile.read(buffer);
        } else {
            init();
            mnistFile.read(buffer);
        }
    }

    public static int nextLabel() throws IOException {
        if (mnistLabelFile.available() >= 1) {
            return (int) mnistLabelFile.read();
        } else {
            init();
            return (int) mnistLabelFile.read();
        }
    }

    public static Matrix nextLabelOneHot() throws IOException {
        Matrix out = new Matrix(1, numClasses);
        int label = nextLabel();
        for (int i = 0; i < numClasses; i++) {
            out.data[0].data[i] = (i==label) ? 1. : 0.;
        }
        return out;
    }

    public static void readNextImageToMatrix(Matrix mat) throws IOException {
        assert (mat.shape[0] == 1) && (mat.shape[1] == size) : String.format("Matrix size must be (1x%d)", size);
        byte[] byteBuf = new byte[size];
        readNextImageTo(byteBuf);
        for (int i=0 ; i<size; i++) {
            mat.data[0].data[i] = (double) (byteBuf[i] & 0xFF) / 256;
        }
    }

    public static void closeFiles() throws IOException {
        mnistFile.close();
        mnistLabelFile.close();
    }
}