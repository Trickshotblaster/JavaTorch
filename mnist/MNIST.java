package mnist;
import javatorch.*;
import java.io.*;

public class MNIST {
    public static String pathToMNIST;
    public static FileInputStream mnistFile;
    public static int rows;
    public static int cols;
    public static int size;
    public static void main(String[] args) throws IOException {
        init();
        for (int num = 0; num < 5; num++) {
            byte[] buffer = new byte[size];
            readNextImageTo(buffer);
            showImage(buffer);
        }
        closeFile();
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
        pathToMNIST = "data/train-images.idx3-ubyte";
        
        mnistFile = new FileInputStream(pathToMNIST);
        mnistFile.read(new byte[16]); // read metadata

        rows = 28;
        cols = 28;
        size = rows*cols;
        
    }

    public static void readNextImageTo(byte[] buffer) throws IOException {
        if (mnistFile.available() >= buffer.length) {
            mnistFile.read(buffer);
        } else {
            init();
        }
    }

    public static void closeFile() throws IOException {
        mnistFile.close();
    }
}