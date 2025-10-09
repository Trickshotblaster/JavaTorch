import javatorch.*;
import java.util.Scanner;
import java.io.*;

public class MNIST {
    public static void main(String[] args) throws IOException {
        String pathToMNIST = "data/train-images.idx3-ubyte";
        
        
        FileInputStream mnistFile = new FileInputStream(pathToMNIST);
        for (int i = 0; i < 16; i++) {
            System.out.println(mnistFile.read());
        }

        for (int num = 0; num < 5; num++) {
            for (int i = 0; i < 28; i++) {
                for (int j = 0; j < 28; j++) {
                    System.out.printf("%4d", mnistFile.read());
                }
                System.out.println();
            }
        }
        
        mnistFile.close();
        

    }


}