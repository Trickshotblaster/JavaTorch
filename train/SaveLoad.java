package train;

import javatorch.*;
import java.io.PrintStream;
import java.io.File;
import java.util.Scanner;
import java.io.FileNotFoundException;
import java.io.IOException;

public class SaveLoad {
    public static void saveToFile(Matrix mat, String filepath) throws FileNotFoundException, IOException {
        // save to file
        File outFile = new File(filepath);
        outFile.createNewFile();
        PrintStream out = new PrintStream(outFile);
        out.print(mat.formatSave());
        out.close();
    }

    public static Matrix readFromFile(String filepath) throws FileNotFoundException {
        // read saved
        Scanner reader = new Scanner(new File(filepath));
        int rows = reader.nextInt();
        int cols = reader.nextInt();
        Matrix out = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                out.data[i].data[j] = reader.nextDouble();
            }
        }
        reader.close();
        return out;
    }
}
