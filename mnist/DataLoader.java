package mnist;

import java.io.FileInputStream;
import java.io.IOException;

import javatorch.Matrix;

public class DataLoader {
    public String typeOf;
    public String filePath;
    public FileInputStream file;
    public static int imageSize = 28 * 28;
    public static int numClasses = 10;
    public int readSize;

    public DataLoader(String typeOf, String filePath) throws IOException {
        this.typeOf = typeOf;
        this.filePath = filePath;
        resetFile();
        this.readSize = this.typeOf.equals("image") ? imageSize : 1;
        this.file.read(new byte[(this.typeOf.equals("image") ? 16 : 8)]);
    }

    public void resetFile() throws IOException {
        this.file = new FileInputStream(this.filePath);
    }

    public boolean hasNext() throws IOException {
        return this.file.available() > readSize;
    }

    public void readNextImageTo(byte[] buffer) throws IOException {
        assert this.typeOf.equals("image");
        if (this.hasNext()) {
            this.file.read(buffer);
        } else {
            resetFile();
            this.file.read(buffer);
        }
    }

    public int nextLabel() throws IOException {
        assert this.typeOf.equals("label");
        if (this.hasNext()) {
            return (int) this.file.read();
        } else {
            resetFile();
            return (int) this.file.read();
        }
    }

    public void readNextLabelToMatrix(Matrix mat) throws IOException {
        assert this.typeOf.equals("label");
        int label = nextLabel();
        for (int i = 0; i < numClasses; i++) {
            mat.data[0].data[i] = (i == label) ? 1. : 0.;
        }
    }

    public void readNextImageToMatrix(Matrix mat) throws IOException {
        assert this.typeOf.equals("image");
        assert (mat.shape[0] == 1) && (mat.shape[1] == imageSize)
                : String.format("Matrix size must be (1x%d)", imageSize);
        byte[] byteBuf = new byte[imageSize];
        readNextImageTo(byteBuf);
        for (int i = 0; i < imageSize; i++) {
            mat.data[0].data[i] = (double) (byteBuf[i] & 0xFF) / 256;
        }
    }

    public void closeFile() throws IOException {
        this.file.close();
    }
}
