/*
DataLoader class to make loading MNIST data easier
*/

package mnist;

import java.io.FileInputStream;
import java.io.IOException;

import javatorch.Matrix;

public class DataLoader {
    // dataloader attributes
    public String typeOf;
    public String filePath;
    public FileInputStream file;
    public int readSize;
    // image data properties
    public static final int imageSize = 28 * 28;
    public static final int numClasses = 10;

    public DataLoader(String typeOf, String filePath) throws IOException {
        // make sure type is either image or label
        assert typeOf.equals("image") || typeOf.equals("label");
        // set attributes
        this.typeOf = typeOf;
        this.filePath = filePath;
        // load the file
        resetFile();
        // how many bytes to read at a time? image -> 28*28, label -> 1
        this.readSize = this.typeOf.equals("image") ? imageSize : 1;
        // looking at metadata is for cowards; just move 16 bytes forward for idx3 and 8
        // for idx1
        this.file.read(new byte[(this.typeOf.equals("image") ? 16 : 8)]);
    }

    public void resetFile() throws IOException {
        // reset the fileinput stream
        this.file = new FileInputStream(this.filePath);
    }

    public boolean hasNext() throws IOException {
        // is there another example left? have we reached the end?
        return this.file.available() > readSize;
    }

    public void readNextImageTo(byte[] buffer) throws IOException {
        // if there is >= 1 example left, read it to the buffer, otherwise reset
        assert this.typeOf.equals("image");
        if (this.hasNext()) {
            this.file.read(buffer);
        } else {
            resetFile();
            this.file.read(buffer);
        }
    }

    public int nextLabel() throws IOException {
        // if there is an example left, return the label
        assert this.typeOf.equals("label");
        if (this.hasNext()) {
            return (int) this.file.read();
        } else {
            resetFile();
            return (int) this.file.read();
        }
    }

    public void readNextLabelToMatrix(Matrix mat) throws IOException {
        // read to a one-hot vector (1 on correct index else 0)
        assert this.typeOf.equals("label");
        int label = nextLabel();
        for (int i = 0; i < numClasses; i++) {
            mat.data[0].data[i] = (i == label) ? 1. : 0.;
        }
    }

    public void readNextImageToMatrix(Matrix mat) throws IOException {
        // read next example to a byte buffer, then cast to double and normalize
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
