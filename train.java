import java.io.IOException;

import javatorch.*;
import mnist.*;

public class train {
    public static int rows = 28;
    public static int cols = 28;
    public static int size = rows * cols;
    public static int numClasses = 10;

    public static int hiddenDim = 20;
    public static double lr = 0.1;
    public static int numSteps = 2000;
    public static int gradAccumSteps = 8;

    public static int logEvery = 100;

    static Matrix w1;
    static Matrix w2;

    public static DataLoader trainX;
    public static DataLoader trainY;

    public static DataLoader valX;
    public static DataLoader valY;

    public static void main(String[] args) throws IOException {
        // build model
        w1 = new Matrix(size, hiddenDim);
        w2 = new Matrix(hiddenDim, numClasses);
        w1._rand();
        w2._rand();

        // get data
        trainX = new DataLoader("image", "data/train-images.idx3-ubyte");
        trainY = new DataLoader("label", "data/train-labels.idx1-ubyte");

        valX = new DataLoader("image", "data/t10k-images.idx3-ubyte");
        valY = new DataLoader("label", "data/t10k-labels.idx1-ubyte");

        for (int step = 0; step < numSteps; step++) {
            Matrix gradw1 = new Matrix(w1.shape[0], w1.shape[1]);
            Matrix gradw2 = new Matrix(w2.shape[0], w2.shape[1]);
            double lossAccum = 0.0;
            double t0 = System.nanoTime();
            for (int gradAccumStep = 0; gradAccumStep < gradAccumSteps; gradAccumStep++) {
                Matrix x = new Matrix(1, size); // 1, 784
                trainX.readNextImageToMatrix(x);
                Matrix y = new Matrix(1, numClasses);
                trainY.readNextLabelToMatrix(y); // 1, 10

                Matrix l1preact = x.matmul(w1); // 1, 784 x 784, 20 => 1, 20
                Matrix l1 = l1preact.tanh(); // 1, 20

                Matrix out = l1.matmul(w2); // 1, 20 x 20, 10 => 1, 10
                Matrix diff = y.subtract(out); // 1, 10

                Matrix diff2 = diff.op(k -> Math.pow(k, 2)); // 1, 10
                double loss = diff2.sum() / diff2.numel(); // 1, 10

                // gradients
                Matrix ddiff2 = new Matrix(diff2.shape[0], diff2.shape[1]).op(k -> k + 1. / diff2.numel()); // 1, 10
                Matrix ddiff = ddiff2.multiply(diff.op(k -> k * 2.)); // 1, 10
                Matrix dout = ddiff.op(k -> -k); // 1, 10
                // dl1 should be 1, 20
                // w2 is 20, 10 -- transposed is 10, 20
                // dout is 1, 10
                Matrix dl1 = dout.matmul(w2.transpose()); // 1, 20
                // need 20, 10
                // dout is 1, 10
                Matrix dw2 = l1.transpose().matmul(dout);

                Matrix dl1preact = dl1.tanhDerivative().multiply(dl1); // 1, 20
                // x is 1, 784
                // dl1preact is 1, 20
                Matrix dw1 = x.transpose().matmul(dl1preact); // 784 * 20

                gradw1 = gradw1.add(dw1.op(k -> k / gradAccumSteps));
                gradw2 = gradw2.add(dw2.op(k -> k / gradAccumSteps));

                lossAccum += loss / gradAccumSteps;
            }
            w1 = w1.subtract(gradw1.op(k -> k * lr));
            w2 = w2.subtract(gradw2.op(k -> k * lr));
            if (step % logEvery == 0) {
                System.out.printf("step %10d | loss %10.4f | time %10.4fs\n", step, lossAccum,
                        (System.nanoTime() - t0) / 1e+9);
            }
        }

        for (int i = 0; i < 5; i++) {
            showImagePredictionPair();
        }

        System.out.printf("Validation Accuracy: %6.3f\n", getValAccuracy());
        trainX.closeFile();
        trainY.closeFile();
        valX.closeFile();
        valY.closeFile();
    }

    public static int getPrediction(Matrix x) {
        Matrix l1preact = x.matmul(w1); // 1, 784 x 784, 20 => 1, 20
        Matrix l1 = l1preact.tanh(); // 1, 20

        Matrix out = l1.matmul(w2); // 1, 20 x 20, 10 => 1, 10
        return out.argmax1Dim();
    }

    public static void showImagePredictionPair() throws IOException {
        Matrix x = new Matrix(1, size);
        byte[] xbuf = new byte[size];
        trainX.readNextImageTo(xbuf);
        for (int i = 0; i < size; i++) {
            x.data[0].data[i] = (double) (xbuf[i] & 0xFF) / 256;
        }

        int label = trainY.nextLabel();

        int pred = getPrediction(x);
        MNIST.showImageAscii(xbuf);
        System.out.printf("| REAL: %d | \t | Predicted: %d |\n", label, pred);
    }

    public static double getValAccuracy() throws IOException {
        double sum = 0;
        int count = 0;
        while (valX.hasNext()) {
            count++;
            Matrix x = new Matrix(1, size);
            valX.readNextImageToMatrix(x);
            sum += (getPrediction(x) == valY.nextLabel()) ? 1. : 0.;
        }
        return sum / count;
    }
}