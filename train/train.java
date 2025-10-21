package train;
/*
Training code: trains a 2-layer MLP on the MNIST dataset with specified hyperparams
*/

import java.io.IOException;

import javatorch.*;
import mnist.*;

public class train {
    // image data characteristics
    public static final int rows = 28;
    public static final int cols = 28;
    public static final int size = rows * cols;
    public static final int numClasses = 10;
    // hyperparams
    public static final int hiddenDim = 20;
    public static double lr = .8;
    public static double lr_final = 0.02;
    public static int numSteps = 2000;
    public static int gradAccumSteps = 16;
    // how often to log results
    public static int logEvery = 100;
    // parameters
    public static Matrix w1;
    public static Matrix w2;
    public static Matrix b;
    // train dataloaders
    public static DataLoader trainX;
    public static DataLoader trainY;
    // validation dataloaders
    public static DataLoader valX;
    public static DataLoader valY;

    public static void main(String[] args) throws IOException {
        run();
        // show the fruits of our labor in ascii form
        for (int i = 0; i < 5; i++) {
            showImagePredictionPair();
        }

        // print out final accuracy
        System.out.printf("Validation Accuracy: %6.3f\n", getValAccuracy());

        // close all the files to be nice
        trainX.closeFile();
        trainY.closeFile();
        valX.closeFile();
        valY.closeFile();
    }

    public static void run() throws IOException {
        // build model, assign empty matrices to parameters then randomize
        w1 = new Matrix(size, hiddenDim);
        w2 = new Matrix(hiddenDim, numClasses);
        b = new Matrix(1, hiddenDim);

        w1._rand();
        w2._rand();
        b._rand();

        // do some normalization tricks (divide by sqrt(fan_in) and multiply by gain
        // (5/3 for tanh))
        // this is VERY IMPORTANT (accuracy is like 0.6 without)
        w1._op(k -> k / Math.sqrt(size) * 5. / 3.);
        w2._op(k -> k / Math.sqrt(hiddenDim));
        b._op(k -> k * 0.1);

        // get data (see mnist/DataLoader.java)
        trainX = new DataLoader("image", "data/train-images.idx3-ubyte");
        trainY = new DataLoader("label", "data/train-labels.idx1-ubyte");

        valX = new DataLoader("image", "data/t10k-images.idx3-ubyte");
        valY = new DataLoader("label", "data/t10k-labels.idx1-ubyte");

        // training loop
        for (int step = 0; step < numSteps; step++) {
            // initialize empty gradients for grad accum
            Matrix gradw1 = new Matrix(w1.shape[0], w1.shape[1]);
            Matrix gradw2 = new Matrix(w2.shape[0], w2.shape[1]);
            Matrix gradb = new Matrix(b.shape[0], b.shape[1]);
            // init accumulated loss
            double lossAccum = 0.0;
            // keep step time
            double t0 = System.nanoTime();
            for (int gradAccumStep = 0; gradAccumStep < gradAccumSteps; gradAccumStep++) {
                // get (x, y) pair from dataloader
                Matrix x = new Matrix(1, size); // [1, 784]
                trainX.readNextImageToMatrix(x);
                Matrix y = new Matrix(1, numClasses); // [1, 10]
                trainY.readNextLabelToMatrix(y);

                // apply the first linear layer to x, store for gradient calculation
                Matrix l1preact = x.matmul(w1); // [1, 784] x [784, 20] => [1, 20]

                // add bias
                Matrix l1biased = l1preact.add(b); // [1, 20]

                // apply actication function
                Matrix l1 = l1biased.tanh(); // [1, 20]

                // apply second linear layer (no sigmoid after because i'm lazy)
                Matrix out = l1.matmul(w2); // [1, 20] x [20, 10] => [1, 10]

                // calculate loss
                // l_mse = 1/n sum[(y-out)^2]
                Matrix diff = y.subtract(out); // [1, 10]
                // op() applies an elementwise operation to the matrix, in this case squaring
                Matrix diff2 = diff.op(k -> Math.pow(k, 2)); // [1, 10]
                // average
                double loss = diff2.sum() / diff2.numel(); // [1, 10]

                // gradients
                // d<variable> is shorthand for gradient of loss with respect to variable
                // L(diff2) = 1/n * diff2 -> dL/ddiff2 = 1/n everywhere in shape of diff2
                // since matrices zero init, this can be done with an elementwise op
                Matrix ddiff2 = new Matrix(diff2.shape[0], diff2.shape[1]).op(k -> k + 1. / diff2.numel()); // [1, 10]
                // L(diff2, diff) = diff2(diff) -> dL/ddiff = dL/ddiff2 * ddiff2/ddiff = ddiff2
                // * 2
                Matrix ddiff = ddiff2.multiply(diff.op(k -> k * 2.)); // [1, 10]
                // L(diff2, diff, out) = diff2(diff(out)) -> dL/dout = dL/ddiff * ddiff/dout =
                // ddiff * -1
                Matrix dout = ddiff.op(k -> -k); // [1, 10]

                // out = (l1 @ w2)
                // dL/dl1 = dL/dout * dout/dl1 = dout * w2.T (just try stuff until the shapes
                // work)
                Matrix dl1 = dout.matmul(w2.transpose()); // [1, 20]
                // dL/dw2 = dL/dout * dout/dw2 = l1 * dout (again just make the shapes work)
                Matrix dw2 = l1.transpose().matmul(dout);

                // dL/dl1biased = dL/dl1 * dl1/dl1biased = dl1 * tanhDerivative(l1biased)
                Matrix dl1biased = dl1.multiply(l1biased.tanhDerivative());
                // dL/db = dL/dl1biased * dl1biased/db = dl1biased * 1
                Matrix db = dl1biased;

                // dL/dl1preact = dL/dl1biased * dl1biased/dl1preact = dl1 * 1
                Matrix dl1preact = dl1biased; // [1, 20]

                // dL/dw1 = dL/dl1preact * dl1preact/dw1 = x.T * dl1preact
                Matrix dw1 = x.transpose().matmul(dl1preact); // [784, 20]
                // accumulate gradients
                gradw1 = gradw1.add(dw1.op(k -> k / gradAccumSteps));
                gradw2 = gradw2.add(dw2.op(k -> k / gradAccumSteps));
                gradb = gradb.add(db.op(k -> k / gradAccumSteps));
                // accumulate loss
                lossAccum += loss / gradAccumSteps;
            }
            // lr schedule
            lr = lr_final + (lr - lr_final) * (1 + Math.cos((step+1)*Math.PI / numSteps)) / (1 + Math.cos(step*Math.PI / numSteps));
            // parameter updates
            w1 = w1.subtract(gradw1.op(k -> k * lr));
            w2 = w2.subtract(gradw2.op(k -> k * lr));
            b = b.subtract(gradb.op(k -> k * lr));
            // log
            if (step % logEvery == 0) {
                System.out.printf("step %10d | loss %10.4f | time %10.4fs | lr %10.4f |\n", step, lossAccum,
                        (System.nanoTime() - t0) / 1e+9, lr);
            }
            
        }

        
    }

    public static int getPrediction(Matrix in) {
        // runs a forward pass of the model on x and returns the highest probability
        // output
        Matrix l1preact = in.matmul(w1); // [1, 784] x [784, 20] => [1, 20]
        // add bias
        Matrix l1biased = l1preact.add(b); // [1, 20]

        // apply actication function
        Matrix l1 = l1biased.tanh(); // [1, 20]
        Matrix out = l1.matmul(w2); // [1, 20] x [20, 10] => [1, 10]
        return out.argmax1Dim();
    }

    public static Matrix getProbs(Matrix in) {
        // output
        Matrix l1preact = in.matmul(w1); // [1, 784] x [784, 20] => [1, 20]
        // add bias
        Matrix l1biased = l1preact.add(b); // [1, 20]

        // apply actication function
        Matrix l1 = l1biased.tanh(); // [1, 20]
        Matrix out = l1.matmul(w2); // [1, 20] x [20, 10] => [1, 10]
        Matrix exponentiated = out.op(k -> Math.exp(k));
        return exponentiated.op(j -> j / exponentiated.sum());
    }

    public static Matrix getOutput(Matrix in) {
        // runs a forward pass of the model on x and returns the highest probability
        // output
        Matrix l1preact = in.matmul(w1); // [1, 784] x [784, 20] => [1, 20]
        // add bias
        Matrix l1biased = l1preact.add(b); // [1, 20]

        // apply actication function
        Matrix l1 = l1biased.tanh(); // [1, 20]
        Matrix out = l1.matmul(w2); // [1, 20] x [20, 10] => [1, 10]
        return out;
    }

    public static void showImagePredictionPair() throws IOException {
        // get an input image (from bytes)
        Matrix x = new Matrix(1, size);
        byte[] xbuf = new byte[size];
        valX.readNextImageTo(xbuf);
        // normalize and assign to x
        for (int i = 0; i < size; i++) {
            x.data[0].data[i] = (double) (xbuf[i] & 0xFF) / 256;
        }
        // get the label
        int label = valY.nextLabel();
        // predict on x and display results
        int pred = getPrediction(x);
        MNIST.showImageAscii(xbuf);
        System.out.println(getProbs(x).toString());
        System.out.println(getOutput(x).toString());
        System.out.printf("| REAL: %d | \t | PREDICTED: %d |\n", label, pred);
    }

    public static double getValAccuracy() throws IOException {
        // initialize count and sum to 0
        double sum = 0;
        int count = 0;
        // loop through validation set and add 1 to sum only if correct
        while (valX.hasNext()) {
            count++;
            Matrix x = new Matrix(1, size);
            valX.readNextImageToMatrix(x);
            sum += (getPrediction(x) == valY.nextLabel()) ? 1. : 0.;
        }
        // get average
        return sum / count;
    }
}