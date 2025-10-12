import java.io.IOException;
import javatorch.*;
import mnist.*;

public class train {
    public static int rows = 28;
    public static int cols = 28;
    public static int size = rows*cols;
    public static int numClasses = 10;

    public static int hiddenDim = 20;
    public static double lr = 3e-4;
    public static int numSteps = 1000;
    public static int gradAccumSteps = 4;

    public static void main(String[] args) throws IOException {
        // build model
        Matrix w1 = new Matrix(size, hiddenDim);
        Matrix w2 = new Matrix(hiddenDim, numClasses);

        // get data
        MNIST.init();
        
        for (int step = 0; step < numSteps; step++) {
            double t0 = System.nanoTime();
            Matrix x = new Matrix(1, size);
            MNIST.readNextImageToMatrix(x);
            Matrix y = MNIST.nextLabelOneHot();

            Matrix l1preact = x.matmul(w1);
            Matrix l1 = l1preact.tanh();

            Matrix out = l1.matmul(w2);
            Matrix diff = y.subtract(out);

            Matrix diff2 = diff.op(k -> Math.pow(k, 2));
            double loss = diff2.sum() / diff2.numel();
            System.out.printf("step %10d | loss %10.4f | time %10.4fs\n", step, loss, (System.nanoTime() - t0) / 1e+9);
        }
        

    }
} 