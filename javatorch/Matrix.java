/*
Matrix class, supports matmul and other functions
*/

package javatorch;

import java.lang.Math;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;
import java.util.Random;

public class Matrix {
    // random for _rand function
    Random random = new Random();
    // need to store a list of values and their shape
    public Vec[] data;
    public int[] shape;

    public Matrix(Vec[] data) {
        // if given an array of vectors, use them as data and their lengths as shape
        this.data = data;
        this.shape = new int[] { data.length, data[0].data.length };
    }

    public Matrix(int rows, int cols) {
        // if given shape, zero-init
        this.data = new Vec[rows];
        for (int i = 0; i < rows; i++) {
            this.data[i] = new Vec(cols);
        }
        this.shape = new int[] { rows, cols };
    }

    public int numel() {
        // count number of elements in the matrix
        return this.shape[0] * this.shape[1];
    }

    public double[] flatten() {
        // squish into 1 dimension
        double[] out = new double[this.shape[0] * this.shape[1]];
        for (int i = 0; i < this.shape[0]; i++) {
            for (int j = 0; j < this.shape[1]; j++) {
                out[i * this.shape[1] + j] = this.data[i].data[j];
            }
        }
        return out;
    }

    public Matrix view(int rows, int cols) {
        // try to reconstruct (m x n) matrix from (mn) data
        Matrix out = new Matrix(rows, cols);

        double[] flattened = this.flatten();

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                out.data[i].data[j] = flattened[i * (cols - 1) + j];
            }
        }

        return out;
    }

    public Matrix transpose() {
        // loop through rows and cols and flip flop
        Matrix out = new Matrix(this.shape[1], this.shape[0]);
        for (int i = 0; i < this.shape[0]; i++) {
            for (int j = 0; j < this.shape[1]; j++) {
                out.data[j].data[i] = this.data[i].data[j];
            }
        }

        return out;
    }

    public Matrix matmul(Matrix other) {
        // if A is (h x i) and B is (j x k), make sure i==j
        assert this.shape[1] == other.shape[0] : "inner shapes must match for matmul";
        // out is (h x k)
        Matrix out = new Matrix(this.shape[0], other.shape[1]);
        // dot product-ing the rows of A with the columns of B is the same as rows of A
        // with rows of B.T
        Matrix transposed = other.transpose();

        // do dot products
        for (int i = 0; i < this.shape[0]; i++) {
            for (int j = 0; j < other.shape[1]; j++) {
                out.data[i].data[j] = this.data[i].dot(transposed.data[j]);
            }
        }

        return out;
    }

    public String toString() {
        // toString for printing
        String out = "{\n";
        for (int i = 0; i < this.shape[0]; i++) {
            out += this.data[i].toString() + ",\n";
        }
        out += "}";
        return out;
    }

    public void _op(DoubleUnaryOperator op) {
        // apply elementwise function in-place
        for (int i = 0; i < this.shape[0]; i++) {
            for (int j = 0; j < this.shape[1]; j++) {
                this.data[i].data[j] = op.applyAsDouble(this.data[i].data[j]);
            }
        }
    }

    public Matrix op(DoubleUnaryOperator op) {
        // apply elementwise function and return new matrix
        Matrix out = new Matrix(this.shape[0], this.shape[1]);
        for (int i = 0; i < this.shape[0]; i++) {
            for (int j = 0; j < this.shape[1]; j++) {
                out.data[i].data[j] = op.applyAsDouble(this.data[i].data[j]);
            }
        }
        return out;
    }

    public void _rand() {
        // randomize the values of a matrix
        for (int i = 0; i < this.shape[0]; i++) {
            for (int j = 0; j < this.shape[1]; j++) {
                this.data[i].data[j] = random.nextDouble() * 2. - 1.;
            }
        }
    }

    public Matrix relationalOp(Matrix other, DoubleBinaryOperator op) {
        // perform elementwise operation between corresponding values of matrix A and B
        // eg A[i, j] = op(A[i, j], B[i, j])
        assert (this.shape[0] == other.shape[0]) && (this.shape[1] == other.shape[1]);
        Matrix out = new Matrix(this.shape[0], this.shape[1]);
        for (int i = 0; i < this.shape[0]; i++) {
            for (int j = 0; j < this.shape[1]; j++) {
                out.data[i].data[j] = op.applyAsDouble(this.data[i].data[j], other.data[i].data[j]);
            }
        }
        return out;
    }

    public Matrix add(Matrix other) {
        return relationalOp(other, (x, y) -> (x + y));
    }

    public Matrix subtract(Matrix other) {
        return relationalOp(other, (x, y) -> (x - y));
    }

    public Matrix multiply(Matrix other) {
        return relationalOp(other, (x, y) -> (x * y));
    }

    public Matrix divide(Matrix other) {
        return relationalOp(other, (x, y) -> (x / y));
    }

    public int argmax1Dim() {
        // just argmax the only vector
        assert this.shape[0] == 1;
        return this.data[0].argmax();
    }

    public double sum() {
        // sum over all elements of the matrix
        double sum = 0.0;
        for (Vec row : this.data) {
            for (double val : row.data) {
                sum += val;
            }
        }
        return sum;
    }

    public Matrix sumRows() {
        // return a row-wise sum of the matrix
        /*
         * eg
         * [ [
         * [0, 5, -2], [3],
         * [-1, 3, 4], => [6],
         * [2, 1, 3], [6],
         * ] ]
         */
        Matrix out = new Matrix(this.shape[0], 1);
        for (int i = 0; i < this.data.length; i++) {
            Vec row = this.data[i];
            double sum = 0.0;
            for (double val : row.data) {
                sum += val;
            }
            out.data[0].data[i] = sum;
        }
        return out;
    }

    public Matrix sumCols() {
        // galaxy brain activities
        return this.transpose().sumRows();
    }

    public Matrix tanh() {
        // shortcut for tanh
        return (this.op(x -> (Math.exp(2. * x) - 1) / (Math.exp(2. * x) + 1)));
    }

    public Matrix tanhDerivative() {
        return this.op(x -> (4 * Math.exp(2. * x)) / Math.pow(Math.exp(2. * x) + 1, 2));
    }
}
