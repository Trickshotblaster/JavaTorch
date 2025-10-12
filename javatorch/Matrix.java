package javatorch;
import java.lang.Math;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

public class Matrix {
    public Vec[] data;
    public int[] shape;

    public Matrix(Vec[] data) {
        this.data = data;
        this.shape = new int[] { data.length, data[0].data.length };
    }

    public Matrix(int rows, int cols) {
        this.data = new Vec[rows];
        for (int i = 0; i < rows; i++) {
            this.data[i] = new Vec(cols);
        }
        this.shape = new int[] { rows, cols };
    }

    public int numel() {
        return this.shape[0] * this.shape[1];
    }

    public double[] flatten() {
        double[] out = new double[this.shape[0] * this.shape[1]];
        for (int i = 0; i < this.shape[0]; i++) {
            for (int j = 0; j < this.shape[1]; j++) {
                out[i * this.shape[1] + j] = this.data[i].data[j];
            }
        }
        return out;
    }

    public Matrix view(int rows, int cols) {
        /*
        * if we have 15 elements and shape is 3 x 5,
        * take chunks of 5 and concatenate into a matrix
        */

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
        Matrix out = new Matrix(this.shape[1], this.shape[0]);
        for (int i = 0; i < this.shape[0]; i++) {
            for (int j = 0; j < this.shape[1]; j++) {
                out.data[j].data[i] = this.data[i].data[j];
            }
        }

        return out;
    }
    
    public Matrix matmul(Matrix other) {
        assert this.shape[1] == other.shape[0] : "inner shapes must match for matmul";

        Matrix out = new Matrix(this.shape[0], other.shape[1]);

        Matrix transposed = other.transpose();

        for (int i = 0; i < this.shape[0]; i++) {
            for (int j = 0; j < other.shape[1]; j++) {
                out.data[i].data[j] = this.data[i].dot(transposed.data[j]);
            }

        }

        return out;
    }
    
    public String toString() {
        String out = "{\n";
        for (int i = 0; i < this.shape[0]; i++) {
            out += this.data[i].toString() + ",\n";
        }
        out += "}";
        return out;
    }

    public void _op(DoubleUnaryOperator op) {
        for (int i = 0;  i < this.shape[0]; i++) {
            for (int j = 0; j < this.shape[1]; j++) {
                this.data[i].data[j] = op.applyAsDouble(this.data[i].data[j]);
            }
        }
    }

    public Matrix op(DoubleUnaryOperator op) {
        Matrix out = new Matrix(this.shape[0], this.shape[1]);
        for (int i = 0;  i < this.shape[0]; i++) {
            for (int j = 0; j < this.shape[1]; j++) {
                out.data[i].data[j] = op.applyAsDouble(this.data[i].data[j]);
            }
        }
        return out;
    }
    
    public Matrix relationalOp(Matrix other, DoubleBinaryOperator op) {
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

    public double sum() {
        double sum = 0.0;
        for (Vec row: this.data) {
            for (double val: row.data) {
                sum += val;
            }
        }
        return sum;
    }

    public Matrix sumRows() {
        Matrix out = new Matrix(1, this.shape[0]);
        for (int i = 0; i < this.data.length; i++) {
            Vec row = this.data[i]
            double sum = 0.0;
            for (double val: row.data) {
                sum += val;
            }
            out.data[i].data[0] = sum;
        }
        return out;
    }

    public Matrix sumCols() {
        // galaxy brain activities
        return this.transpose().sumRows();
    }

    public Matrix tanh() {
        return (this.op(x -> (Math.exp(2.* x) - 1) / (Math.exp(2.*x) + 1)));
    }

    public Matrix tanhDerivative() {
        return this.op(x -> (4*Math.exp(2.*x)) / Math.pow(Math.exp(2.*x) + 1, 2));
    }
}
