package javatorch;


public class Matrix {
    Vec[] data;
    int[] shape;

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
}
