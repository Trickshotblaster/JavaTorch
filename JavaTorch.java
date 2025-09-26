import java.util.Random;

public class JavaTorch {
    public static void main(String[] args) {
        System.out.println("hello world\t\to");
        // double[] data_a = {0.2, 0.3, 1., 2.};
        // double[] data_b = { 0.1, 0.2, 1.3, 2.4 };
        Vec a = new Vec(new double[] { 0.2, 0.3 });
        Vec b = new Vec(new double[] { 0.1, 0.2 });
        double c = a.dot(b);

        Vec d = new Vec(new double[] { 0.1, 0.2, 0.3 });
        Vec e = new Vec(new double[] { -0.3, 0.2, 0.4 });
        Matrix A = new Matrix(new Vec[] { a, b });
        Matrix B = new Matrix(new Vec[] { d, e });
        System.out.println(A.toString());
        System.out.println(B.toString());
        Matrix C = A.matmul(B);
        System.out.println("a dot b:");
        System.out.println(c);
        System.out.println("A x B:");
        System.out.println(C.toString());
    }

}

class Vec {
    public double[] data;
    Random random = new Random();

    public Vec(double[] data) {
        this.data = data;
    }

    public Vec(int length) {
        this.data = new double[length];
        for (int i = 0; i < length; i++) {
            this.data[i] = random.nextDouble() * 2. - 1.;
        }
    }

    public double dot(Vec other) {
        assert this.data.length == other.data.length : "vectors must be of same length for dot product";
        double out = 0.0;

        for (int i = 0; i < this.data.length; i++) {
            out += this.data[i] * other.data[i];
        }
        return out;
    }

    public String toString() {
        String res = "{";
        for (int i = 0; i < this.data.length; i++) {
            res += String.format("%.3f", this.data[i]);
            if (i < this.data.length - 1) {
                res += " ";
            }
        }
        res += "}";
        return res;
    }
}

class Matrix {
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