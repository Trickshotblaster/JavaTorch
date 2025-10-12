

package javatorch;

import java.util.Random;

public class Vec {
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

