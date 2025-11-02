/*
Vector class for all your vector needs
*/
package javatorch;

public class Vec {
    // store an array of doubles
    public double[] data;

    public Vec(double[] data) {
        // if we are given data to use, use it
        this.data = data;
    }

    public Vec(int length) {
        // if we are given a length, zero-init
        this.data = new double[length];
    }

    public double dot(Vec other) {
        // dot_prod = sum(this[i] * other[i])
        assert this.data.length == other.data.length : "vectors must be of same length for dot product";
        double out = 0.0;

        for (int i = 0; i < this.data.length; i++) {
            out += this.data[i] * other.data[i];
        }
        return out;
    }

    public int argmax() {
        // simple argmax impl, starting with first value as max
        double max = this.data[0];
        int idx = 0;
        for (int i = 1; i < this.data.length; i++) {
            if (this.data[i] > max) {
                max = this.data[i];
                idx = i;
            }
        }
        return idx;
    }

    public boolean equals(Vec other) {
        if (other.data.length != this.data.length)
            return false;
        for (int i = 0; i < this.data.length; i++) {
            if (this.data[i] != other.data[i])
                return false;
        }
        return true;
    }

    public String toString() {
        // toString method for printing
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
