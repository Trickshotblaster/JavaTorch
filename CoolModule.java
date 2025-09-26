public class CoolModule {
    public static void main(String[] args) {
        System.out.println("hello world\t\to");
        // double[] data_a = {0.2, 0.3, 1., 2.};
        // double[] data_b = { 0.1, 0.2, 1.3, 2.4 };
        Vec a = new Vec(new double[] { 0.2, 0.3, 1., 2. });
        Vec b = new Vec(new double[] { 0.1, 0.2, 1.3, 2.4 });
        Vec c = a.dot(b);
        System.out.println(c.toString());
    }

}

class Vec {
    public double[] data;

    public Vec(double[] data) {
        this.data = data;
    }

    public Vec dot(Vec other) {
        Vec out = new Vec(this.data);
        for (int i = 0; i < data.length; i++) {
            out.data[i] = this.data[i] * other.data[i];
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