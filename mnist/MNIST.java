package mnist;

public class MNIST {
    public static int rows = 28;
    public static int cols = 28;
    public static int size = rows * cols;
    public static String valueScale = " .:1S#";

    public static void showImage(byte[] buffer) {
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                System.out.printf("%4d", buffer[i * 28 + j] & 0xFF);
            }
            System.out.println();
        }
    }

    public static void showImageAscii(byte[] buffer) {
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                System.out.printf("%2s",
                        valueScale.charAt((int) ((((buffer[i * 28 + j] & 0xFF)) / 256.) * valueScale.length())));
            }
            System.out.println();
        }
    }

}