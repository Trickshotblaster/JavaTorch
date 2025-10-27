import java.awt.Color;
import java.awt.Container;
import java.awt.FlowLayout;
import java.awt.GridLayout;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.io.IOException;
import javatorch.*;
import javax.swing.*;

import mnist.*;
import train.SaveLoad;
import train.train;

public class gui {
    // image properties
    public static final int rows = 28;
    public static final int cols = 28;
    public static final int size = rows * cols;
    // how big each pixel is on the drawing canvas
    public static final int pixelSize = 32;
    // matrix to use for prediction
    public static Matrix imageBuf = new Matrix(1, size);
    // store mouse state
    public static boolean down = false;
    // container for the drawing canvas
    public static Container canvas;
    // ui labels
    public static JLabel predictionLabel;
    public static JLabel confidenceLabel;
    public static JLabel realLabel;
    // model to load
    public static final String modelPath = "models/H100S10k/";
    // model parameters
    public static Matrix w1, w2, b;
    // dataloaders for loading random images
    public static DataLoader valX;
    public static DataLoader valY;

    public static void main(String[] args) throws IOException {
        // init dataloaders
        valX = new DataLoader("image", "data/t10k-images.idx3-ubyte");
        valY = new DataLoader("label", "data/t10k-labels.idx1-ubyte");

        // read the weights in
        w1 = SaveLoad.readFromFile(modelPath + "w1.txt");
        w2 = SaveLoad.readFromFile(modelPath + "w2.txt");
        b = SaveLoad.readFromFile(modelPath + "b.txt");

        // init main ui
        JFrame frMain = new JFrame();
        frMain.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        // container to hold 28*28 grid
        canvas = new Container();
        canvas.setLayout(new GridLayout(rows, cols));

        // create grid of pixels
        for (int i = 0; i < size; i++) {
            JPanel pan = new JPanel();
            pan.setName(Integer.toString(i));
            pan.setBackground(new Color(0, 0, 0));
            pan.setSize(pixelSize, pixelSize);
            // add mouse listener to each pixel (very bad but doesn't seem to work otherwise
            // and perf is fine)
            pan.addMouseListener(new MouseAdapter() {
                public void mouseEntered(MouseEvent e) {
                    if (down) {
                        // if the mouse is held, color the pixel and neighboring pixels
                        e.getComponent().setBackground(new Color(255, 255, 255));
                        // get pixels around the current one
                        int thisIndex = Integer.parseInt(e.getComponent().getName());
                        int upIndex = (thisIndex > cols) ? thisIndex - cols : thisIndex;
                        int downIndex = (thisIndex < rows * (cols - 1)) ? thisIndex + cols : thisIndex;
                        int leftIndex = (thisIndex % cols != 0) ? thisIndex - 1 : thisIndex;
                        int rightIndex = (thisIndex % cols != 0) ? thisIndex + 1 : thisIndex;
                        int upRightIndex = upIndex + 1;
                        int upLeftIndex = upIndex - 1;
                        int downRightIndex = downIndex + 1;
                        int downLeftIndex = downIndex - 1;
                        // make nearby pixels bright
                        for (int h : new int[] { upIndex, downIndex, leftIndex, rightIndex }) {

                            if (imageBuf.data[0].data[h] + 0.25 < 1) {
                                imageBuf.data[0].data[h] += 0.25;
                            }

                            int val = (int) (imageBuf.data[0].data[h] * 255);
                            canvas.getComponent(h).setBackground(new Color(val, val, val));
                        }
                        // make corners slightly less bright
                        for (int j : new int[] { upRightIndex, upLeftIndex, downRightIndex, downLeftIndex }) {
                            try {
                                if (imageBuf.data[0].data[j] + 0.1 < 1) {
                                    imageBuf.data[0].data[j] += 0.1;
                                }

                                int val = (int) (imageBuf.data[0].data[j] * 255);
                                canvas.getComponent(j).setBackground(new Color(val, val, val));
                            } catch (Exception exc) {
                            }
                            ;
                        }
                        // crop and predict
                        imageBuf.data[0].data[thisIndex] = 1.;
                        e.getComponent().setBackground(new Color(255, 255, 255));
                        Matrix probs = train.getProbs(MNIST.cropMatrix(imageBuf), w1, w2, b);
                        int prediction = probs.argmax1Dim();
                        predictionLabel.setText(String.format("Prediction: %d", prediction));
                        confidenceLabel.setText(String.format("Confidence: %.3f", probs.data[0].data[prediction]));
                    }
                }

                // keep track of mouse state
                public void mousePressed(MouseEvent e) {
                    down = true;
                }

                public void mouseReleased(MouseEvent e) {
                    down = false;
                }
            });
            canvas.add(pan);
        }

        canvas.setBounds(0, 0, rows * pixelSize, cols * pixelSize);

        // container to store various side buttons and labels
        Container predictionHolder = new Container();
        predictionHolder.setBounds(cols * pixelSize, 0, 200, rows * pixelSize);
        predictionHolder.setBackground(new Color(50, 50, 50));
        predictionHolder.setLayout(new FlowLayout());

        // button to clear the canvas
        JButton clearButton = new JButton("Clear");
        clearButton.setSize(100, 20);
        predictionHolder.add(clearButton);
        clearButton.addActionListener(e -> clearCanvas());

        // panel to hold current prediction and confidence
        JPanel predictionPanel = new JPanel();
        predictionPanel.setLayout(new GridLayout(2, 1));
        predictionPanel.setSize(100, 40);
        predictionPanel.setBackground(new Color(100, 100, 100));
        // label to show prediction
        predictionLabel = new JLabel("Prediction: _");
        predictionLabel.setBounds(cols * pixelSize, 50, 80, 20);
        // label for confidence of prediction
        confidenceLabel = new JLabel("Confidence: _");
        confidenceLabel.setBounds(cols * pixelSize, 70, 80, 20);

        predictionPanel.add(predictionLabel);
        predictionPanel.add(confidenceLabel);

        // button for viewing a sample from the dataloader
        JButton viewRandomButton = new JButton("View Random Sample");
        viewRandomButton.setSize(100, 20);
        predictionHolder.add(viewRandomButton);
        viewRandomButton.addActionListener(e -> showExample());

        // label to show the actual label of dataloader sample
        realLabel = new JLabel("Real: _");
        realLabel.setBounds(cols * pixelSize, 130, 80, 20);
        predictionHolder.add(realLabel);

        predictionHolder.add(predictionPanel);
        frMain.add(canvas);
        frMain.add(predictionHolder);
        frMain.setLayout(null);

        // nice background
        JPanel bg = new JPanel();
        bg.setBounds(0, 0, cols * pixelSize + 200, rows * pixelSize);
        bg.setBackground(new Color(50, 50, 50));

        frMain.add(bg);
        frMain.setSize(cols * pixelSize + 200, rows * pixelSize);
        frMain.setVisible(true);
    }

    public static void clearCanvas() {
        // show the cropped image, for debugging
        MNIST.showImageMatrixAscii(MNIST.cropMatrix(imageBuf));
        // loop through elements and set to black
        for (java.awt.Component p : canvas.getComponents()) {
            p.setBackground(new Color(0, 0, 0));
        }
        // reset the image
        imageBuf = new Matrix(1, size);
        predictionLabel.setText("Prediction: _");
        confidenceLabel.setText("Confidence: _");
    }

    public static void showExample() {
        try {
            // empty byte array to write image to
            byte[] xbuf = new byte[size];

            // read the next image and label
            valX.readNextImageTo(xbuf);
            int label = valY.nextLabel();

            // set pixel values accordingly
            for (int i = 0; i < size; i++) {
                int val = (int) xbuf[i] & 0xFF;
                canvas.getComponent(i).setBackground(new Color(val, val, val));
                imageBuf.data[0].data[i] = (double) val / 256.;
            }
            // predict on image
            Matrix probs = train.getProbs(MNIST.cropMatrix(imageBuf), w1, w2, b);
            int prediction = probs.argmax1Dim();
            double confidence = probs.data[0].data[prediction];
            predictionLabel.setText("Prediction: " + prediction);
            realLabel.setText("Real: " + label);
            confidenceLabel.setText(String.format("Confidence: %.3f", confidence));
        } catch (IOException e) {
            System.out.println(e);
        }
    }
}
