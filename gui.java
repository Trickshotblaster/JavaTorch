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
import train.train;

public class gui {
    public static final int rows = 28;
    public static final int cols = 28;
    public static final int size = rows * cols;
    public static final int pixelSize = 32;
    public static Matrix imageBuf = new Matrix(1, size);
    public static boolean down = false;
    public static Container canvas;
    public static JLabel predictionLabel;
    public static JLabel confidenceLabel;

    public void main(String[] args) throws IOException {
        train.run();
        JFrame frMain = new JFrame();

        canvas = new Container();
        canvas.setLayout(new GridLayout(rows, cols));

        for (int i = 0; i < size; i++) {
            JPanel pan = new JPanel();
            pan.setName(Integer.toString(i));
            pan.setBackground(new Color(0, 0, 0));
            pan.setSize(pixelSize, pixelSize);
            pan.addMouseListener(new MouseAdapter() {
                public void mouseEntered(MouseEvent e) {
                    if (down) {
                        e.getComponent().setBackground(new Color(255, 255, 255));
                        int thisIndex = Integer.parseInt(e.getComponent().getName());
                        int upIndex = (thisIndex > cols) ? thisIndex - cols : thisIndex;
                        int downIndex = (thisIndex < rows * (cols - 1)) ? thisIndex + cols : thisIndex;
                        int leftIndex = (thisIndex % cols != 0) ? thisIndex - 1 : thisIndex;
                        int rightIndex = (thisIndex % cols != 0) ? thisIndex + 1 : thisIndex;
                        int upRightIndex = upIndex + 1;
                        int upLeftIndex = upIndex - 1;
                        int downRightIndex = downIndex + 1;
                        int downLeftIndex = downIndex -1;
                        for (int h : new int[] { upIndex, downIndex, leftIndex, rightIndex }) {
                            
                            if (imageBuf.data[0].data[h] + 0.5 < 1) {
                                imageBuf.data[0].data[h] += 0.5;
                            }
                            
                            int val = (int) (imageBuf.data[0].data[h] * 255);
                            canvas.getComponent(h).setBackground(new Color(val, val, val));
                        }
                        for (int j: new int[] {upRightIndex, upLeftIndex, downRightIndex, downLeftIndex}) {
                            try {
                                if (imageBuf.data[0].data[j] + 0.25 < 1) {
                                    imageBuf.data[0].data[j] += 0.25;
                                }
                                
                                int val = (int) (imageBuf.data[0].data[j] * 255);
                                canvas.getComponent(j).setBackground(new Color(val, val, val));
                            } catch (Exception _) {};
                        }
                        imageBuf.data[0].data[thisIndex] = 1.;
                        e.getComponent().setBackground(new Color(255, 255, 255));
                        Matrix probs = train.getProbs(imageBuf);
                        int prediction = probs.argmax1Dim();
                        predictionLabel.setText(String.format("Prediction: %d", prediction));
                        confidenceLabel.setText(String.format("Confidence: %.3f", probs.data[0].data[prediction]));
                    }
                }

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

        Container predictionHolder = new Container();
        predictionHolder.setBounds(cols * pixelSize, 0, 100, rows * pixelSize);
        predictionHolder.setBackground(new Color(50, 50, 50));
        predictionHolder.setLayout(new FlowLayout());
        JButton clearButton = new JButton("Clear");
        clearButton.setSize(80, 10);
        predictionHolder.add(clearButton);
        clearButton.addActionListener(_ -> clearCanvas());
        JPanel predictionPanel = new JPanel();
        predictionPanel.setLayout(new GridLayout(2, 1));
        predictionPanel.setSize(80, 40);
        predictionPanel.setBackground(new Color(100, 100, 100));
        predictionLabel = new JLabel("Prediction: _");
        predictionLabel.setBounds(cols * pixelSize, 50, 80, 20);
        confidenceLabel = new JLabel("Confidence: _");
        confidenceLabel.setBounds(cols * pixelSize, 70, 80, 20);
        predictionPanel.add(predictionLabel);
        predictionPanel.add(confidenceLabel);
        
        predictionHolder.add(predictionPanel);
        frMain.add(canvas);
        frMain.add(predictionHolder);
        frMain.setLayout(null);

        JPanel bg = new JPanel();
        bg.setBounds(0, 0, cols * pixelSize + 200, rows * pixelSize);
        bg.setBackground(new Color(50, 50, 50));

        frMain.add(bg);
        frMain.setSize(cols * pixelSize + 200, rows * pixelSize);
        frMain.setVisible(true);

    }

    public void clearCanvas() {
        MNIST.showImageMatrixAscii(imageBuf);
        System.out.println(train.getProbs(imageBuf).toString());
        System.out.println(train.getOutput(imageBuf).toString());
        for (java.awt.Component p : canvas.getComponents()) {

            p.setBackground(new Color(0, 0, 0));

        }
        imageBuf = new Matrix(1, size);
        predictionLabel.setText("Prediciton: _");
        confidenceLabel.setText("Confidence: _");
        try {
            train.showImagePredictionPair();
        } catch (IOException e) {
            System.out.println(e);
        }

    }
}
