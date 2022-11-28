import javax.swing.*;
import java.awt.*;

public class Main {
    public static void main(String[] args) {
        System.out.println("Hello world!");
        JFrame frame1 = new JFrame("Test App 1");
        frame1.setContentPane(new Login_page().panel1);
        frame1.setPreferredSize(new Dimension(500,500));
        frame1.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame1.pack();
        frame1.setVisible(true);
    }
}