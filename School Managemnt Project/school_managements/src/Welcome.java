import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;

public class Welcome {

    JPanel panel2;
    private JButton button1;
    private JButton FEESUBMISSIONMODULEButton;
    private JButton TEACHERSMODULEButton;
    private JButton STUDENTSREPORTCARDSButton;
    private JLabel logoutLabel;

    public Welcome() {
        button1.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {

            }
        });
        logoutLabel.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                super.mouseClicked(e);
                Login_page obj = new Login_page();
                JFrame frame10 = new JFrame("Login_Page");
                frame10.setContentPane(new Login_page().panel1);
                frame10.pack();
                frame10.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                frame10.setVisible(true);
            }
        });
        button1.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                studentsmodule obj = new studentsmodule();
                JFrame frame3 = new JFrame("Students Module");
                frame3.setContentPane(new studentsmodule().panelSM);
                frame3.pack();
                frame3.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                frame3.setVisible(true);
            }
        });
        FEESUBMISSIONMODULEButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                feeSubmission obj = new feeSubmission();
                JFrame frame4 = new JFrame("Fee Submission");
                frame4.setContentPane(new feeSubmission().panel4);
                frame4.pack();
                frame4.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                frame4.setVisible(true);
            }
        });
        TEACHERSMODULEButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                teachersmodule obj = new teachersmodule();
                JFrame frame5 = new JFrame("Teachers Module");
                frame5.setContentPane(new teachersmodule().panel5);
                frame5.pack();
                frame5.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                frame5.setVisible(true);
            }
        });
        STUDENTSREPORTCARDSButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                reportcard obj = new reportcard();
                JFrame frame6 = new JFrame("Report Card");
                frame6.setContentPane(new reportcard().panel6);
                frame6.pack();
                frame6.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                frame6.setVisible(true);
            }
        });
    }
}
