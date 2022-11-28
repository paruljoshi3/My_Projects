import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;

public class studentsmodule {

    JPanel panelSM;
    private JButton ADDSTUDENTButton;
    private JButton EDITSTUDENTSButton;
    private JButton SEARCHDELETERECORDButton;
    private JLabel BackLabel;

    public studentsmodule() {
        ADDSTUDENTButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                //Addstu obj = new Addstu();
                JFrame frame7 = new JFrame("Add Student");
                frame7.setContentPane(new Addstu().panel7);
                frame7.pack();
                frame7.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                frame7.setVisible(true);
            }
        });

        EDITSTUDENTSButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                editstu obj = new editstu();
                JFrame frame8 = new JFrame("Edit Student");
                frame8.setContentPane(new editstu().panel8);
                frame8.pack();
                frame8.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                frame8.setVisible(true);
            }
        });
        SEARCHDELETERECORDButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                SDstud obj = new SDstud();
                JFrame frame9 = new JFrame("Search or Delete Record");
                frame9.setContentPane(new SDstud().panel9);
                frame9.pack();
                frame9.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                frame9.setVisible(true);
            }
        });

        BackLabel.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                super.mouseClicked(e);
                Welcome obj = new Welcome();
                JFrame frame11 = new JFrame("Welcome_Page");
                frame11.setContentPane(new Welcome().panel2);
                frame11.pack();
                frame11.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                frame11.setVisible(true);
            }
        });
    }
}
