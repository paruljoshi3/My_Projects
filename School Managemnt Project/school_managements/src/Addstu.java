import javax.swing.*;
import javax.swing.event.AncestorEvent;
import javax.swing.event.AncestorListener;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;

public class Addstu {
    JPanel panel7;
    private JTextField sid;
    private JTextField sn;
    private JTextField fn;
    private JTextField pn;
    private JTextField fphn;
    private JTextField cl;
    private JTextField roll;
    private JTextField add;
    private JButton SUBMITButton;
    private JLabel BACKLabel;


    public Addstu() {
        BACKLabel.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                super.mouseClicked(e);
                studentsmodule obj = new studentsmodule();
                JFrame frame12 = new JFrame("Students_Module");
                frame12.setContentPane(new studentsmodule().panelSM);
                frame12.pack();
                frame12.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                frame12.setVisible(true);
            }
        });
        SUBMITButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                try{
                    Class.forName("com.mysql.jdbc.Driver");
                    Connection conn = DriverManager.getConnection("jdbc:mysql://localhost:3306/parul","root","tiger");
                    String sql = "insert into stureg values(?,?,?,?,?,?,?,?)";

                    PreparedStatement ptstmt = conn.prepareStatement(sql);
                    ptstmt.setString(1, sid.getText());
                    ptstmt.setString(2, sn.getText());
                    ptstmt.setString(3, fn.getText());
                    ptstmt.setString(4, pn.getText());
                    ptstmt.setString(5, fphn.getText());
                    ptstmt.setString(6, cl.getText());
                    ptstmt.setString(7, roll.getText());
                    ptstmt.setString(8, add.getText());

                    ptstmt.executeUpdate();
                    JOptionPane.showMessageDialog(null,"Data stored successfully");
                    conn.close();
                }
            catch (Exception err){
                    JOptionPane.showMessageDialog(null,err);
                }
            }
        });
    }
}
