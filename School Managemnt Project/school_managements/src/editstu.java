import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;

public class editstu {
    JPanel panel8;
    private JTextField sid;
    private JTextField sn;
    private JTextField fn;
    private JTextField pn;
    private JTextField fphn;
    private JTextField cl;
    private JTextField roll;
    private JTextField add;
    private JButton button1;

    public editstu() {
        button1.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                String id = sid.getText();
                String name = sn.getText();
                String fname = fn.getText();
                String cla = cl.getText();
                String pnum = pn.getText();
                String fnumb = fphn.getText();
                String address = add.getText();
                String rn = roll.getText();

                try{
                    Class.forName("com.mysql.jdbc.Driver");
                    Connection conn = DriverManager.getConnection("jdbc://mysql://localhost:3306/parul","root","tiger");
                    String sql = "update 'stureg' SET 'fname' = '"+fname+"','name'='"+name+"',class = '"+cla+"',phone = '"+pnum+"'WHERE id='"+id+"'";
                    PreparedStatement ptst = conn.prepareStatement(sql);
                    ptst.execute();

                    JOptionPane.showMessageDialog(null,"Record has been updated successfully");
                }
                catch (Exception err){
                    JOptionPane.showMessageDialog(null,err);
                }
            }
        });
    }
}
