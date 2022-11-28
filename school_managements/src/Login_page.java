import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.Statement;

public class Login_page {
    public JPanel panel1;
    private JTextField user;
    private JPasswordField pass;
    private JButton submitButton;
    private JTextField sc;
    private JButton submitButton1;

    public Login_page() {
        submitButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                String un = user.getText();
                String p = String.valueOf(pass.getPassword());

                try {
                    Connection conn = DriverManager.getConnection("jdbc:mysql://localhost:3306/parul", "root", "tiger");

                    Statement st = conn.createStatement();
                    String sql = "select * from user_login";

                    ResultSet rs = st.executeQuery(sql);
                    boolean loginSucc = false;

                    while (rs.next()) {
                        String username = rs.getString("username");
                        String Password = rs.getString("password");
//                        System.out.println(username + " Hello " + Password);
//                        System.out.println(un + " Hello " + p);
                        if (un.equals(username) && (p.equals(Password))) {
//                            System.out.println("Hello");
                            loginSucc = true;
                            JFrame frame2 = new JFrame("Second Screen");
                            frame2.setContentPane(new Welcome().panel2);
                            frame2.pack();
                            frame2.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                            frame2.setVisible(true);
                        }
                    }

                    if (!loginSucc){
                        JOptionPane.showMessageDialog(null,"Username or Password is incorrect!");
                    }
                } catch (Exception err) {
                    err.printStackTrace();
                    JOptionPane.showMessageDialog(null, "Error while establishing the connection");
                }
            }
        });
        submitButton1.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                String secret = sc.getText();

                if (secret.equals("admin@123")) {
                   JOptionPane.showMessageDialog(null,"The username and password is 'admin'");
                }
            }
        });
    }

    private void createUIComponents() {
        // TODO: place custom component creation code here
    }
}
