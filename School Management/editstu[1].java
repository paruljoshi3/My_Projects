/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/GUIForms/JFrame.java to edit this template
 */
package schoolmanagementsystem;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.Statement;
import javax.swing.JOptionPane;
import javax.swing.table.DefaultTableModel;

/**
 *
 * @author parul
 */
public class editstu extends javax.swing.JFrame {

    /**
     * Creates new form editstu
     */
    public editstu() {
        initComponents();
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jLabel9 = new javax.swing.JLabel();
        sid = new javax.swing.JTextField();
        jLabel1 = new javax.swing.JLabel();
        sn = new javax.swing.JTextField();
        jLabel2 = new javax.swing.JLabel();
        fn = new javax.swing.JTextField();
        jLabel3 = new javax.swing.JLabel();
        pn = new javax.swing.JTextField();
        jLabel4 = new javax.swing.JLabel();
        fpn = new javax.swing.JTextField();
        jLabel5 = new javax.swing.JLabel();
        cl = new javax.swing.JTextField();
        jLabel6 = new javax.swing.JLabel();
        roll = new javax.swing.JTextField();
        jLabel7 = new javax.swing.JLabel();
        add = new javax.swing.JTextField();
        jLabel8 = new javax.swing.JLabel();
        jButton1 = new javax.swing.JButton();
        jPanel1 = new javax.swing.JPanel();
        jScrollPane1 = new javax.swing.JScrollPane();
        studsTable = new javax.swing.JTable();
        jLabel10 = new javax.swing.JLabel();
        jLabel11 = new javax.swing.JLabel();
        jButton2 = new javax.swing.JButton();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        getContentPane().setLayout(new org.netbeans.lib.awtextra.AbsoluteLayout());

        jLabel9.setFont(new java.awt.Font("Times New Roman", 1, 15)); // NOI18N
        jLabel9.setForeground(new java.awt.Color(255, 255, 255));
        jLabel9.setText("Address :");
        getContentPane().add(jLabel9, new org.netbeans.lib.awtextra.AbsoluteConstraints(130, 400, -1, -1));
        getContentPane().add(sid, new org.netbeans.lib.awtextra.AbsoluteConstraints(260, 120, 345, -1));

        jLabel1.setFont(new java.awt.Font("Times New Roman", 1, 36)); // NOI18N
        jLabel1.setText("UPDATION OF RECORDS");
        getContentPane().add(jLabel1, new org.netbeans.lib.awtextra.AbsoluteConstraints(160, 50, -1, -1));
        getContentPane().add(sn, new org.netbeans.lib.awtextra.AbsoluteConstraints(260, 160, 345, -1));

        jLabel2.setFont(new java.awt.Font("Times New Roman", 1, 15)); // NOI18N
        jLabel2.setForeground(new java.awt.Color(255, 255, 255));
        jLabel2.setText("ID :");
        getContentPane().add(jLabel2, new org.netbeans.lib.awtextra.AbsoluteConstraints(170, 120, -1, -1));
        getContentPane().add(fn, new org.netbeans.lib.awtextra.AbsoluteConstraints(260, 200, 345, -1));

        jLabel3.setFont(new java.awt.Font("Times New Roman", 1, 15)); // NOI18N
        jLabel3.setForeground(new java.awt.Color(255, 255, 255));
        jLabel3.setText("Name : ");
        getContentPane().add(jLabel3, new org.netbeans.lib.awtextra.AbsoluteConstraints(140, 160, -1, -1));
        getContentPane().add(pn, new org.netbeans.lib.awtextra.AbsoluteConstraints(260, 240, 345, -1));

        jLabel4.setFont(new java.awt.Font("Times New Roman", 1, 15)); // NOI18N
        jLabel4.setForeground(new java.awt.Color(255, 255, 255));
        jLabel4.setText("Father Name :");
        getContentPane().add(jLabel4, new org.netbeans.lib.awtextra.AbsoluteConstraints(100, 200, -1, -1));
        getContentPane().add(fpn, new org.netbeans.lib.awtextra.AbsoluteConstraints(260, 280, 345, -1));

        jLabel5.setFont(new java.awt.Font("Times New Roman", 1, 15)); // NOI18N
        jLabel5.setForeground(new java.awt.Color(255, 255, 255));
        jLabel5.setText("Phone :");
        getContentPane().add(jLabel5, new org.netbeans.lib.awtextra.AbsoluteConstraints(140, 240, -1, -1));
        getContentPane().add(cl, new org.netbeans.lib.awtextra.AbsoluteConstraints(260, 320, 345, -1));

        jLabel6.setFont(new java.awt.Font("Times New Roman", 1, 15)); // NOI18N
        jLabel6.setForeground(new java.awt.Color(255, 255, 255));
        jLabel6.setText("Father Phone :");
        getContentPane().add(jLabel6, new org.netbeans.lib.awtextra.AbsoluteConstraints(100, 280, -1, -1));
        getContentPane().add(roll, new org.netbeans.lib.awtextra.AbsoluteConstraints(260, 360, 345, -1));

        jLabel7.setFont(new java.awt.Font("Times New Roman", 1, 15)); // NOI18N
        jLabel7.setForeground(new java.awt.Color(255, 255, 255));
        jLabel7.setText("Class :");
        getContentPane().add(jLabel7, new org.netbeans.lib.awtextra.AbsoluteConstraints(150, 320, -1, -1));
        getContentPane().add(add, new org.netbeans.lib.awtextra.AbsoluteConstraints(260, 400, 345, -1));

        jLabel8.setFont(new java.awt.Font("Times New Roman", 1, 15)); // NOI18N
        jLabel8.setForeground(new java.awt.Color(255, 255, 255));
        jLabel8.setText("Roll Number :");
        getContentPane().add(jLabel8, new org.netbeans.lib.awtextra.AbsoluteConstraints(100, 360, -1, -1));

        jButton1.setFont(new java.awt.Font("Times New Roman", 1, 18)); // NOI18N
        jButton1.setText("SUBMIT");
        jButton1.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton1ActionPerformed(evt);
            }
        });
        getContentPane().add(jButton1, new org.netbeans.lib.awtextra.AbsoluteConstraints(210, 460, 110, 50));

        jPanel1.setBackground(new java.awt.Color(0, 153, 153));

        studsTable.setModel(new javax.swing.table.DefaultTableModel(
            new Object [][] {
                {null, null, null, null, null, null, null, null},
                {null, null, null, null, null, null, null, null},
                {null, null, null, null, null, null, null, null},
                {null, null, null, null, null, null, null, null}
            },
            new String [] {
                "ID", "Name", "Father's Name", "Phone", "Father's Phone", "Class", "Roll Number", "Address"
            }
        ));
        jScrollPane1.setViewportView(studsTable);

        jLabel10.setFont(new java.awt.Font("Times New Roman", 1, 18)); // NOI18N
        jLabel10.setForeground(new java.awt.Color(51, 0, 153));
        jLabel10.setText("BACK");
        jLabel10.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                jLabel10MouseClicked(evt);
            }
        });

        jLabel11.setFont(new java.awt.Font("Times New Roman", 1, 18)); // NOI18N
        jLabel11.setText("STUDENTS DETAILS");

        jButton2.setFont(new java.awt.Font("Times New Roman", 1, 18)); // NOI18N
        jButton2.setText("STUDENTS INFORMATION");
        jButton2.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton2ActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout jPanel1Layout = new javax.swing.GroupLayout(jPanel1);
        jPanel1.setLayout(jPanel1Layout);
        jPanel1Layout.setHorizontalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel1Layout.createSequentialGroup()
                .addContainerGap(662, Short.MAX_VALUE)
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, jPanel1Layout.createSequentialGroup()
                        .addComponent(jLabel10)
                        .addGap(16, 16, 16))
                    .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, jPanel1Layout.createSequentialGroup()
                        .addComponent(jScrollPane1, javax.swing.GroupLayout.PREFERRED_SIZE, 912, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addContainerGap())
                    .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, jPanel1Layout.createSequentialGroup()
                        .addComponent(jLabel11)
                        .addGap(390, 390, 390))
                    .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, jPanel1Layout.createSequentialGroup()
                        .addComponent(jButton2)
                        .addGap(363, 363, 363))))
        );
        jPanel1Layout.setVerticalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel1Layout.createSequentialGroup()
                .addGap(28, 28, 28)
                .addComponent(jLabel10)
                .addGap(16, 16, 16)
                .addComponent(jLabel11)
                .addGap(18, 18, 18)
                .addComponent(jScrollPane1, javax.swing.GroupLayout.PREFERRED_SIZE, 324, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(18, 18, 18)
                .addComponent(jButton2)
                .addContainerGap(182, Short.MAX_VALUE))
        );

        getContentPane().add(jPanel1, new org.netbeans.lib.awtextra.AbsoluteConstraints(0, 0, 1580, 660));

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void jLabel10MouseClicked(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_jLabel10MouseClicked
        // TODO add your handling code here:
        studentsmodule obj = new studentsmodule();
        obj.setVisible(true);
        dispose();
    }//GEN-LAST:event_jLabel10MouseClicked

    private void jButton1ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton1ActionPerformed
        // TODO add your handling code here:
        String id = sid.getText();
        String name = sn.getText();
        String fname = fn.getText();
        String cla = cl.getText();
        String pnum = pn.getText();
        String fnumb = fpn.getText();
        String address = add.getText();
        String rn = roll.getText();

        try {
//            Class.forName("com.mysql.cj.jdbc.Driver");
//            Connection conn = (Connection) DriverManager.getConnection("jdbc:mysql://localhost:3306/parul", "root", "tiger");
//            Statement st = (Statement) conn.createStatement();
            Class.forName("com.mysql.cj.jdbc.Driver");
            Connection conn = (Connection)DriverManager.getConnection("jdbc:mysql://localhost:3306/parul", "root", "tiger");
            String sql = "update `stureg` SET `fname` = '" + fname + "',`name`='" + name + "',`class` = '" + cla + "',`phone` = '" + pnum + "'WHERE id='" + id + "'";
            PreparedStatement ptst = conn.prepareStatement(sql);
            ptst.execute();

            JOptionPane.showMessageDialog(null, "Record has been updated successfully");
        } catch (Exception err) {
            JOptionPane.showMessageDialog(null, err);
        }
    }//GEN-LAST:event_jButton1ActionPerformed

    private void jButton2ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton2ActionPerformed
        // TODO add your handling code here:
        try {
            Class.forName("com.mysql.jdbc.Driver");
            Connection conn = DriverManager.getConnection("jdbc:mysql://localhost:3306/parul", "root", "tiger");
            Statement st = conn.createStatement();
            String sql = "select*from stureg";
            PreparedStatement ptst = conn.prepareStatement(sql);
            ResultSet rs = ptst.executeQuery();
            DefaultTableModel tm = (DefaultTableModel) studsTable.getModel();
            tm.setRowCount(0);
            while (rs.next()) {
                Object o[] = {rs.getInt("ID"), rs.getString("Name"), rs.getString("fname"), rs.getInt("phone"),
                    rs.getInt("fatherphone"), rs.getString("class"), rs.getString("roll"), rs.getString("address")};
                tm.addRow(o);
            }
        } catch (Exception e) {
            JOptionPane.showMessageDialog(null, e);
        }
    }//GEN-LAST:event_jButton2ActionPerformed

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(editstu.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(editstu.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(editstu.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(editstu.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {
            public void run() {
                new editstu().setVisible(true);
            }
        });
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JTextField add;
    private javax.swing.JTextField cl;
    private javax.swing.JTextField fn;
    private javax.swing.JTextField fpn;
    private javax.swing.JButton jButton1;
    private javax.swing.JButton jButton2;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JLabel jLabel10;
    private javax.swing.JLabel jLabel11;
    private javax.swing.JLabel jLabel2;
    private javax.swing.JLabel jLabel3;
    private javax.swing.JLabel jLabel4;
    private javax.swing.JLabel jLabel5;
    private javax.swing.JLabel jLabel6;
    private javax.swing.JLabel jLabel7;
    private javax.swing.JLabel jLabel8;
    private javax.swing.JLabel jLabel9;
    private javax.swing.JPanel jPanel1;
    private javax.swing.JScrollPane jScrollPane1;
    private javax.swing.JTextField pn;
    private javax.swing.JTextField roll;
    private javax.swing.JTextField sid;
    private javax.swing.JTextField sn;
    private javax.swing.JTable studsTable;
    // End of variables declaration//GEN-END:variables
}