import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5.uic import loadUi
import mysql.connector as mc


class Login(QDialog):
    def __init__(self):
        super(Login,self).__init__()
        loadUi("Login.ui",self)
        self.loginBtn.clicked.connect(self.loginfunction)
        self.logToSign.clicked.connect(self.gotosign)
        self.bgimage.setStyleSheet("border-image: url(images/A.jpg);")
        
    
    def gotosign(self):
        signup = Signup()
        widget.addWidget(signup)
        widget.setCurrentIndex(widget.currentIndex()+1)


    def loginfunction(self):
        username= self.logUsername.text()
        password= self.logPassword.text()
        #database 
        mydb = mc.connect(
            host = "localhost",
            user = "root",
            password = "",
            database = "pyqt5"
            )
        mycursor = mydb.cursor()


class Signup(QDialog):
    def __init__(self):
        super(Signup,self).__init__()
        loadUi("Signup.ui",self)
        self.signupBtn.clicked.connect(self.signupfunction)
        self.signToLog.clicked.connect(self.gotolog)
        self.bgimage.setStyleSheet("border-image: url(images/A.jpg);")
        
    
    def gotolog(self):
        login = Login()
        widget.addWidget(login)
        widget.setCurrentIndex(widget.currentIndex()+1)
        
        
    def signupfunction(self):
        username= self.signUsername.text()
        password= self.signPassword.text()
        #database
        
        

app=QApplication(sys.argv)

mainwindow=Login()
widget = QtWidgets.QStackedWidget()
widget.addWidget(mainwindow)
widget.setFixedHeight(431)
widget.setFixedWidth(471)
widget.show()

app.exec_()        