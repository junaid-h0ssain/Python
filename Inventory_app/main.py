import sys
from PyQt5.QtCore import * # type: ignore
from PyQt5.QtWidgets import * # type: ignore
from PyQt5.QtGui import * # type: ignore
#print('debug 2')
import add_items
#print('debug 3')
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Inventory App')
        self.setGeometry(600,300,800,600)
        self.setWindowIcon(QIcon('profile.png'))
        self.label = QLabel('Inventory App',self)
        self.button = QPushButton('View Inventory',self)
        self.textbox = QLineEdit(self)
        self.initUI()
        #print('debug')
        
    def initUI(self):
        
        self.label.setFont(QFont('Arial',25))
        #self.label.setGeometry(0,0,800,100)
        self.label.setStyleSheet('color: #ffffff;''background-color: #030303;')
        self.label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter) # type: ignore

        viewButtonLayout = QVBoxLayout()
        viewButtonLayout.addWidget(self.button)

        #self.button.setGeometry(400,100,300,100)
        self.button.setStyleSheet('font-size:30px;')
        self.button.clicked.connect(self.onclick)
        self.blabel = QLabel('hello',self)
        #self.blabel.setGeometry(100,220,800,400)

        self.blabel.setStyleSheet('font-size:25px;')
        self.blabel.setWordWrap(True)

        #self.textbox.setGeometry(100,100,200,100)

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.label)
        mainLayout.addLayout(viewButtonLayout)
        mainLayout.addWidget(self.blabel)
        mainLayout.addWidget(self.textbox)

        widget = QWidget()
        widget.setLayout(mainLayout)
        self.setCentralWidget(widget)

    def onclick(self):
        self.blabel.setText(str(add_items.view()))
        #self.button.setDisabled(True)

        
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()