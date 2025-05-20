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
        self.setGeometry(400,200,1000,800)
        self.setWindowIcon(QIcon('profile.png'))
        self.label = QLabel('Inventory App',self)
        self.button = QPushButton('View Inventory',self)
        self.textbox = QLineEdit(self)
        self.textbox1 = QLineEdit(self)
        self.textbox2 = QLineEdit(self)
        self.textbox3 = QLineEdit(self)
        self.flabel1 = QLabel('Name :',self)
        self.flabel2 = QLabel('Price :',self)
        self.flabel3 = QLabel('Amount :',self)
        self.flabel4 = QLabel('Image :',self)
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

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.blabel.setStyleSheet('font-size:25px;')
        #self.blabel.setWordWrap(True)
        self.scroll_area.setWidget(self.blabel)

        addformlayout = QGridLayout()
        #self.textbox.setGeometry(100,100,200,100)
        self.textbox.setStyleSheet('font-size:25px;')
        self.textbox1.setStyleSheet('font-size:25px;')
        self.textbox2.setStyleSheet('font-size:25px;')
        self.textbox3.setStyleSheet('font-size:25px;')
        self.textboxbutton = QPushButton('Submit',self)
        self.textboxbutton.setStyleSheet('font-size:30px;')

        self.flabel1.setStyleSheet('font-size:25px;')
        self.flabel2.setStyleSheet('font-size:25px;')
        self.flabel3.setStyleSheet('font-size:25px;')
        self.flabel4.setStyleSheet('font-size:25px;')

        addformlayout.addWidget(self.flabel1,0,0)
        addformlayout.addWidget(self.flabel2,1,0)
        addformlayout.addWidget(self.flabel3,2,0)
        addformlayout.addWidget(self.flabel4,3,0)
        addformlayout.addWidget(self.textbox,0,1)
        addformlayout.addWidget(self.textbox1,1,1)
        addformlayout.addWidget(self.textbox2,2,1)
        addformlayout.addWidget(self.textbox3,3,1)
        addformlayout.addWidget(self.textboxbutton,3,2)
        self.textboxbutton.clicked.connect(self.submit)

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.label)
        mainLayout.addLayout(viewButtonLayout)
        mainLayout.addWidget(self.scroll_area)
        mainLayout.addLayout(addformlayout)

        widget = QWidget()
        widget.setLayout(mainLayout)
        self.setCentralWidget(widget)

    def onclick(self):
        self.blabel.setText(str(add_items.view()))
        #self.button.setDisabled(True)

    def submit(self):
        name = self.textbox.text()
        price = self.textbox1.text()
        amount = self.textbox2.text()

        if name and price and amount:
            self.blabel.setText(add_items.add(self.textbox.text(),
                                float(self.textbox1.text()),
                                int(self.textbox2.text()),
                                self.textbox3.text())) # type: ignore
            self.textbox.clear()
            self.textbox1.clear()
            self.textbox2.clear()
            self.textbox3.clear()
        else:
            self.blabel.setText(f'Name, price or amount was empty.')

        
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()