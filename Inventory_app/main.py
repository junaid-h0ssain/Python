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

        # self.scroll_area = QScrollArea(self)
        # self.scroll_area.setWidgetResizable(True)
        # self.blabel.setStyleSheet('font-size:25px;')
        # #self.blabel.setWordWrap(True)
        # self.scroll_area.setWidget(self.blabel)
        
        # new table add

        self.scroll_widget = QWidget()

        table = QGridLayout(self.scroll_widget)
        self.name_item = QLabel('Name',self)
        self.price_item = QLabel('Price',self)
        self.amount_item = QLabel('Amount',self)
        self.pic_item = QLabel('Image',self)

        self.name_item.setStyleSheet('font-size:35px;')
        self.price_item.setStyleSheet('font-size:35px;')
        self.amount_item.setStyleSheet('font-size:35px;')
        self.pic_item.setStyleSheet('font-size:35px;')

        table.addWidget(self.name_item,0,0)
        table.addWidget(self.price_item,0,1)
        table.addWidget(self.amount_item,0,2)
        table.addWidget(self.pic_item,0,3)

        def table_items(array,pos):
            items = []
            i=2
            j=0
            for item in (array):
                label = QLabel(item,self)
                label.setStyleSheet('font-size:25px;')
                items.append(label)
                # self.items[0].setStyleSheet('font-size:25px;')
                table.addWidget(items[j],i,pos,)
                i = i+1
                j = j+1

        table_items(add_items.view_name(),0)
        table_items(add_items.view_price(),1)
        table_items(add_items.view_amount(),2)
        table_items(add_items.view_img(),3)

        # table end

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
        mainLayout.addWidget(self.scroll_widget)
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