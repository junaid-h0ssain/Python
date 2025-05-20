import sys
from PyQt5.QtCore import * # type: ignore
from PyQt5.QtWidgets import * # type: ignore
from PyQt5.QtGui import * # type: ignore
#print('debug 2')
import add_items

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Inventory App')
        self.setGeometry(400,200,1000,800)
        self.setWindowIcon(QIcon('profile.png'))

        table = QGridLayout()
        self.name_item = QLabel('Name',self)
        self.price_item = QLabel('Price',self)
        self.amount_item = QLabel('Amount',self)
        self.pic_item = QLabel('Image',self)

        self.name_item.setStyleSheet('font-size:25px;')
        self.price_item.setStyleSheet('font-size:25px;')
        self.amount_item.setStyleSheet('font-size:25px;')
        self.pic_item.setStyleSheet('font-size:25px;')

        table.addWidget(self.name_item,0,0)
        table.addWidget(self.price_item,0,1)
        table.addWidget(self.amount_item,0,2)
        table.addWidget(self.pic_item,0,3)
        items = []
        for item in range(4):
            i=1
            j=1
            items[item] = QLabel(add_items.view_name(),self)
            table.addWidget(items[item],i,j)
            j = j+1

        mainLayout = QVBoxLayout()
        mainLayout.addLayout(table)

        widget = QWidget()
        widget.setLayout(mainLayout)
        self.setCentralWidget(widget)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()