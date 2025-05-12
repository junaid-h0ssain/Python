from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

app = QApplication([])
window = QWidget()

window.setWindowTitle('Inventory App')
window.resize(500,400)

title = QLabel('Inventory App')
# addinfo = QLabel('?')
# deleteinfo = QLabel('?')
# viewinfo = QLabel('?')

addbutton = QPushButton('Add')
deletebutton = QPushButton('Delete/Remove')
viewbutton = QPushButton('View/List Items')

masterlayout = QVBoxLayout()
row1 = QHBoxLayout()
row2 = QHBoxLayout()
row3 = QHBoxLayout()

row1.addWidget(title,alignment=Qt.AlignCenter)
# row2.addWidget(addinfo,alignment=Qt.AlignCenter)
# row2.addWidget(deleteinfo,alignment=Qt.AlignCenter)
# row2.addWidget(viewinfo,alignment=Qt.AlignCenter)
row3.addWidget(addbutton)
row3.addWidget(deletebutton)
row3.addWidget(viewbutton)

masterlayout.addLayout(row1)
# masterlayout.addLayout(row2)
masterlayout.addLayout(row3)

window.setLayout(masterlayout)


window.show()
app.exec_()

