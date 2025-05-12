from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *

app = QApplication([])
window = QWidget()
window.setWindowTitle('Hello Worlds')
window.resize(500,500)

window.show()
app.exec_()

