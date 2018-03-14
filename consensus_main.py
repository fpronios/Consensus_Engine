import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, \
    QPushButton, QListWidget,QComboBox,QGridLayout,QGroupBox, QLabel,QLineEdit,QTextEdit
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from database_handlers import *

import random


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.left = 10
        self.top = 10
        self.title = 'Consensus ROC Plotter'
        self.width = 640
        self.height = 400
        self.initUI()

    def initUI(self):


        title = QLabel('Title')
        author = QLabel('Author')
        review = QLabel('Review')

        titleEdit = QLineEdit()
        authorEdit = QLineEdit()
        reviewEdit = QTextEdit()

        grid = QGridLayout()
        grid.setSpacing(10)

        grid.addWidget(title, 1, 0)
        grid.addWidget(titleEdit, 1, 1)

        grid.addWidget(author, 2, 0)
        grid.addWidget(authorEdit, 2, 1)

        grid.addWidget(review, 3, 0)
        grid.addWidget(reviewEdit, 3, 1, 5, 1)



        """ 
        method_list = QComboBox()
        method_list.addItem("Item 1")
        method_list.addItem("Item 1")
        method_list.addItem("Item 1")
        method_list.addItem("Item 1")
        method_list.addItem("Item 1")
        method_list.addItem("Item 1")
        method_list.addItem("Item 1")
        method_list.addItem("Item 1")
        #self.method_list.move(500, 140)

        target_list = QComboBox()
        target_list.addItem("Aggregates")
        target_list.addItem("CDK5")
        #self.target_list.currentIndexChanged.connect(self.selectionchange)
        """



        self.setLayout(grid)

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.show()


    def selectionchange(self, i):
        print ("Items in the list are :")

        for count in range(self.target_list.count()):
            print(self.target_list.itemText(count))
        print("Current index", i, "selection changed ", self.target_list.currentText())




if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())