import sys , os
from PyQt5.QtWidgets import QApplication,  QPushButton,  QGroupBox, QDialog, QVBoxLayout, \
    QGridLayout,QComboBox, QRadioButton
from PyQt5.QtGui import QIcon
#
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from database_handlers import *

class App(QDialog):

    def __init__(self):
        super().__init__()
        self.title = 'Consensus Plotter'
        self.left = 10
        self.top = 10
        self.width = 320
        self.height = 100
        self.initUI()
        self.setWindowIcon(QIcon(resource_path('app_icon.ico')))

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.createGridLayout()

        windowLayout = QVBoxLayout()
        windowLayout.addWidget(self.horizontalGroupBox)
        self.setLayout(windowLayout)

        self.show()

    def createGridLayout(self):
        self.horizontalGroupBox = QGroupBox("ROC Plot options")
        layout = QGridLayout()
        layout.setColumnStretch(1, 4)
        layout.setColumnStretch(2, 4)

        self.target_list = QComboBox()
        self.target_list.addItem('Select Target')
        self.target_list.addItem('CDK5')
        self.target_list.addItem('GSK3b')
        self.target_list.addItem('CK1')
        self.target_list.addItem('DYK1a')

        #trg_lst = get_target

        self.method_list = QComboBox()
        self.method_list.addItem('Select Method',None)

        self.plot_button = QPushButton('Plot')
        self.invert_opt = QRadioButton('Invert ROC plot')
        self.populate_methods()

        layout.addWidget(self.target_list, 0, 0)
        layout.addWidget(self.method_list, 1, 0)
        layout.addWidget(self.plot_button, 1, 1)
        layout.addWidget(self.invert_opt, 0, 1)


        self.plot_button.clicked.connect(self.plot_fig)
        self.target_list.currentIndexChanged.connect(self.selectionchange)
        self.horizontalGroupBox.setLayout(layout)

    def populate_methods(self):
        self.method_list.clear()
        self.method_list.addItem('Select Method', None)
        mthds = get_targets_methods()
        for mtd, idx in zip(mthds,range (len(mthds))):
            self.method_list.addItem(mtd,idx)



    def plot_fig(self):
        #if self.target_list.currentData() != None:
        method = self.method_list.currentData()
        target = self.target_list.currentText()
        if method != None:
            plot_remote(target, method , self.invert_opt.isChecked())
            plot_remote_show()

    def selectionchange(self, i):
        print ("Items in the list are :")
        self.populate_methods()
        #for count in range(self.target_list.count()):
        #    print(self.target_list.itemText(count))
        #print("Current index", i, "selection changed ", self.target_list.currentText())

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())