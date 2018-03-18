import sys , os
from PyQt5.QtWidgets import QApplication,  QPushButton,  QGroupBox, QDialog, QVBoxLayout, \
    QGridLayout,QComboBox, QRadioButton,QLineEdit,QCheckBox
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
        self.target_list.addItem('DYRK1a')

        #trg_lst = get_target

        self.method_list = QComboBox()
        self.method_list.addItem('Select Method',None)

        self.plot_button = QPushButton('Plot')
        self.fig_button = QPushButton('New Figure')
        self.invert_opt = QCheckBox('Invert ROC plot')
        self.new_window = QCheckBox('Plot on new')
        self.exp_value  = QLineEdit('1.0')
        self.best_auc = QLineEdit('10')
        self.clear_plot = QPushButton('Clear Plot')

        self.best_roc = QPushButton('Best ROC Areas')
        self.populate_methods()

        layout.addWidget(self.target_list, 0, 0)
        layout.addWidget(self.method_list, 1, 0 ,1,2)
        layout.addWidget(self.plot_button, 1, 2)
        layout.addWidget(self.invert_opt, 0, 2)
        layout.addWidget(self.exp_value,2,0)
        layout.addWidget(self.clear_plot, 2, 2)
        layout.addWidget(self.new_window, 1, 3)
        layout.addWidget(self.best_roc, 2, 3)
        layout.addWidget(self.best_auc, 2, 4)

        #layout.addWidget(self.fig_button, 0, 2)


        self.plot_button.clicked.connect(self.plot_fig)
        self.clear_plot.clicked.connect(self.clear_fig)
        self.best_roc.clicked.connect(self.plot_best_roc)
        self.fig_button.clicked.connect(self.new_fig)
        self.target_list.currentIndexChanged.connect(self.selectionchange)
        self.horizontalGroupBox.setLayout(layout)

    def plot_best_roc(self):
        find_best_auc(self.target_list.currentText(),self.invert_opt.isChecked(),float(self.exp_value.text()),int(self.best_auc.text()))
        #cumulative_best_roc()


    def populate_methods(self):
        self.method_list.clear()
        #self.target_list.addItem('Select Method', None)
        if self.target_list.currentText() != 'Select Target':
            mthds = get_targets_methods(self.target_list.currentText())
            for mtd, idx in zip(mthds,range (len(mthds))):
                self.method_list.addItem(mtd,idx)

    def clear_fig(self):
        cear_plot_fig()

    def new_fig(self):
        new_fig_mpl()

    def plot_fig(self):
        #if self.target_list.currentData() != None:
        method = self.method_list.currentData()
        target = self.target_list.currentText()
        if method != None and 'Exponential Mean' != self.method_list.currentText() and 'Mean' != self.method_list.currentText() \
                and 'Linear Reduction Mean'!= self.method_list.currentText() :
            if self.new_window.isChecked():
                plot_remote(target, method, self.invert_opt.isChecked())
                plot_remote_show()
            else:
                plot_remote_v2(target, method , self.invert_opt.isChecked())
                plot_remote_show()
        if self.method_list.currentText() == 'Mean':
            get_mean_roc(target,self.invert_opt.isChecked())
            plot_remote_show()
        if self.method_list.currentText() == 'Exponential Mean':
            get_mean_exp_roc(target,self.invert_opt.isChecked(),float(self.exp_value.text()))
            plot_remote_show()
        if 'Linear Reduction Mean' == self.method_list.currentText():
            get_mean_exp_roc_v2(target, self.invert_opt.isChecked(), float(self.exp_value.text()))
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