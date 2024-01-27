import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from functools import partial
import sys
import matplotlib
matplotlib.use("Qt5Agg")  # 声明使用QT5
matplotlib.rcParams['savefig.dpi'] = 600
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import os
import re
from scipy.interpolate import interp1d
import pickle

"""
================
Title: Spectrum Merging GUI
Author: Siyuan Qiu
Create Date: 2022/9/1
Institution: Columbia University, Department of Physics
=================
"""

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # set the size and title of the window
        self.setGeometry(200, 100, 1500, 900)
        self.setWindowTitle('MergeSpec')

        self.init_UI()

    def init_UI(self):
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        global spectrum_widget
        global merged_spec_widget

        # create the up widget for image loading and angle selection
        spectrum_widget = Spectrum()
        merged_spec_widget = MergedSpecDisplayManager()

        # Note that this command needs to be declared outside spectrum_widget, otherwise
        # there is cross declaration problem
        spectrum_widget.show_manager_btn.clicked.connect(merged_spec_widget.show)

        splitter.addWidget(spectrum_widget)

        self.show()

        self._createActions()
        self._createMenuBar()

    def _createMenuBar(self):
        menuBar = self.menuBar()
        menuBar.clear()
        # Creating menus using a title
        windowMenu = menuBar.addMenu("&Window")
        # windowMenu.addSeparator()
        windowMenu.addAction(self.initializeAction)
        windowMenu.addAction(self.exitAction)

    def _createActions(self):
        # Creating actions using the second constructor
        self.initializeAction = QAction("&Reinitialize...", self)
        self.initializeAction.triggered.connect(self.reinitialize)
        self.exitAction = QAction("&Exit...", self)
        self.exitAction.triggered.connect(self.exit)

    def reinitialize(self):
        buttonReply = QMessageBox.question(self, 'Reintialize', "Are you sure to reinitialize everything?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if buttonReply == QMessageBox.Yes:
            self.init_UI()
        else:
            return

    def exit(self):
        app.quit()

    def closeEvent(self, event):
        app.quit()

class MergedSpecDisplayManager(QFrame):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.counter = 0
        self.setGeometry(150, 100, 250, 50)

    def initUI(self):
        main_vbox = QVBoxLayout()
        main_grid = QHBoxLayout()
        main_grid.setSpacing(10)
        self.setLayout(main_vbox)
        self.setWindowTitle("Merged spectrum display")

        btn_hbox = QHBoxLayout()

        self.load_btn = QPushButton("Load spectrum")
        self.load_btn.setFixedWidth(100)
        self.load_btn.clicked.connect(self.load_mergedSpec)
        btn_hbox.addWidget(self.load_btn)

        self.load_folder_btn = QPushButton("Load spectrum from folder")
        self.load_folder_btn.setFixedWidth(150)
        self.load_folder_btn.clicked.connect(self.load_mergedSpec_from_folder)
        btn_hbox.addWidget(self.load_folder_btn)

        color_lb = QLabel("Color")
        color_lb.setFixedHeight(20)
        color_lb.setAlignment(Qt.AlignCenter)
        name_lb = QLabel("Name")
        name_lb.setFixedHeight(20)
        name_lb.setAlignment(Qt.AlignCenter)
        display_lb = QLabel("Display")
        display_lb.setFixedHeight(20)
        display_lb.setAlignment(Qt.AlignCenter)
        unload_lb = QLabel("Unload")
        unload_lb.setFixedHeight(20)
        unload_lb.setAlignment(Qt.AlignCenter)

        self.color_vbox = QVBoxLayout()
        self.name_vbox = QVBoxLayout()
        self.display_vbox = QVBoxLayout()
        self.unload_vbox = QVBoxLayout()

        self.color_vbox.addWidget(color_lb)
        self.name_vbox.addWidget(name_lb)
        self.display_vbox.addWidget(display_lb)
        self.unload_vbox.addWidget(unload_lb)

        main_grid.addLayout(self.color_vbox)
        main_grid.addLayout(self.name_vbox)
        main_grid.addLayout(self.display_vbox)
        main_grid.addLayout(self.unload_vbox)

        main_vbox.addLayout(btn_hbox)
        main_vbox.addLayout(main_grid)

    def read_refFIT_data(self, path):
        try:
            file = np.loadtxt(path).transpose()
            freq = file[0]
            reflectance = file[1]
        except:
            try:
                file = np.loadtxt(path, delimiter=",").transpose()
                freq = file[0]
                reflectance = file[1]
            except:
                try:
                    file = np.loadtxt(path, delimiter=" ").transpose()
                    freq = file[0]
                    reflectance = file[1]
                except:
                    return
        return np.array(reflectance), np.array(freq)

    def load_mergedSpec(self):
        path = QFileDialog.getOpenFileName(self, "Select a file", r"~\PycharmProjects/Transfer Matrix Method", "Text Files (*.txt *.csv *.dat)")[0]
        if path != "":
            try:
                reflectance, freq = self.read_refFIT_data(path)
                name = os.path.basename(path)
                self.create_spec(freq, reflectance, name)
            except:
                QMessageBox.warning(self, "Load mergedSpec", "You are not selecting a correct file!")
                return

    def sort_nicely(self, l):
        """ Sort the given list in the way that humans expect.
        """
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
        l.sort(key=alphanum_key)
        return l

    def load_mergedSpec_from_folder(self):
        folderpath = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if folderpath != "":
            try:
                files = self.sort_nicely(os.listdir(folderpath))
                for f in files:
                    if f[-4:] == ".txt" or f[-4:] == ".csv" or f[-4:] == ".dat":
                        file = open(folderpath + "\\" + f, "r")
                        reflectance, freq = self.read_refFIT_data(file)
                        self.create_spec(freq, reflectance, f)
            except:
                QMessageBox.warning(self, "Load mergedSpec from folder", "Cannot read files in the selected folder!")
                return

    # use partial to perform lambda functions in exec!
    def create_spec(self, freq, reflectance, name):
        self.counter += 1
        exec("self.color_select_btn{} = QPushButton()".format(self.counter))
        exec("self.color_select_btn{}.setFixedHeight(25)".format(self.counter))
        exec("self.name_le{} = QLineEdit()".format(self.counter))
        exec("self.name_le{}.setFixedHeight(25)".format(self.counter))
        exec("self.name_le{}.setFixedWidth(100)".format(self.counter))
        exec("self.name_le{}.setText(name)".format(self.counter))
        exec("self.display_btn{} = QPushButton(\"OFF\")".format(self.counter))
        exec("self.display_btn{}.setFixedHeight(25)".format(self.counter))
        exec("self.unload_btn{} = QPushButton(\"unload\")".format(self.counter))
        exec("self.unload_btn{}.setFixedHeight(25)".format(self.counter))
        exec("self.spec{}, = spectrum_widget.axes.plot(freq, reflectance, label=name, zorder=-1)".format(self.counter))
        exec("color = self.spec{}.get_color()".format(self.counter))
        exec("self.color_select_btn{}.setStyleSheet(\"background-color: {}\")".format(self.counter, locals()["color"]))
        exec("self.color_select_btn{}.clicked.connect(partial(self.change_color, self.color_select_btn{}, self.spec{}))".format(self.counter, self.counter, self.counter))
        exec("self.name_le{}.editingFinished.connect(partial(self.change_name, self.name_le{}, self.spec{}))".format(self.counter, self.counter, self.counter))
        exec("self.display_btn{}.clicked.connect(partial(self.change_display, self.display_btn{}, self.spec{}))".format(self.counter, self.counter, self.counter))
        exec("self.unload_btn{}.clicked.connect(partial(self.unload, self.color_select_btn{}, self.name_le{}, self.display_btn{}, self.unload_btn{}, self.spec{}))".format(self.counter, self.counter, self.counter, self.counter, self.counter, self.counter))

        exec("self.color_vbox.addWidget(self.color_select_btn{})".format(self.counter))
        exec("self.name_vbox.addWidget(self.name_le{})".format(self.counter))
        exec("self.display_vbox.addWidget(self.display_btn{})".format(self.counter))
        exec("self.unload_vbox.addWidget(self.unload_btn{})".format(self.counter))

        spectrum_widget.axes.legend()
        spectrum_widget.F.draw()

    def change_color(self, btn, spec):
        color = QColorDialog.getColor()
        btn.setStyleSheet("background-color: {}".format(color.name()))
        spec.set_color(color.name())
        spectrum_widget.axes.legend()
        spectrum_widget.F.draw()

    def change_name(self, le, spec):
        spec.set_label(le.text())
        spectrum_widget.axes.legend()
        spectrum_widget.F.draw()

    def change_display(self, display, spec):
        if display.text() == "OFF":
            spec.set_visible(False)
            display.setText("ON")
        else:
            spec.set_visible(True)
            display.setText("OFF")
        spectrum_widget.axes.legend()
        spectrum_widget.F.draw()

    def unload(self, color, name, display, unload, spec):

        self.color_vbox.removeWidget(color)
        self.name_vbox.removeWidget(name)
        self.display_vbox.removeWidget(display)
        self.unload_vbox.removeWidget(unload)

        color.deleteLater()
        name.deleteLater()
        display.deleteLater()
        unload.deleteLater()

        spec.remove()
        spectrum_widget.axes.legend()
        spectrum_widget.F.draw()

class Spectrum(QFrame):
    def __init__(self):
        super().__init__()
        self.reflectance = [[], [], [], [], []]
        self.freq = [[], [], [], [], []]
        self.range = [[], [], [], [], []]
        self.offset = np.zeros(5)
        self.multiplier = np.ones(5)
        self.is_auto_fill = [[], [], [], [], []]
        self.auto_fill_order = [1, 1, 1]
        self.R_curve = [None, None, None, None, None]
        self.break_line = [None, None, None, None]
        self.R_curve_color = ["#FF0000", "#FFA500", "#228B22", "#0000FF", "#8A2BE2"]
        self.break_line_color = ["#FF0000", "#FFA500", "#228B22", "#0000FF"]
        Ag = self.loadpickle("Ag_Epsilon_Reflectance_400-35000cm-1.pickle")
        self.Ag_refl = interp1d(Ag["Yang2015PRB"].freq, Ag["Yang2015PRB"].R)
        Au = self.loadpickle("Au_Eps_Reflectance_Olmon2012PRB.pickle")
        self.Au_refl = interp1d(Au.freq, Au.R)
        self.initUI()

    def initUI(self):
        # create main grid to organize layout
        main_grid = QGridLayout()
        main_grid.setSpacing(10)
        self.setLayout(main_grid)
        self.setWindowTitle("Spectrum")

        self.figure = plt.figure(figsize=(200, 100))
        self.F = FigureCanvas(self.figure)
        main_grid.addWidget(NavigationToolbar(self.F, self), 0, 2, 1, 1, Qt.AlignCenter)
        main_grid.addWidget(self.F, 1, 0, 1, 5)

        save_hbox = QHBoxLayout()
        self.ref_cb = QComboBox()
        self.ref_cb.addItems(["no reference", "Au", "Ag"])
        self.save_spec_cb = QCheckBox("Save spectrum")
        self.save_spec_cb.setChecked(True)
        self.save_params_cb = QCheckBox("Save params")
        self.save_params_cb.setChecked(True)
        self.save_btn = QPushButton("Save selected items")
        self.save_btn.setFixedHeight(30)
        self.save_btn.clicked.connect(self.save_items)
        save_hbox.addWidget(self.ref_cb)
        save_hbox.addWidget(self.save_spec_cb)
        save_hbox.addWidget(self.save_params_cb)
        save_hbox.addWidget(self.save_btn)
        main_grid.addLayout(save_hbox, 0, 0, 1, 2, Qt.AlignCenter)

        load_hbox = QHBoxLayout()
        self.show_manager_btn = QPushButton("Show merged spectrum manager")
        self.show_manager_btn.setFixedHeight(30)
        self.load_params_btn = QPushButton("Load params")
        self.load_params_btn.setFixedHeight(30)
        self.load_params_btn.clicked.connect(self.load_params)
        load_hbox.addWidget(self.show_manager_btn)
        load_hbox.addWidget(self.load_params_btn)
        main_grid.addLayout(load_hbox, 0, 3, 1, 2, Qt.AlignCenter)

        self.slider_hb = QHBoxLayout()
        main_grid.addLayout(self.slider_hb, 2, 0, 1, 5)

        breakPoint1_vb = QVBoxLayout()
        breakPoint1_hb = QGridLayout()
        self.breakPoint1_color_btn = QPushButton()
        self.breakPoint1_color_btn.setFixedHeight(15)
        self.breakPoint1_color_btn.setFixedWidth(20)
        self.breakPoint1_color_btn.setStyleSheet("background-color: {}".format(self.break_line_color[0]))
        self.breakPoint1_color_btn.clicked.connect(lambda: self.setColor("breakPoint1"))
        breakPoint1_lb = QLabel("Break point 1")
        breakPoint1_lb2 = QLabel("")
        breakPoint1_hb.addWidget(self.breakPoint1_color_btn, 0, 0, 1, 1, Qt.AlignRight)
        breakPoint1_hb.addWidget(breakPoint1_lb, 0, 1, 1, 1, Qt.AlignCenter)
        breakPoint1_hb.addWidget(breakPoint1_lb2, 0, 2, 1, 1, Qt.AlignCenter)
        self.breakPoint1_sld = QDoubleSlider(Qt.Horizontal)
        self.breakPoint1_sld.setTickPosition(QSlider.TicksBelow)
        self.breakPoint1_sld.setMaximum(100)
        self.breakPoint1_sld.setMinimum(0)
        self.breakPoint1_sld.setSingleStep(1)
        self.breakPoint1_sld.setTickInterval(200)
        self.breakPoint1_sld.setEnabled(False)
        self.breakPoint1_sld.valueChanged.connect(lambda: self.setSliderPos("1", "breakpoint"))
        breakPoint1_sb_hb = QHBoxLayout()
        breakPoint1_min_lb = QLabel("0")
        breakPoint1_min_lb.setAlignment(Qt.AlignLeft)
        self.breakPoint1_sb = QClickableSpinBox()
        self.breakPoint1_sb.setDecimals(4)
        self.breakPoint1_sb.setFixedWidth(100)
        self.breakPoint1_sb.setMaximum(100)
        self.breakPoint1_sb.setMinimum(0)
        self.breakPoint1_sb.setAlignment(Qt.AlignCenter)
        self.breakPoint1_sb.setEnabled(False)
        self.breakPoint1_sb.editingFinished.connect(lambda: self.setSbPos("1", "breakpoint"))
        breakPoint1_max_lb = QLabel("100")
        breakPoint1_max_lb.setAlignment(Qt.AlignRight)
        breakPoint1_sb_hb.addWidget(breakPoint1_min_lb)
        breakPoint1_sb_hb.addWidget(self.breakPoint1_sb)
        breakPoint1_sb_hb.addWidget(breakPoint1_max_lb)
        breakPoint1_vb.addLayout(breakPoint1_hb)
        breakPoint1_vb.addWidget(self.breakPoint1_sld)
        breakPoint1_vb.addLayout(breakPoint1_sb_hb)
        self.slider_hb.addLayout(breakPoint1_vb)

        Separador1 = QFrame()
        Separador1.setFrameShape(QFrame.VLine)
        Separador1.setLineWidth(1)
        self.slider_hb.addWidget(Separador1)

        breakPoint2_vb = QVBoxLayout()
        breakPoint2_hb = QGridLayout()
        self.breakPoint2_color_btn = QPushButton()
        self.breakPoint2_color_btn.setFixedHeight(15)
        self.breakPoint2_color_btn.setFixedWidth(20)
        self.breakPoint2_color_btn.setStyleSheet("background-color: {}".format(self.break_line_color[1]))
        self.breakPoint2_color_btn.clicked.connect(lambda: self.setColor("breakPoint2"))
        breakPoint2_lb = QLabel("Break point 2")
        breakPoint2_lb2 = QLabel("")
        breakPoint2_hb.addWidget(self.breakPoint2_color_btn, 0, 0, 1, 1, Qt.AlignRight)
        breakPoint2_hb.addWidget(breakPoint2_lb, 0, 1, 1, 1, Qt.AlignCenter)
        breakPoint2_hb.addWidget(breakPoint2_lb2, 0, 2, 1, 1, Qt.AlignCenter)
        self.breakPoint2_sld = QDoubleSlider(Qt.Horizontal)
        self.breakPoint2_sld.setTickPosition(QSlider.TicksBelow)
        self.breakPoint2_sld.setMaximum(900)
        self.breakPoint2_sld.setMinimum(10)
        self.breakPoint2_sld.setSingleStep(1)
        self.breakPoint2_sld.setTickInterval(200)
        self.breakPoint2_sld.setEnabled(False)
        self.breakPoint2_sld.valueChanged.connect(lambda: self.setSliderPos("2", "breakpoint"))
        breakPoint2_sb_hb = QHBoxLayout()
        breakPoint2_min_lb = QLabel("10")
        breakPoint2_min_lb.setAlignment(Qt.AlignLeft)
        self.breakPoint2_sb = QClickableSpinBox()
        self.breakPoint2_sb.setDecimals(4)
        self.breakPoint2_sb.setFixedWidth(100)
        self.breakPoint2_sb.setMaximum(900)
        self.breakPoint2_sb.setMinimum(10)
        self.breakPoint2_sb.setAlignment(Qt.AlignCenter)
        self.breakPoint2_sb.setEnabled(False)
        self.breakPoint2_sb.editingFinished.connect(lambda: self.setSbPos("2", "breakpoint"))
        breakPoint2_max_lb = QLabel("900")
        breakPoint2_max_lb.setAlignment(Qt.AlignRight)
        breakPoint2_sb_hb.addWidget(breakPoint2_min_lb)
        breakPoint2_sb_hb.addWidget(self.breakPoint2_sb)
        breakPoint2_sb_hb.addWidget(breakPoint2_max_lb)
        breakPoint2_vb.addLayout(breakPoint2_hb)
        breakPoint2_vb.addWidget(self.breakPoint2_sld)
        breakPoint2_vb.addLayout(breakPoint2_sb_hb)
        self.slider_hb.addLayout(breakPoint2_vb)

        Separador2 = QFrame()
        Separador2.setFrameShape(QFrame.VLine)
        Separador2.setLineWidth(1)
        self.slider_hb.addWidget(Separador2)

        breakPoint3_vb = QVBoxLayout()
        breakPoint3_hb = QGridLayout()
        self.breakPoint3_color_btn = QPushButton()
        self.breakPoint3_color_btn.setFixedHeight(15)
        self.breakPoint3_color_btn.setFixedWidth(20)
        self.breakPoint3_color_btn.setStyleSheet("background-color: {}".format(self.break_line_color[2]))
        self.breakPoint3_color_btn.clicked.connect(lambda: self.setColor("breakPoint3"))
        breakPoint3_lb = QLabel("Break point 3")
        breakPoint3_lb2 = QLabel("")
        breakPoint3_hb.addWidget(self.breakPoint3_color_btn, 0, 0, 1, 1, Qt.AlignRight)
        breakPoint3_hb.addWidget(breakPoint3_lb, 0, 1, 1, 1, Qt.AlignCenter)
        breakPoint3_hb.addWidget(breakPoint3_lb2, 0, 2, 1, 1, Qt.AlignCenter)
        self.breakPoint3_sld = QDoubleSlider(Qt.Horizontal)
        self.breakPoint3_sld.setTickPosition(QSlider.TicksBelow)
        self.breakPoint3_sld.setMaximum(9000)
        self.breakPoint3_sld.setMinimum(500)
        self.breakPoint3_sld.setSingleStep(1)
        self.breakPoint3_sld.setTickInterval(200)
        self.breakPoint3_sld.setEnabled(False)
        self.breakPoint3_sld.valueChanged.connect(lambda: self.setSliderPos("3", "breakpoint"))
        breakPoint3_sb_hb = QHBoxLayout()
        breakPoint3_min_lb = QLabel("500")
        breakPoint3_min_lb.setAlignment(Qt.AlignLeft)
        self.breakPoint3_sb = QClickableSpinBox()
        self.breakPoint3_sb.setDecimals(4)
        self.breakPoint3_sb.setFixedWidth(100)
        self.breakPoint3_sb.setMaximum(9000)
        self.breakPoint3_sb.setMinimum(500)
        self.breakPoint3_sb.setAlignment(Qt.AlignCenter)
        self.breakPoint3_sb.setEnabled(False)
        self.breakPoint3_sb.editingFinished.connect(lambda: self.setSbPos("3", "breakpoint"))
        breakPoint3_max_lb = QLabel("9000")
        breakPoint3_max_lb.setAlignment(Qt.AlignRight)
        breakPoint3_sb_hb.addWidget(breakPoint3_min_lb)
        breakPoint3_sb_hb.addWidget(self.breakPoint3_sb)
        breakPoint3_sb_hb.addWidget(breakPoint3_max_lb)
        breakPoint3_vb.addLayout(breakPoint3_hb)
        breakPoint3_vb.addWidget(self.breakPoint3_sld)
        breakPoint3_vb.addLayout(breakPoint3_sb_hb)
        self.slider_hb.addLayout(breakPoint3_vb)

        Separador3 = QFrame()
        Separador3.setFrameShape(QFrame.VLine)
        Separador3.setLineWidth(1)
        self.slider_hb.addWidget(Separador3)

        breakPoint4_vb = QVBoxLayout()
        breakPoint4_hb = QGridLayout()
        self.breakPoint4_color_btn = QPushButton()
        self.breakPoint4_color_btn.setFixedHeight(15)
        self.breakPoint4_color_btn.setFixedWidth(20)
        self.breakPoint4_color_btn.setStyleSheet("background-color: {}".format(self.break_line_color[3]))
        self.breakPoint4_color_btn.clicked.connect(lambda: self.setColor("breakPoint4"))
        breakPoint4_lb = QLabel("Break point 4")
        breakPoint4_lb2 = QLabel("")
        breakPoint4_hb.addWidget(self.breakPoint4_color_btn, 0, 0, 1, 1, Qt.AlignRight)
        breakPoint4_hb.addWidget(breakPoint4_lb, 0, 1, 1, 1, Qt.AlignCenter)
        breakPoint4_hb.addWidget(breakPoint4_lb2, 0, 2, 1, 1, Qt.AlignCenter)
        self.breakPoint4_sld = QDoubleSlider(Qt.Horizontal)
        self.breakPoint4_sld.setTickPosition(QSlider.TicksBelow)
        self.breakPoint4_sld.setMaximum(12000)
        self.breakPoint4_sld.setMinimum(7000)
        self.breakPoint4_sld.setSingleStep(1)
        self.breakPoint4_sld.setTickInterval(200)
        self.breakPoint4_sld.setEnabled(False)
        self.breakPoint4_sld.valueChanged.connect(lambda: self.setSliderPos("4", "breakpoint"))
        breakPoint4_sb_hb = QHBoxLayout()
        breakPoint4_min_lb = QLabel("7000")
        breakPoint4_min_lb.setAlignment(Qt.AlignLeft)
        self.breakPoint4_sb = QClickableSpinBox()
        self.breakPoint4_sb.setDecimals(4)
        self.breakPoint4_sb.setFixedWidth(100)
        self.breakPoint4_sb.setMaximum(12000)
        self.breakPoint4_sb.setMinimum(7000)
        self.breakPoint4_sb.setAlignment(Qt.AlignCenter)
        self.breakPoint4_sb.setEnabled(False)
        self.breakPoint4_sb.editingFinished.connect(lambda: self.setSbPos("4", "breakpoint"))
        breakPoint4_max_lb = QLabel("12000")
        breakPoint4_max_lb.setAlignment(Qt.AlignRight)
        breakPoint4_sb_hb.addWidget(breakPoint4_min_lb)
        breakPoint4_sb_hb.addWidget(self.breakPoint4_sb)
        breakPoint4_sb_hb.addWidget(breakPoint4_max_lb)
        breakPoint4_vb.addLayout(breakPoint4_hb)
        breakPoint4_vb.addWidget(self.breakPoint4_sld)
        breakPoint4_vb.addLayout(breakPoint4_sb_hb)
        self.slider_hb.addLayout(breakPoint4_vb)

        EEIR_hb = QHBoxLayout()
        EEIR_vb = QVBoxLayout()
        EEIR_lb_hb = QGridLayout()
        self.EEIR_color_btn = QPushButton()
        self.EEIR_color_btn.setFixedHeight(15)
        self.EEIR_color_btn.setFixedWidth(20)
        self.EEIR_color_btn.setStyleSheet("background-color: {}".format(self.R_curve_color[0]))
        self.EEIR_color_btn.clicked.connect(lambda: self.setColor("EEIR"))
        EEIR_lb = QLabel("THz")
        EEIR_lb2 = QLabel("")
        EEIR_lb_hb.addWidget(self.EEIR_color_btn, 0, 0, 1, 1, Qt.AlignRight)
        EEIR_lb_hb.addWidget(EEIR_lb, 0, 1, 1, 1, Qt.AlignCenter)
        EEIR_lb_hb.addWidget(EEIR_lb2, 0, 2, 1, 1, Qt.AlignCenter)
        EEIR_offset_hb = QHBoxLayout()
        EEIR_offset_lb = QLabel("offset")
        self.EEIR_offset_sld = QDoubleSlider(Qt.Horizontal)
        # self.EEIR_offset_sld.setEnabled(False)
        self.EEIR_offset_sld.setRange(-0.5, 0.5)
        self.EEIR_offset_sld.setValue(0)
        self.EEIR_offset_sld.setSingleStep(20)
        self.EEIR_offset_sld.valueChanged.connect(lambda: self.setSliderPos("EEIR", "offset"))
        self.EEIR_offset_sb = QDoubleSpinBox()
        self.EEIR_offset_sb.setRange(-0.5, 0.5)
        self.EEIR_offset_sb.setDecimals(4)
        self.EEIR_offset_sb.setSingleStep(0.01)
        self.EEIR_offset_sb.setValue(0)
        self.EEIR_offset_sb.setFixedWidth(60)
        self.EEIR_offset_sb.editingFinished.connect(lambda: self.setSbPos("EEIR", "offset"))
        EEIR_offset_hb.addWidget(EEIR_offset_lb)
        EEIR_offset_hb.addWidget(self.EEIR_offset_sld)
        EEIR_offset_hb.addWidget(self.EEIR_offset_sb)
        EEIR_multiplier_hb = QHBoxLayout()
        EEIR_multiplier_lb = QLabel("multiplier")
        self.EEIR_multiplier_sld = QDoubleSlider(Qt.Horizontal)
        # self.EEIR_offset_sld.setEnabled(False)
        self.EEIR_multiplier_sld.setRange(0, 2)
        self.EEIR_multiplier_sld.setValue(1)
        self.EEIR_multiplier_sld.setSingleStep(20)
        self.EEIR_multiplier_sld.valueChanged.connect(lambda: self.setSliderPos("EEIR", "multiplier"))
        self.EEIR_multiplier_sb = QDoubleSpinBox()
        self.EEIR_multiplier_sb.setRange(0, 2)
        self.EEIR_multiplier_sb.setDecimals(4)
        self.EEIR_multiplier_sb.setSingleStep(0.01)
        self.EEIR_multiplier_sb.setValue(1)
        self.EEIR_multiplier_sb.setFixedWidth(60)
        self.EEIR_multiplier_sb.editingFinished.connect(lambda: self.setSbPos("EEIR", "multiplier"))
        EEIR_multiplier_hb.addWidget(EEIR_multiplier_lb)
        EEIR_multiplier_hb.addWidget(self.EEIR_multiplier_sld)
        EEIR_multiplier_hb.addWidget(self.EEIR_multiplier_sb)
        EEIR_R_hb = QHBoxLayout()
        self.EEIR_reset_btn = QPushButton("Reset")
        self.EEIR_reset_btn.setFixedWidth(50)
        self.EEIR_reset_btn.clicked.connect(lambda: self.reset(0))
        self.EEIR_R_lb = QLabel(u'\u274c')
        self.EEIR_R_lb.setAlignment(Qt.AlignRight)
        self.EEIR_R_lb.setFixedHeight(15)
        self.EEIR_R_btn = QPushButton("Load data")
        self.EEIR_R_btn.clicked.connect(lambda: self.load_reflectance(0))
        self.EEIR_path_lb = QLabel()
        EEIR_R_hb.addWidget(self.EEIR_reset_btn)
        EEIR_R_hb.addWidget(self.EEIR_R_lb)
        EEIR_R_hb.addWidget(self.EEIR_R_btn)
        EEIR_vb.addLayout(EEIR_lb_hb)
        EEIR_vb.addLayout(EEIR_offset_hb)
        EEIR_vb.addLayout(EEIR_multiplier_hb)
        EEIR_vb.addLayout(EEIR_R_hb)
        EEIR_vb.addWidget(self.EEIR_path_lb)
        EEIR_hb.addLayout(EEIR_vb)
        EEIR_Separador = QFrame()
        EEIR_Separador.setFrameShape(QFrame.VLine)
        EEIR_Separador.setLineWidth(1)
        EEIR_hb.addWidget(EEIR_Separador)
        main_grid.addLayout(EEIR_hb, 3, 0, 1, 1, Qt.AlignCenter)

        FIR_hb = QHBoxLayout()
        FIR_vb = QVBoxLayout()
        FIR_lb_hb = QGridLayout()
        self.FIR_color_btn = QPushButton()
        self.FIR_color_btn.setFixedHeight(15)
        self.FIR_color_btn.setFixedWidth(20)
        self.FIR_color_btn.setStyleSheet("background-color: {}".format(self.R_curve_color[1]))
        self.FIR_color_btn.clicked.connect(lambda: self.setColor("FIR"))
        FIR_lb = QLabel("FIR")
        FIR_lb2 = QLabel("")
        FIR_lb_hb.addWidget(self.FIR_color_btn, 0, 0, 1, 1, Qt.AlignRight)
        FIR_lb_hb.addWidget(FIR_lb, 0, 1, 1, 1, Qt.AlignCenter)
        FIR_lb_hb.addWidget(FIR_lb2, 0, 2, 1, 1, Qt.AlignCenter)
        FIR_offset_hb = QHBoxLayout()
        FIR_offset_lb = QLabel("offset")
        self.FIR_offset_sld = QDoubleSlider(Qt.Horizontal)
        # self.EEIR_offset_sld.setEnabled(False)
        self.FIR_offset_sld.setRange(-0.5, 0.5)
        self.FIR_offset_sld.setValue(0)
        self.FIR_offset_sld.setSingleStep(20)
        self.FIR_offset_sld.valueChanged.connect(lambda: self.setSliderPos("FIR", "offset"))
        self.FIR_offset_sb = QDoubleSpinBox()
        self.FIR_offset_sb.setRange(-0.5, 0.5)
        self.FIR_offset_sb.setDecimals(4)
        self.FIR_offset_sb.setSingleStep(0.01)
        self.FIR_offset_sb.setValue(0)
        self.FIR_offset_sb.setFixedWidth(60)
        self.FIR_offset_sb.editingFinished.connect(lambda: self.setSbPos("FIR", "offset"))
        FIR_offset_hb.addWidget(FIR_offset_lb)
        FIR_offset_hb.addWidget(self.FIR_offset_sld)
        FIR_offset_hb.addWidget(self.FIR_offset_sb)
        FIR_multiplier_hb = QHBoxLayout()
        FIR_multiplier_lb = QLabel("multiplier")
        self.FIR_multiplier_sld = QDoubleSlider(Qt.Horizontal)
        # self.EEIR_offset_sld.setEnabled(False)
        self.FIR_multiplier_sld.setRange(0, 2)
        self.FIR_multiplier_sld.setValue(1)
        self.FIR_multiplier_sld.setSingleStep(20)
        self.FIR_multiplier_sld.valueChanged.connect(lambda: self.setSliderPos("FIR", "multiplier"))
        self.FIR_multiplier_sb = QDoubleSpinBox()
        self.FIR_multiplier_sb.setRange(0, 2)
        self.FIR_multiplier_sb.setDecimals(4)
        self.FIR_multiplier_sb.setSingleStep(0.01)
        self.FIR_multiplier_sb.setValue(1)
        self.FIR_multiplier_sb.setFixedWidth(60)
        self.FIR_multiplier_sb.editingFinished.connect(lambda: self.setSbPos("FIR", "multiplier"))
        FIR_multiplier_hb.addWidget(FIR_multiplier_lb)
        FIR_multiplier_hb.addWidget(self.FIR_multiplier_sld)
        FIR_multiplier_hb.addWidget(self.FIR_multiplier_sb)
        FIR_R_hb = QHBoxLayout()
        self.FIR_reset_btn = QPushButton("Reset")
        self.FIR_reset_btn.setFixedWidth(50)
        self.FIR_reset_btn.clicked.connect(lambda: self.reset(1))
        self.FIR_autoFill_cb = QCheckBox("")
        self.FIR_autoFill_combobox = QComboBox()
        self.FIR_autoFill_combobox.addItems(["1st order fill", "2nd order fill", "3rd order fill"])
        self.FIR_autoFill_cb.stateChanged.connect(lambda: self.auto_fill(1, self.FIR_autoFill_cb.isChecked(), self.FIR_autoFill_combobox.currentIndex()))
        self.FIR_autoFill_combobox.currentIndexChanged.connect(lambda: self.auto_fill(1, self.FIR_autoFill_cb.isChecked(), self.FIR_autoFill_combobox.currentIndex()))
        self.FIR_R_lb = QLabel(u'\u274c')
        self.FIR_R_lb.setAlignment(Qt.AlignRight)
        self.FIR_R_lb.setFixedHeight(15)
        self.FIR_R_btn = QPushButton("Load data")
        self.FIR_R_btn.clicked.connect(lambda: self.load_reflectance(1))
        self.FIR_path_lb = QLabel()
        FIR_R_hb.addWidget(self.FIR_reset_btn)
        FIR_R_hb.addWidget(self.FIR_autoFill_cb)
        FIR_R_hb.addWidget(self.FIR_autoFill_combobox)
        FIR_R_hb.addWidget(self.FIR_R_lb)
        FIR_R_hb.addWidget(self.FIR_R_btn)
        FIR_vb.addLayout(FIR_lb_hb)
        FIR_vb.addLayout(FIR_offset_hb)
        FIR_vb.addLayout(FIR_multiplier_hb)
        FIR_vb.addLayout(FIR_R_hb)
        FIR_vb.addWidget(self.FIR_path_lb)
        FIR_hb.addLayout(FIR_vb)
        FIR_Separador = QFrame()
        FIR_Separador.setFrameShape(QFrame.VLine)
        FIR_Separador.setLineWidth(1)
        FIR_hb.addWidget(FIR_Separador)
        main_grid.addLayout(FIR_hb, 3, 1, 1, 1, Qt.AlignCenter)

        MIR_hb = QHBoxLayout()
        MIR_vb = QVBoxLayout()
        MIR_lb_hb = QGridLayout()
        self.MIR_color_btn = QPushButton()
        self.MIR_color_btn.setFixedHeight(15)
        self.MIR_color_btn.setFixedWidth(20)
        self.MIR_color_btn.setStyleSheet("background-color: {}".format(self.R_curve_color[2]))
        self.MIR_color_btn.clicked.connect(lambda: self.setColor("MIR"))
        MIR_lb = QLabel("MIR")
        MIR_lb2 = QLabel("")
        MIR_lb_hb.addWidget(self.MIR_color_btn, 0, 0, 1, 1, Qt.AlignRight)
        MIR_lb_hb.addWidget(MIR_lb, 0, 1, 1, 1, Qt.AlignCenter)
        MIR_lb_hb.addWidget(MIR_lb2, 0, 2, 1, 1, Qt.AlignCenter)
        MIR_offset_hb = QHBoxLayout()
        MIR_offset_lb = QLabel("offset")
        self.MIR_offset_sld = QDoubleSlider(Qt.Horizontal)
        # self.EEIR_offset_sld.setEnabled(False)
        self.MIR_offset_sld.setRange(-0.5, 0.5)
        self.MIR_offset_sld.setValue(0)
        self.MIR_offset_sld.setSingleStep(20)
        self.MIR_offset_sld.valueChanged.connect(lambda: self.setSliderPos("MIR", "offset"))
        self.MIR_offset_sb = QDoubleSpinBox()
        self.MIR_offset_sb.setRange(-0.5, 0.5)
        self.MIR_offset_sb.setDecimals(4)
        self.MIR_offset_sb.setSingleStep(0.01)
        self.MIR_offset_sb.setValue(0)
        self.MIR_offset_sb.setFixedWidth(60)
        self.MIR_offset_sb.editingFinished.connect(lambda: self.setSbPos("MIR", "offset"))
        MIR_offset_hb.addWidget(MIR_offset_lb)
        MIR_offset_hb.addWidget(self.MIR_offset_sld)
        MIR_offset_hb.addWidget(self.MIR_offset_sb)
        MIR_multiplier_hb = QHBoxLayout()
        MIR_multiplier_lb = QLabel("multiplier")
        self.MIR_multiplier_sld = QDoubleSlider(Qt.Horizontal)
        # self.EEIR_offset_sld.setEnabled(False)
        self.MIR_multiplier_sld.setRange(0, 2)
        self.MIR_multiplier_sld.setValue(1)
        self.MIR_multiplier_sld.setSingleStep(20)
        self.MIR_multiplier_sld.valueChanged.connect(lambda: self.setSliderPos("MIR", "multiplier"))
        self.MIR_multiplier_sb = QDoubleSpinBox()
        self.MIR_multiplier_sb.setRange(0, 2)
        self.MIR_multiplier_sb.setDecimals(4)
        self.MIR_multiplier_sb.setSingleStep(0.01)
        self.MIR_multiplier_sb.setValue(1)
        self.MIR_multiplier_sb.setFixedWidth(60)
        self.MIR_multiplier_sb.editingFinished.connect(lambda: self.setSbPos("MIR", "multiplier"))
        MIR_multiplier_hb.addWidget(MIR_multiplier_lb)
        MIR_multiplier_hb.addWidget(self.MIR_multiplier_sld)
        MIR_multiplier_hb.addWidget(self.MIR_multiplier_sb)
        MIR_R_hb = QHBoxLayout()
        self.MIR_reset_btn = QPushButton("Reset")
        self.MIR_reset_btn.setFixedWidth(50)
        self.MIR_reset_btn.clicked.connect(lambda: self.reset(2))
        self.MIR_autoFill_cb = QCheckBox("")
        self.MIR_autoFill_combobox = QComboBox()
        self.MIR_autoFill_combobox.addItems(["1st order fill", "2nd order fill", "3rd order fill"])
        self.MIR_autoFill_cb.stateChanged.connect(lambda: self.auto_fill(2, self.MIR_autoFill_cb.isChecked(), self.MIR_autoFill_combobox.currentIndex()))
        self.MIR_autoFill_combobox.currentIndexChanged.connect(lambda: self.auto_fill(2, self.MIR_autoFill_cb.isChecked(), self.MIR_autoFill_combobox.currentIndex()))
        self.MIR_R_lb = QLabel(u'\u274c')
        self.MIR_R_lb.setAlignment(Qt.AlignRight)
        self.MIR_R_lb.setFixedHeight(15)
        self.MIR_R_btn = QPushButton("Load data")
        self.MIR_R_btn.clicked.connect(lambda: self.load_reflectance(2))
        self.MIR_path_lb = QLabel()
        MIR_R_hb.addWidget(self.MIR_reset_btn)
        MIR_R_hb.addWidget(self.MIR_autoFill_cb)
        MIR_R_hb.addWidget(self.MIR_autoFill_combobox)
        MIR_R_hb.addWidget(self.MIR_R_lb)
        MIR_R_hb.addWidget(self.MIR_R_btn)
        MIR_vb.addLayout(MIR_lb_hb)
        MIR_vb.addLayout(MIR_offset_hb)
        MIR_vb.addLayout(MIR_multiplier_hb)
        MIR_vb.addLayout(MIR_R_hb)
        MIR_vb.addWidget(self.MIR_path_lb)
        MIR_hb.addLayout(MIR_vb)
        MIR_Separador = QFrame()
        MIR_Separador.setFrameShape(QFrame.VLine)
        MIR_Separador.setLineWidth(1)
        MIR_hb.addWidget(MIR_Separador)
        main_grid.addLayout(MIR_hb, 3, 2, 1, 1, Qt.AlignCenter)

        NIR_hb = QHBoxLayout()
        NIR_vb = QVBoxLayout()
        NIR_lb_hb = QGridLayout()
        self.NIR_color_btn = QPushButton()
        self.NIR_color_btn.setFixedHeight(15)
        self.NIR_color_btn.setFixedWidth(20)
        self.NIR_color_btn.setStyleSheet("background-color: {}".format(self.R_curve_color[3]))
        self.NIR_color_btn.clicked.connect(lambda: self.setColor("NIR"))
        NIR_lb = QLabel("NIR")
        NIR_lb2 = QLabel("")
        NIR_lb_hb.addWidget(self.NIR_color_btn, 0, 0, 1, 1, Qt.AlignRight)
        NIR_lb_hb.addWidget(NIR_lb, 0, 1, 1, 1, Qt.AlignCenter)
        NIR_lb_hb.addWidget(NIR_lb2, 0, 2, 1, 1, Qt.AlignCenter)
        NIR_offset_hb = QHBoxLayout()
        NIR_offset_lb = QLabel("offset")
        self.NIR_offset_sld = QDoubleSlider(Qt.Horizontal)
        # self.EEIR_offset_sld.setEnabled(False)
        self.NIR_offset_sld.setRange(-0.5, 0.5)
        self.NIR_offset_sld.setValue(0)
        self.NIR_offset_sld.setSingleStep(20)
        self.NIR_offset_sld.valueChanged.connect(lambda: self.setSliderPos("NIR", "offset"))
        self.NIR_offset_sb = QDoubleSpinBox()
        self.NIR_offset_sb.setRange(-0.5, 0.5)
        self.NIR_offset_sb.setDecimals(4)
        self.NIR_offset_sb.setSingleStep(0.01)
        self.NIR_offset_sb.setValue(0)
        self.NIR_offset_sb.setFixedWidth(60)
        self.NIR_offset_sb.editingFinished.connect(lambda: self.setSbPos("NIR", "offset"))
        NIR_offset_hb.addWidget(NIR_offset_lb)
        NIR_offset_hb.addWidget(self.NIR_offset_sld)
        NIR_offset_hb.addWidget(self.NIR_offset_sb)
        NIR_multiplier_hb = QHBoxLayout()
        NIR_multiplier_lb = QLabel("multiplier")
        self.NIR_multiplier_sld = QDoubleSlider(Qt.Horizontal)
        # self.EEIR_offset_sld.setEnabled(False)
        self.NIR_multiplier_sld.setRange(0, 2)
        self.NIR_multiplier_sld.setValue(1)
        self.NIR_multiplier_sld.setSingleStep(20)
        self.NIR_multiplier_sld.valueChanged.connect(lambda: self.setSliderPos("NIR", "multiplier"))
        self.NIR_multiplier_sb = QDoubleSpinBox()
        self.NIR_multiplier_sb.setRange(0, 2)
        self.NIR_multiplier_sb.setDecimals(4)
        self.NIR_multiplier_sb.setSingleStep(0.01)
        self.NIR_multiplier_sb.setValue(1)
        self.NIR_multiplier_sb.setFixedWidth(60)
        self.NIR_multiplier_sb.editingFinished.connect(lambda: self.setSbPos("NIR", "multiplier"))
        NIR_multiplier_hb.addWidget(NIR_multiplier_lb)
        NIR_multiplier_hb.addWidget(self.NIR_multiplier_sld)
        NIR_multiplier_hb.addWidget(self.NIR_multiplier_sb)
        NIR_R_hb = QHBoxLayout()
        self.NIR_reset_btn = QPushButton("Reset")
        self.NIR_reset_btn.setFixedWidth(50)
        self.NIR_reset_btn.clicked.connect(lambda: self.reset(3))
        self.NIR_autoFill_cb = QCheckBox("")
        self.NIR_autoFill_combobox = QComboBox()
        self.NIR_autoFill_combobox.addItems(["1st order fill", "2nd order fill", "3rd order fill"])
        self.NIR_autoFill_cb.stateChanged.connect(lambda: self.auto_fill(3, self.NIR_autoFill_cb.isChecked(), self.NIR_autoFill_combobox.currentIndex()))
        self.NIR_autoFill_combobox.currentIndexChanged.connect(lambda: self.auto_fill(3, self.NIR_autoFill_cb.isChecked(), self.NIR_autoFill_combobox.currentIndex()))
        self.NIR_R_lb = QLabel(u'\u274c')
        self.NIR_R_lb.setAlignment(Qt.AlignRight)
        self.NIR_R_lb.setFixedHeight(15)
        self.NIR_R_btn = QPushButton("Load data")
        self.NIR_R_btn.clicked.connect(lambda: self.load_reflectance(3))
        self.NIR_path_lb = QLabel()
        NIR_R_hb.addWidget(self.NIR_reset_btn)
        NIR_R_hb.addWidget(self.NIR_autoFill_cb)
        NIR_R_hb.addWidget(self.NIR_autoFill_combobox)
        NIR_R_hb.addWidget(self.NIR_R_lb)
        NIR_R_hb.addWidget(self.NIR_R_btn)
        NIR_vb.addLayout(NIR_lb_hb)
        NIR_vb.addLayout(NIR_offset_hb)
        NIR_vb.addLayout(NIR_multiplier_hb)
        NIR_vb.addLayout(NIR_R_hb)
        NIR_vb.addWidget(self.NIR_path_lb)
        NIR_hb.addLayout(NIR_vb)
        NIR_Separador = QFrame()
        NIR_Separador.setFrameShape(QFrame.VLine)
        NIR_Separador.setLineWidth(1)
        NIR_hb.addWidget(NIR_Separador)
        main_grid.addLayout(NIR_hb, 3, 3, 1, 1, Qt.AlignCenter)

        VIS_hb = QHBoxLayout()
        VIS_vb = QVBoxLayout()
        VIS_lb_hb = QGridLayout()
        self.VIS_color_btn = QPushButton()
        self.VIS_color_btn.setFixedHeight(15)
        self.VIS_color_btn.setFixedWidth(20)
        self.VIS_color_btn.setStyleSheet("background-color: {}".format(self.R_curve_color[4]))
        self.VIS_color_btn.clicked.connect(lambda: self.setColor("VIS"))
        VIS_lb = QLabel("VIS")
        VIS_lb2 = QLabel("")
        VIS_lb_hb.addWidget(self.VIS_color_btn, 0, 0, 1, 1, Qt.AlignRight)
        VIS_lb_hb.addWidget(VIS_lb, 0, 1, 1, 1, Qt.AlignCenter)
        VIS_lb_hb.addWidget(VIS_lb2, 0, 2, 1, 1, Qt.AlignCenter)
        VIS_offset_hb = QHBoxLayout()
        VIS_offset_lb = QLabel("offset")
        self.VIS_offset_sld = QDoubleSlider(Qt.Horizontal)
        # self.EEIR_offset_sld.setEnabled(False)
        self.VIS_offset_sld.setRange(-0.5, 0.5)
        self.VIS_offset_sld.setValue(0)
        self.VIS_offset_sld.setSingleStep(20)
        self.VIS_offset_sld.valueChanged.connect(lambda: self.setSliderPos("VIS", "offset"))
        self.VIS_offset_sb = QDoubleSpinBox()
        self.VIS_offset_sb.setRange(-0.5, 0.5)
        self.VIS_offset_sb.setDecimals(4)
        self.VIS_offset_sb.setSingleStep(0.01)
        self.VIS_offset_sb.setValue(0)
        self.VIS_offset_sb.setFixedWidth(60)
        self.VIS_offset_sb.editingFinished.connect(lambda: self.setSbPos("VIS", "offset"))
        VIS_offset_hb.addWidget(VIS_offset_lb)
        VIS_offset_hb.addWidget(self.VIS_offset_sld)
        VIS_offset_hb.addWidget(self.VIS_offset_sb)
        VIS_multiplier_hb = QHBoxLayout()
        VIS_multiplier_lb = QLabel("multiplier")
        self.VIS_multiplier_sld = QDoubleSlider(Qt.Horizontal)
        # self.EEIR_offset_sld.setEnabled(False)
        self.VIS_multiplier_sld.setRange(0, 2)
        self.VIS_multiplier_sld.setValue(1)
        self.VIS_multiplier_sld.setSingleStep(20)
        self.VIS_multiplier_sld.valueChanged.connect(lambda: self.setSliderPos("VIS", "multiplier"))
        self.VIS_multiplier_sb = QDoubleSpinBox()
        self.VIS_multiplier_sb.setRange(0, 2)
        self.VIS_multiplier_sb.setDecimals(4)
        self.VIS_multiplier_sb.setSingleStep(0.01)
        self.VIS_multiplier_sb.setValue(1)
        self.VIS_multiplier_sb.setFixedWidth(60)
        self.VIS_multiplier_sb.editingFinished.connect(lambda: self.setSbPos("VIS", "multiplier"))
        VIS_multiplier_hb.addWidget(VIS_multiplier_lb)
        VIS_multiplier_hb.addWidget(self.VIS_multiplier_sld)
        VIS_multiplier_hb.addWidget(self.VIS_multiplier_sb)
        VIS_R_hb = QHBoxLayout()
        self.VIS_reset_btn = QPushButton("Reset")
        self.VIS_reset_btn.setFixedWidth(50)
        self.VIS_reset_btn.clicked.connect(lambda: self.reset(4))
        self.VIS_removeHeNe_cb = QCheckBox("Remove HeNe")
        self.VIS_removeHeNe_cb.stateChanged.connect(self.remove_HeNe)
        self.VIS_R_lb = QLabel(u'\u274c')
        self.VIS_R_lb.setAlignment(Qt.AlignRight)
        self.VIS_R_lb.setFixedHeight(15)
        self.VIS_R_btn = QPushButton("Load data")
        self.VIS_R_btn.clicked.connect(lambda: self.load_reflectance(4))
        self.VIS_path_lb = QLabel()
        VIS_R_hb.addWidget(self.VIS_reset_btn)
        VIS_R_hb.addWidget(self.VIS_removeHeNe_cb)
        VIS_R_hb.addWidget(self.VIS_R_lb)
        VIS_R_hb.addWidget(self.VIS_R_btn)
        VIS_vb.addLayout(VIS_lb_hb)
        VIS_vb.addLayout(VIS_offset_hb)
        VIS_vb.addLayout(VIS_multiplier_hb)
        VIS_vb.addLayout(VIS_R_hb)
        VIS_vb.addWidget(self.VIS_path_lb)
        VIS_hb.addLayout(VIS_vb)
        main_grid.addLayout(VIS_hb, 3, 4, 1, 1, Qt.AlignCenter)

        self.initialize_graph()

    def loadpickle(self, fname):
        with open(fname, "rb") as fp:
            data = pickle.load(fp)
        return data

    def setColor(self, id):
        color = QColorDialog.getColor().name()
        if id == "breakPoint1":
            self.breakPoint1_color_btn.setStyleSheet("background-color: {}".format(color))
            self.break_line_color[0] = color
            if self.break_line[0] is not None:
                self.break_line[0].set_color(color)
                self.F.draw()
        elif id == "breakPoint2":
            self.breakPoint2_color_btn.setStyleSheet("background-color: {}".format(color))
            self.break_line_color[1] = color
            if self.break_line[1] is not None:
                self.break_line[1].set_color(color)
                self.F.draw()
        elif id == "breakPoint3":
            self.breakPoint3_color_btn.setStyleSheet("background-color: {}".format(color))
            self.break_line_color[2] = color
            if self.break_line[2] is not None:
                self.break_line[2].set_color(color)
                self.F.draw()
        elif id == "breakPoint4":
            self.breakPoint4_color_btn.setStyleSheet("background-color: {}".format(color))
            self.break_line_color[3] = color
            if self.break_line[3] is not None:
                self.break_line[3].set_color(color)
                self.F.draw()
        elif id == "EEIR":
            self.EEIR_color_btn.setStyleSheet("background-color: {}".format(color))
            self.R_curve_color[0] = color
            if self.R_curve[0] is not None:
                self.R_curve[0].set_color(color)
                self.F.draw()
        elif id == "FIR":
            self.FIR_color_btn.setStyleSheet("background-color: {}".format(color))
            self.R_curve_color[1] = color
            if self.R_curve[1] is not None:
                self.R_curve[1].set_color(color)
                self.F.draw()
        elif id == "MIR":
            self.MIR_color_btn.setStyleSheet("background-color: {}".format(color))
            self.R_curve_color[2] = color
            if self.R_curve[2] is not None:
                self.R_curve[2].set_color(color)
                self.F.draw()
        elif id == "NIR":
            self.NIR_color_btn.setStyleSheet("background-color: {}".format(color))
            self.R_curve_color[3] = color
            if self.R_curve[3] is not None:
                self.R_curve[3].set_color(color)
                self.F.draw()
        elif id == "VIS":
            self.VIS_color_btn.setStyleSheet("background-color: {}".format(color))
            self.R_curve_color[4] = color
            if self.R_curve[4] is not None:
                self.R_curve[4].set_color(color)
                self.F.draw()

    def setSliderPos(self, id, type):
        if type == "offset":
            if id == "EEIR":
                self.EEIR_offset_sb.setValue(self.EEIR_offset_sld.value())
                self.offset[0] = self.EEIR_offset_sld.value()
                self.scale_graph(0, self.EEIR_offset_sb.value(), self.EEIR_multiplier_sb.value())
            elif id == "FIR":
                self.FIR_offset_sb.setValue(self.FIR_offset_sld.value())
                self.offset[1] = self.FIR_offset_sld.value()
                self.scale_graph(1, self.FIR_offset_sb.value(), self.FIR_multiplier_sb.value())
            elif id == "MIR":
                self.MIR_offset_sb.setValue(self.MIR_offset_sld.value())
                self.offset[2] = self.MIR_offset_sld.value()
                self.scale_graph(2, self.MIR_offset_sb.value(), self.MIR_multiplier_sb.value())
            elif id == "NIR":
                self.NIR_offset_sb.setValue(self.NIR_offset_sld.value())
                self.offset[3] = self.NIR_offset_sld.value()
                self.scale_graph(3, self.NIR_offset_sb.value(), self.NIR_multiplier_sb.value())
            elif id == "VIS":
                self.VIS_offset_sb.setValue(self.VIS_offset_sld.value())
                self.offset[4] = self.VIS_offset_sld.value()
                self.scale_graph(4, self.VIS_offset_sb.value(), self.VIS_multiplier_sb.value())
        elif type == "multiplier":
            if id == "EEIR":
                self.EEIR_multiplier_sb.setValue(self.EEIR_multiplier_sld.value())
                self.multiplier[0] = self.EEIR_multiplier_sld.value()
                self.scale_graph(0, self.EEIR_offset_sb.value(), self.EEIR_multiplier_sb.value())
            elif id == "FIR":
                self.FIR_multiplier_sb.setValue(self.FIR_multiplier_sld.value())
                self.multiplier[1] = self.FIR_multiplier_sld.value()
                self.scale_graph(1, self.FIR_offset_sb.value(), self.FIR_multiplier_sb.value())
            elif id == "MIR":
                self.MIR_multiplier_sb.setValue(self.MIR_multiplier_sld.value())
                self.multiplier[2] = self.MIR_multiplier_sld.value()
                self.scale_graph(2, self.MIR_offset_sb.value(), self.MIR_multiplier_sb.value())
            elif id == "NIR":
                self.NIR_multiplier_sb.setValue(self.NIR_multiplier_sld.value())
                self.multiplier[3] = self.NIR_multiplier_sld.value()
                self.scale_graph(3, self.NIR_offset_sb.value(), self.NIR_multiplier_sb.value())
            elif id == "VIS":
                self.VIS_multiplier_sb.setValue(self.VIS_multiplier_sld.value())
                self.multiplier[4] = self.VIS_multiplier_sld.value()
                self.scale_graph(4, self.VIS_offset_sb.value(), self.VIS_multiplier_sb.value())
        elif type == "breakpoint":
            if id == "1":
                if self.breakPoint1_sld.value() > self.breakPoint2_sld.value():
                    self.breakPoint1_sld.setValue(self.breakPoint2_sld.value())
                self.breakPoint1_sb.setValue(self.breakPoint1_sld.value())
                self.merge_graph(id, self.breakPoint1_sld.value(), None, self.breakPoint2_sb.value())
                self.scale_graph(0, self.EEIR_offset_sb.value(), self.EEIR_multiplier_sb.value())
                self.scale_graph(1, self.FIR_offset_sb.value(), self.FIR_multiplier_sb.value())
            elif id == "2":
                if self.breakPoint2_sld.value() > self.breakPoint3_sld.value():
                    self.breakPoint2_sld.setValue(self.breakPoint3_sld.value())
                elif self.breakPoint2_sld.value() < self.breakPoint1_sld.value():
                    self.breakPoint2_sld.setValue(self.breakPoint1_sld.value())
                self.breakPoint2_sb.setValue(self.breakPoint2_sld.value())
                self.merge_graph(id, self.breakPoint2_sld.value(), self.breakPoint1_sb.value(), self.breakPoint3_sb.value())
                self.scale_graph(1, self.FIR_offset_sb.value(), self.FIR_multiplier_sb.value())
                self.scale_graph(2, self.MIR_offset_sb.value(), self.MIR_multiplier_sb.value())
            elif id == "3":
                if self.breakPoint3_sld.value() > self.breakPoint4_sld.value():
                    self.breakPoint3_sld.setValue(self.breakPoint4_sld.value())
                elif self.breakPoint3_sld.value() < self.breakPoint2_sld.value():
                    self.breakPoint3_sld.setValue(self.breakPoint2_sld.value())
                self.breakPoint3_sb.setValue(self.breakPoint3_sld.value())
                self.merge_graph(id, self.breakPoint3_sld.value(), self.breakPoint2_sb.value(), self.breakPoint4_sb.value())
                self.scale_graph(2, self.MIR_offset_sb.value(), self.MIR_multiplier_sb.value())
                self.scale_graph(3, self.NIR_offset_sb.value(), self.NIR_multiplier_sb.value())
            elif id == "4":
                if self.breakPoint4_sld.value() < self.breakPoint3_sld.value():
                    self.breakPoint4_sld.setValue(self.breakPoint3_sld.value())
                self.breakPoint4_sb.setValue(self.breakPoint4_sld.value())
                self.merge_graph(id, self.breakPoint4_sld.value(), self.breakPoint3_sb.value(), None)
                self.scale_graph(3, self.NIR_offset_sb.value(), self.NIR_multiplier_sb.value())
                self.scale_graph(4, self.VIS_offset_sb.value(), self.VIS_multiplier_sb.value())

    def setSbPos(self, id, type):
        if type == "offset":
            if id == "EEIR":
                self.EEIR_offset_sld.setValue(self.EEIR_offset_sb.value())
                self.offset[0] = self.EEIR_offset_sb.value()
                self.scale_graph(0, self.EEIR_offset_sb.value(), self.EEIR_multiplier_sb.value())
            elif id == "FIR":
                self.FIR_offset_sld.setValue(self.FIR_offset_sb.value())
                self.offset[1] = self.FIR_offset_sb.value()
                self.scale_graph(1, self.FIR_offset_sb.value(), self.FIR_multiplier_sb.value())
            elif id == "MIR":
                self.MIR_offset_sld.setValue(self.MIR_offset_sb.value())
                self.offset[2] = self.MIR_offset_sb.value()
                self.scale_graph(2, self.MIR_offset_sb.value(), self.MIR_multiplier_sb.value())
            elif id == "NIR":
                self.NIR_offset_sld.setValue(self.NIR_offset_sb.value())
                self.offset[3] = self.NIR_offset_sb.value()
                self.scale_graph(3, self.NIR_offset_sb.value(), self.NIR_multiplier_sb.value())
            elif id == "VIS":
                self.VIS_offset_sld.setValue(self.VIS_offset_sb.value())
                self.offset[4] = self.VIS_offset_sb.value()
                self.scale_graph(4, self.VIS_offset_sb.value(), self.VIS_multiplier_sb.value())
        elif type == "multiplier":
            if id == "EEIR":
                self.EEIR_multiplier_sld.setValue(self.EEIR_multiplier_sb.value())
                self.multiplier[0] = self.EEIR_multiplier_sb.value()
                self.scale_graph(0, self.EEIR_offset_sb.value(), self.EEIR_multiplier_sb.value())
            elif id == "FIR":
                self.FIR_multiplier_sld.setValue(self.FIR_multiplier_sb.value())
                self.multiplier[1] = self.FIR_multiplier_sb.value()
                self.scale_graph(1, self.FIR_offset_sb.value(), self.FIR_multiplier_sb.value())
            elif id == "MIR":
                self.MIR_multiplier_sld.setValue(self.MIR_multiplier_sb.value())
                self.multiplier[2] = self.MIR_multiplier_sb.value()
                self.scale_graph(2, self.MIR_offset_sb.value(), self.MIR_multiplier_sb.value())
            elif id == "NIR":
                self.NIR_multiplier_sld.setValue(self.NIR_multiplier_sb.value())
                self.multiplier[3] = self.NIR_multiplier_sb.value()
                self.scale_graph(3, self.NIR_offset_sb.value(), self.NIR_multiplier_sb.value())
            elif id == "VIS":
                self.VIS_multiplier_sld.setValue(self.VIS_multiplier_sb.value())
                self.multiplier[4] = self.VIS_multiplier_sb.value()
                self.scale_graph(4, self.VIS_offset_sb.value(), self.VIS_multiplier_sb.value())
        elif type == "breakpoint":
            if id == "1":
                if self.breakPoint1_sb.value() > self.breakPoint2_sb.value():
                    self.breakPoint1_sb.setValue(self.breakPoint2_sb.value())
                self.breakPoint1_sld.setValue(self.breakPoint1_sb.value())
                self.merge_graph(id, self.breakPoint1_sb.value(), None, self.breakPoint2_sb.value())
                self.scale_graph(0, self.EEIR_offset_sb.value(), self.EEIR_multiplier_sb.value())
                self.scale_graph(1, self.FIR_offset_sb.value(), self.FIR_multiplier_sb.value())
            elif id == "2":
                if self.breakPoint2_sb.value() > self.breakPoint3_sb.value():
                    self.breakPoint2_sb.setValue(self.breakPoint3_sb.value())
                elif self.breakPoint2_sb.value() < self.breakPoint1_sb.value():
                    self.breakPoint2_sb.setValue(self.breakPoint1_sb.value())
                self.breakPoint2_sld.setValue(self.breakPoint2_sb.value())
                self.merge_graph(id, self.breakPoint2_sb.value(), self.breakPoint1_sb.value(), self.breakPoint3_sb.value())
                self.scale_graph(1, self.FIR_offset_sb.value(), self.FIR_multiplier_sb.value())
                self.scale_graph(2, self.MIR_offset_sb.value(), self.MIR_multiplier_sb.value())
            elif id == "3":
                if self.breakPoint3_sb.value() > self.breakPoint4_sb.value():
                    self.breakPoint3_sb.setValue(self.breakPoint4_sb.value())
                elif self.breakPoint3_sb.value() < self.breakPoint2_sb.value():
                    self.breakPoint3_sb.setValue(self.breakPoint2_sb.value())
                self.breakPoint3_sld.setValue(self.breakPoint3_sb.value())
                self.merge_graph(id, self.breakPoint3_sb.value(), self.breakPoint2_sb.value(), self.breakPoint4_sb.value())
                self.scale_graph(2, self.MIR_offset_sb.value(), self.MIR_multiplier_sb.value())
                self.scale_graph(3, self.NIR_offset_sb.value(), self.NIR_multiplier_sb.value())
            elif id == "4":
                if self.breakPoint4_sb.value() < self.breakPoint3_sb.value():
                    self.breakPoint4_sb.setValue(self.breakPoint3_sb.value())
                self.breakPoint4_sld.setValue(self.breakPoint4_sb.value())
                self.merge_graph(id, self.breakPoint4_sb.value(), self.breakPoint3_sb.value(), None)
                self.scale_graph(3, self.NIR_offset_sb.value(), self.NIR_multiplier_sb.value())
                self.scale_graph(4, self.VIS_offset_sb.value(), self.VIS_multiplier_sb.value())

    def reset(self, code):
        if code == 0:
            self.EEIR_offset_sb.setValue(0)
            self.EEIR_offset_sld.setValue(0)
            self.offset[0] = 0
            self.EEIR_multiplier_sld.setValue(1)
            self.EEIR_multiplier_sb.setValue(1)
            self.multiplier[0] = 1
            self.scale_graph(0, self.EEIR_offset_sb.value(), self.EEIR_multiplier_sb.value())
        elif code == 1:
            self.FIR_offset_sb.setValue(0)
            self.FIR_offset_sld.setValue(0)
            self.offset[1] = 0
            self.FIR_multiplier_sld.setValue(1)
            self.FIR_multiplier_sb.setValue(1)
            self.multiplier[1] = 1
            self.scale_graph(1, self.FIR_offset_sb.value(), self.FIR_multiplier_sb.value())
        elif code == 2:
            self.MIR_offset_sb.setValue(0)
            self.MIR_offset_sld.setValue(0)
            self.offset[2] = 0
            self.MIR_multiplier_sld.setValue(1)
            self.MIR_multiplier_sb.setValue(1)
            self.multiplier[2] = 1
            self.scale_graph(2, self.MIR_offset_sb.value(), self.MIR_multiplier_sb.value())
        elif code == 3:
            self.NIR_offset_sb.setValue(0)
            self.NIR_offset_sld.setValue(0)
            self.offset[3] = 0
            self.NIR_multiplier_sld.setValue(1)
            self.NIR_multiplier_sb.setValue(1)
            self.multiplier[3] = 1
            self.scale_graph(3, self.NIR_offset_sb.value(), self.NIR_multiplier_sb.value())
        elif code == 4:
            self.VIS_offset_sb.setValue(0)
            self.VIS_offset_sld.setValue(0)
            self.offset[4] = 0
            self.VIS_multiplier_sld.setValue(1)
            self.VIS_multiplier_sb.setValue(1)
            self.multiplier[4] = 1
            self.scale_graph(4, self.VIS_offset_sb.value(), self.VIS_multiplier_sb.value())

    def read_refFIT_data(self, path):
        try:
            file = np.loadtxt(path).transpose()
            freq = file[0]
            reflectance = file[1]
        except:
            try:
                file = np.loadtxt(path, delimiter=",").transpose()
                freq = file[0]
                reflectance = file[1]
            except:
                try:
                    file = np.loadtxt(path, delimiter=" ").transpose()
                    freq = file[0]
                    reflectance = file[1]
                except:
                    return
        return np.array(reflectance), np.array(freq)

    def load_reflectance(self, code):
        try:
            path = QFileDialog.getOpenFileName(self, "Select a file", r"~\PycharmProjects/Transfer Matrix Method", "Text Files (*.txt *.csv *.dat)")[0]
            filename = os.path.basename(path)
            if path == "":
                if code == 0:
                    self.EEIR_R_lb.setText(u'\u274c')
                    self.EEIR_path_lb.setText("")
                elif code == 1:
                    self.FIR_R_lb.setText(u'\u274c')
                    self.FIR_path_lb.setText("")
                elif code == 2:
                    self.MIR_R_lb.setText(u'\u274c')
                    self.NIR_autoFill_cb.setChecked(False)
                    self.MIR_path_lb.setText("")
                elif code == 3:
                    self.NIR_R_lb.setText(u'\u274c')
                    self.NIR_autoFill_cb.setEnabled(True)
                    self.NIR_path_lb.setText("")
                elif code == 4:
                    self.VIS_R_lb.setText(u'\u274c')
                    self.NIR_autoFill_cb.setChecked(False)
                    self.VIS_path_lb.setText("")
                self.reflectance[code] = []
                self.freq[code] = []
                self.range[code] = []
            else:
                if code == 0:
                    self.EEIR_R_lb.setText(u'\u2705')
                    self.EEIR_path_lb.setText(filename)
                elif code == 1:
                    self.FIR_R_lb.setText(u'\u2705')
                    self.FIR_path_lb.setText(filename)
                elif code == 2:
                    self.MIR_R_lb.setText(u'\u2705')
                    self.MIR_path_lb.setText(filename)
                elif code == 3:
                    self.NIR_autoFill_cb.setChecked(False)
                    self.NIR_autoFill_cb.setEnabled(False)
                    self.NIR_R_lb.setText(u'\u2705')
                    self.NIR_path_lb.setText(filename)
                elif code == 4:
                    self.VIS_R_lb.setText(u'\u2705')
                    self.VIS_path_lb.setText(filename)
                reflectance, freq = self.read_refFIT_data(path)
                self.reflectance[code] = reflectance
                self.freq[code] = freq
            self.renew_graph()
            self.reset(code)
        except:
            QMessageBox.warning(self, "Load reflectance", "You are not selecting a correct file!")
            return

    def remake_auto_fill_data(self, code):
        if self.auto_fill_order[code-1] == 1:
            kind = "quadratic"
        elif self.auto_fill_order[code-1] == 2:
            kind = "cubic"
        else:
            kind = "linear"
        # f = interp1d(self.is_auto_fill[code][0], self.is_auto_fill[code][1])
        f = interp1d(np.append(self.is_auto_fill[code][0][0], self.is_auto_fill[code][0][1]), np.append(self.is_auto_fill[code][1][0], self.is_auto_fill[code][1][1]), kind=kind)
        # freq = np.arange(self.is_auto_fill[code][0][0], self.is_auto_fill[code][0][1], self.freq[code-1][-1]-self.freq[code-1][-2])
        freq = np.arange(self.is_auto_fill[code][0][0][-1], self.is_auto_fill[code][0][1][0], self.freq[code-1][-1]-self.freq[code-1][-2])
        self.freq[code] = freq
        self.reflectance[code] = f(freq)

    def auto_fill(self, code, auto, order):
        self.auto_fill_order[code-1] = order
        if auto and len(self.reflectance[code-1]) > 0 and len(self.reflectance[code+1]) > 0:
            self.reset(code-1)
            self.reset(code+1)
            # self.is_auto_fill[code] = [[self.freq[code-1][-1], self.freq[code+1][0]], [self.reflectance[code-1][-1], self.reflectance[code+1][0]]]
            self.is_auto_fill[code] = [[self.freq[code-1][-100:], self.freq[code+1][:100]], [self.reflectance[code-1][-100:], self.reflectance[code+1][:100]]]
            self.remake_auto_fill_data(code)
            if code == 0:
                self.EEIR_R_lb.setText(u'\u2705')
            elif code == 1:
                self.FIR_R_lb.setText(u'\u2705')
            elif code == 2:
                self.MIR_R_lb.setText(u'\u2705')
            elif code == 3:
                self.NIR_R_lb.setText(u'\u2705')
            elif code == 4:
                self.VIS_R_lb.setText(u'\u2705')
        else:
            self.reflectance[code] = []
            self.freq[code] = []
            self.is_auto_fill[code] = []
            if code == 0:
                self.EEIR_R_lb.setText(u'\u274c')
            elif code == 1:
                self.FIR_R_lb.setText(u'\u274c')
            elif code == 2:
                self.MIR_R_lb.setText(u'\u274c')
            elif code == 3:
                self.NIR_R_lb.setText(u'\u274c')
            elif code == 4:
                self.VIS_R_lb.setText(u'\u274c')
        self.renew_graph()
        self.merge_graph("1", self.breakPoint1_sb.value(), None, self.breakPoint2_sb.value())
        self.merge_graph("2", self.breakPoint2_sb.value(), self.breakPoint1_sb.value(), self.breakPoint3_sb.value())
        self.merge_graph("3", self.breakPoint3_sb.value(), self.breakPoint2_sb.value(), self.breakPoint4_sb.value())
        self.merge_graph("4", self.breakPoint4_sb.value(), self.breakPoint3_sb.value(), None)
        self.reset(code)

    def remove_HeNe(self):
        if self.VIS_removeHeNe_cb.isChecked() and len(self.reflectance[4]) > 0:
            index = []
            for i in range(len(self.freq[4])):
                if 15785 <= self.freq[4][i] <= 15815:
                    index.append(i)
            func = interp1d([self.freq[4][int(min(index)-1)], self.freq[4][(max(index)+1)]], [self.reflectance[4][int(min(index)-1)], self.reflectance[4][(max(index)+1)]])
            for i in index:
                self.reflectance[4][i] = func(self.freq[4][i])
            self.scale_graph(4, self.VIS_offset_sb.value(), self.VIS_multiplier_sb.value())
            self.F.draw()

    def initialize_graph(self):
        self.figure.clf()
        self.axes = self.figure.add_subplot()
        self.axes2 = self.axes.twiny()
        # self.axes.set_title("Merge Spec", fontsize=12)
        self.axes.set_xlabel(r'Frequency (cm$^{-1}$)', fontsize=9)
        self.axes.set_ylim([0, 1])
        self.axes.set_xlim([0, 25000])
        self.axes2.set_xlabel(r'Energy (eV)', fontsize=9)
        self.axes2.set_xlim([0, 25000/8065.5])
        self.F.figure.subplots_adjust(left=0.03,
                        bottom=0.1,
                        right=0.97,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
        self.F.draw()

    def renew_graph(self):
        for i in range(len(self.reflectance)-1):
            if self.break_line[i] is not None:
                    self.break_line[i].remove()
                    self.break_line[i] = None
            if len(self.reflectance[i]) > 0 and len(self.reflectance[i+1]) > 0:
                exec("self.breakPoint{}_sld.setEnabled(True)".format(i+1))
                exec("self.breakPoint{}_sb.setEnabled(True)".format(i+1))
                exec("self.breakPoint{}_sld.setValue((self.freq[i][-1] + self.freq[i+1][0])/2)".format(i+1))
                exec("self.breakPoint{}_sb.setValue((self.freq[i][-1] + self.freq[i+1][0])/2)".format(i+1))
                if self.break_line[i] is None:
                    self.break_line[i] = self.axes.axvline(x = (self.freq[i][-1] + self.freq[i+1][0])/2, color = self.break_line_color[i], linestyle = '--')
            else:
                exec("self.breakPoint{}_sld.setEnabled(False)".format(i+1))
                exec("self.breakPoint{}_sb.setEnabled(False)".format(i+1))
        for i in range(len(self.reflectance)):
            if self.R_curve[i] is not None:
                self.R_curve[i].remove()
                self.R_curve[i] = None
            if len(self.reflectance[i]) > 0:
                color = self.R_curve_color[i]
                if i == 0:
                    if self.break_line[i] is None:
                        self.range[i] = [np.where(self.freq[i] >= 0)]
                        self.R_curve[i], = self.axes.plot(self.freq[i], self.reflectance[i], color = color, linestyle = '-')
                    else:
                        freq = self.freq[i][np.where(self.freq[i] <= (self.freq[i][-1] + self.freq[i+1][0])/2)]
                        reflectance = self.reflectance[i][np.where(self.freq[i] <= (self.freq[i][-1] + self.freq[i+1][0])/2)]
                        self.range[i] = [np.where(self.freq[i] <= (self.freq[i][-1] + self.freq[i+1][0])/2)]
                        self.R_curve[i], = self.axes.plot(freq, reflectance, color = color, linestyle = '-')
                elif i < 4:
                    if self.break_line[i] is None and self.break_line[i-1] is None:
                        self.range[i] = [np.where(self.freq[i] >= 0)]
                        self.R_curve[i], = self.axes.plot(self.freq[i], self.reflectance[i], color = color, linestyle = '-')
                    elif self.break_line[i-1] is None:
                        freq = self.freq[i][np.where(self.freq[i] <= (self.freq[i][-1] + self.freq[i+1][0])/2)]
                        reflectance = self.reflectance[i][np.where(self.freq[i] <= (self.freq[i][-1] + self.freq[i+1][0])/2)]
                        self.range[i] = [np.where(self.freq[i] <= (self.freq[i][-1] + self.freq[i+1][0])/2)]
                        self.R_curve[i], = self.axes.plot(freq, reflectance, color = color, linestyle = '-')
                    elif self.break_line[i] is None:
                        freq = self.freq[i][np.where(self.freq[i] > (self.freq[i-1][-1] + self.freq[i][0])/2)]
                        reflectance = self.reflectance[i][np.where(self.freq[i] > (self.freq[i-1][-1] + self.freq[i][0])/2)]
                        self.range[i] = [np.where(self.freq[i] > (self.freq[i-1][-1] + self.freq[i][0])/2)]
                        self.R_curve[i], = self.axes.plot(freq, reflectance, color = color, linestyle = '-')
                    else:
                        freq = self.freq[i][np.where(((self.freq[i-1][-1] + self.freq[i][0])/2 < self.freq[i]) & (self.freq[i] <= (self.freq[i][-1] + self.freq[i+1][0])/2))]
                        reflectance = self.reflectance[i][np.where(((self.freq[i-1][-1] + self.freq[i][0])/2 < self.freq[i]) & (self.freq[i] <= (self.freq[i][-1] + self.freq[i+1][0])/2))]
                        self.range[i] = [np.where(((self.freq[i-1][-1] + self.freq[i][0])/2 < self.freq[i]) & (self.freq[i] <= (self.freq[i][-1] + self.freq[i+1][0])/2))]
                        self.R_curve[i], = self.axes.plot(freq, reflectance, color = color, linestyle = '-')
                else:
                    if self.break_line[i-1] is None:
                        self.range[i] = [np.where(self.freq[i] >= 0)]
                        self.R_curve[i], = self.axes.plot(self.freq[i], self.reflectance[i], color = color, linestyle = '-')
                    else:
                        freq = self.freq[i][np.where(self.freq[i] > (self.freq[i-1][-1] + self.freq[i][0])/2)]
                        reflectance = self.reflectance[i][np.where(self.freq[i] > (self.freq[i-1][-1] + self.freq[i][0])/2)]
                        self.range[i] = [np.where(self.freq[i] > (self.freq[i-1][-1] + self.freq[i][0])/2)]
                        self.R_curve[i], = self.axes.plot(freq, reflectance, color = color, linestyle = '-')
        self.scale_graph(0, self.EEIR_offset_sb.value(), self.EEIR_multiplier_sb.value())
        self.scale_graph(1, self.FIR_offset_sb.value(), self.FIR_multiplier_sb.value())
        self.scale_graph(2, self.MIR_offset_sb.value(), self.MIR_multiplier_sb.value())
        self.scale_graph(3, self.NIR_offset_sb.value(), self.NIR_multiplier_sb.value())
        self.scale_graph(4, self.VIS_offset_sb.value(), self.VIS_multiplier_sb.value())
        self.remove_HeNe()
        self.F.draw()

    def merge_graph(self, id, x, left, right):
        i = int(id)-1
        if len(self.is_auto_fill[i]) > 0:
            # self.is_auto_fill[i][0][1] = self.freq[i+1][np.where(self.freq[i+1] > x)][0]
            self.is_auto_fill[i][0][1] = self.freq[i+1][np.where(self.freq[i+1] > x)][:100]
            self.remake_auto_fill_data(i)
        if len(self.is_auto_fill[i+1]) > 0:
            # self.is_auto_fill[i+1][0][0] = self.freq[i][np.where(self.freq[i] <= x)][-1]
            self.is_auto_fill[i+1][0][0] = self.freq[i][np.where(self.freq[i] <= x)][-100:]
            self.remake_auto_fill_data(i+1)
        if self.break_line[i] is not None:
            self.break_line[i].remove()
            self.break_line[i] = None
        if self.R_curve[i] is not None:
                self.R_curve[i].remove()
                self.R_curve[i] = None
        if self.R_curve[i+1] is not None:
                self.R_curve[i+1].remove()
                self.R_curve[i+1] = None
        if len(self.reflectance[i]) > 0 and len(self.reflectance[i+1]) > 0:
            self.break_line[i] = self.axes.axvline(x = x, color = self.break_line_color[i], linestyle = '--')
        if len(self.reflectance[i]) > 0:
            if left is not None and self.break_line[i-1] is not None:
                freq1 = self.freq[i][np.where((self.freq[i] <= x) & (self.freq[i] > left))]
                reflectance1 = self.reflectance[i][np.where((self.freq[i] <= x) & (self.freq[i] > left))]
                self.range[i] = [np.where((self.freq[i] <= x) & (self.freq[i] > left))]
            else:
                freq1 = self.freq[i][np.where(self.freq[i] <= x)]
                reflectance1 = self.reflectance[i][np.where(self.freq[i] <= x)]
                self.range[i] = [np.where(self.freq[i] <= x)]
            self.R_curve[i], = self.axes.plot(freq1, reflectance1, color = self.R_curve_color[i], linestyle = '-')
        if len(self.reflectance[i+1]) > 0:
            if right is not None and self.break_line[i+1] is not None:
                freq2 = self.freq[i+1][np.where((self.freq[i+1] > x) & (self.freq[i+1] <= right))]
                reflectance2 = self.reflectance[i+1][np.where((self.freq[i+1] > x) & (self.freq[i+1] <= right))]
                self.range[i+1] = [np.where((self.freq[i+1] > x) & (self.freq[i+1] <= right))]
            else:
                freq2 = self.freq[i+1][np.where(self.freq[i+1] > x)]
                reflectance2 = self.reflectance[i+1][np.where(self.freq[i+1] > x)]
                self.range[i+1] = [np.where(self.freq[i+1] > x)]
            self.R_curve[i+1], = self.axes.plot(freq2, reflectance2, color = self.R_curve_color[i+1], linestyle = '-')
        self.F.draw()

    def scale_graph(self, i, offset, multiplier):
        if i > 0 and len(self.is_auto_fill[i-1]) > 0:
            # self.is_auto_fill[i-1][1][1] = (np.array(self.reflectance[i][self.range[i][0]])*multiplier+offset)[0]
            self.is_auto_fill[i-1][1][1] = (np.array(self.reflectance[i][self.range[i][0]])*multiplier+offset)[:100]
            self.remake_auto_fill_data(i-1)
            self.R_curve[i-1].set_ydata(np.array(self.reflectance[i-1][self.range[i-1][0]]))
        if i < 4 and len(self.is_auto_fill[i+1]) > 0:
            # self.is_auto_fill[i+1][1][0] = (np.array(self.reflectance[i][self.range[i][0]])*multiplier+offset)[-1]
            self.is_auto_fill[i+1][1][0] = (np.array(self.reflectance[i][self.range[i][0]])*multiplier+offset)[-100:]
            self.remake_auto_fill_data(i+1)
            self.R_curve[i+1].set_ydata(np.array(self.reflectance[i+1][self.range[i+1][0]]))
        if self.R_curve[i] is not None:
            self.R_curve[i].set_ydata(np.array(self.reflectance[i][self.range[i][0]])*multiplier+offset)
            self.F.draw()

    def save_mergedSpec(self):
        path = QFileDialog.getSaveFileName(self, "Save your file", r"~\PycharmProjects/Transfer Matrix Method/merged_spectrum", "TXT Files (*.txt) ;; CSV Files (*.csv) ;; DAT Files (*.dat)")[0]
        if path != "":
            file = open(path, 'w')
            for i in range(len(self.freq)):
                if len(self.freq[i]) > 0:
                    reflectance = np.array(self.reflectance[i][self.range[i][0]]) * self.multiplier[i] + self.offset[i]
                    freq = np.array(self.freq[i][self.range[i][0]])
                    if self.ref_cb.currentText() == "Au":
                        reflectance *= self.Au_refl(freq)
                    elif self.ref_cb.currentText() == "Ag":
                        reflectance *= self.Ag_refl(freq)
                    for j in range(len(freq)):
                        file.write("{}\t{}\n".format(freq[j], reflectance[j]))
            file.close()

    def save_params(self):
        path = QFileDialog.getSaveFileName(self, "Save your file", r"~\PycharmProjects/Transfer Matrix Method/merging_params", "TXT Files (*.txt) ;; CSV Files (*.csv) ;; DAT Files (*.dat)")[0]
        if path != "":
            file = open(path, 'w')
            file.write("Breakpoint1, {}\n".format(self.breakPoint1_sb.value()))
            file.write("Breakpoint2, {}\n".format(self.breakPoint2_sb.value()))
            file.write("Breakpoint3, {}\n".format(self.breakPoint3_sb.value()))
            file.write("Breakpoint4, {}\n".format(self.breakPoint4_sb.value()))
            file.write("THz, {}, {}\n".format(self.EEIR_offset_sb.value(), self.EEIR_multiplier_sb.value()))
            file.write("FIR, {}, {}\n".format(self.FIR_offset_sb.value(), self.FIR_multiplier_sb.value()))
            file.write("MIR, {}, {}\n".format(self.MIR_offset_sb.value(), self.MIR_multiplier_sb.value()))
            file.write("NIR, {}, {}\n".format(self.NIR_offset_sb.value(), self.NIR_multiplier_sb.value()))
            file.write("VIS, {}, {}\n".format(self.VIS_offset_sb.value(), self.VIS_multiplier_sb.value()))
            file.close()

    def save_items(self):
        if self.save_spec_cb.isChecked():
            self.save_mergedSpec()
        if self.save_params_cb.isChecked():
            self.save_params()

    def load_params(self):
        path = QFileDialog.getOpenFileName(self, "Select a file", r"~\PycharmProjects/Transfer Matrix Method/merging_params", "Text Files (*.txt *.csv *.dat)")[0]
        if path != "":
            file = open(path, 'r')
            try:
                for line_index, line_str in enumerate(file):
                    line_list = self.split_string_to_data(line_str)
                    if line_list[0] == "Breakpoint1":
                        self.breakPoint1_sb.setValue(float(line_list[1]))
                        self.breakPoint1_sld.setValue(float(line_list[1]))
                    elif line_list[0] == "Breakpoint2":
                        self.breakPoint2_sb.setValue(float(line_list[1]))
                        self.breakPoint2_sld.setValue(float(line_list[1]))
                    elif line_list[0] == "Breakpoint3":
                        self.breakPoint3_sb.setValue(float(line_list[1]))
                        self.breakPoint3_sld.setValue(float(line_list[1]))
                    elif line_list[0] == "Breakpoint4":
                        self.breakPoint4_sb.setValue(float(line_list[1]))
                        self.breakPoint4_sld.setValue(float(line_list[1]))
                    elif line_list[0] == "EEIR" or line_list[0] == "THz":
                        self.EEIR_offset_sb.setValue(float(line_list[1]))
                        self.EEIR_offset_sld.setValue(float(line_list[1]))
                        self.offset[0] = float(line_list[1])
                        self.EEIR_multiplier_sb.setValue(float(line_list[2]))
                        self.EEIR_multiplier_sld.setValue(float(line_list[2]))
                        self.multiplier[0] = float(line_list[2])
                    elif line_list[0] == "FIR":
                        self.FIR_offset_sb.setValue(float(line_list[1]))
                        self.FIR_offset_sld.setValue(float(line_list[1]))
                        self.offset[1] = float(line_list[1])
                        self.FIR_multiplier_sb.setValue(float(line_list[2]))
                        self.FIR_multiplier_sld.setValue(float(line_list[2]))
                        self.multiplier[1] = float(line_list[2])
                    elif line_list[0] == "MIR":
                        self.MIR_offset_sb.setValue(float(line_list[1]))
                        self.MIR_offset_sld.setValue(float(line_list[1]))
                        self.offset[2] = float(line_list[1])
                        self.MIR_multiplier_sb.setValue(float(line_list[2]))
                        self.MIR_multiplier_sld.setValue(float(line_list[2]))
                        self.multiplier[2] = float(line_list[2])
                    elif line_list[0] == "NIR":
                        self.NIR_offset_sb.setValue(float(line_list[1]))
                        self.NIR_offset_sld.setValue(float(line_list[1]))
                        self.offset[3] = float(line_list[1])
                        self.NIR_multiplier_sb.setValue(float(line_list[2]))
                        self.NIR_multiplier_sld.setValue(float(line_list[2]))
                        self.multiplier[3] = float(line_list[2])
                    elif line_list[0] == "VIS":
                        self.VIS_offset_sb.setValue(float(line_list[1]))
                        self.VIS_offset_sld.setValue(float(line_list[1]))
                        self.offset[4] = float(line_list[1])
                        self.VIS_multiplier_sb.setValue(float(line_list[2]))
                        self.VIS_multiplier_sld.setValue(float(line_list[2]))
                        self.multiplier[4] = float(line_list[2])
                self.merge_graph("1", self.breakPoint1_sb.value(), None, self.breakPoint2_sb.value())
                self.merge_graph("2", self.breakPoint2_sb.value(), self.breakPoint1_sb.value(), self.breakPoint3_sb.value())
                self.merge_graph("3", self.breakPoint3_sb.value(), self.breakPoint2_sb.value(), self.breakPoint4_sb.value())
                self.merge_graph("4", self.breakPoint4_sb.value(), self.breakPoint3_sb.value(), None)
                self.scale_graph(0, self.EEIR_offset_sb.value(), self.EEIR_multiplier_sb.value())
                self.scale_graph(1, self.FIR_offset_sb.value(), self.FIR_multiplier_sb.value())
                self.scale_graph(2, self.MIR_offset_sb.value(), self.MIR_multiplier_sb.value())
                self.scale_graph(3, self.NIR_offset_sb.value(), self.NIR_multiplier_sb.value())
                self.scale_graph(4, self.VIS_offset_sb.value(), self.VIS_multiplier_sb.value())
                self.F.draw()
            except:
                QMessageBox.warning(self, "Load params", "You are not selecting a correct file!")
                return

    def split_string_to_data(self, string):
        string = string.replace('\n', '') # delete tail '\n'
        string = string.replace(',', ' ') # replace ',' by ' '
        string = string.replace('\t', ' ') # replace '\t' by ' '
        string = string.replace(';', ' ') # replace ';' by ' '
        while '  ' in string:
            # replace multiple spaces by one space
            string = string.replace('  ', ' ')
        # split with delimiter ' ' and store them in a list
        var_list = string.split(' ')
        while '' in var_list:
            # remove empty strings from the list
            var_list.remove('')
        return var_list


class QDoubleSlider(QSlider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decimals = 4
        self._max_int = 10 ** self.decimals

        super().setMinimum(0)
        super().setMaximum(self._max_int)

        self._min_value = 0.0
        self._max_value = 1.0

    @property
    def _value_range(self):
        return self._max_value - self._min_value

    def value(self):
        return float(super().value()) / self._max_int * self._value_range + self._min_value

    def setValue(self, value):
        super().setValue(int((value - self._min_value) / self._value_range * self._max_int))

    def setMinimum(self, value):
        if value > self._max_value:
            raise ValueError("Minimum limit cannot be higher than maximum")

        self._min_value = value
        self.setValue(self.value())

    def setMaximum(self, value):
        if value < self._min_value:
            raise ValueError("Minimum limit cannot be higher than maximum")

        self._max_value = value
        self.setValue(self.value())

    def setRange(self, minval, maxval):
        self.setMinimum(minval)
        self.setMaximum(maxval)

    def minimum(self):
        return self._min_value

    def maximum(self):
        return self._max_value

    def setDecimals(self, value):
        if type(value) != int:
            raise ValueError('Number of decimals must be an int')
        else:
            self.decimals = value

class QClickableSpinBox(QDoubleSpinBox):

    def __init__(self):
        super(QClickableSpinBox, self).__init__()
        self.installEventFilter(self)

    def eventFilter(self, QObject, event):
        if (event.type() == QEvent.MouseButtonPress and event.button() == Qt.RightButton) or event.type() == QEvent.MouseButtonDblClick:
            if self.isEnabled():
                self.setEnabled(False)
            else:
                self.setEnabled(True)
        return False

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
