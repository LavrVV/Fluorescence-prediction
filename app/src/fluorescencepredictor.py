# -*- mode: python ; coding: utf-8 -*-
# This Python file uses the following encoding: utf-8
import sys

import pydicom
import zipfile
import numpy as np
from PIL import Image
import pandas as pd
import pickle
import sklearn

import pydicom.encoders.pylibjpeg
import pydicom.encoders.gdcm

from scipy.special import expit

from PySide6 import QtGui
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QErrorMessage

# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py
from ui_form import Ui_FluorescencePredictor


class Encoder:
    def __init__(self, model):
        self.n_comp = model.n_comp
        self.l1 = model.encoder[0].weight.data.numpy()
        self.l1b = model.encoder[0].bias.data.numpy()

        self.l2 = model.encoder[2].weight.data.numpy()
        self.l2b = model.encoder[2].bias.data.numpy()

        self.l3 = model.encoder[4].weight.data.numpy()
        self.l3b = model.encoder[4].bias.data.numpy()

    def encode(self, X):
        X = X.dot(self.l1.T) + self.l1b
        X = expit(X)
        X = X.dot(self.l2.T) + self.l2b
        X = expit(X)
        X = X.dot(self.l3.T) + self.l3b

        #sparce
        z = X
        z = np.sort(X, axis=1)[:,::-1][:, self.n_comp - 1]#.resize((len(x), 1))
        z = np.array([z])
        z.resize((len(X), 1))
        X[X < z] = 0
        #delete 0
        X[X != 0].resize((len(X), self.n_comp))
        k = len(X)
        z = np.array([X[X != 0]])
        z.resize((k, self.n_comp))

        return z


class FluorescencePredictor(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_FluorescencePredictor()
        self.ui.setupUi(self)
        self.setWindowTitle("Предсказатель флуоресценции")
        self.setWindowIcon(QtGui.QIcon('brain.ico'))
        self.menuBar().setHidden(True) # todo add help info
        self.ui.pathToFile.setReadOnly(True)
        self.ui.open.clicked.connect(self.the_button_was_clicked)

    def _raise_error(self, message):
        self.ui.answer.setStyleSheet('QLabel { color : black; font-size : 14pt; }')
        self.ui.answer.setText('Ответ')
        em = QErrorMessage(self)
        em.setWindowTitle("Error")
        em.showMessage(message)

    def the_button_was_clicked(self, checked):
        self.ui.answer.setStyleSheet('QLabel { color : black; font-size : 14pt; }')
        self.ui.answer.setText('Ответ')
        file_to_load = QFileDialog.getOpenFileName(self, 'Open file', '',"zip archive (*.zip)")[0]
        self.ui.pathToFile.setText(file_to_load)
        if file_to_load == '':
            return True
        try:
            data, other = self._extractData(file_to_load)
        except ValueError as e:
            self._raise_error(str(e))
            return True
        except Exception as e:
            self._raise_error(str(e))
            return True
        res = self._predict(data, other)
        if np.mean(res) > 0.5:
            self.ui.answer.setStyleSheet('QLabel { color : green; font-size : 14pt; }')
            self.ui.answer.setText('будет светиться')
        else:
            self.ui.answer.setStyleSheet('QLabel { color : red; font-size : 14pt; }')
            self.ui.answer.setText('НЕ будет светиться')

    def _extractData(self, MRIarchive):
        with zipfile.ZipFile(MRIarchive, 'r') as MRIs:
            data = []
            sh = 64
            count = 0
            sex = -1
            age = -1
            weight = -1

            for fName in MRIs.namelist():
                if fName[-4:] == '.dcm':
                    with MRIs.open(fName) as fileDCM:
                        ds = pydicom.read_file(fileDCM)
                        try:
                            t = ds[(0x8, 0x103e)].value
                        except:
                            continue

                        if not ('T1' in t or 'T2' in t or 't1' in t or 't2' in t):
                            continue
                        try:
                            sex = ds[(0x10, 0x40)].value
                            sex = 1 if sex == 'M' else 0
                            age = ds[(0x10, 0x1010)].value
                            age = int(age[:-1])
                            weight = ds[(0x10, 0x1030)].value
                            weight = float(weight)
                        except:
                            pass

                        ds = ds.pixel_array

                        if len(ds.shape) > 2:
                            count += 1
                            continue


                        if ds.shape[0] != sh:
                        
                            gg = Image.fromarray(ds)
                            gg = gg.resize((sh, sh))
                            curr = np.asarray(gg)

                        curr = curr.reshape(sh**2)
                        mmax = float(np.max(curr))
                        if mmax != 0:
                            curr = curr / mmax 

                        data.append(curr)
            data = np.array(data, dtype=float)
            
        if len(data) == 0:
            raise Exception('В архиве не найдены снимки МРТ формата dicom.')
        other = []
        for i in range(len(data)):
            other.append([sex, age, weight])

        other = np.array(other)
        other = pd.DataFrame(other)
        
        if sex == -1 or age == -1 or weight == -1:
            raise ValueError('В снимках не обнаружены аттрибуты пол возраст или вес.')
        
        return data, other

    def _predict(self, data, other):
        with open('model/encoder.pickle', 'rb') as f:
            encoder = pickle.load(f)
        with open('model/predictor-1.1.1.pickle', 'rb') as f:
            predictor = pickle.load(f)

        encoded = encoder.encode(data)
        encoded = pd.DataFrame(encoded)

        ddata = pd.concat([encoded, other], axis=1)
        res = predictor.predict(ddata)
        return res

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = FluorescencePredictor()
    widget.show()
    sys.exit(app.exec())
