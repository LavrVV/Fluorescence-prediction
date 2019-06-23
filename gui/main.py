import pyforms
from   pyforms.basewidget          import BaseWidget
# from   pyforms.controls import ControlText
# from   pyforms.controls import ControlButton
from   pyforms.controls import ControlFile
from   pyforms.controls import ControlLabel
import pydicom
import zipfile
import numpy as np
from PIL import Image
import pandas as pd
import pickle

from scipy.special import expit

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

class FluorescencePredictor(BaseWidget):

    def __init__(self):
        super(FluorescencePredictor, self).__init__('Предсказатель Флуоресенции')

        #Definition of the forms fields
        self._answer = ControlLabel('Ответ')
        self._load = ControlFile(label='Загрузить снимок')
        
        self._load.changed_event = self._fileOpen
    def _extractData(self):
        with zipfile.ZipFile(self._load.value, 'r') as MRIs:
            data = []
            other = []
            sh = 64
            count = 0
            for fName in MRIs.namelist():
                if fName[-4:] == '.dcm':
                    with MRIs.open(fName) as fileDCM:
                        ds = pydicom.read_file(fileDCM)

                        t = ds[(0x8, 0x103e)].value

                        if not ('T1' in t or 'T2' in t or 't1' in t or 't2' in t):
                            continue

                        sex = ds[(0x10, 0x40)].value
                        sex = 1 if sex == 'M' else 0
                        age = ds[(0x10, 0x1010)].value
                        age = int(age[:-1])
                        weight = ds[(0x10, 0x1030)].value
                        weight = float(weight)

                        ds = ds.pixel_array

                        if len(ds.shape) > 2:
                            count += 1
                            continue

                        other.append([sex, age, weight])

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
            other = np.array(other)
            other = pd.DataFrame(other)

        return data, other
    def _predict(self, data, other):
        with open('encoder.pickle', 'rb') as f:
            encoder = pickle.load(f)
        
        with open('predictor.pickle', 'rb') as f:
            predictor = pickle.load(f)
        
        encoded = encoder.encode(data)
        encoded = pd.DataFrame(encoded)

        ddata = pd.concat([encoded, other], axis=1)
        res = predictor.predict(ddata)
        return res

    def _fileOpen(self):
        if self._load.value == '':
            pass
        elif self._load.value[-4:] == '.zip':
            data, other = self._extractData()
            res = self._predict(data, other)
            if np.mean(res) > 0.5:
                self._answer.value = 'будет светиться'
            else:
                self._answer.value = 'не будет светиться'    
            
        else:
            pass
        


        return True

#Execute the application
if __name__ == "__main__":   
    pyforms.start_app( FluorescencePredictor )