from django.shortcuts import render
from django.views import View
from django import forms
import os
import pickle

class PatientData(forms.Form):
    field0 = forms.FloatField(label='База Агеева')
    field1 = forms.FloatField(label='База по долгожителям')
    field2 = forms.FloatField(label='РНФ')
    field3 = forms.FloatField(label='Информация')
    field4 = forms.FloatField(label='Противоэпилептические')
    field5 = forms.FloatField(label='Unnamed: 74')
    field6 = forms.FloatField(label='Эпиприступы')
    field7 = forms.FloatField(label='Кисты')
    field8 = forms.FloatField(label='Речевые зоны')
    field9 = forms.FloatField(label='Изменение ИО')
    field10 = forms.FloatField(label='Хирургия с пробуждением')
    field11 = forms.FloatField(label='Проводилось?')
    field12 = forms.FloatField(label='Локализация')
    field13 = forms.FloatField(label='Зад')
    field14 = forms.FloatField(label='Пол')
    field15 = forms.FloatField(label='Возраст')
    field16 = forms.FloatField(label='Отделение')
    field17 = forms.FloatField(label='Grade')
    field18 = forms.FloatField(label='Сторона')
    field19 = forms.FloatField(label='Висок')
    field20 = forms.FloatField(label='Затылок')
    field21 = forms.FloatField(label='СТ')
    field22 = forms.FloatField(label='П/ПР')
    field23 = forms.FloatField(label='ASL-перфузия')
    field24 = forms.FloatField(label='HARDI: ')
    field25 = forms.FloatField(label='спектроскопия')
    field26 = forms.FloatField(label='Гистология')
    field27 = forms.FloatField(label='Лоб')
    field28 = forms.FloatField(label='Темя')
    field29 = forms.FloatField(label='Островок')
    field30 = forms.FloatField(label='фМРТ')
    field31 = forms.FloatField(label='Нейрофизиологический мониторинг')
    field32 = forms.FloatField(label='Стимуляция коры Проведение')
    field33 = forms.FloatField(label='Стимуляция коры Нашли')
    field34 = forms.FloatField(label='Ложе удаленной опухоли Проведение')
    field35 = forms.FloatField(label='Ложе удаленной опухоли Нашли')
    field36 = forms.FloatField(label='Ассоциативные пути')
    field37 = forms.FloatField(label='ИК До операции')
    field38 = forms.FloatField(label='ИК На момент выписки')
    field39 = forms.FloatField(label='ИК Динамика')

BASE_DIR = os.path.dirname(__file__)

class Prediction(View):
    def get(self, request, *args, **kwargs):
        patientdata = PatientData()
        return render(request, "index.html", {"form": patientdata})
    def post(self, request, *args, **kwargs):
        patient = []
        for i in range(40):
            patient.append(float(request.POST.get("field" + str(i))))
        
        pca_path = os.path.join(BASE_DIR, 'static\\pca.pkl')
        with open(pca_path, 'rb') as file:  
            pca = pickle.load(file)

        model_path = os.path.join(BASE_DIR, 'static\\fs_prediction.pkl')
        with open(model_path, 'rb') as file:  
            prediction = pickle.load(file)
        X = pca.transform([patient])
        res = prediction.predict(X)
        if(res[0] == 1):
            res = 'yes'
        else:
            res = 'no'
        return render(request, "result.html", {"result": res})