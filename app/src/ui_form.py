# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 6.3.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QLayout,
    QLineEdit, QMainWindow, QMenu, QMenuBar,
    QPushButton, QSizePolicy, QWidget)

class Ui_FluorescencePredictor(object):
    def setupUi(self, FluorescencePredictor):
        if not FluorescencePredictor.objectName():
            FluorescencePredictor.setObjectName(u"FluorescencePredictor")
        FluorescencePredictor.resize(817, 314)
        FluorescencePredictor.setMinimumSize(QSize(817, 314))
        FluorescencePredictor.setMaximumSize(QSize(817, 314))
        self.centralwidget = QWidget(FluorescencePredictor)
        self.centralwidget.setObjectName(u"centralwidget")
        self.centralwidget.setMinimumSize(QSize(817, 288))
        self.centralwidget.setMaximumSize(QSize(817, 288))
        self.layoutWidget = QWidget(self.centralwidget)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.layoutWidget.setGeometry(QRect(50, 150, 691, 61))
        self.horizontalLayout = QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.loadLabel = QLabel(self.layoutWidget)
        self.loadLabel.setObjectName(u"loadLabel")

        self.horizontalLayout.addWidget(self.loadLabel)

        self.pathToFile = QLineEdit(self.layoutWidget)
        self.pathToFile.setObjectName(u"pathToFile")

        self.horizontalLayout.addWidget(self.pathToFile)

        self.open = QPushButton(self.layoutWidget)
        self.open.setObjectName(u"open")

        self.horizontalLayout.addWidget(self.open)

        self.answer = QLabel(self.centralwidget)
        self.answer.setObjectName(u"answer")
        self.answer.setGeometry(QRect(330, 50, 151, 61))
        self.answer.setAlignment(Qt.AlignCenter)
        FluorescencePredictor.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(FluorescencePredictor)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 817, 26))
        self.menu = QMenu(self.menubar)
        self.menu.setObjectName(u"menu")
        FluorescencePredictor.setMenuBar(self.menubar)

        self.menubar.addAction(self.menu.menuAction())

        self.retranslateUi(FluorescencePredictor)

        QMetaObject.connectSlotsByName(FluorescencePredictor)
    # setupUi

    def retranslateUi(self, FluorescencePredictor):
        FluorescencePredictor.setWindowTitle(QCoreApplication.translate("FluorescencePredictor", u"FluorescencePredictor", None))
        self.loadLabel.setText(QCoreApplication.translate("FluorescencePredictor", u"\u0417\u0430\u0433\u0440\u0443\u0437\u0438\u0442\u044c \u0441\u043d\u0438\u043c\u043e\u043a", None))
        self.open.setText(QCoreApplication.translate("FluorescencePredictor", u"open", None))
        self.answer.setText(QCoreApplication.translate("FluorescencePredictor", u"<html><head/><body><p><span style=\" font-size:14pt;\">\u041e\u0442\u0432\u0435\u0442</span></p></body></html>", None))
        self.menu.setTitle(QCoreApplication.translate("FluorescencePredictor", u"\u041e \u043f\u0440\u043e\u0433\u0440\u0430\u043c\u043c\u0435", None))
    # retranslateUi

