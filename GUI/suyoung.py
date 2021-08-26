from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, QObject, pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import QMessageBox
import cv2
import numpy as np
import mediapipe as mp
import time
import math
import tensorflow as tf
import keras.backend
import serial
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
import win32gui
from PIL import ImageGrab
from multiprocessing import Process, Queue, Value, Array
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
tf.config.list_physical_devices(device_type='GPU')

import pandas as pd

def data_predict(p_list, maximum):
    model = keras.models.load_model(
        'C:/Users/user/PycharmProjects/MLbasic/model_save/my_model_63.h5'
    )
    while True:
        mark_p_list = p_list.get()
        df = pd.DataFrame(mark_p_list, columns=[str(i) for i in range(0, 63)])
        col_name = [str(i) for i in range(0, 63)]
        gstr = df[col_name].to_numpy()
        prediction = model.predict(gstr[[0]])
        prediction = prediction[0]
        MAX = prediction.argmax()  # sigmoid 모델을 사용할 때는 쓰지 않는다.
        if prediction[MAX] > 0.9:
            maximum.value = MAX
        else:
            maximum.value = 2

def data_predict_2(p_list_2, maximum_2):
    model = keras.models.load_model(
        'C:/Users/user/PycharmProjects/MLbasic/model_save/my_model_63.h5'
    )
    while True:
        mark_p_list = p_list_2.get()
        df = pd.DataFrame(mark_p_list, columns=[str(i) for i in range(0, 63)])
        col_name = [str(i) for i in range(0, 63)]
        gstr = df[col_name].to_numpy()
        prediction = model.predict(gstr[[0]])
        prediction = prediction[0]
        MAX = prediction.argmax()  # sigmoid 모델을 사용할 때는 쓰지 않는다.
        if prediction[MAX] > 0.9:
            maximum_2.value = MAX
        else:
            maximum_2.value = 2

def cv_gui_arduino(p_list, maximum, p_list_2, maximum_2):
    # ard = serial.Serial('COM4', 115200)
    class Handmark():
        def __init__(self, mark_p):
            self._p_list = mark_p
            self.finger_state = [0 for _ in range(5)]

        @property
        def p_list(self):
            return self._p_list

        @p_list.setter
        def p_list(self, new_p):
            self._p_list = new_p

        def get_finger_angle(self, finger): #finger는 self에서 정의된 손가락들, 4크기 배열
            l1 = finger[0] - finger[1]
            l2 = finger[3] - finger[1]
            l1_ = np.array([l1[0], l1[1], l1[2]])
            l2_ = np.array([l2[0], l2[1], l2[2]])
            return np.arccos(np.dot(l1_, l2_) / (np.linalg.norm(l1) * np.linalg.norm(l2)))

        def get_angle(self, l1, l2):
            l1_ = np.array([l1[0], l1[1], l1[2]])
            l2_ = np.array([l2[0], l2[1], l2[2]])
            return np.arccos(np.dot(l1_, l2_) / (np.linalg.norm(l1) * np.linalg.norm(l2)))

        def get_finger_angle_thumb(self, finger):
            l1 = finger[0] - finger[1]
            l2 = finger[1] - finger[2]
            return self.get_angle(l1, l2)

        def get_palm_vector(self):
            l1 = self._p_list[17] - self._p_list[0]
            l2 = self._p_list[5] - self._p_list[0]
            l1_ = np.array([l1[0], l1[1], l1[2]])
            l2_ = np.array([l2[0], l2[1], l2[2]])

            self.palm_vector = np.cross(l1_, l2_)
            self.palm_vector = self.palm_vector / vector_magnitude(self.palm_vector)
            self.z_orthogonality = self.palm_vector[2]
            return self.palm_vector, self.z_orthogonality

        def return_finger_info(self):
            self.thumb = [self._p_list[i] for i in range(1, 5)]
            self.index = [self._p_list[i] for i in range(5, 9)]
            self.middle = [self._p_list[i] for i in range(9, 13)]
            self.ring = [self._p_list[i] for i in range(13, 17)]
            self.pinky = [self._p_list[i] for i in range(17, 21)]

            # TODO 각 손가락 각도 근거로 손가락 굽힘 판단
            self.finger_angle_list = np.array([self.get_finger_angle(self.thumb),
                                               self.get_finger_angle(self.index),
                                               self.get_finger_angle(self.middle),
                                               self.get_finger_angle(self.ring),
                                               self.get_finger_angle(self.pinky)])

            finger_angle_threshold = np.array([2.8, 1.7, 2.2, 2.2, 2.4])
            self.finger_state_angle = np.array(self.finger_angle_list > finger_angle_threshold, dtype=int)
            # TODO 각 손가락 거리정보 근거로 손가락 굽힘 판단

            self.finger_distance_list = np.array([get_distance(self.thumb[3], self.pinky[0]) / get_distance(self.index[0], self.pinky[0]),
                                                  get_distance(self.index[3], self.index[0]) / get_distance(self.index[0], self.index[1]),
                                                  get_distance(self.middle[3], self.middle[0]) / get_distance(self.middle[0], self.middle[1]),
                                                  get_distance(self.ring[3], self.ring[0]) / get_distance(self.ring[0], self.ring[1]),
                                                  get_distance(self.pinky[3], self.pinky[0]) / get_distance(self.pinky[0], self.pinky[1])])
            finger_distance_threshold = np.array([1.5, 2, 2, 2, 2])
            self.finger_state_distance = np.array(self.finger_distance_list > finger_distance_threshold, dtype=int)

            # TODO 손가락과 손바닥 이용해 손가락 굽힘 판단
            self.hand_angle_list = np.array([self.get_angle(self.thumb[1] - self._p_list[0], self.thumb[3] - self.thumb[1]),
                                             self.get_angle(self.index[0] - self._p_list[0], self.index[3] - self.index[0]),
                                             self.get_angle(self.middle[0] - self._p_list[0], self.middle[3] - self.middle[0]),
                                             self.get_angle(self.ring[0] - self._p_list[0], self.ring[3] - self.ring[0]),
                                             self.get_angle(self.pinky[0] - self._p_list[0], self.pinky[3] - self.pinky[0])])
            hand_angle_threshold = np.array([0.7, 1.5, 1.5, 1.5, 1.3])
            self.hand_state_angle = np.array(self.hand_angle_list < hand_angle_threshold, dtype=int)
            self.input = np.concatenate((self.finger_angle_list, self.finger_distance_list, self.hand_angle_list))
            result = self.finger_state_angle + self.finger_state_distance + self.hand_state_angle > 1
            result = result.tolist()
            for i in range(len(result)):
                result[i] = int(result[i])
            self.result = result
            return self.result

        def predict_gesture(self, index_list, max):
            self.return_63_indices()
            index_list.put(self.output)
            self.MAX = max.value
            return self.MAX

        def return_63_indices(self):
            output = []
            self.output = []
            for i in range(21):
                output.append(self._p_list[i].x)
                output.append(self._p_list[i].y)
                output.append(self._p_list[i].z)
            self.output.append(output)
            return self.output

    def vector_magnitude(one_D_array):
        return math.sqrt(np.sum(one_D_array * one_D_array))

    def get_distance(p1, p2):
        try:
            return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)
        except:
            return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    class Mark_pixel():
        def __init__(self, x, y, z=0, LR=0):
            self.x = x
            self.y = y
            self.z = z
            self.LR = LR

        def __str__(self):
            return str(self.x) + '   ' + str(self.y) + '   ' + str(self.z)

        def to_list(self):
            return [self.x, self.y, self.z]

        def __sub__(self, other):
            return self.x - other.x, self.y - other.y, self.z - other.z

    def data_normalize(list_1, list_2):
        xvalue = []
        yvalue = []
        zvalue = []
        for i in range(21):
            xval, yval, zval = list_2.landmark[i].x - list_2.landmark[0].x, \
                               list_2.landmark[i].y - list_2.landmark[0].y, \
                               list_2.landmark[i].z - list_2.landmark[0].z
            list_1.append(xval)
            list_1.append(yval)
            list_1.append(zval)  # 마크 픽셀 클래스 인스턴스를 사용하여 mark_p 리스트에 넣음
            xvalue.append(abs(xval))
            yvalue.append(abs(yval))
            zvalue.append(abs(zval))
        xmax = max(xvalue)
        ymax = max(yvalue)
        zmax = max(zvalue)
        for i in range(21):
            i = 3 * i
            list_1[i] = list_1[i] / xmax
            list_1[i + 1] = list_1[i + 1] / ymax
            list_1[i + 2] = list_1[i + 2] / zmax

    class touch_control(QThread):
        def __init__(self):
            super().__init__()
            self.control_direction = 2 # default는 멈춤

        @pyqtSlot(int)
        def new_control_direction(self, new_control_direction):
            self.control_direction = new_control_direction

        def run(self):
            while True:
                if self.control_direction == 0:
                    print('U')
                    # ard.write(b'U')
                elif self.control_direction == 1:
                    print('D')
                    # ard.write(b'D')
                elif self.control_direction == 2:
                    print('S')
                    # ard.write(b'S')
                    self.control_direction = 3
                else :
                    pass

    class opcv(QThread):

        change_pixmap_signal = pyqtSignal(np.ndarray)

        def __init__(self):
            super().__init__()
            self.bool_state = True

        @pyqtSlot(bool)
        def new_bool_state(self, new_bool_state):
            self.bool_state = new_bool_state

        def run(self):  # p를 보는 emit 함수
            hands = mp_hands.Hands(max_num_hands=6, min_detection_confidence=0.3, min_tracking_confidence=0.8)
            prevTime = 0
            toplist, winlist = [], []

            def enum_cb(hwnd, results):
                winlist.append((hwnd, win32gui.GetWindowText(hwnd)))

            while self.bool_state:
                win32gui.EnumWindows(enum_cb, toplist)
                skype = [(hwnd, title) for hwnd, title in winlist if 'skype' in title.lower()]
                # just grab the hwnd for first window matching firefox
                skype = skype[0]
                hwnd = skype[0]
                # win32gui.SetForegroundWindow(hwnd)
                bbox = win32gui.GetWindowRect(hwnd)
                image = ImageGrab.grab((bbox[0] + 10, bbox[1] + 30, bbox[2] - 10, bbox[3] - 30))
                image = np.array(image)

                toplist, winlist = [], []
                curTime = time.time()
                sec = curTime - prevTime
                prevTime = curTime
                fps = 1 / sec
                # Flip the image horizontally for a later selfie-view display, and convert
                # the BGR image to RGB.
                # image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_RGB2BGR)
                image = cv2.flip(image, 1)
                # x_size, y_size, channel = image.shape
                # To improve performance, optionally mark the image as not writeable to
                # p
                # ass by reference.
                image.flags.writeable = False
                results = hands.process(image)
                # Draw the hand annotations on the image.
                image.flags.writeable = True
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                hands_list = []
                handedness_list = []
                z_orthogonality_list = []
                prediction_list = []
                turn = 0
                if results.multi_hand_landmarks:
                    mark_p_list = []
                    size_list = []
                    for hand_landmarks in results.multi_hand_landmarks:  # hand_landmarks는 감지된 손의 갯수만큼의 원소 수를 가진 list 자료구조
                        mark_p = []
                        normalized_list = []
                        data_normalize(normalized_list, hand_landmarks)  # 여기서 나온 mark_p_list data_predict로 보내주기
                        size_list.append(get_distance(hand_landmarks.landmark[5], hand_landmarks.landmark[17]))
                        for i in range(21):
                            i = 3 * i
                            mark_p.append(Mark_pixel(normalized_list[i], normalized_list[i+1], normalized_list[i+2]))
                        mark_p_list.append(mark_p)
                    if len(mark_p_list) >= 2:
                        idx1 = size_list.index(max(size_list))
                        size_list[idx1] = -10
                        idx2 = size_list.index(max(size_list))
                        target_list = [idx1, idx2]
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        for i in target_list:
                            mp_drawing.draw_landmarks(
                                image, results.multi_hand_landmarks[i], mp_hands.HAND_CONNECTIONS)
                            #exit()
                            HM = Handmark(mark_p_list[i])
                            HM.return_finger_info()
                            if turn == 0:
                                HM.predict_gesture(p_list, maximum)
                            else:
                                HM.predict_gesture(p_list_2, maximum_2)
                            HM.get_palm_vector()
                            hands_list.append(HM.result)
                            z_orthogonality_list.append(HM.z_orthogonality)
                            prediction_list.append(HM.MAX)
                            handedness_list.append(results.multi_handedness[i].classification[0].label)
                            turn += 1
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        if [1, 1, 1, 1, 1] in hands_list:
                            i = hands_list.index([1, 1, 1, 1, 1])
                            left_or_right = len(handedness_list[i])
                            orthogonality = z_orthogonality_list[i]
                            if left_or_right == 4 and orthogonality > 0.4:
                                if 0 in prediction_list :
                                    cv2.putText(image, 'Increase', (30, 40), cv2.FONT_HERSHEY_PLAIN, 3, color=(0,255,255), thickness=3)
                                    # ard.write(b'U')
                                elif 1 in prediction_list :
                                    cv2.putText(image, 'Decrease', (30, 40), cv2.FONT_HERSHEY_PLAIN, 3, color=(0,255,255), thickness=3)
                                    # ard.write(b'D')
                                else:
                                    # ard.write(b'S')
                                    pass
                            elif left_or_right == 5 and orthogonality < -0.4:
                                if 0 in prediction_list :
                                    cv2.putText(image, 'Increase', (30, 40), cv2.FONT_HERSHEY_PLAIN, 3, color=(0,255,255), thickness=3)
                                    # ard.write(b'U')
                                elif 1 in prediction_list :
                                    cv2.putText(image, 'Decrease', (30, 40), cv2.FONT_HERSHEY_PLAIN, 3, color=(0,255,255), thickness=3)
                                    # ard.write(b'D')
                                else:
                                    # ard.write(b'S')
                                    pass
                            else:
                                # ard.write(b'S')
                                pass
                        else:
                            # ard.write(b'S')
                            pass
                    else:
                        # ard.write(b'S')
                        pass
                else:
                    # ard.write(b'S')
                    pass
                cv2.putText(image, "FPS : %0.1f" % fps, (1050, 30), cv2.FONT_HERSHEY_PLAIN, 2, 4)
                # if self.bool_state == True:
                self.change_pixmap_signal.emit(image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
            self.change_pixmap_signal.emit(cv2.cvtColor(cv2.imread('C:/Users/user/PycharmProjects/MLbasic/GUI/Image/Inbody.png'), cv2.COLOR_BGR2RGB))
            hands.close()


    class Ui_MainWindow(QObject):
        power_signal = pyqtSignal(bool)
        increase_or_decrease = pyqtSignal(int)
        def setupUi(self, MainWindow):
            MainWindow.setObjectName("MainWindow")
            MainWindow.resize(1600, 900)
            MainWindow.setStyleSheet("background-color: rgb(240, 255, 240);")
            self.Mywindow = MyWindow()
            self.centralwidget = QtWidgets.QWidget(MainWindow)
            self.centralwidget.setObjectName("centralwidget")
            self.label = QtWidgets.QLabel(self.centralwidget)
            self.label.setGeometry(QtCore.QRect(0, 50, 1360, 765))
            self.label.setObjectName("label")
            self.label.setPixmap(QtGui.QPixmap('C:/Users/user/PycharmProjects/MLbasic/GUI/Image/Inbody.png'))
            self.label.setScaledContents(True)
            self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
            self.groupBox.setGeometry(QtCore.QRect(1370, 225, 220, 375))
            self.groupBox.setAlignment(QtCore.Qt.AlignCenter)
            self.groupBox.setObjectName("groupBox")
            self.pushButton = QtWidgets.QPushButton(self.groupBox)
            self.pushButton.setGeometry(QtCore.QRect(22, 21, 157, 157))
            self.pushButton.setAutoFillBackground(False)
            self.pushButton.setText("")
            self.pushButton.setStyleSheet("border-radius : 78; border : 2px solid white")

            self.pushButton.setStyleSheet(
                '''
                QPushButton{image:url(C:/Users/user/PycharmProjects/MLbasic/GUI/Image/Increase.png); border:0px;}
                QPushButton:checked{image:url(C:/Users/user/PycharmProjects/MLbasic/GUI/Image/Increase_checked.png); border:0px;}
                ''')
            self.pushButton.setCheckable(True)
            self.pushButton.setObjectName("pushButton")
            self.pushButton.setEnabled(False)
            self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
            self.pushButton_2.setGeometry(QtCore.QRect(22, 199, 157, 157))
            self.pushButton_2.setText("")
            self.pushButton_2.setStyleSheet("border-radius : 78; border : 2px solid white")
            self.pushButton_2.setStyleSheet(
                '''
                QPushButton{image:url(C:/Users/user/PycharmProjects/MLbasic/GUI/Image/Decrease.png); border:0px;}
                QPushButton:checked{image:url(C:/Users/user/PycharmProjects/MLbasic/GUI/Image/Decrease_checked.png); border:0px;}
                ''')
            self.pushButton_2.setCheckable(True)
            self.pushButton_2.setObjectName("pushButton_2")
            self.pushButton_2.setEnabled(False)
            self.checkBox = QtWidgets.QCheckBox(self.centralwidget)
            self.checkBox.setGeometry(QtCore.QRect(1370, 170, 220, 34))
            self.checkBox.setObjectName("checkBox")
            self.checkBox.setStyleSheet(" width: 20px; height: 20px;")
            self.checkBox.setCheckState(False)
            self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
            self.groupBox_2.setGeometry(QtCore.QRect(1370, 620, 220, 157))
            self.groupBox_2.setAlignment(QtCore.Qt.AlignCenter)
            self.groupBox_2.setObjectName("groupBox_2")
            self.pushButton_3 = QtWidgets.QPushButton(self.groupBox_2)
            self.pushButton_3.setGeometry(QtCore.QRect(10, 31, 95,95))
            self.pushButton_3.setText("")
            self.pushButton_3.setStyleSheet("border-radius : 47; border : 2px solid white")
            self.pushButton_3.setStyleSheet(
                '''
                QPushButton{image:url(C:/Users/user/PycharmProjects/MLbasic/GUI/Image/refresh.png); border:0px;}
                QPushButton:hover{image:url(C:/Users/user/PycharmProjects/MLbasic/GUI/Image/refresh_checked.png); border:0px;}
                ''')
            self.pushButton_3.setObjectName("pushButton_3")
            self.pushButton_4 = QtWidgets.QPushButton(self.groupBox_2)
            self.pushButton_4.setGeometry(QtCore.QRect(115, 31, 95, 95))
            self.pushButton_4.setText("")
            self.pushButton_4.setStyleSheet("border-radius : 47; border : 2px solid white")
            self.pushButton_4.setStyleSheet(
                '''
                QPushButton{image:url(C:/Users/user/PycharmProjects/MLbasic/GUI/Image/power.png); border:0px;}
                QPushButton:hover{image:url(C:/Users/user/PycharmProjects/MLbasic/GUI/Image/power_checked.png); border:0px;}
                ''')
            self.pushButton_4.setObjectName("pushButton_4")
            self.label_2 = QtWidgets.QLabel(self.centralwidget)
            self.label_2.setGeometry(QtCore.QRect(1370, 70, 220, 80))
            self.label_2.setText("")
            self.label_2.setPixmap(QtGui.QPixmap("C:/Users/user/PycharmProjects/MLbasic/GUI/Image/인바디.png"))
            self.label_2.setScaledContents(True)
            self.label_2.setObjectName("label_2")
            MainWindow.setCentralWidget(self.centralwidget)
            self.menubar = QtWidgets.QMenuBar(MainWindow)
            self.menubar.setGeometry(QtCore.QRect(0, 0, 1600, 21))
            self.menubar.setObjectName("menubar")
            MainWindow.setMenuBar(self.menubar)
            self.statusbar = QtWidgets.QStatusBar(MainWindow)
            self.statusbar.setObjectName("statusbar")
            MainWindow.setStatusBar(self.statusbar)

            self.retranslateUi(MainWindow)
            QtCore.QMetaObject.connectSlotsByName(MainWindow)

            self.pushButton.clicked.connect(lambda: self.send_increase_signal(MainWindow))
            self.pushButton_2.clicked.connect(lambda: self.send_decrease_signal(MainWindow))
            self.checkBox.stateChanged.connect(self.touch_mode)
            self.pushButton_4.clicked.connect(self.Mywindow.closeEvent)
            self.thread = opcv()
            self.thread.change_pixmap_signal.connect(self.update_img)
            self.power_signal.connect(self.thread.new_bool_state)
            self.pushButton_3.clicked.connect(self.rebooting)
            self.thread.start()

            self.thread_2 = touch_control()
            self.increase_or_decrease.connect(self.thread_2.new_control_direction)

        def rebooting(self):
            self.checkBox.setEnabled(False)
            self.power_signal.emit(False)
            time.sleep(3)
            self.thread.start()
            self.power_signal.emit(True)
            self.checkBox.setEnabled(True)

        def touch_mode(self):
            if self.checkBox.isChecked():
                print('Woo')
                self.pushButton_3.setEnabled(False)
                self.thread_2.start()
                self.power_signal.emit(False)
                self.pushButton.setChecked(False)
                self.pushButton_2.setChecked(False)
                self.pushButton.setEnabled(True)
                self.pushButton_2.setEnabled(True)
            if not self.checkBox.isChecked():
                print('WoW')
                self.thread_2.terminate()
                self.thread.start()
                self.power_signal.emit(True)
                self.pushButton.setChecked(False)
                self.pushButton_2.setChecked(False)
                self.pushButton.setEnabled(False)
                self.pushButton_2.setEnabled(False)
                self.pushButton_3.setEnabled(True)

        def cvt_qt(self, img):
            # rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv 이미지 파일 rgb 색계열로 바꿔주기
            h, w, ch = img.shape  # image 쉐입 알기
            bytes_per_line = ch * w  # 차원?
            convert_to_Qt_format = QtGui.QImage(img.data, w, h, bytes_per_line,
                                                QtGui.QImage.Format_RGB888)  # qt 포맷으로 바꾸기
            p = convert_to_Qt_format.scaled(1360, 765, QtCore.Qt.KeepAspectRatio)  # 디스클레이 크기로 바꿔주기.

            return QtGui.QPixmap.fromImage(p)  # 진정한 qt 이미지 생성

        @pyqtSlot(np.ndarray)
        def update_img(self, img):
            qt_img = self.cvt_qt(img)
            self.label.setPixmap(qt_img)


        def send_increase_signal(self, MainWindow):
            if self.pushButton_2.isChecked():
                self.pushButton_2.toggle()
            if self.pushButton.isChecked():
                self.increase_or_decrease.emit(0)
            else:
                self.increase_or_decrease.emit(2)
                pass

        def send_decrease_signal(self, MainWindow):
            if self.pushButton.isChecked():
                self.pushButton.toggle()
            if self.pushButton_2.isChecked():
                self.increase_or_decrease.emit(1)
            else:
                self.increase_or_decrease.emit(2)
                pass

        # 토글로 바꾸고 시그널 받으면 계속 돌릴 와일문 만들기

        def retranslateUi(self, MainWindow):
            _translate = QtCore.QCoreApplication.translate
            MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
            self.groupBox.setTitle(_translate("MainWindow", "Control Box"))
            self.checkBox.setText(_translate("MainWindow", "터치 제어 사용"))
            self.groupBox_2.setTitle(_translate("MainWindow", "Refresh/Power"))

    class MyWindow(QtWidgets.QMainWindow):

        power_off_signal = pyqtSignal(bool)

        def __init__(self):
            super().__init__()
            self.setStyleSheet('''QMessageBox{background-color: rgb(225, 225, 225);}''')
            self.setStyleSheet('''QMainWindow{background-color : rgb(0, 0, 0);}''')
            self.msg = QMessageBox()
        def closeEvent(self, event):
            result = self.msg.question(self,
                                 "Confirm Exit...",
                                 "Are you sure you want to exit ?",
                                 self.msg.Yes | self.msg.No)
            if result == self.msg.Yes:
                event.accept()
            else :
                event.ignore()

    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    p_list = Queue(1)
    maximum = Value('i', 0)
    p_list_2 = Queue(1)
    maximum_2 = Value('i', 0)
    process1 = Process(target=cv_gui_arduino, args=(p_list, maximum, p_list_2, maximum_2,))
    process2 = Process(target=data_predict, args=(p_list, maximum,))
    process3 = Process(target=data_predict_2, args=(p_list_2, maximum_2,))
    process1.start()
    process2.start()
    process3.start()
    while process1.is_alive():
        pass
    process2.terminate()
    process3.terminate()
    process1.join()
    process2.join()
    process3.terminate()
