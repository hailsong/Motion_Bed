from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, QObject, pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import QMessageBox

import cv2
import numpy as np
import mediapipe as mp
import time
import math
# import tensorflow as tf
# import keras.backend
import serial
# import keras
# config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
# session = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(session)
from multiprocessing import Process, Queue, Value

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mode_global = 0

import pandas as pd

def cv_gui_arduino(p_list, maximum, p_list_2, maximum_2):
    """
    opencv, GUI, arduino 돌리는 메인 프로세스
    """
    class Loading(object):
        def setupUi(self, Form):
            Form.setObjectName("Form")
            Form.resize(250, 125)
            Form.setStyleSheet("background-color : rgb(224,244,253)")
            Form.setWindowFlag(QtCore.Qt.FramelessWindowHint)
            self.Label = QtWidgets.QLabel(Form)
            self.Label.setGeometry(QtCore.QRect(50, 40, 151, 41))
            font = QtGui.QFont()
            font.setFamily("에스코어 드림 6 Bold")
            font.setPointSize(24)
            font.setBold(True)
            font.setWeight(75)
            self.Label.setFont(font)
            self.Label.setObjectName("label")
            # self.Label_2 = QtWidgets.QLabel(Form)
            # self.Label_2.setGeometry(QtCore.QRect(170, 25, 71, 71))
            # # set qmovie as label
            # self.movie = QtGui.QMovie("Image/loading.gif")
            # self.Label_2.setMovie(self.movie)
            # self.Label_2.setScaledContents(True)
            # self.movie.start()

            self.retranslateUi(Form)
            QtCore.QMetaObject.connectSlotsByName(Form)

        def retranslateUi(self, Form):
            _translate = QtCore.QCoreApplication.translate
            Form.setWindowTitle(_translate("Form", "Form"))
            self.Label.setText(_translate("Form", "로딩 중..."))

    try :
        ard = serial.Serial('COM4', 115200)
    except :
        pass

    class Handmark():
        def __init__(self, mark_p):
            self._p_list = mark_p
            self.finger_state = [0 for _ in range(5)]
            self.return_finger_info()
            self.get_palm_vector()

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
            if self._p_list[0].y > self.middle[0].y :
                result.append(1)
            else :
                result.append(0)
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

    class Enter_mode:
        """
        전체 MODE 결정하기 위한 Class
        """
        QUEUE_SIZE = 20

        def __init__(self):
            self.right = [0] * self.QUEUE_SIZE
            # select_mode

        def update_right(self, right, select_idx):
            self.right.append(right)
            self.right.pop(0)

            # print(self.right)

            # print(self.right)

            select_idx = self.select_mode(select_idx)
            # print(select_idx)+
            return select_idx

        def select_mode(self, select_idx):
            global mode_global
            # print('mode_origin :', mode_global)
            right_idx_1 = 0
            for right in self.right:
                if right == 1:
                    right_idx_1 += 1
                # print('right, rightidx', right, right_idx_1, select_idx)
            if select_idx < 0:
                # print(right_idx_1)
                if right_idx_1 > 10 and mode_global == 0:
                    mode_global = 1
                    print('mode 1')
                    select_idx = 30
                elif right_idx_1 < 10 and mode_global == 1:
                    mode_global = 0
                    print('mode 0')

                    select_idx = 30
            select_idx -= 1
            # print(select_idx)
            return select_idx
            # return mode

        # mode = 0, 기본 상태(인식은 하지만 제어는 불가) mode = 1, 제스처 제어 가능 상태
        # mode 가 바뀔 때 사용자에게 인식시켜주는 방법 > 이미지 바꾸기

    class touch_control(QThread):
        def __init__(self):
            super().__init__()
            self.control_direction = 2 # default는 멈춤

        @pyqtSlot(int) # increase_or_decrease
        def new_control_direction(self, new_control_direction):
            self.control_direction = new_control_direction

        def run(self):
            while True:
                if self.control_direction == 0:
                    print('U')
                    try:
                        ard.write(b'U')
                    except:
                        pass
                    self.control_direction = 3
                elif self.control_direction == 1:
                    print('D')
                    try:
                        ard.write(b'D')
                    except:
                        pass
                    self.control_direction = 3
                elif self.control_direction == 2:
                    print('S')
                    try:
                        ard.write(b'S')
                    except:
                        pass
                        # self.is_connect_2.emit(0)
                    self.control_direction = 3
                else :
                    pass

    class opcv(QThread):
        change_pixmap_signal = pyqtSignal(np.ndarray)

        def __init__(self):
            super().__init__()
            self.bool_state = True
            self.det_time = time.time()

        @pyqtSlot(bool)
        def new_bool_state(self, new_bool_state):
            self.bool_state = new_bool_state

        @pyqtSlot(int, int)
        def mode_setting(self, mode, mode_before):  # 1
            global GESTURE_CONTROL
            if mode != mode_before:
                self.mode_signal.emit(int(mode - 1))  # 2 / #2-4

                if mode == 1 and mode_global != mode:
                    GESTURE_CONTROL = True
                    print('MODE 1, 제스처 제어 모드')
                    mode_global = mode

        def run(self):  # p를 보는 emit 함수
            self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            cap = self.capture
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # cap.set(cv2.CAP_PROP_FPS, int(10))
            hands = mp_hands.Hands(max_num_hands=6, min_detection_confidence=0.5, min_tracking_confidence=0.8)
            prevTime = 0
            command_status = 3
            gesture_list = [3 for _ in range(4)]
            enter_mode = Enter_mode()

            select_idx = 1
            mode_before = 3

            while self.bool_state and cap.isOpened():
                success, image = cap.read()
                curTime = time.time()
                sec = curTime - prevTime
                prevTime = curTime
                fps = 1 / sec
                nohand = True
                if not success:
                    exit()
                # Flip the image horizontally for a later selfie-view display, and convert
                # the BGR image to RGB.
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                # x_size, y_size, channel = image.shape
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                results = hands.process(image)
                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                det_duration = time.time() - self.det_time
                if results.multi_hand_landmarks:
                    mark_p_list = []
                    size_list = []
                    for hand_landmarks in results.multi_hand_landmarks:  # hand_landmarks는 감지된 손의 갯수만큼의 원소 수를 가진 list 자료구조
                        mark_p = []
                        normalized_list = []
                        data_normalize(normalized_list, hand_landmarks)  # 여기서 나온 mark_p_list cv_gui_arduino.data_predict로 보내주기
                        size_list.append(get_distance(hand_landmarks.landmark[5], hand_landmarks.landmark[17]))
                        for i in range(21):
                            i = 3 * i
                            mark_p.append(Mark_pixel(normalized_list[i], normalized_list[i+1], normalized_list[i+2]))
                        mark_p_list.append(mark_p)
                    self.det_time = time.time()
                    target_idx = size_list.index(max(size_list))
                    HM = Handmark(mark_p_list[target_idx])
                    HM.predict_gesture(p_list, maximum)


                    mp_drawing.draw_landmarks(
                        image, results.multi_hand_landmarks[target_idx], mp_hands.HAND_CONNECTIONS)
                    left_or_right = results.multi_handedness[target_idx].classification[0].label
                    if HM.result == [1, 1, 1, 1, 1, 1] and len(left_or_right) == 5 and HM.z_orthogonality < -0.5:
                        select_idx = enter_mode.update_right(1, select_idx)
                    else :
                        select_idx = enter_mode.update_right(0, select_idx)

                    # print(mode_global)

                    if mode_global:
                        command_status = 3
                        gesture_list.append(HM.MAX)
                        gesture_list.pop(0)
                        idx_0 = 0
                        idx_11 = 0
                        idx_2 = 0
                        for i in gesture_list:
                            if i == 0 :
                                idx_0 += 1
                            if i == 11 :
                                idx_11 += 1
                            if i == 2 :
                                idx_2 += 1
                        if idx_0 >= 3 :
                            command_status = 0
                        if idx_11 >= 3 :
                            command_status = 1
                        if idx_2 >= 3 :
                            command_status = 2

                        # print('command_status :', command_status)
                        # print(gesture_list)

                    elif mode_global == 0 :
                        command_status == 2
                else:
                    select_idx = enter_mode.update_right(0, select_idx)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if command_status == 0 :
                    cv2.putText(image, 'Increase', (30, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, 1)
                    try:
                        ard.write(b'U')
                        # self.is_connect.emit(1)
                    except:
                        pass
                        # self.is_connect.emit(0)
                elif command_status == 1 :
                    cv2.putText(image, 'Decrease', (30, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, 1)
                    try:
                        ard.write(b'D')
                        # self.is_connect.emit(1)
                    except:
                        pass
                        # self.is_connect.emit(0)
                elif command_status == 2 :
                    cv2.putText(image, 'STOP', (30, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, 1)
                    try:
                        ard.write(b'S')
                        # self.is_connect.emit(1)
                    except:
                        pass
                        # self.is_connect.emit(0)
                    command_status = 3
                else :
                    pass
                    # self.is_connect.emit(1)
                cv2.putText(image, "FPS : %0.1f" % fps, (450, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, 1)

                if det_duration > 60 :
                    self.change_pixmap_signal.emit(cv2.cvtColor(cv2.imread('Image/wating_screen.png'), cv2.COLOR_BGR2RGB))
                else :
                    self.change_pixmap_signal.emit(image)

                if cv2.waitKey(5) & 0xFF == 27:
                    break
                mode_before = mode_global
            try:
                ard.write(b'S')
                # self.is_connect.emit(1)
            except:
                pass
                # self.is_connect.emit(0)
            self.change_pixmap_signal.emit(cv2.cvtColor(cv2.imread('Image/wating_screen_2.png'), cv2.COLOR_BGR2RGB))
            hands.close()
            self.capture.release()

    class Ui_MainWindow(QtWidgets.QMainWindow):

        power_signal = pyqtSignal(bool)
        increase_or_decrease = pyqtSignal(int)

        def setupUi(self, MainWindow):
            MainWindow.setObjectName("MainWindow")
            MainWindow.resize(1600, 900)
            self.centralwidget = QtWidgets.QWidget(MainWindow)
            self.centralwidget.setObjectName("centralwidget")
            self.frame = QtWidgets.QFrame(self.centralwidget)
            self.frame.setGeometry(QtCore.QRect(0, 0, 1150, 900))
            self.frame.setStyleSheet("background-color : #FFFFFF")
            self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
            self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
            self.frame.setObjectName("frame")
            self.label = QtWidgets.QLabel(self.frame)
            self.label.setGeometry(QtCore.QRect(75, 75, 1000, 750))
            self.label.setObjectName("label")
            self.frame_2 = QtWidgets.QFrame(self.centralwidget)
            self.frame_2.setGeometry(QtCore.QRect(1150, 0, 450, 900))
            self.frame_2.setStyleSheet("background-color : #FAFAFA")
            self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
            self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
            self.frame_2.setObjectName("frame_2")

            self.label_2 = QtWidgets.QLabel(self.frame_2)
            self.label_2.setGeometry(QtCore.QRect(40, 60, 375, 120))
            self.label_2.setObjectName("label_2")
            self.label_2.setPixmap(QtGui.QPixmap("./Image/인바디.png"))
            self.label_2.setScaledContents(True)

            self.pushButton = QtWidgets.QPushButton(self.frame_2)
            self.pushButton.setGeometry(QtCore.QRect(39, 220, 191, 80))
            self.pushButton.setObjectName("pushButton")
            self.pushButton.setStyleSheet("border-radius : 10;")
            self.pushButton.setStyleSheet(
                '''
                QPushButton{image:url(./Image/모션제어.png); border:0px;}
                QPushButton:checked{image:url(./Image/모션제어_checked.png); border:0px;}
                ''')
            self.pushButton.setEnabled(True)
            self.pushButton.setCheckable(True)
            self.pushButton.setChecked(True)

            self.pushButton_2 = QtWidgets.QPushButton(self.frame_2)
            self.pushButton_2.setGeometry(QtCore.QRect(230, 220, 191, 80))
            self.pushButton_2.setObjectName("pushButton_2")
            self.pushButton_2.setStyleSheet("border-radius : 10;")
            self.pushButton_2.setStyleSheet(
                '''
                QPushButton{image:url(./Image/터치제어.png); border:0px;}
                QPushButton:checked{image:url(./Image/터치제어_checked.png); border:0px;}
                ''')
            self.pushButton_2.setEnabled(True)
            self.pushButton_2.setCheckable(True)

            self.stackedWidget = QtWidgets.QStackedWidget(self.frame_2)
            self.stackedWidget.setGeometry(QtCore.QRect(40, 300, 381, 350))
            self.stackedWidget.setObjectName("stackedWidget")
            self.page = QtWidgets.QWidget()
            self.page.setObjectName("page")
            self.frame_3 = QtWidgets.QFrame(self.page)
            self.frame_3.setGeometry(QtCore.QRect(0, 0, 381, 350))
            self.frame_3.setStyleSheet("background-color : #ECECEC")

            self.label_3 = QtWidgets.QLabel(self.frame_3)
            self.label_3.setGeometry(QtCore.QRect(10, 20, 360, 20))
            self.label_3.setObjectName("label_3")

            self.label_5 = QtWidgets.QLabel(self.frame_3)
            self.label_5.setGeometry(QtCore.QRect(10, 50, 360, 20))
            self.label_5.setObjectName("label_5")

            self.label_6 = QtWidgets.QLabel(self.frame_3)
            self.label_6.setGeometry(QtCore.QRect(60, 85, 270, 250))
            self.label_6.setObjectName("label_5")
            self.label_6.setPixmap(QtGui.QPixmap("Image/gesture_before.PNG"))
            self.label_6.setScaledContents(True)

            self.stackedWidget.addWidget(self.page)
            self.page_2 = QtWidgets.QWidget()
            self.page_2.setObjectName("page_2")
            self.frame_4 = QtWidgets.QFrame(self.page_2)
            self.frame_4.setGeometry(QtCore.QRect(0, 0, 381, 280))
            self.frame_4.setStyleSheet("background-color : #ECECEC")

            self.pushButton_3 = QtWidgets.QPushButton(self.frame_4)
            self.pushButton_3.setGeometry(QtCore.QRect(10, 50, 111, 111))
            self.pushButton_3.setObjectName("pushButton_3")
            self.pushButton_3.setStyleSheet("border-radius : 10;")
            self.pushButton_3.setStyleSheet(
                '''
                QPushButton{image:url(./Image/up.png); border:0px;}
                QPushButton:checked{image:url(./Image/up_checked.png); border:0px;}
                ''')
            self.pushButton_3.setCheckable(True)

            self.pushButton_4 = QtWidgets.QPushButton(self.frame_4)
            self.pushButton_4.setGeometry(QtCore.QRect(140, 50, 101, 101))
            self.pushButton_4.setObjectName("pushButton_4")
            self.pushButton_4.setStyleSheet("border-radius : 10;")
            self.pushButton_4.setStyleSheet(
                '''
                QPushButton{image:url(./Image/stop.png); border:0px;}
                QPushButton:checked{image:url(./Image/stop_checked.png); border:0px;}
                ''')
            self.pushButton_4.setCheckable(True)

            self.pushButton_5 = QtWidgets.QPushButton(self.frame_4)
            self.pushButton_5.setGeometry(QtCore.QRect(260, 50, 111, 111))
            self.pushButton_5.setObjectName("pushButton_5")
            self.pushButton_5.setStyleSheet("border-radius : 10;")
            self.pushButton_5.setStyleSheet(
                '''
                QPushButton{image:url(./Image/down.png); border:0px;}
                QPushButton:checked{image:url(./Image/down_checked.png); border:0px;}
                ''')
            self.pushButton_5.setCheckable(True)

            self.label_4 = QtWidgets.QLabel(self.frame_4)
            self.label_4.setGeometry(QtCore.QRect(10, 200, 361, 16))
            self.label_4.setObjectName("label_4")
            self.stackedWidget.addWidget(self.page_2)

            self.pushButton_6 = QtWidgets.QPushButton(self.frame_2)
            self.pushButton_6.setGeometry(QtCore.QRect(40, 710, 101, 101))
            self.pushButton_6.setObjectName("pushButton_6")
            self.pushButton_6.setStyleSheet("border-radius : 10;")
            self.pushButton_6.setStyleSheet(
                '''
                QPushButton{image:url(./Image/refresh.png); border:0px;}
                QPushButton:hover{image:url(./Image/refresh_hover.png); border:0px;}
                QPushButton:disabled{image:url(); border:0px;}
                ''')

            self.pushButton_7 = QtWidgets.QPushButton(self.frame_2)
            self.pushButton_7.setGeometry(QtCore.QRect(180, 710, 101, 101))
            self.pushButton_7.setObjectName("pushButton_7")

            self.pushButton_7.setStyleSheet("border-radius : 10;")
            self.pushButton_7.setStyleSheet(
                '''
                QPushButton{image:url(./Image/power.png); border:0px;}
                QPushButton:hover{image:url(./Image/power_hover.png); border:0px;}
                ''')

            self.pushButton_8 = QtWidgets.QPushButton(self.frame_2)
            self.pushButton_8.setGeometry(QtCore.QRect(320, 710, 101, 101))
            self.pushButton_8.setObjectName("pushButton_8")
            self.pushButton_8.setStyleSheet("border-radius : 10;")
            self.pushButton_8.setStyleSheet(
                '''
                QPushButton{image:url(./Image/qmark.png); border:0px;}
                QPushButton:hover{image:url(./Image/qmark_hover.png); border:0px;}
                ''')

            MainWindow.setCentralWidget(self.centralwidget)
            self.menubar = QtWidgets.QMenuBar(MainWindow)
            self.menubar.setGeometry(QtCore.QRect(0, 0, 1600, 21))
            self.menubar.setObjectName("menubar")
            MainWindow.setMenuBar(self.menubar)
            self.statusbar = QtWidgets.QStatusBar(MainWindow)
            self.statusbar.setObjectName("statusbar")
            MainWindow.setStatusBar(self.statusbar)

            self.retranslateUi(MainWindow)
            self.stackedWidget.setCurrentIndex(0)
            QtCore.QMetaObject.connectSlotsByName(MainWindow)

            self.Dialog = QtWidgets.QDialog()
            self.guide = Ui_Dialog()
            self.guide.setupUi(self.Dialog)

            self.Mywindow = MyWindow()

            self.pushButton.toggled.connect(self.motion_clicked)
            self.pushButton_2.toggled.connect(self.touch_clicked)
            self.pushButton_3.clicked.connect(lambda : self.up_clicked(MainWindow))
            self.pushButton_4.clicked.connect(lambda : self.stop_clicked(MainWindow))
            self.pushButton_5.clicked.connect(lambda : self.down_clicked(MainWindow))
            self.pushButton_6.clicked.connect(self.rebooting)
            self.pushButton_7.clicked.connect(self.Mywindow.closeEvent)
            self.pushButton_8.clicked.connect(self.to_guide_window)

            self.thread = opcv()
            self.thread.change_pixmap_signal.connect(self.update_img)
            self.power_signal.connect(self.thread.new_bool_state)
            self.thread.start()

            self.thread_2 = touch_control()
            self.increase_or_decrease.connect(self.thread_2.new_control_direction)

            self.msg = QMessageBox()

        def retranslateUi(self, MainWindow):
            _translate = QtCore.QCoreApplication.translate
            MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
            font = QtGui.QFont()
            font.setFamily("에스코어 드림 6 Bold")
            font.setPointSize(12)
            font.setBold(True)
            font.setWeight(75)
            self.label_4.setFont(font)
            self.label_4.setText(_translate("MainWindow", "버튼을 터치해 침대 경사를 조절해보세요!"))
            self.label_4.setAlignment(QtCore.Qt.AlignCenter)

            font_2 = QtGui.QFont()
            font_2.setFamily("에스코어 드림 6 Bold")
            font_2.setPointSize(16)
            font_2.setBold(True)
            font_2.setWeight(75)

            self.label_3.setFont(font_2)
            self.label_3.setText(_translate("MainWindow", "아래의 제스처 조합을 취해"))
            self.label_3.setAlignment(QtCore.Qt.AlignCenter)

            self.label_5.setFont(font_2)
            self.label_5.setText(_translate("MainWindow", "침대 기울기를 조절해보세요."))
            self.label_5.setAlignment(QtCore.Qt.AlignCenter)

        def rebooting(self):
            result = self.msg.question(self,
                                 "재부팅 확인...",
                                 "모션 인식을 새로고침하시겠습니까?",
                                 self.msg.Yes | self.msg.No)
            if result == self.msg.Yes:
                self.pushButton_2.setEnabled(False)
                self.power_signal.emit(False)
                self.thread.terminate()
                self.thread.start()
                self.power_signal.emit(True)
                self.pushButton_2.setEnabled(True)
                try :
                    global ard
                    ard = serial.Serial('COM4', 115200)
                except :
                    pass
            else :
                pass
            # self.pushButton_2.setEnabled(False)
            # self.power_signal.emit(False)
            # self.thread.terminate()
            # self.thread.start()
            # self.power_signal.emit(True)
            # self.pushButton_2.setEnabled(True)

        def cvt_qt(self, img):
            h, w, ch = img.shape  # image 쉐입 알기
            bytes_per_line = ch * w  # 차원?
            convert_to_Qt_format = QtGui.QImage(img.data, w, h, bytes_per_line,
                                                QtGui.QImage.Format_RGB888)
            p = convert_to_Qt_format.scaled(1000, 750, QtCore.Qt.KeepAspectRatio)
            return QtGui.QPixmap.fromImage(p)  # 진정한 qt 이미지 생성

        @pyqtSlot(np.ndarray)
        def update_img(self, img):
            qt_img = self.cvt_qt(img)
            self.label.setPixmap(qt_img)
            if mode_global == 1:
                self.label.setStyleSheet("color: blue;"
                              "background-color: #87CEFA;"
                              "border-style: dashed;"
                              "border-width: 10px;"
                              "border-color: #1E90FF")
            else:
                self.label.setStyleSheet(
                    "border-style: none;"
                )

        def motion_clicked(self):
            if self.pushButton.isChecked():
                if self.pushButton_2.isChecked():
                    self.pushButton_2.toggle()
                self.thread_2.terminate()
                self.stackedWidget.setCurrentIndex(0)
            else:
                if not self.pushButton_2.isChecked():
                    self.pushButton_2.toggle()

        def touch_clicked(self):
            touch_mode_buttons = [self.pushButton_3, self.pushButton_4, self.pushButton_5]
            if self.pushButton_2.isChecked():
                if self.pushButton.isChecked():
                    self.pushButton.toggle()
                self.pushButton_6.setEnabled(False)
                self.stackedWidget.setCurrentIndex(1)
                self.increase_or_decrease.emit(2)
                self.power_signal.emit(False)
                self.thread_2.start()
                for button in touch_mode_buttons :
                    button.setEnabled(True)
            else:
                self.increase_or_decrease.emit(2)
                self.power_signal.emit(True)
                self.thread.start()
                for button in touch_mode_buttons :
                    button.setChecked(False)
                    button.setEnabled(False)
                self.pushButton_6.setEnabled(True)
                if not self.pushButton.isChecked():
                    self.pushButton.toggle()

        def up_clicked(self, MainWindow):
            if self.pushButton_4.isChecked():
                self.pushButton_4.toggle()
            if self.pushButton_5.isChecked():
                self.pushButton_5.toggle()
            if self.pushButton_3.isChecked():
                self.increase_or_decrease.emit(0) # 경사 오름
            else:
                self.increase_or_decrease.emit(2) # 멈춰라

        def stop_clicked(self, MainWindow):
            if self.pushButton_3.isChecked():
                self.pushButton_3.toggle()
            if self.pushButton_5.isChecked():
                self.pushButton_5.toggle()
            self.increase_or_decrease.emit(2) # 멈춰라

        def down_clicked(self, MainWindow):
            if self.pushButton_4.isChecked():
                self.pushButton_4.toggle()
            if self.pushButton_3.isChecked():
                self.pushButton_3.toggle()
            if self.pushButton_5.isChecked():
                self.increase_or_decrease.emit(1) # 경사 내림
            else:
                self.increase_or_decrease.emit(2) # 멈춰라

        def to_guide_window(self): # 가이드 윈도우를 열어라
            self.Dialog.show()

    class MyWindow(QtWidgets.QMainWindow): # 프로그램 종료
        def __init__(self):
            super().__init__()
            self.setStyleSheet('''QMessageBox{background-color: rgb(224, 244, 253);}''')
            self.setStyleSheet('''QMainWindow{background-color : rgb(0, 0, 0);}''')
            self.msg = QMessageBox()

        def closeEvent(self, event):
            result = self.msg.question(self,
                                 "종료 확인...",
                                 "프로그램을 종료하시겠습니까?",
                                 self.msg.Yes | self.msg.No)
            if result == self.msg.Yes:
                sys.exit()
            else :
                pass

    class Ui_Dialog(object): # 가이드 윈도우
        def setupUi(self, Dialog):
            Dialog.setObjectName("Dialog")
            Dialog.resize(800, 700)
            Dialog.setStyleSheet("background-color : #F3F3F3")
            Dialog.setSizeGripEnabled(False)
            Dialog.setModal(False)
            self.label = QtWidgets.QLabel(Dialog)
            self.label.setGeometry(QtCore.QRect(250, 30, 301, 71))
            font = QtGui.QFont()
            font.setFamily("에스코어 드림 6 Bold")
            font.setPointSize(26)
            font.setBold(True)
            font.setWeight(75)
            self.label.setFont(font)
            self.label.setScaledContents(False)
            self.label.setAlignment(QtCore.Qt.AlignCenter)
            self.label.setObjectName("label")
            self.scrollArea_2 = QtWidgets.QScrollArea(Dialog)
            self.scrollArea_2.setGeometry(QtCore.QRect(10, 110, 780, 570))
            self.scrollArea_2.setMinimumSize(QtCore.QSize(780, 570))
            self.scrollArea_2.setFrameShape(QtWidgets.QFrame.NoFrame)
            self.scrollArea_2.setWidgetResizable(True)
            self.scrollArea_2.setObjectName("scrollArea_2")
            self.scrollAreaWidgetContents_2 = QtWidgets.QWidget()
            self.scrollAreaWidgetContents_2.setGeometry(QtCore.QRect(0, 0, 763, 2418))
            self.scrollAreaWidgetContents_2.setObjectName("scrollAreaWidgetContents_2")
            self.scrollArea_2.setStyleSheet('background-color : #FFFFFF')
            self.horizontalLayout = QtWidgets.QHBoxLayout(self.scrollAreaWidgetContents_2)
            self.horizontalLayout.setObjectName("horizontalLayout")
            self.frame = QtWidgets.QFrame(self.scrollAreaWidgetContents_2)
            self.frame.setMinimumSize(QtCore.QSize(0, 2400))
            self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
            self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
            self.frame.setObjectName("frame")
            self.frame.setStyleSheet('background-color : #FFFFFF')
            self.label_10 = QtWidgets.QLabel(self.frame)
            self.label_10.setGeometry(QtCore.QRect(30, 2200, 711, 41))
            font = QtGui.QFont()
            font.setFamily("에스코어 드림 3 Light")
            font.setPointSize(12)
            self.label_10.setFont(font)
            self.label_10.setObjectName("label_10")
            self.label_17 = QtWidgets.QLabel(self.frame)
            self.label_17.setGeometry(QtCore.QRect(20, 620, 711, 41))
            font = QtGui.QFont()
            font.setFamily("에스코어 드림 4 Regular")
            font.setPointSize(14)
            self.label_17.setFont(font)
            self.label_17.setObjectName("label_17")
            self.label_30 = QtWidgets.QLabel(self.frame)
            self.label_30.setGeometry(QtCore.QRect(490, 1500, 711, 41))
            font = QtGui.QFont()
            font.setFamily("에스코어 드림 3 Light")
            font.setPointSize(12)
            self.label_30.setFont(font)
            self.label_30.setObjectName("label_30")
            self.label_8 = QtWidgets.QLabel(self.frame)
            self.label_8.setGeometry(QtCore.QRect(10, 2110, 711, 41))
            font = QtGui.QFont()
            font.setFamily("에스코어 드림 4 Regular")
            font.setPointSize(14)
            self.label_8.setFont(font)
            self.label_8.setObjectName("label_8")
            self.label_31 = QtWidgets.QLabel(self.frame)
            self.label_31.setGeometry(QtCore.QRect(30, 1730, 321, 291))
            self.label_31.setText("")
            self.label_31.setPixmap(QtGui.QPixmap("Image/guide_13.PNG"))
            self.label_31.setScaledContents(True)
            self.label_31.setObjectName("label_31")
            self.label_2 = QtWidgets.QLabel(self.frame)
            self.label_2.setGeometry(QtCore.QRect(30, 1380, 421, 241))
            self.label_2.setText("")
            self.label_2.setPixmap(QtGui.QPixmap("Image/guide_10.png"))
            self.label_2.setScaledContents(True)
            self.label_2.setObjectName("label_2")
            self.label_34 = QtWidgets.QLabel(self.frame)
            self.label_34.setGeometry(QtCore.QRect(440, 1860, 711, 41))
            font = QtGui.QFont()
            font.setFamily("에스코어 드림 3 Light")
            font.setPointSize(12)
            self.label_34.setFont(font)
            self.label_34.setObjectName("label_34")
            self.label_26 = QtWidgets.QLabel(self.frame)
            self.label_26.setGeometry(QtCore.QRect(40, 720, 711, 41))
            font = QtGui.QFont()
            font.setFamily("에스코어 드림 3 Light")
            font.setPointSize(12)
            self.label_26.setFont(font)
            self.label_26.setObjectName("label_26")
            self.label_38 = QtWidgets.QLabel(self.frame)
            self.label_38.setGeometry(QtCore.QRect(380, 1910, 51, 51))
            self.label_38.setText("")
            self.label_38.setPixmap(QtGui.QPixmap("Image/down_activation.png"))
            self.label_38.setScaledContents(True)
            self.label_38.setObjectName("label_38")
            self.label_25 = QtWidgets.QLabel(self.frame)
            self.label_25.setGeometry(QtCore.QRect(120, 1000, 711, 41))
            font = QtGui.QFont()
            font.setFamily("에스코어 드림 3 Light")
            font.setPointSize(12)
            self.label_25.setFont(font)
            self.label_25.setObjectName("label_25")
            self.label_7 = QtWidgets.QLabel(self.frame)
            self.label_7.setGeometry(QtCore.QRect(10, 2060, 271, 41))
            font = QtGui.QFont()
            font.setFamily("에스코어 드림 5 Medium")
            font.setPointSize(18)
            self.label_7.setFont(font)
            self.label_7.setObjectName("label_7")
            self.label_14 = QtWidgets.QLabel(self.frame)
            self.label_14.setGeometry(QtCore.QRect(30, 80, 681, 401))
            self.label_14.setText("")
            self.label_14.setPixmap(QtGui.QPixmap("Image/program.png"))
            self.label_14.setScaledContents(True)
            self.label_14.setObjectName("label_14")
            self.label_22 = QtWidgets.QLabel(self.frame)
            self.label_22.setGeometry(QtCore.QRect(40, 990, 61, 61))
            self.label_22.setText("")
            self.label_22.setPixmap(QtGui.QPixmap("Image/qmark.png"))
            self.label_22.setScaledContents(True)
            self.label_22.setObjectName("label_22")
            self.label_32 = QtWidgets.QLabel(self.frame)
            self.label_32.setGeometry(QtCore.QRect(380, 1740, 711, 41))
            font = QtGui.QFont()
            font.setFamily("에스코어 드림 3 Light")
            font.setPointSize(12)
            self.label_32.setFont(font)
            self.label_32.setObjectName("label_32")
            self.label_3 = QtWidgets.QLabel(self.frame)
            self.label_3.setGeometry(QtCore.QRect(20, 20, 181, 41))
            font = QtGui.QFont()
            font.setFamily("에스코어 드림 5 Medium")
            font.setPointSize(18)
            self.label_3.setFont(font)
            self.label_3.setObjectName("label_3")
            self.label_4 = QtWidgets.QLabel(self.frame)
            self.label_4.setGeometry(QtCore.QRect(9, 1079, 191, 41))
            font = QtGui.QFont()
            font.setFamily("에스코어 드림 5 Medium")
            font.setPointSize(18)
            self.label_4.setFont(font)
            self.label_4.setObjectName("label_4")
            self.label_19 = QtWidgets.QLabel(self.frame)
            self.label_19.setGeometry(QtCore.QRect(20, 790, 711, 41))
            font = QtGui.QFont()
            font.setFamily("에스코어 드림 4 Regular")
            font.setPointSize(14)
            self.label_19.setFont(font)
            self.label_19.setObjectName("label_19")
            self.label_33 = QtWidgets.QLabel(self.frame)
            self.label_33.setGeometry(QtCore.QRect(380, 1790, 711, 41))
            font = QtGui.QFont()
            font.setFamily("에스코어 드림 3 Light")
            font.setPointSize(12)
            self.label_33.setFont(font)
            self.label_33.setObjectName("label_33")
            self.label_36 = QtWidgets.QLabel(self.frame)
            self.label_36.setGeometry(QtCore.QRect(440, 1980, 711, 41))
            font = QtGui.QFont()
            font.setFamily("에스코어 드림 3 Light")
            font.setPointSize(12)
            self.label_36.setFont(font)
            self.label_36.setObjectName("label_36")
            self.label_37 = QtWidgets.QLabel(self.frame)
            self.label_37.setGeometry(QtCore.QRect(380, 1850, 51, 51))
            self.label_37.setText("")
            self.label_37.setPixmap(QtGui.QPixmap("Image/up_activation.png"))
            self.label_37.setScaledContents(True)
            self.label_37.setObjectName("label_37")
            self.label_24 = QtWidgets.QLabel(self.frame)
            self.label_24.setGeometry(QtCore.QRect(120, 930, 711, 41))
            font = QtGui.QFont()
            font.setFamily("에스코어 드림 3 Light")
            font.setPointSize(12)
            self.label_24.setFont(font)
            self.label_24.setObjectName("label_24")
            self.label_9 = QtWidgets.QLabel(self.frame)
            self.label_9.setGeometry(QtCore.QRect(30, 2160, 711, 41))
            font = QtGui.QFont()
            font.setFamily("에스코어 드림 3 Light")
            font.setPointSize(12)
            self.label_9.setFont(font)
            self.label_9.setObjectName("label_9")
            self.label_11 = QtWidgets.QLabel(self.frame)
            self.label_11.setGeometry(QtCore.QRect(30, 2290, 711, 41))
            font = QtGui.QFont()
            font.setFamily("에스코어 드림 3 Light")
            font.setPointSize(12)
            self.label_11.setFont(font)
            self.label_11.setObjectName("label_11")
            self.label_27 = QtWidgets.QLabel(self.frame)
            self.label_27.setGeometry(QtCore.QRect(40, 1260, 711, 41))
            font = QtGui.QFont()
            font.setFamily("에스코어 드림 3 Light")
            font.setPointSize(12)
            self.label_27.setFont(font)
            self.label_27.setObjectName("label_27")
            self.label_35 = QtWidgets.QLabel(self.frame)
            self.label_35.setGeometry(QtCore.QRect(440, 1920, 711, 41))
            font = QtGui.QFont()
            font.setFamily("에스코어 드림 3 Light")
            font.setPointSize(12)
            self.label_35.setFont(font)
            self.label_35.setObjectName("label_35")
            self.label_28 = QtWidgets.QLabel(self.frame)
            self.label_28.setGeometry(QtCore.QRect(40, 1300, 711, 41))
            font = QtGui.QFont()
            font.setFamily("에스코어 드림 3 Light")
            font.setPointSize(12)
            self.label_28.setFont(font)
            self.label_28.setObjectName("label_28")
            self.label_16 = QtWidgets.QLabel(self.frame)
            self.label_16.setGeometry(QtCore.QRect(40, 560, 711, 41))
            font = QtGui.QFont()
            font.setFamily("에스코어 드림 3 Light")
            font.setPointSize(12)
            self.label_16.setFont(font)
            self.label_16.setObjectName("label_16")
            self.label_13 = QtWidgets.QLabel(self.frame)
            self.label_13.setGeometry(QtCore.QRect(10, 2240, 711, 41))
            font = QtGui.QFont()
            font.setFamily("에스코어 드림 4 Regular")
            font.setPointSize(14)
            self.label_13.setFont(font)
            self.label_13.setObjectName("label_13")
            self.label_15 = QtWidgets.QLabel(self.frame)
            self.label_15.setGeometry(QtCore.QRect(20, 510, 711, 41))
            font = QtGui.QFont()
            font.setFamily("에스코어 드림 4 Regular")
            font.setPointSize(14)
            self.label_15.setFont(font)
            self.label_15.setObjectName("label_15")
            self.label_21 = QtWidgets.QLabel(self.frame)
            self.label_21.setGeometry(QtCore.QRect(40, 920, 61, 61))
            self.label_21.setText("")
            self.label_21.setPixmap(QtGui.QPixmap("Image/power.png"))
            self.label_21.setScaledContents(True)
            self.label_21.setObjectName("label_21")
            self.label_39 = QtWidgets.QLabel(self.frame)
            self.label_39.setGeometry(QtCore.QRect(380, 1970, 51, 51))
            self.label_39.setText("")
            self.label_39.setPixmap(QtGui.QPixmap("Image/stop_activation.png"))
            self.label_39.setScaledContents(True)
            self.label_39.setObjectName("label_39")
            self.label_40 = QtWidgets.QLabel(self.frame)
            self.label_40.setGeometry(QtCore.QRect(40, 1180, 261, 51))
            self.label_40.setText("")
            self.label_40.setPixmap(QtGui.QPixmap("Image/guide_12.PNG"))
            self.label_40.setScaledContents(True)
            self.label_40.setObjectName("label_40")
            self.label_23 = QtWidgets.QLabel(self.frame)
            self.label_23.setGeometry(QtCore.QRect(120, 860, 711, 41))
            font = QtGui.QFont()
            font.setFamily("에스코어 드림 3 Light")
            font.setPointSize(12)
            self.label_23.setFont(font)
            self.label_23.setObjectName("label_23")
            self.label_20 = QtWidgets.QLabel(self.frame)
            self.label_20.setGeometry(QtCore.QRect(40, 850, 61, 61))
            self.label_20.setText("")
            self.label_20.setPixmap(QtGui.QPixmap("Image/refresh.png"))
            self.label_20.setScaledContents(True)
            self.label_20.setObjectName("label_20")
            self.label_12 = QtWidgets.QLabel(self.frame)
            self.label_12.setGeometry(QtCore.QRect(30, 2330, 711, 41))
            font = QtGui.QFont()
            font.setFamily("에스코어 드림 3 Light")
            font.setPointSize(12)
            self.label_12.setFont(font)
            self.label_12.setObjectName("label_12")
            self.label_18 = QtWidgets.QLabel(self.frame)
            self.label_18.setGeometry(QtCore.QRect(40, 670, 711, 41))
            font = QtGui.QFont()
            font.setFamily("에스코어 드림 3 Light")
            font.setPointSize(12)
            self.label_18.setFont(font)
            self.label_18.setObjectName("label_18")
            self.label_6 = QtWidgets.QLabel(self.frame)
            self.label_6.setGeometry(QtCore.QRect(10, 1660, 191, 41))
            font = QtGui.QFont()
            font.setFamily("에스코어 드림 5 Medium")
            font.setPointSize(18)
            self.label_6.setFont(font)
            self.label_6.setObjectName("label_6")
            self.label_29 = QtWidgets.QLabel(self.frame)
            self.label_29.setGeometry(QtCore.QRect(490, 1450, 711, 41))
            font = QtGui.QFont()
            font.setFamily("에스코어 드림 3 Light")
            font.setPointSize(12)
            self.label_29.setFont(font)
            self.label_29.setObjectName("label_29")
            self.label_5 = QtWidgets.QLabel(self.frame)
            self.label_5.setGeometry(QtCore.QRect(40, 1130, 711, 41))
            font = QtGui.QFont()
            font.setFamily("에스코어 드림 3 Light")
            font.setPointSize(12)
            self.label_5.setFont(font)
            self.label_5.setObjectName("label_5")
            self.horizontalLayout.addWidget(self.frame)
            self.scrollArea_2.setWidget(self.scrollAreaWidgetContents_2)

            self.retranslateUi(Dialog)
            QtCore.QMetaObject.connectSlotsByName(Dialog)

        def retranslateUi(self, Dialog):
            _translate = QtCore.QCoreApplication.translate
            Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
            self.label.setText(_translate("Dialog", "프로그램 사용 방법"))
            self.label_10.setText(_translate("Dialog", "- 카메라와 컴퓨터의 연결선을 확인한 뒤 새로고침 버튼을 눌러보세요."))
            self.label_17.setText(_translate("Dialog", "(2) 제어 옵션 선택부"))
            self.label_30.setText(_translate("Dialog", "침대의 운동을 제어할 수 있습니다."))
            self.label_8.setText(_translate("Dialog", "(1) 영상이 송출되지 않을 때"))
            self.label_34.setText(_translate("Dialog", "를 누르면, 침대의 경사가 증가합니다."))
            self.label_26.setText(_translate("Dialog", "- 두 제어 옵션을 동시에 사용하는 것은 불가능합니다."))
            self.label_25.setText(_translate("Dialog", "버튼을 누르면, 프로그램의 사용 방법에 대한 도움창를 띄울 수 있습니다. "))
            self.label_7.setText(_translate("Dialog", "4. 작동 오류시 해결 방법"))
            self.label_32.setText(_translate("Dialog", "\'터치 제어\' 버튼을 누르면 좌측과 같이"))
            self.label_3.setText(_translate("Dialog", "1. 프로그램 구성"))
            self.label_4.setText(_translate("Dialog", "2. 모션 제어 방법"))
            self.label_19.setText(_translate("Dialog", "(3) 새로고침/프로그램 종료/도움말"))
            self.label_33.setText(_translate("Dialog", "침대를 제어할 수 있는 버튼이 제공됩니다."))
            self.label_36.setText(_translate("Dialog", "를 눌러, 침대 운동를 중지할 수 있습니다."))
            self.label_24.setText(_translate("Dialog", "버튼을 누르면, 프로그램을 종료할 수 있습니다. "))
            self.label_9.setText(_translate("Dialog", "- 우측 하단의 새로고침 버튼을 눌러보세요."))
            self.label_11.setText(_translate("Dialog", "- 컨트롤 박스와 컴퓨터의 연결을 확인한 뒤 프로그램을 다시 시작해보세요."))
            self.label_27.setText(_translate("Dialog", "영상 송출부를 통해서 제스처 인식 영상을 볼 수 있습니다. 이때, 두 손이 카메라에 모두 노출되어야만"))
            self.label_35.setText(_translate("Dialog", "를 누르면, 침대의 경사가 감소합니다."))
            self.label_28.setText(_translate("Dialog", "제스처 인식이 시작됩니다."))
            self.label_16.setText(_translate("Dialog", "- 유저에게 실시간 핸드 트래킹 영상을 보여줍니다."))
            self.label_13.setText(_translate("Dialog", "(2) 침대 제어가 안될 때"))
            self.label_15.setText(_translate("Dialog", "(1) 영상 송출부"))
            self.label_23.setText(_translate("Dialog", "버튼을 누르면, 모션 인식 기능을 재부팅할 수 있습니다. "))
            self.label_12.setText(_translate("Dialog", "- 침대의 전원을 확인한 뒤 프로그램을 다시 시작해보세요."))
            self.label_18.setText(_translate("Dialog", "- 모션으로 침대를 제어할지, 스크린을 터치하여 침대를 제어할지 선택할 수 있습니다."))
            self.label_6.setText(_translate("Dialog", "3. 터치 제어 방법"))
            self.label_29.setText(_translate("Dialog", "좌측의 제스처 조합을 이용하여"))
            self.label_5.setText(_translate("Dialog", "\'모션 제어\' 버튼을 누르면 손을 이용한 제스처로 침대를 제어할 수 있게 됩니다. "))

    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

def data_predict(p_list, maximum):

    import tensorflow as tf
    import keras.backend
    import keras
    config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)

    try:
        model = keras.models.load_model(
            '../model_save/my_model_63_report.h5'
        )
    except:
        print('ML 모델 로딩 실패')
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
            maximum.value = 3

if __name__ == '__main__':
    p_list = Queue(1)
    maximum = Value('i', 0)
    p_list_2 = Queue(1)
    maximum_2 = Value('i', 0)
    process1 = Process(target=cv_gui_arduino, args=(p_list, maximum, p_list_2, maximum_2))
    process2 = Process(target=data_predict, args=(p_list, maximum,))
    process3 = Process(target=data_predict, args=(p_list_2, maximum_2,))
    process1.start()
    process2.start()
    process3.start()
    while process1.is_alive():
        pass
    process2.terminate()
    process3.terminate()
    process1.join()
    process2.join()
    process3.join()


