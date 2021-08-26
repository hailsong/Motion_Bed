import pandas as pd
import numpy as np
import cv2
from tensorflow import keras
import mediapipe as mp
import math
import serial
import threading
from mediapipe.framework.formats import location_data_pb2
import time
from multiprocessing import Queue
# ard = serial.Serial('COM4',115200)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection


model = keras.models.load_model(
    'model_save/my_model_63_mode.h5'
)

class mark_pixel():
    def __init__(self, x, y, z = 0, LR = 0):
        self.x = x
        self.y = y
        self.z = z
        self.LR = LR
    def __str__(self):
        return str(self.x) +'   '+ str(self.y) +'   ' + str(self.z)
    def to_list(self):
        return [self.x, self.y, self.z]
    def to_pixel(self):
        global x_size
        global y_size
        return mark_2d(self.x * x_size, self.y * y_size)
    def __sub__(self, other):
        return self.x - other.x, self.y - other.y, self.z - other.z

class mark_2d():
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __str__(self):
        return tuple(self.x, self.y)


def BlurFunction(src):
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:  # with 문, mp_face_detection.FaceDetection 클래스를 face_detection으로서 사용
        image = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)  # image 파일의 BGR 색상 베이스를 RGB 베이스로 바꾸기
        results = face_detection.process(image)  # 튜플 형태
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_rows, image_cols, _ = image.shape
        c_mask: ndarray = np.zeros((image_rows, image_cols), np.uint8)
        if results.detections:
            for detection in results.detections:
                if not detection.location_data:
                    break
                if image.shape[2] != 3:
                    raise ValueError('Input image must contain three channel rgb data.')
                location = detection.location_data
                if location.format != location_data_pb2.LocationData.RELATIVE_BOUNDING_BOX:
                    raise ValueError('LocationData must be relative for this drawing funtion to work.')
                # Draws bounding box if exists.
                if not location.HasField('relative_bounding_box'):
                    break
                relative_bounding_box = location.relative_bounding_box
                rect_start_point = mp_drawing._normalized_to_pixel_coordinates(
                    relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
                    image_rows)
                rect_end_point = mp_drawing._normalized_to_pixel_coordinates(
                    relative_bounding_box.xmin + relative_bounding_box.width,
                    relative_bounding_box.ymin + relative_bounding_box.height, image_cols,
                    image_rows)
                try :
                    x1 = int((rect_start_point[0]+rect_end_point[0])/2)
                    y1 = int((rect_start_point[1]+rect_end_point[1])/2)
                    a = int(rect_end_point[0]-rect_start_point[0])
                    b = int(rect_end_point[1]-rect_start_point[1])
                    radius = int(math.sqrt(a*a+b*b)/2*0.8)
                    # 원 스펙 설정
                    cv2.circle(c_mask, (x1,y1), radius, (255,255,255), -1)
                except :
                    pass
            img_all_blurred = cv2.blur(image, (17,17))
            c_mask = cv2.cvtColor(c_mask, cv2.COLOR_GRAY2BGR)
            #print(c_mask.shape)
            image = np.where(c_mask > 0, img_all_blurred, image)
    return image

def data_predict():
    df = pd.DataFrame(mark_p_list, columns=[str(i) for i in range(0, 63)])
    col_name = [str(i) for i in range(0, 63)]
    gstr = df[col_name].to_numpy()
    print(gstr)
    prediction = model.predict(gstr[[0]])
    prediction = prediction[0]
    MAX = prediction.argmax()  # sigmoid 모델을 사용할 때는 쓰지 않는다.
    if prediction[MAX] > 0.9:
        ex_gesture.append(MAX)
    if MAX != 4:
        if ex_gesture == [MAX for i in range(5)]:
            cv2.putText(image, 'Mode{}'.format(MAX), (30, 30), cv2.FONT_HERSHEY_PLAIN, 2, 4)
    if len(ex_gesture) == 5:
        del ex_gesture[0]

def data_normalize(list_1, list_2, list_3):
    xvalue = []
    yvalue = []
    zvalue = []
    for i in range(21):
        xval, yval, zval = list_2.landmark[i].x - list_2.landmark[0].x, \
                           list_2.landmark[i].y - list_2.landmark[0].y, \
                           list_2.landmark[i].z - list_2.landmark[0].z
        list_1.append(xval)
        list_1.append(yval)
        list_1.append(zval)# 마크 픽셀 클래스 인스턴스를 사용하여 mark_p 리스트에 넣음
        xvalue.append(abs(xval))
        yvalue.append(abs(yval))
        zvalue.append(abs(zval))
    xmax = max(xvalue)
    ymax = max(yvalue)
    zmax = max(zvalue)
    for i in range(21):
        i = 3*i
        list_1[i] = list_1[i] / xmax
        list_1[i+1] = list_1[i+1] / ymax
        list_1[i+2] = list_1[i+2] / zmax
    list_3.append(list_1)  # 21 개 index에 대한 모든 객체가 삽입된 mark_p 리스트를 mark_p_list에 넣는다.


##### 데이터 준비 #######
width = 960  # 너비
height = 540  # 높이
bpp = 3  # 표시 채널(grayscale:1, bgr:3, transparent: 4)
img = np.full((height, width, bpp), 255, np.uint8)  # 빈 화면 표시

hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

before_c = mark_pixel(0, 0, 0)
pixel_c = mark_pixel(0, 0, 0)
hm_idx = False
finger_open_ = [False for _ in range(5)]
mode = 0
ex_gesture = []

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
prevTime = 0
while cap.isOpened():
    success, image = cap.read()

    curTime = time.time()
    sec = curTime - prevTime
    prevTime = curTime
    fps = 1 / sec
    if not success:
        exit()
    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image = BlurFunction(image)
    # x_size, y_size, channel = image.shape
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mode_predict = threading.Thread(target=data_predict)
    if results.multi_hand_landmarks:
        mark_p_list = []
        for hand_landmarks in results.multi_hand_landmarks:  # hand_landmarks는 감지된 손의 갯수만큼의 원소 수를 가진 list 자료구조
            mark_p = []
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            data_normalize(mark_p, hand_landmarks, mark_p_list)
        # mode_predict = threading.Thread(target=data_predict, args=(mark_p_list, ex_gesture, image))
        mode_predict.start()
        #     xvalue = []
        #     yvalue = []
        #     zvalue = []
        #     for i in range(21):
        #         xval, yval, zval = hand_landmarks.landmark[i].x - hand_landmarks.landmark[0].x, \
        #                            hand_landmarks.landmark[i].y - hand_landmarks.landmark[0].y, \
        #                            hand_landmarks.landmark[i].z - hand_landmarks.landmark[0].z
         #         mark_p.append(xval)
        #         mark_p.append(yval)
        #         mark_p.append(zval)# 마크 픽셀 클래스 인스턴스를 사용하여 mark_p 리스트에 넣음
        #         xvalue.append(abs(xval))
        #         yvalue.append(abs(yval))
        #         zvalue.append(abs(zval))
        #     xmax = max(xvalue)
        #     ymax = max(yvalue)
        #     zmax = max(zvalue)
        #     for i in range(21):
        #         i = 3*i
        #         mark_p[i] = mark_p[i] / xmax
        #         mark_p[i+1] = mark_p[i+1] / ymax
        #         mark_p[i+2] = mark_p[i+2] / zmax
        #     mark_p_list.append(mark_p)  # 21 개 index에 대한 모든 객체가 삽입된 mark_p 리스트를 mark_p_list에 넣는다.
        # data_predict(mark_p_list)
        # df = pd.DataFrame(mark_p_list, columns=[str(i) for i in range(0,63)])
        # col_name = [str(i) for i in range(0,63)]
        # gstr = df[col_name].to_numpy()
        # prediction = model.predict(gstr[[0]])
        # prediction = prediction[0]
        # MAX = prediction.argmax()  # sigmoid 모델을 사용할 때는 쓰지 않는다.
        # if prediction[MAX] > 0.9:
        #     ex_gesture.append(MAX)
        # if MAX != 4:
        #     if ex_gesture == [MAX for i in range(5)] :
        #         cv2.putText(image, 'Mode{}'.format(MAX), (30,30), cv2.FONT_HERSHEY_PLAIN, 2, 4)
        #     # if mode != MAX:
            #     ard.write(b'MAX')
            # mode = MAX
        # if ex_gesture == [0,0,0,0,0]:
        #     cv2.putText(image, 'Mode0', (30,30), cv2.FONT_HERSHEY_PLAIN, 2, 4)
        #     if mode != 0:
        #         ard.write(b'0')
        #     mode = 0
        # elif ex_gesture == [1,1,1,1,1]:
        #     cv2.putText(image, 'Mode1', (30,30), cv2.FONT_HERSHEY_PLAIN, 2, 4)
        #     if mode != 1:
        #         ard.write(b'1')
        #     mode = 1
        # elif ex_gesture == [2,2,2,2,2]:
        #     cv2.putText(image, 'Mode2', (30,30), cv2.FONT_HERSHEY_PLAIN, 2, 4)
        #     if mode != 2:
        #         ard.write(b'2')
        #     mode = 2
        # elif ex_gesture == [3,3,3,3,3]:
        #     cv2.putText(image, 'Mode3', (30,30), cv2.FONT_HERSHEY_PLAIN, 2, 4)
        #     if mode != 3:
        #         ard.write(b'3')
        #     mode = 3
        # if len(ex_gesture) == 5:
        #     del ex_gesture[0]
        mode_predict.join()
    cv2.putText(image, "FPS : %0.1f" % fps, (400, 30), cv2.FONT_HERSHEY_PLAIN, 2, 4)
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

hands.close()
cap.release()
cv2.destroyAllWindows()