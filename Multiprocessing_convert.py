

import os
import pandas as pd
import multiprocessing
import time

path=os.getcwd() # 현재 디렉토리 위치 반환

FOLDER_NAME = 'DATASET_tentative/increase/'
split_list = FOLDER_NAME.split('/') # '/' 문자열을 기준으로 싹뚝 잘라 리스트에 넣기
dir_1 = split_list[0] # DATASET
dir_2 = split_list[1] # slope_increase

init_list = [[0. for _ in range(4+21*3)],]
init_list[0][:4] = ['dummy', 'dummy', 'dummy', True]
column_name = ['FILENAME', 'real', 'LR', 'match',]
print(init_list)
for i in range(21*3):
    column_name.append(str(i)) # 63개의 좌표값을 넣을 칼럼 만들기

if not(os.path.isdir("../working/" + dir_1)): # 현재의 디렉토리에 /video_output/LSTM_DATASET2 디렉토리가 없다면
    os.makedirs(os.path.join("../working/" + dir_1 + "/")) # /video_output/LSTM_DATASET2 디렉토리를 만들어라.

if not os.path.isdir("../working/" + dir_1 + "/" + dir_2): # 현재의 디렉토리에 /video_output/LSTM_DATASET2/POSE_+POSE_NAME 디렉토리가 없다면
    print("../working/" + dir_1 + "/" + dir_2 + "/")
    os.makedirs(os.path.join("../working/" + dir_1 + "/" + dir_2 + "/")) # 만들어라

experiment_df = pd.DataFrame.from_records(init_list, columns = column_name) # ndarray, 튜플의 리스트, 딕셔너리 혹은 DataFrame을 입력받는다. 사용하기 위한 열의 레이블을 입력받는다. init_list 의 값들을 입력받음.
#print(experiment_df)
experiment_df.to_csv("../working/" + FOLDER_NAME + "/output_63.csv") # ,로 구분된 csv 파일을 괄호 안의 디렉토리에 만든다.

# multiprocessing
def file_convert(i):
    filename = FOLDER_NAME + str(i) + '.avi'
    progress = round((i+1) * 100 / 8, 2)
    #print(filename)
    os.system('python get_data_norm.py --target={}'.format(filename)) # 현재의 디렉토리에서 괄호 안의 시스템 명령어를 사용한다.
    print('PROCESSING VIDEO : {},    PROGRESS : {}%'.format(filename, progress)) # 현재 변환 중인 파일 이름과 진행 정도를 프린트한다.

if __name__ == '__main__':
    start_time = time.time()
    pool = multiprocessing.Pool(processes=6) # 프로세스를 6개 돌린다.
    num = range(0,4) # 200개의 num 인풋값
    pool.map(file_convert, num) # file_convert 함수에 num 인풋을 매핑해준다.
    print(time.time()-start_time) # processing이 얼마나 걸렸는지 프린트

    if os.path.isfile("../working/DATASET_tentative/csv_list_LSTM.txt"): # csv_list_LSTM.txt 파일이 있다면
        for i in num: # 200 번 반복
            txt = open("../working/DATASET_tentative/csv_list_LSTM.txt", 'a') # 파일을 'a'(파일의 마지막에 새로운 내용 추가) 모드로 연다.
            print('txt append', i) # 반복 횟수와 POSE 종류를 txt에 넣을 것을 알려줌.
            data = FOLDER_NAME + str(i) + '_63.csv\n' # 폴더 이름과 반복 횟수 + _63.csv\n(줄바꿈) 으로 이뤄진 string 생성
            txt.write(data) # 파일에 저장한다.