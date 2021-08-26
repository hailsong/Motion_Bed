import pandas as pd
import glob
import os

input_file = r'C:\Users\user\PycharmProjects\working\DATASET_tentative\increase'
input_file2 = r'C:\Users\user\PycharmProjects\working\DATASET_tentative\decrease'
input_file3 = r'C:\Users\user\PycharmProjects\working\DATASET_tentative\stop'
# input_file4 = r'C:\Users\user\PycharmProjects\MLbasic\DATASET_MODE\Mode3'
# input_file5 = r'C:\Users\user\PycharmProjects\MLbasic\DATASET_MODE\ANY'
output_file = r'C:\Users\user\PycharmProjects\working\output_sum_63_tentative.csv'

allFile_list = glob.glob(os.path.join(input_file, '*_63.csv')) # glob함수로 _csv로 끝나는 파일들을 모은다
allFile_list = allFile_list[:-1]
allFile_list2 = glob.glob(os.path.join(input_file2, '*_63.csv'))
allFile_list2 = allFile_list2[:-1]
allFile_list3 = glob.glob(os.path.join(input_file3, '*_63.csv'))
allFile_list3 = allFile_list3[:-1]
# allFile_list4 = glob.glob(os.path.join(input_file4, '*_63.csv'))
# allFile_list4 = allFile_list4[:-1]
# allFile_list5 = glob.glob(os.path.join(input_file5, '*_63.csv'))
# allFile_list5 = allFile_list5[:-1]
allFile_list = allFile_list + allFile_list2 + allFile_list3 # + allFile_list4 + allFile_list5

allData = [] # 읽어 들인 csv파일 내용을 저장할 빈 리스트를 하나 만든다

for file in allFile_list:
    df = pd.read_csv(file) # for구문으로 csv파일들을 읽어 들인다
    allData.append(df) # 빈 리스트에 읽어 들인 내용을 추가한다

dataCombine = pd.concat(allData, axis=0, ignore_index=True) # concat함수를 이용해서 리스트의 내용을 병합
# axis=0은 수직으로 병합함. axis=1은 수평. ignore_index=True는 인데스 값이 기존 순서를 무시하고 순서대로 정렬되도록 한다.
dataCombine.to_csv(output_file, index=False) # to_csv함수로 저장한다. 인데스를 빼려면 False로 설정
