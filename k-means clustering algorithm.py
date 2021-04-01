import numpy as np
import pandas as pd

# import file
###[input1] file of 5-vectors ###
fileAdr = input("input할 파일의 주소를 입력하시오 : ")
file = pd.read_csv(fileAdr, encoding="utf-8", sep=' ', header=None)

#Data Preprocessing
file = file[[0, 1, 2, 3, 4]] #5-vectors외의 값은 모두 삭제
file = file.to_numpy() #dataframe to numpy.ndarray

# k-means Algorithm
###[input2] cluster갯수(k) ###
k = int(input("몇 개의 cluster로 clustering할 건가요? : "))
###[input3] the maximum number of iterations ###
iter_max = int(input("몇 번 iteration(반복)할 건가요? : "))

#5차원에서 두 점 a,b 사이의 거리를 구하는 함수 (L5-Norm)
def distance(a, b):
  return sum([abs(a_col - b_col)**5  for a_col, b_col in list(zip(a, b))]) ** 0.2

#1. initial list of k group representative vectors (cntrds[0],..,cntrds[k-1])
#랜덤으로 좌표(cntrds_1, cntrds_2, cntrds_3, cntrds_4, cntrds_5) k개 생성
cntrds_1= np.random.uniform(min(file[:,0]), max(file[:,0]), k)
cntrds_2= np.random.uniform(min(file[:,1]), max(file[:,1]), k)
cntrds_3= np.random.uniform(min(file[:,2]), max(file[:,2]), k)
cntrds_4= np.random.uniform(min(file[:,3]), max(file[:,3]), k)
cntrds_5= np.random.uniform(min(file[:,4]), max(file[:,4]), k)
cntrds = list(zip(cntrds_1, cntrds_2, cntrds_3, cntrds_4, cntrds_5))

df_cntrds = pd.DataFrame(cntrds)
cntrds = df_cntrds.to_numpy() #dataframe to numpy.ndarray

#2. 각 데이터가 속한 그룹index(0~k-1)를 element로 가지는 벡터 labels를 0으로 초기화
labels = np.zeros(file.shape[0])

###iteration 시작###
for iter in range(iter_max+1):

    ### Partitioning the 5-vectors into k groups ###
    #3. data.column에 "labels"를 추가하기 위해서 for루프를 돌음
    for i in range(file.shape[0]):
    #해당 데이터로부터 각 group representative vector의 거리를 element로 가지는 벡터 distances를 0으로 초기화
        distances = np.zeros(k)
        for n in range(k):
            distances[n] = distance(file[i], cntrds[n])
            #cluster에 벡터 distances의 element 중 가장 작은 값의 index를 저장
            #즉, cluster는 각 데이터와 가장 가까운 group representative vector의 index
            cluster = np.argmin(distances) 
            #labal[i]에는 0~k-1 중 하나가 들어가게 됨
            labels[i] = int(cluster)

    ### Update representatives ###        
    #각 그룹별 해당하는 벡터들을 알기위해서 for루프를 돌음
    for n in range(k):
        points= [file[i] for i in range(file.shape[0]) if n==labels[i]] #k group representative vectors의 index와 각 데이터의 clust가 일치하는 데이터 추출
        df_points = pd.DataFrame(points)
        if df_points.empty==True: #해당 그룹에 속하는 데이터가 아예 없을 경우 pass
            pass
        else:
            points = df_points.to_numpy() #list로 저장된, 해당 그룹의 points를 array로 변환
            cntrds[n] = np.mean(points, axis=0) #update representatives : 기존 group representative vector을 해당 그룹의 points의 평균으로 변경

# Outputs
###[output1] the centroids of each cluster ###
print("# of actual iterations : %d"%iter_max)
print("centroids : ", end='')
for n in range(k-1):
    print(tuple(cntrds[n]), end=",")
print(tuple(cntrds[k-1]))

###[output2] # of vectors in each cluster ###
for n in range(k):
    points= [file[i] for i in range(file.shape[0]) if n==labels[i]] #k group representative vectors의 index와 각 데이터의 clust가 일치하는 데이터 추출
    df_points = pd.DataFrame(points)
    if df_points.empty==True: #해당 그룹에 속하는 데이터가 아예 없을 경우 pass
        print("# of vectors for cluster %d = "%(n+1), 0)
    else:
        points = df_points.to_numpy() #list로 저장된, 해당 그룹의 points를 array로 변환
        print("# of vectors for cluster %d = "%(n+1), len(points))