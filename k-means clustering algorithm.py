import numpy as np
import pandas as pd

### Import file ###
fileAdr = input("input할 파일의 주소를 입력하시오 : ") #[input1] file
file = pd.read_csv(fileAdr, encoding="utf-8", sep=' ', header=None)

### Data Processing ###
file = file[[0, 1, 2, 3, 4]] 
file = file.to_numpy()

### k-means clustering algorithm ###
k = int(input("몇 개의 cluster로 clustering할 건가요? : ")) #[input2] cluster갯수(k)
iter_max = int(input("몇 번 iteration(반복)할 건가요? : ")) #[input3] the maximum number of iterations
def distance(a, b): #L5-Norm
  return sum([abs(a_col - b_col)**5  for a_col, b_col in list(zip(a, b))]) ** 0.2

### 1) initial list of k group representative vectors ###
cntrds_1= np.random.uniform(min(file[:,0]), max(file[:,0]), k)
cntrds_2= np.random.uniform(min(file[:,1]), max(file[:,1]), k)
cntrds_3= np.random.uniform(min(file[:,2]), max(file[:,2]), k)
cntrds_4= np.random.uniform(min(file[:,3]), max(file[:,3]), k)
cntrds_5= np.random.uniform(min(file[:,4]), max(file[:,4]), k)
cntrds = list(zip(cntrds_1, cntrds_2, cntrds_3, cntrds_4, cntrds_5))

df_cntrds = pd.DataFrame(cntrds)
cntrds = df_cntrds.to_numpy()

### 2) iteration ###
for iter in range(iter_max+1):
### 2-1) Partitioning the 5-vectors into k groups ###
    labels = np.zeros(file.shape[0])
    for i in range(file.shape[0]):
        distances = np.zeros(k)
        for n in range(k):
            distances[n] = distance(file[i], cntrds[n])
            cluster = np.argmin(distances) 
            labels[i] = int(cluster)
### 2-2) Update representatives ###
    for n in range(k):
        points= [file[i] for i in range(file.shape[0]) if n==labels[i]]
        df_points = pd.DataFrame(points)
        if df_points.empty==True: 
            pass
        else:
            points = df_points.to_numpy() 
            cntrds[n] = np.mean(points, axis=0)

### Outputs ###             
#[output1] the centroids of each cluster
print("# of actual iterations : %d"%iter_max)
print("centroids : ", end='')
for n in range(k-1):
    print(tuple(cntrds[n]), end=",")
print(tuple(cntrds[k-1]))

#[output2] # of vectors in each cluster
for n in range(k):
    points= [file[i] for i in range(file.shape[0]) if n==labels[i]]
    df_points = pd.DataFrame(points)
    if df_points.empty==True:
        print("# of vectors for cluster %d = "%(n+1), 0)
    else:
        points = df_points.to_numpy()
        print("# of vectors for cluster %d = "%(n+1), len(points)) 
