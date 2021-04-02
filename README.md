# K-means Clustering Algorithm of 5-vectors

이미 작성되어 있는 k-means clustering 관련 library함수를 쓰지 않고
*python* 으로 k-means clustering algorithm을 직접 구현했습니다.

#### K-means Clustering Algorithm of 5-vectors은 다음과 같은 과정을 수행합니다.

5-vectors가 나열되어 있는 데이터, clustering할 cluster의 갯수(k), iteration횟수를 입력받습니다.

1. 데이터값을 바탕으로 임의의 k개의 group representative vectors, 즉 centroids를 지정합니다.
2. 입력받은 iteration횟수만큼 다음 과정을 반복합니다.
   * 각 데이터를 그와 가장 가까운 centriod가 속한 cluster에 할당합니다.
   * 각 cluster별 centroid를 할당된 데이터들의 평균으로 update합니다.
 
centriods와 각 cluster별로 속한 데이터의 갯수(# of vectors in each cluster)를 출력합니다.


## Getting Started
### Prerequisites / 선행 조건

```python
pip install pandas
pip install numpy
```

### Installing / 설치

```python
import pandas as pd
import numpy as np
```

## Running the tests / 프로그램 동작

### 1. import file

>#### [input1] file of 5-vectors
>* 첫번째 input은 5-vector들이 나열되어 있는 파일입니다.
>* 파일을 읽기 오기 위해 해당 파일의 주소를 입력받습니다.

```python
fileAdr = input("input할 파일의 주소를 입력하시오 : ")
file = pd.read_csv(fileAdr, encoding="utf-8", sep=' ', header=None)
```

### 2. Data Preprocessing
* 5-vectors외의 값은 모두 삭제합니다.
* dataframe인 file을 numpy.ndarray로 바꿔줍니다.

```python
file = file[[0, 1, 2, 3, 4]]
file = file.to_numpy()
```

### 3. K-means Clustering Algorithm

>#### [input2] cluster갯수(k)
>* 두번째 input으로 몇 개의 cluster로 clustering할 지 입력받습니다.

```python
k = int(input("몇 개의 cluster로 clustering할 건가요? : "))
```

>#### [input3] the maximum number of iterations
>* 세번째 input으로 몇 번의 iteration을 수행할 것인지 입력받습니다.

```python
iter_max = int(input("몇 번 iteration(반복)할 건가요? : "))
```

#### distance()
* 5차원에서 두 점(데이터) a,b 사이의 거리, 즉 L5-Norm을 구하는 함수입니다.

```python
def distance(a, b):
  return sum([abs(a_col - b_col)**5  for a_col, b_col in list(zip(a, b))]) ** 0.2
```  

### 1) initial list of k group representative vectors (cntrds[0],..,cntrds[k-1])
* 데이터값을 바탕으로 k개의 임의의 group representative vectors, 즉 centroids를 지정합니다.
* centroid는 (cntrds_1, cntrds_2, cntrds_3, cntrds_4, cntrds_5)로 이루어져있습니다.

  + 변수 cntrds_?은 각 데이터(5-vector)의 ?번째 속성(element)에 대한 값들의 최소~최대 범위에서 정한 k개의 랜덤한 수들을 의미합니다.

```python
cntrds_1= np.random.uniform(min(file[:,0]), max(file[:,0]), k)
cntrds_2= np.random.uniform(min(file[:,1]), max(file[:,1]), k)
cntrds_3= np.random.uniform(min(file[:,2]), max(file[:,2]), k)
cntrds_4= np.random.uniform(min(file[:,3]), max(file[:,3]), k)
cntrds_5= np.random.uniform(min(file[:,4]), max(file[:,4]), k)
cntrds = list(zip(cntrds_1, cntrds_2, cntrds_3, cntrds_4, cntrds_5))

df_cntrds = pd.DataFrame(cntrds)
cntrds = df_cntrds.to_numpy() #dataframe to numpy.ndarray
```

### 2) iteration
* _2-1) Partitioning the 5-vectors into k groups_ ~ _2-2) Update representatives_ 과정을 입력받은 iteration횟수만큼 반복해서 수행합니다.

```python
for iter in range(iter_max+1):
  ...
```

### 2-1) Partitioning the 5-vectors into k groups
* 각 데이터(5-vector)를 그와 가장 가까운 centriod가 속한 cluster에 할당합니다.

  + 변수 labels에 각 데이터가 속한 cluster의 index를 저장하기 위해서 for루프를 수행합니다.
    - 변수 distances는 해당 데이터로부터 각 cluster의 centroid의 거리를 element로 가짐
    - 변수 cluster는 distances의 element 중 가장 작은 값의 index를 저장함
    - 즉, 변수 cluster는 각 데이터와 가장 가까운 centriod가 속한 cluster의 index를 저장함

```python
labels = np.zeros(file.shape[0])
for i in range(file.shape[0]):
    distances = np.zeros(k)
    for n in range(k):
       distances[n] = distance(file[i], cntrds[n])
       cluster = np.argmin(distances) 
       #labal[i]에는 0~k-1 중 하나가 들어가게 됨
       labels[i] = int(cluster)
```

### 2-2) Update representatives
* 각 cluster별 centroid를, 해당 cluster에 할당된 데이터들의 평균(mean of 5-vectors in each cluster)으로 update합니다.

  + 각 cluster에 속한 데이터(5-vectors in each cluster)을 알기위해서 for루프를 수행합니다.
    - 각 centroid의 index와 각 데이터가 속한 cluster의 index(변수 cluster)가 일치하는 데이터(5-vectors)를 추출함
  + 기존 centroid를 해당 cluster에 할당된 데이터들의 평균(mean of 5-vectors in each cluster)으로 변경합니다.
    - 단, 해당 cluster에 속하는 데이터가 아예 없을 경우 pass
    
```python
for n in range(k):
    points= [file[i] for i in range(file.shape[0]) if n==labels[i]] 
    df_points = pd.DataFrame(points)
    if df_points.empty==True: 
        pass
    else:
        points = df_points.to_numpy()
        cntrds[n] = np.mean(points, axis=0)
```

### 4. Outputs

>#### [output1] the centroids of each cluster
>* iteration한 횟수를 출력합니다.
>* iteration한 최종, 각 cluster에 속한 centroid를 모두 출력합니다.

```python
print("# of actual iterations : %d"%iter_max)
print("centroids : ", end='')
for n in range(k-1):
    print(tuple(cntrds[n]), end=",")
print(tuple(cntrds[k-1]))
```

>#### [output2] # of vectors in each cluster
>* 각 cluster별로 속한 데이터의 갯수를 출력합니다.
>
>    + 각 cluster에 속한 데이터(5-vectors in each cluster)을 알기위해서 for루프를 수행합니다.
>      - 각 centroid의 index와 각 데이터가 속한 cluster의 index(변수 cluster)가 일치하는 데이터(5-vectors)를 추출함
>    + 이를 통해 구한, 각 cluster별로 속한 데이터의 갯수를 출력합니다.
>      - 만약 해당 cluster에 데이터가 없다면 0개라고 출력

```python
for n in range(k):
    points= [file[i] for i in range(file.shape[0]) if n==labels[i]] 
    df_points = pd.DataFrame(points)
    if df_points.empty==True:
        print("# of vectors for cluster %d = "%(n+1), 0)
    else:
        points = df_points.to_numpy() 
        print("# of vectors for cluster %d = "%(n+1), len(points))
```
