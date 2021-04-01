# k-means clustering algorithm 
 프로젝트의 전반적인 내용에 대한 요약

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
#### [input1] file of 5-vectors
첫번째 input은 5-vectors에 대한 데이터가 있는 파일입니다.
파일을 읽기 오기 위해 사용자는 해당 파일의 주소를 입력합니다.

```python
fileAdr = input("input할 파일의 주소를 입력하시오 : ")
file = pd.read_csv(fileAdr, encoding="utf-8", sep=' ', header=None)
```

### 2. Data Preprocessing
5-vectors외의 값은 모두 삭제합니다.
dataframe인 file을 numpy.ndarray로 바꿔줍니다.

```python
file = file[[0, 1, 2, 3, 4]]
file = file.to_numpy()
```

### 3. Data Preprocessing
#### [input2] cluster갯수(k)
#### [input3] the maximum number of iterations
```python
k = int(input("몇 개의 cluster로 clustering할 건가요? : "))
iter_max = int(input("몇 번 iteration(반복)할 건가요? : "))
```
