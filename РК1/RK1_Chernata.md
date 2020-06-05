```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline 
sns.set(style="ticks")
```

Загрузка данных


```python
data = pd.read_csv('MMO/fifa19/data.csv', sep=",")
```


```python
data.shape
```




    (18207, 89)




```python
data.dtypes
```




    Unnamed: 0          int64
    ID                  int64
    Name               object
    Age                 int64
    Photo              object
                       ...   
    GKHandling        float64
    GKKicking         float64
    GKPositioning     float64
    GKReflexes        float64
    Release Clause     object
    Length: 89, dtype: object




```python
data.isnull().sum()
```




    Unnamed: 0           0
    ID                   0
    Name                 0
    Age                  0
    Photo                0
                      ... 
    GKHandling          48
    GKKicking           48
    GKPositioning       48
    GKReflexes          48
    Release Clause    1564
    Length: 89, dtype: int64




```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>ID</th>
      <th>Name</th>
      <th>Age</th>
      <th>Photo</th>
      <th>Nationality</th>
      <th>Flag</th>
      <th>Overall</th>
      <th>Potential</th>
      <th>Club</th>
      <th>...</th>
      <th>Composure</th>
      <th>Marking</th>
      <th>StandingTackle</th>
      <th>SlidingTackle</th>
      <th>GKDiving</th>
      <th>GKHandling</th>
      <th>GKKicking</th>
      <th>GKPositioning</th>
      <th>GKReflexes</th>
      <th>Release Clause</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>158023</td>
      <td>L. Messi</td>
      <td>31</td>
      <td>https://cdn.sofifa.org/players/4/19/158023.png</td>
      <td>Argentina</td>
      <td>https://cdn.sofifa.org/flags/52.png</td>
      <td>94</td>
      <td>94</td>
      <td>FC Barcelona</td>
      <td>...</td>
      <td>96.0</td>
      <td>33.0</td>
      <td>28.0</td>
      <td>26.0</td>
      <td>6.0</td>
      <td>11.0</td>
      <td>15.0</td>
      <td>14.0</td>
      <td>8.0</td>
      <td>€226.5M</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>20801</td>
      <td>Cristiano Ronaldo</td>
      <td>33</td>
      <td>https://cdn.sofifa.org/players/4/19/20801.png</td>
      <td>Portugal</td>
      <td>https://cdn.sofifa.org/flags/38.png</td>
      <td>94</td>
      <td>94</td>
      <td>Juventus</td>
      <td>...</td>
      <td>95.0</td>
      <td>28.0</td>
      <td>31.0</td>
      <td>23.0</td>
      <td>7.0</td>
      <td>11.0</td>
      <td>15.0</td>
      <td>14.0</td>
      <td>11.0</td>
      <td>€127.1M</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>190871</td>
      <td>Neymar Jr</td>
      <td>26</td>
      <td>https://cdn.sofifa.org/players/4/19/190871.png</td>
      <td>Brazil</td>
      <td>https://cdn.sofifa.org/flags/54.png</td>
      <td>92</td>
      <td>93</td>
      <td>Paris Saint-Germain</td>
      <td>...</td>
      <td>94.0</td>
      <td>27.0</td>
      <td>24.0</td>
      <td>33.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>15.0</td>
      <td>15.0</td>
      <td>11.0</td>
      <td>€228.1M</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>193080</td>
      <td>De Gea</td>
      <td>27</td>
      <td>https://cdn.sofifa.org/players/4/19/193080.png</td>
      <td>Spain</td>
      <td>https://cdn.sofifa.org/flags/45.png</td>
      <td>91</td>
      <td>93</td>
      <td>Manchester United</td>
      <td>...</td>
      <td>68.0</td>
      <td>15.0</td>
      <td>21.0</td>
      <td>13.0</td>
      <td>90.0</td>
      <td>85.0</td>
      <td>87.0</td>
      <td>88.0</td>
      <td>94.0</td>
      <td>€138.6M</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>192985</td>
      <td>K. De Bruyne</td>
      <td>27</td>
      <td>https://cdn.sofifa.org/players/4/19/192985.png</td>
      <td>Belgium</td>
      <td>https://cdn.sofifa.org/flags/7.png</td>
      <td>91</td>
      <td>92</td>
      <td>Manchester City</td>
      <td>...</td>
      <td>88.0</td>
      <td>68.0</td>
      <td>58.0</td>
      <td>51.0</td>
      <td>15.0</td>
      <td>13.0</td>
      <td>5.0</td>
      <td>10.0</td>
      <td>13.0</td>
      <td>€196.4M</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 89 columns</p>
</div>




```python
total_count = data.shape[0]
print('Всего строк: {}'.format(total_count))
```

    Всего строк: 18207
    

## Обработка пропусков в данных
### Удаление или заполнение нулями


```python
data_new_1 = data.dropna(axis=1, how='any')
(data.shape, data_new_1.shape)
```




    ((18207, 89), (18207, 13))




```python
data_new_2 = data.dropna(axis=0, how='any')
(data.shape, data_new_2.shape)
```




    ((18207, 89), (0, 89))




```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>ID</th>
      <th>Name</th>
      <th>Age</th>
      <th>Photo</th>
      <th>Nationality</th>
      <th>Flag</th>
      <th>Overall</th>
      <th>Potential</th>
      <th>Club</th>
      <th>...</th>
      <th>Composure</th>
      <th>Marking</th>
      <th>StandingTackle</th>
      <th>SlidingTackle</th>
      <th>GKDiving</th>
      <th>GKHandling</th>
      <th>GKKicking</th>
      <th>GKPositioning</th>
      <th>GKReflexes</th>
      <th>Release Clause</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>158023</td>
      <td>L. Messi</td>
      <td>31</td>
      <td>https://cdn.sofifa.org/players/4/19/158023.png</td>
      <td>Argentina</td>
      <td>https://cdn.sofifa.org/flags/52.png</td>
      <td>94</td>
      <td>94</td>
      <td>FC Barcelona</td>
      <td>...</td>
      <td>96.0</td>
      <td>33.0</td>
      <td>28.0</td>
      <td>26.0</td>
      <td>6.0</td>
      <td>11.0</td>
      <td>15.0</td>
      <td>14.0</td>
      <td>8.0</td>
      <td>€226.5M</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>20801</td>
      <td>Cristiano Ronaldo</td>
      <td>33</td>
      <td>https://cdn.sofifa.org/players/4/19/20801.png</td>
      <td>Portugal</td>
      <td>https://cdn.sofifa.org/flags/38.png</td>
      <td>94</td>
      <td>94</td>
      <td>Juventus</td>
      <td>...</td>
      <td>95.0</td>
      <td>28.0</td>
      <td>31.0</td>
      <td>23.0</td>
      <td>7.0</td>
      <td>11.0</td>
      <td>15.0</td>
      <td>14.0</td>
      <td>11.0</td>
      <td>€127.1M</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>190871</td>
      <td>Neymar Jr</td>
      <td>26</td>
      <td>https://cdn.sofifa.org/players/4/19/190871.png</td>
      <td>Brazil</td>
      <td>https://cdn.sofifa.org/flags/54.png</td>
      <td>92</td>
      <td>93</td>
      <td>Paris Saint-Germain</td>
      <td>...</td>
      <td>94.0</td>
      <td>27.0</td>
      <td>24.0</td>
      <td>33.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>15.0</td>
      <td>15.0</td>
      <td>11.0</td>
      <td>€228.1M</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>193080</td>
      <td>De Gea</td>
      <td>27</td>
      <td>https://cdn.sofifa.org/players/4/19/193080.png</td>
      <td>Spain</td>
      <td>https://cdn.sofifa.org/flags/45.png</td>
      <td>91</td>
      <td>93</td>
      <td>Manchester United</td>
      <td>...</td>
      <td>68.0</td>
      <td>15.0</td>
      <td>21.0</td>
      <td>13.0</td>
      <td>90.0</td>
      <td>85.0</td>
      <td>87.0</td>
      <td>88.0</td>
      <td>94.0</td>
      <td>€138.6M</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>192985</td>
      <td>K. De Bruyne</td>
      <td>27</td>
      <td>https://cdn.sofifa.org/players/4/19/192985.png</td>
      <td>Belgium</td>
      <td>https://cdn.sofifa.org/flags/7.png</td>
      <td>91</td>
      <td>92</td>
      <td>Manchester City</td>
      <td>...</td>
      <td>88.0</td>
      <td>68.0</td>
      <td>58.0</td>
      <td>51.0</td>
      <td>15.0</td>
      <td>13.0</td>
      <td>5.0</td>
      <td>10.0</td>
      <td>13.0</td>
      <td>€196.4M</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 89 columns</p>
</div>



### Импьютация 
#### Обработка пропусков в числовых данных


```python
data_new_3 = data.fillna(0)
data_new_3.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>ID</th>
      <th>Name</th>
      <th>Age</th>
      <th>Photo</th>
      <th>Nationality</th>
      <th>Flag</th>
      <th>Overall</th>
      <th>Potential</th>
      <th>Club</th>
      <th>...</th>
      <th>Composure</th>
      <th>Marking</th>
      <th>StandingTackle</th>
      <th>SlidingTackle</th>
      <th>GKDiving</th>
      <th>GKHandling</th>
      <th>GKKicking</th>
      <th>GKPositioning</th>
      <th>GKReflexes</th>
      <th>Release Clause</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>158023</td>
      <td>L. Messi</td>
      <td>31</td>
      <td>https://cdn.sofifa.org/players/4/19/158023.png</td>
      <td>Argentina</td>
      <td>https://cdn.sofifa.org/flags/52.png</td>
      <td>94</td>
      <td>94</td>
      <td>FC Barcelona</td>
      <td>...</td>
      <td>96.0</td>
      <td>33.0</td>
      <td>28.0</td>
      <td>26.0</td>
      <td>6.0</td>
      <td>11.0</td>
      <td>15.0</td>
      <td>14.0</td>
      <td>8.0</td>
      <td>€226.5M</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>20801</td>
      <td>Cristiano Ronaldo</td>
      <td>33</td>
      <td>https://cdn.sofifa.org/players/4/19/20801.png</td>
      <td>Portugal</td>
      <td>https://cdn.sofifa.org/flags/38.png</td>
      <td>94</td>
      <td>94</td>
      <td>Juventus</td>
      <td>...</td>
      <td>95.0</td>
      <td>28.0</td>
      <td>31.0</td>
      <td>23.0</td>
      <td>7.0</td>
      <td>11.0</td>
      <td>15.0</td>
      <td>14.0</td>
      <td>11.0</td>
      <td>€127.1M</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>190871</td>
      <td>Neymar Jr</td>
      <td>26</td>
      <td>https://cdn.sofifa.org/players/4/19/190871.png</td>
      <td>Brazil</td>
      <td>https://cdn.sofifa.org/flags/54.png</td>
      <td>92</td>
      <td>93</td>
      <td>Paris Saint-Germain</td>
      <td>...</td>
      <td>94.0</td>
      <td>27.0</td>
      <td>24.0</td>
      <td>33.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>15.0</td>
      <td>15.0</td>
      <td>11.0</td>
      <td>€228.1M</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>193080</td>
      <td>De Gea</td>
      <td>27</td>
      <td>https://cdn.sofifa.org/players/4/19/193080.png</td>
      <td>Spain</td>
      <td>https://cdn.sofifa.org/flags/45.png</td>
      <td>91</td>
      <td>93</td>
      <td>Manchester United</td>
      <td>...</td>
      <td>68.0</td>
      <td>15.0</td>
      <td>21.0</td>
      <td>13.0</td>
      <td>90.0</td>
      <td>85.0</td>
      <td>87.0</td>
      <td>88.0</td>
      <td>94.0</td>
      <td>€138.6M</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>192985</td>
      <td>K. De Bruyne</td>
      <td>27</td>
      <td>https://cdn.sofifa.org/players/4/19/192985.png</td>
      <td>Belgium</td>
      <td>https://cdn.sofifa.org/flags/7.png</td>
      <td>91</td>
      <td>92</td>
      <td>Manchester City</td>
      <td>...</td>
      <td>88.0</td>
      <td>68.0</td>
      <td>58.0</td>
      <td>51.0</td>
      <td>15.0</td>
      <td>13.0</td>
      <td>5.0</td>
      <td>10.0</td>
      <td>13.0</td>
      <td>€196.4M</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 89 columns</p>
</div>




```python
num_cols = []
for col in data.columns:
    # Количество пустых значений 
    temp_null_count = data[data[col].isnull()].shape[0]
    dt = str(data[col].dtype)
    if temp_null_count>0 and (dt=='float64' or dt=='int64'):
        num_cols.append(col)
        temp_perc = round((temp_null_count / total_count) * 100.0, 2)
        print('Колонка {}. Тип данных {}. Количество пустых значений {}, {}%.'.format(col, dt, temp_null_count, temp_perc))
```

    Колонка International Reputation. Тип данных float64. Количество пустых значений 48, 0.26%.
    Колонка Weak Foot. Тип данных float64. Количество пустых значений 48, 0.26%.
    Колонка Skill Moves. Тип данных float64. Количество пустых значений 48, 0.26%.
    Колонка Jersey Number. Тип данных float64. Количество пустых значений 60, 0.33%.
    Колонка Crossing. Тип данных float64. Количество пустых значений 48, 0.26%.
    Колонка Finishing. Тип данных float64. Количество пустых значений 48, 0.26%.
    Колонка HeadingAccuracy. Тип данных float64. Количество пустых значений 48, 0.26%.
    Колонка ShortPassing. Тип данных float64. Количество пустых значений 48, 0.26%.
    Колонка Volleys. Тип данных float64. Количество пустых значений 48, 0.26%.
    Колонка Dribbling. Тип данных float64. Количество пустых значений 48, 0.26%.
    Колонка Curve. Тип данных float64. Количество пустых значений 48, 0.26%.
    Колонка FKAccuracy. Тип данных float64. Количество пустых значений 48, 0.26%.
    Колонка LongPassing. Тип данных float64. Количество пустых значений 48, 0.26%.
    Колонка BallControl. Тип данных float64. Количество пустых значений 48, 0.26%.
    Колонка Acceleration. Тип данных float64. Количество пустых значений 48, 0.26%.
    Колонка SprintSpeed. Тип данных float64. Количество пустых значений 48, 0.26%.
    Колонка Agility. Тип данных float64. Количество пустых значений 48, 0.26%.
    Колонка Reactions. Тип данных float64. Количество пустых значений 48, 0.26%.
    Колонка Balance. Тип данных float64. Количество пустых значений 48, 0.26%.
    Колонка ShotPower. Тип данных float64. Количество пустых значений 48, 0.26%.
    Колонка Jumping. Тип данных float64. Количество пустых значений 48, 0.26%.
    Колонка Stamina. Тип данных float64. Количество пустых значений 48, 0.26%.
    Колонка Strength. Тип данных float64. Количество пустых значений 48, 0.26%.
    Колонка LongShots. Тип данных float64. Количество пустых значений 48, 0.26%.
    Колонка Aggression. Тип данных float64. Количество пустых значений 48, 0.26%.
    Колонка Interceptions. Тип данных float64. Количество пустых значений 48, 0.26%.
    Колонка Positioning. Тип данных float64. Количество пустых значений 48, 0.26%.
    Колонка Vision. Тип данных float64. Количество пустых значений 48, 0.26%.
    Колонка Penalties. Тип данных float64. Количество пустых значений 48, 0.26%.
    Колонка Composure. Тип данных float64. Количество пустых значений 48, 0.26%.
    Колонка Marking. Тип данных float64. Количество пустых значений 48, 0.26%.
    Колонка StandingTackle. Тип данных float64. Количество пустых значений 48, 0.26%.
    Колонка SlidingTackle. Тип данных float64. Количество пустых значений 48, 0.26%.
    Колонка GKDiving. Тип данных float64. Количество пустых значений 48, 0.26%.
    Колонка GKHandling. Тип данных float64. Количество пустых значений 48, 0.26%.
    Колонка GKKicking. Тип данных float64. Количество пустых значений 48, 0.26%.
    Колонка GKPositioning. Тип данных float64. Количество пустых значений 48, 0.26%.
    Колонка GKReflexes. Тип данных float64. Количество пустых значений 48, 0.26%.
    


```python
data_num = data[num_cols]
data_num
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>International Reputation</th>
      <th>Weak Foot</th>
      <th>Skill Moves</th>
      <th>Jersey Number</th>
      <th>Crossing</th>
      <th>Finishing</th>
      <th>HeadingAccuracy</th>
      <th>ShortPassing</th>
      <th>Volleys</th>
      <th>Dribbling</th>
      <th>...</th>
      <th>Penalties</th>
      <th>Composure</th>
      <th>Marking</th>
      <th>StandingTackle</th>
      <th>SlidingTackle</th>
      <th>GKDiving</th>
      <th>GKHandling</th>
      <th>GKKicking</th>
      <th>GKPositioning</th>
      <th>GKReflexes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>10.0</td>
      <td>84.0</td>
      <td>95.0</td>
      <td>70.0</td>
      <td>90.0</td>
      <td>86.0</td>
      <td>97.0</td>
      <td>...</td>
      <td>75.0</td>
      <td>96.0</td>
      <td>33.0</td>
      <td>28.0</td>
      <td>26.0</td>
      <td>6.0</td>
      <td>11.0</td>
      <td>15.0</td>
      <td>14.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>7.0</td>
      <td>84.0</td>
      <td>94.0</td>
      <td>89.0</td>
      <td>81.0</td>
      <td>87.0</td>
      <td>88.0</td>
      <td>...</td>
      <td>85.0</td>
      <td>95.0</td>
      <td>28.0</td>
      <td>31.0</td>
      <td>23.0</td>
      <td>7.0</td>
      <td>11.0</td>
      <td>15.0</td>
      <td>14.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>10.0</td>
      <td>79.0</td>
      <td>87.0</td>
      <td>62.0</td>
      <td>84.0</td>
      <td>84.0</td>
      <td>96.0</td>
      <td>...</td>
      <td>81.0</td>
      <td>94.0</td>
      <td>27.0</td>
      <td>24.0</td>
      <td>33.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>15.0</td>
      <td>15.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>17.0</td>
      <td>13.0</td>
      <td>21.0</td>
      <td>50.0</td>
      <td>13.0</td>
      <td>18.0</td>
      <td>...</td>
      <td>40.0</td>
      <td>68.0</td>
      <td>15.0</td>
      <td>21.0</td>
      <td>13.0</td>
      <td>90.0</td>
      <td>85.0</td>
      <td>87.0</td>
      <td>88.0</td>
      <td>94.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>7.0</td>
      <td>93.0</td>
      <td>82.0</td>
      <td>55.0</td>
      <td>92.0</td>
      <td>82.0</td>
      <td>86.0</td>
      <td>...</td>
      <td>79.0</td>
      <td>88.0</td>
      <td>68.0</td>
      <td>58.0</td>
      <td>51.0</td>
      <td>15.0</td>
      <td>13.0</td>
      <td>5.0</td>
      <td>10.0</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>18202</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>22.0</td>
      <td>34.0</td>
      <td>38.0</td>
      <td>40.0</td>
      <td>49.0</td>
      <td>25.0</td>
      <td>42.0</td>
      <td>...</td>
      <td>43.0</td>
      <td>45.0</td>
      <td>40.0</td>
      <td>48.0</td>
      <td>47.0</td>
      <td>10.0</td>
      <td>13.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>18203</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>21.0</td>
      <td>23.0</td>
      <td>52.0</td>
      <td>52.0</td>
      <td>43.0</td>
      <td>36.0</td>
      <td>39.0</td>
      <td>...</td>
      <td>43.0</td>
      <td>42.0</td>
      <td>22.0</td>
      <td>15.0</td>
      <td>19.0</td>
      <td>10.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>5.0</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>18204</th>
      <td>1.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>25.0</td>
      <td>40.0</td>
      <td>46.0</td>
      <td>38.0</td>
      <td>38.0</td>
      <td>45.0</td>
      <td>...</td>
      <td>55.0</td>
      <td>41.0</td>
      <td>32.0</td>
      <td>13.0</td>
      <td>11.0</td>
      <td>6.0</td>
      <td>5.0</td>
      <td>10.0</td>
      <td>6.0</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>18205</th>
      <td>1.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>34.0</td>
      <td>44.0</td>
      <td>50.0</td>
      <td>39.0</td>
      <td>42.0</td>
      <td>40.0</td>
      <td>51.0</td>
      <td>...</td>
      <td>50.0</td>
      <td>46.0</td>
      <td>20.0</td>
      <td>25.0</td>
      <td>27.0</td>
      <td>14.0</td>
      <td>6.0</td>
      <td>14.0</td>
      <td>8.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>18206</th>
      <td>1.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>41.0</td>
      <td>34.0</td>
      <td>46.0</td>
      <td>48.0</td>
      <td>30.0</td>
      <td>43.0</td>
      <td>...</td>
      <td>33.0</td>
      <td>43.0</td>
      <td>40.0</td>
      <td>43.0</td>
      <td>50.0</td>
      <td>10.0</td>
      <td>15.0</td>
      <td>9.0</td>
      <td>12.0</td>
      <td>9.0</td>
    </tr>
  </tbody>
</table>
<p>18207 rows × 38 columns</p>
</div>




```python
for col in data_num:
    plt.hist(data[col], 50)
    plt.xlabel(col)
    plt.show()
```

    c:\users\ncher\appdata\local\programs\python\python37\lib\site-packages\numpy\lib\histograms.py:839: RuntimeWarning: invalid value encountered in greater_equal
      keep = (tmp_a >= first_edge)
    c:\users\ncher\appdata\local\programs\python\python37\lib\site-packages\numpy\lib\histograms.py:840: RuntimeWarning: invalid value encountered in less_equal
      keep &= (tmp_a <= last_edge)
    


![png](output_16_1.png)



![png](output_16_2.png)



![png](output_16_3.png)



![png](output_16_4.png)



![png](output_16_5.png)



![png](output_16_6.png)



![png](output_16_7.png)



![png](output_16_8.png)



![png](output_16_9.png)



![png](output_16_10.png)



![png](output_16_11.png)



![png](output_16_12.png)



![png](output_16_13.png)



![png](output_16_14.png)



![png](output_16_15.png)



![png](output_16_16.png)



![png](output_16_17.png)



![png](output_16_18.png)



![png](output_16_19.png)



![png](output_16_20.png)



![png](output_16_21.png)



![png](output_16_22.png)



![png](output_16_23.png)



![png](output_16_24.png)



![png](output_16_25.png)



![png](output_16_26.png)



![png](output_16_27.png)



![png](output_16_28.png)



![png](output_16_29.png)



![png](output_16_30.png)



![png](output_16_31.png)



![png](output_16_32.png)



![png](output_16_33.png)



![png](output_16_34.png)



![png](output_16_35.png)



![png](output_16_36.png)



![png](output_16_37.png)



![png](output_16_38.png)



```python
data[data['Jersey Number'].isnull()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>ID</th>
      <th>Name</th>
      <th>Age</th>
      <th>Photo</th>
      <th>Nationality</th>
      <th>Flag</th>
      <th>Overall</th>
      <th>Potential</th>
      <th>Club</th>
      <th>...</th>
      <th>Composure</th>
      <th>Marking</th>
      <th>StandingTackle</th>
      <th>SlidingTackle</th>
      <th>GKDiving</th>
      <th>GKHandling</th>
      <th>GKKicking</th>
      <th>GKPositioning</th>
      <th>GKReflexes</th>
      <th>Release Clause</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5018</th>
      <td>5018</td>
      <td>153160</td>
      <td>R. Raldes</td>
      <td>37</td>
      <td>https://cdn.sofifa.org/players/4/19/153160.png</td>
      <td>Bolivia</td>
      <td>https://cdn.sofifa.org/flags/53.png</td>
      <td>70</td>
      <td>70</td>
      <td>NaN</td>
      <td>...</td>
      <td>64.0</td>
      <td>79.0</td>
      <td>70.0</td>
      <td>70.0</td>
      <td>16.0</td>
      <td>8.0</td>
      <td>11.0</td>
      <td>12.0</td>
      <td>13.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6736</th>
      <td>6736</td>
      <td>175393</td>
      <td>J. Arce</td>
      <td>33</td>
      <td>https://cdn.sofifa.org/players/4/19/175393.png</td>
      <td>Bolivia</td>
      <td>https://cdn.sofifa.org/flags/53.png</td>
      <td>68</td>
      <td>68</td>
      <td>NaN</td>
      <td>...</td>
      <td>67.0</td>
      <td>12.0</td>
      <td>34.0</td>
      <td>33.0</td>
      <td>16.0</td>
      <td>13.0</td>
      <td>15.0</td>
      <td>16.0</td>
      <td>8.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7922</th>
      <td>7922</td>
      <td>195905</td>
      <td>L. Gutiérrez</td>
      <td>33</td>
      <td>https://cdn.sofifa.org/players/4/19/195905.png</td>
      <td>Bolivia</td>
      <td>https://cdn.sofifa.org/flags/53.png</td>
      <td>67</td>
      <td>67</td>
      <td>NaN</td>
      <td>...</td>
      <td>54.0</td>
      <td>72.0</td>
      <td>71.0</td>
      <td>64.0</td>
      <td>11.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>12.0</td>
      <td>12.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9905</th>
      <td>9905</td>
      <td>226044</td>
      <td>R. Vargas</td>
      <td>23</td>
      <td>https://cdn.sofifa.org/players/4/19/226044.png</td>
      <td>Bolivia</td>
      <td>https://cdn.sofifa.org/flags/53.png</td>
      <td>66</td>
      <td>69</td>
      <td>NaN</td>
      <td>...</td>
      <td>64.0</td>
      <td>19.0</td>
      <td>24.0</td>
      <td>23.0</td>
      <td>13.0</td>
      <td>11.0</td>
      <td>11.0</td>
      <td>7.0</td>
      <td>9.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10628</th>
      <td>10628</td>
      <td>216751</td>
      <td>D. Bejarano</td>
      <td>26</td>
      <td>https://cdn.sofifa.org/players/4/19/216751.png</td>
      <td>Bolivia</td>
      <td>https://cdn.sofifa.org/flags/53.png</td>
      <td>65</td>
      <td>66</td>
      <td>NaN</td>
      <td>...</td>
      <td>57.0</td>
      <td>68.0</td>
      <td>69.0</td>
      <td>68.0</td>
      <td>14.0</td>
      <td>12.0</td>
      <td>11.0</td>
      <td>14.0</td>
      <td>12.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13236</th>
      <td>13236</td>
      <td>177971</td>
      <td>J. McNulty</td>
      <td>33</td>
      <td>https://cdn.sofifa.org/players/4/19/177971.png</td>
      <td>Scotland</td>
      <td>https://cdn.sofifa.org/flags/42.png</td>
      <td>62</td>
      <td>62</td>
      <td>Rochdale</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13237</th>
      <td>13237</td>
      <td>195380</td>
      <td>J. Barrera</td>
      <td>29</td>
      <td>https://cdn.sofifa.org/players/4/19/195380.png</td>
      <td>Nicaragua</td>
      <td>https://cdn.sofifa.org/flags/86.png</td>
      <td>62</td>
      <td>62</td>
      <td>Boyacá Chicó FC</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13238</th>
      <td>13238</td>
      <td>139317</td>
      <td>J. Stead</td>
      <td>35</td>
      <td>https://cdn.sofifa.org/players/4/19/139317.png</td>
      <td>England</td>
      <td>https://cdn.sofifa.org/flags/14.png</td>
      <td>62</td>
      <td>62</td>
      <td>Notts County</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13239</th>
      <td>13239</td>
      <td>240437</td>
      <td>A. Semprini</td>
      <td>20</td>
      <td>https://cdn.sofifa.org/players/4/19/240437.png</td>
      <td>Italy</td>
      <td>https://cdn.sofifa.org/flags/27.png</td>
      <td>62</td>
      <td>72</td>
      <td>Brescia</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13240</th>
      <td>13240</td>
      <td>209462</td>
      <td>R. Bingham</td>
      <td>24</td>
      <td>https://cdn.sofifa.org/players/4/19/209462.png</td>
      <td>England</td>
      <td>https://cdn.sofifa.org/flags/14.png</td>
      <td>62</td>
      <td>66</td>
      <td>Hamilton Academical FC</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13241</th>
      <td>13241</td>
      <td>219702</td>
      <td>K. Dankowski</td>
      <td>21</td>
      <td>https://cdn.sofifa.org/players/4/19/219702.png</td>
      <td>Poland</td>
      <td>https://cdn.sofifa.org/flags/37.png</td>
      <td>62</td>
      <td>72</td>
      <td>Śląsk Wrocław</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13242</th>
      <td>13242</td>
      <td>225590</td>
      <td>I. Colman</td>
      <td>23</td>
      <td>https://cdn.sofifa.org/players/4/19/225590.png</td>
      <td>Argentina</td>
      <td>https://cdn.sofifa.org/flags/52.png</td>
      <td>62</td>
      <td>70</td>
      <td>Club Atlético Aldosivi</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13243</th>
      <td>13243</td>
      <td>233782</td>
      <td>M. Feeney</td>
      <td>19</td>
      <td>https://cdn.sofifa.org/players/4/19/233782.png</td>
      <td>England</td>
      <td>https://cdn.sofifa.org/flags/14.png</td>
      <td>62</td>
      <td>78</td>
      <td>Everton</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13244</th>
      <td>13244</td>
      <td>239158</td>
      <td>R. Minor</td>
      <td>30</td>
      <td>https://cdn.sofifa.org/players/4/19/239158.png</td>
      <td>Denmark</td>
      <td>https://cdn.sofifa.org/flags/13.png</td>
      <td>62</td>
      <td>62</td>
      <td>Hobro IK</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13245</th>
      <td>13245</td>
      <td>242998</td>
      <td>Klauss</td>
      <td>21</td>
      <td>https://cdn.sofifa.org/players/4/19/242998.png</td>
      <td>Brazil</td>
      <td>https://cdn.sofifa.org/flags/54.png</td>
      <td>62</td>
      <td>69</td>
      <td>HJK Helsinki</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13246</th>
      <td>13246</td>
      <td>244022</td>
      <td>I. Sissoko</td>
      <td>22</td>
      <td>https://cdn.sofifa.org/players/4/19/244022.png</td>
      <td>France</td>
      <td>https://cdn.sofifa.org/flags/18.png</td>
      <td>62</td>
      <td>68</td>
      <td>AS Béziers</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13247</th>
      <td>13247</td>
      <td>189238</td>
      <td>F. Hart</td>
      <td>28</td>
      <td>https://cdn.sofifa.org/players/4/19/189238.png</td>
      <td>Austria</td>
      <td>https://cdn.sofifa.org/flags/4.png</td>
      <td>62</td>
      <td>62</td>
      <td>SV Mattersburg</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13248</th>
      <td>13248</td>
      <td>211511</td>
      <td>L. McCullough</td>
      <td>24</td>
      <td>https://cdn.sofifa.org/players/4/19/211511.png</td>
      <td>Northern Ireland</td>
      <td>https://cdn.sofifa.org/flags/35.png</td>
      <td>62</td>
      <td>69</td>
      <td>Tranmere Rovers</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13249</th>
      <td>13249</td>
      <td>224055</td>
      <td>Li Yunqiu</td>
      <td>27</td>
      <td>https://cdn.sofifa.org/players/4/19/224055.png</td>
      <td>China PR</td>
      <td>https://cdn.sofifa.org/flags/155.png</td>
      <td>62</td>
      <td>62</td>
      <td>Shanghai Greenland Shenhua FC</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13250</th>
      <td>13250</td>
      <td>244535</td>
      <td>F. Garcia</td>
      <td>29</td>
      <td>https://cdn.sofifa.org/players/4/19/244535.png</td>
      <td>Paraguay</td>
      <td>https://cdn.sofifa.org/flags/58.png</td>
      <td>62</td>
      <td>62</td>
      <td>Itagüí Leones FC</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13251</th>
      <td>13251</td>
      <td>134968</td>
      <td>R. Haemhouts</td>
      <td>34</td>
      <td>https://cdn.sofifa.org/players/4/19/134968.png</td>
      <td>Belgium</td>
      <td>https://cdn.sofifa.org/flags/7.png</td>
      <td>62</td>
      <td>62</td>
      <td>NAC Breda</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13252</th>
      <td>13252</td>
      <td>225336</td>
      <td>E. Binaku</td>
      <td>22</td>
      <td>https://cdn.sofifa.org/players/4/19/225336.png</td>
      <td>Albania</td>
      <td>https://cdn.sofifa.org/flags/1.png</td>
      <td>62</td>
      <td>70</td>
      <td>Malmö FF</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13253</th>
      <td>13253</td>
      <td>171320</td>
      <td>G. Miller</td>
      <td>31</td>
      <td>https://cdn.sofifa.org/players/4/19/171320.png</td>
      <td>Scotland</td>
      <td>https://cdn.sofifa.org/flags/42.png</td>
      <td>62</td>
      <td>62</td>
      <td>Carlisle United</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13254</th>
      <td>13254</td>
      <td>246328</td>
      <td>A. Aidonis</td>
      <td>17</td>
      <td>https://cdn.sofifa.org/players/4/19/246328.png</td>
      <td>Germany</td>
      <td>https://cdn.sofifa.org/flags/21.png</td>
      <td>62</td>
      <td>82</td>
      <td>VfB Stuttgart</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13255</th>
      <td>13255</td>
      <td>196921</td>
      <td>L. Sowah</td>
      <td>25</td>
      <td>https://cdn.sofifa.org/players/4/19/196921.png</td>
      <td>Germany</td>
      <td>https://cdn.sofifa.org/flags/21.png</td>
      <td>62</td>
      <td>65</td>
      <td>Hamilton Academical FC</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13256</th>
      <td>13256</td>
      <td>202809</td>
      <td>R. Deacon</td>
      <td>26</td>
      <td>https://cdn.sofifa.org/players/4/19/202809.png</td>
      <td>England</td>
      <td>https://cdn.sofifa.org/flags/14.png</td>
      <td>62</td>
      <td>62</td>
      <td>Dundee FC</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13257</th>
      <td>13257</td>
      <td>226617</td>
      <td>Jang Hyun Soo</td>
      <td>25</td>
      <td>https://cdn.sofifa.org/players/4/19/226617.png</td>
      <td>Korea Republic</td>
      <td>https://cdn.sofifa.org/flags/167.png</td>
      <td>62</td>
      <td>65</td>
      <td>Suwon Samsung Bluewings</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13258</th>
      <td>13258</td>
      <td>230713</td>
      <td>A. Al Malki</td>
      <td>23</td>
      <td>https://cdn.sofifa.org/players/4/19/230713.png</td>
      <td>Saudi Arabia</td>
      <td>https://cdn.sofifa.org/flags/183.png</td>
      <td>62</td>
      <td>67</td>
      <td>Al Wehda</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13259</th>
      <td>13259</td>
      <td>234809</td>
      <td>E. Guerrero</td>
      <td>27</td>
      <td>https://cdn.sofifa.org/players/4/19/234809.png</td>
      <td>Chile</td>
      <td>https://cdn.sofifa.org/flags/55.png</td>
      <td>62</td>
      <td>65</td>
      <td>CD Palestino</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13260</th>
      <td>13260</td>
      <td>246073</td>
      <td>Hernáiz</td>
      <td>20</td>
      <td>https://cdn.sofifa.org/players/4/19/246073.png</td>
      <td>Spain</td>
      <td>https://cdn.sofifa.org/flags/45.png</td>
      <td>62</td>
      <td>69</td>
      <td>Albacete BP</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13261</th>
      <td>13261</td>
      <td>221498</td>
      <td>H. Al Mansour</td>
      <td>25</td>
      <td>https://cdn.sofifa.org/players/4/19/221498.png</td>
      <td>Saudi Arabia</td>
      <td>https://cdn.sofifa.org/flags/183.png</td>
      <td>62</td>
      <td>64</td>
      <td>Al Nassr</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13262</th>
      <td>13262</td>
      <td>244026</td>
      <td>H. Paul</td>
      <td>24</td>
      <td>https://cdn.sofifa.org/players/4/19/244026.png</td>
      <td>Germany</td>
      <td>https://cdn.sofifa.org/flags/21.png</td>
      <td>62</td>
      <td>66</td>
      <td>TSV 1860 München</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13263</th>
      <td>13263</td>
      <td>244538</td>
      <td>S. Bauer</td>
      <td>25</td>
      <td>https://cdn.sofifa.org/players/4/19/244538.png</td>
      <td>Austria</td>
      <td>https://cdn.sofifa.org/flags/4.png</td>
      <td>62</td>
      <td>66</td>
      <td>FC Admira Wacker Mödling</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13264</th>
      <td>13264</td>
      <td>201019</td>
      <td>M. Chergui</td>
      <td>29</td>
      <td>https://cdn.sofifa.org/players/4/19/201019.png</td>
      <td>France</td>
      <td>https://cdn.sofifa.org/flags/18.png</td>
      <td>62</td>
      <td>62</td>
      <td>Grenoble Foot 38</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13265</th>
      <td>13265</td>
      <td>221499</td>
      <td>D. Gardner</td>
      <td>28</td>
      <td>https://cdn.sofifa.org/players/4/19/221499.png</td>
      <td>England</td>
      <td>https://cdn.sofifa.org/flags/14.png</td>
      <td>62</td>
      <td>62</td>
      <td>Oldham Athletic</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13266</th>
      <td>13266</td>
      <td>237371</td>
      <td>L. Bengtsson</td>
      <td>20</td>
      <td>https://cdn.sofifa.org/players/4/19/237371.png</td>
      <td>Sweden</td>
      <td>https://cdn.sofifa.org/flags/46.png</td>
      <td>62</td>
      <td>73</td>
      <td>Hammarby IF</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13267</th>
      <td>13267</td>
      <td>242491</td>
      <td>F. Jaramillo</td>
      <td>22</td>
      <td>https://cdn.sofifa.org/players/4/19/242491.png</td>
      <td>Colombia</td>
      <td>https://cdn.sofifa.org/flags/56.png</td>
      <td>62</td>
      <td>70</td>
      <td>Itagüí Leones FC</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13268</th>
      <td>13268</td>
      <td>153148</td>
      <td>L. Garguła</td>
      <td>37</td>
      <td>https://cdn.sofifa.org/players/4/19/153148.png</td>
      <td>Poland</td>
      <td>https://cdn.sofifa.org/flags/37.png</td>
      <td>62</td>
      <td>62</td>
      <td>Miedź Legnica</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13269</th>
      <td>13269</td>
      <td>244540</td>
      <td>S. Rivera</td>
      <td>26</td>
      <td>https://cdn.sofifa.org/players/4/19/244540.png</td>
      <td>Colombia</td>
      <td>https://cdn.sofifa.org/flags/56.png</td>
      <td>62</td>
      <td>64</td>
      <td>Jaguares de Córdoba</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13270</th>
      <td>13270</td>
      <td>245564</td>
      <td>Vinicius</td>
      <td>19</td>
      <td>https://cdn.sofifa.org/players/4/19/245564.png</td>
      <td>Brazil</td>
      <td>https://cdn.sofifa.org/flags/54.png</td>
      <td>62</td>
      <td>77</td>
      <td>Bologna</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13271</th>
      <td>13271</td>
      <td>213821</td>
      <td>F. Sepúlveda</td>
      <td>26</td>
      <td>https://cdn.sofifa.org/players/4/19/213821.png</td>
      <td>Chile</td>
      <td>https://cdn.sofifa.org/flags/55.png</td>
      <td>62</td>
      <td>63</td>
      <td>CD Antofagasta</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13272</th>
      <td>13272</td>
      <td>240701</td>
      <td>L. Spence</td>
      <td>22</td>
      <td>https://cdn.sofifa.org/players/4/19/240701.png</td>
      <td>Scotland</td>
      <td>https://cdn.sofifa.org/flags/42.png</td>
      <td>62</td>
      <td>70</td>
      <td>Dundee FC</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13273</th>
      <td>13273</td>
      <td>242237</td>
      <td>B. Lepistu</td>
      <td>25</td>
      <td>https://cdn.sofifa.org/players/4/19/242237.png</td>
      <td>Estonia</td>
      <td>https://cdn.sofifa.org/flags/208.png</td>
      <td>62</td>
      <td>67</td>
      <td>Kristiansund BK</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13274</th>
      <td>13274</td>
      <td>244029</td>
      <td>A. Abruscia</td>
      <td>27</td>
      <td>https://cdn.sofifa.org/players/4/19/244029.png</td>
      <td>Italy</td>
      <td>https://cdn.sofifa.org/flags/27.png</td>
      <td>62</td>
      <td>62</td>
      <td>TSV 1860 München</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13275</th>
      <td>13275</td>
      <td>244541</td>
      <td>E. González</td>
      <td>23</td>
      <td>https://cdn.sofifa.org/players/4/19/244541.png</td>
      <td>Venezuela</td>
      <td>https://cdn.sofifa.org/flags/61.png</td>
      <td>62</td>
      <td>70</td>
      <td>Boyacá Chicó FC</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13276</th>
      <td>13276</td>
      <td>211006</td>
      <td>M. Al Amri</td>
      <td>26</td>
      <td>https://cdn.sofifa.org/players/4/19/211006.png</td>
      <td>Saudi Arabia</td>
      <td>https://cdn.sofifa.org/flags/183.png</td>
      <td>62</td>
      <td>63</td>
      <td>Al Raed</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13277</th>
      <td>13277</td>
      <td>215102</td>
      <td>J. Rebolledo</td>
      <td>26</td>
      <td>https://cdn.sofifa.org/players/4/19/215102.png</td>
      <td>Chile</td>
      <td>https://cdn.sofifa.org/flags/55.png</td>
      <td>62</td>
      <td>62</td>
      <td>Deportes Iquique</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13278</th>
      <td>13278</td>
      <td>246078</td>
      <td>C. Mamengi</td>
      <td>17</td>
      <td>https://cdn.sofifa.org/players/4/19/246078.png</td>
      <td>Netherlands</td>
      <td>https://cdn.sofifa.org/flags/34.png</td>
      <td>62</td>
      <td>79</td>
      <td>FC Utrecht</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13279</th>
      <td>13279</td>
      <td>239679</td>
      <td>P. Mazzocchi</td>
      <td>22</td>
      <td>https://cdn.sofifa.org/players/4/19/239679.png</td>
      <td>Italy</td>
      <td>https://cdn.sofifa.org/flags/27.png</td>
      <td>62</td>
      <td>69</td>
      <td>Perugia</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13280</th>
      <td>13280</td>
      <td>244543</td>
      <td>Y. Ammour</td>
      <td>19</td>
      <td>https://cdn.sofifa.org/players/4/19/244543.png</td>
      <td>France</td>
      <td>https://cdn.sofifa.org/flags/18.png</td>
      <td>62</td>
      <td>77</td>
      <td>Montpellier HSC</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13281</th>
      <td>13281</td>
      <td>212800</td>
      <td>Jwa Joon Hyeop</td>
      <td>27</td>
      <td>https://cdn.sofifa.org/players/4/19/212800.png</td>
      <td>Korea Republic</td>
      <td>https://cdn.sofifa.org/flags/167.png</td>
      <td>62</td>
      <td>62</td>
      <td>Gyeongnam FC</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13282</th>
      <td>13282</td>
      <td>231232</td>
      <td>O. Marrufo</td>
      <td>25</td>
      <td>https://cdn.sofifa.org/players/4/19/231232.png</td>
      <td>Mexico</td>
      <td>https://cdn.sofifa.org/flags/83.png</td>
      <td>62</td>
      <td>65</td>
      <td>Tiburones Rojos de Veracruz</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13283</th>
      <td>13283</td>
      <td>232256</td>
      <td>Han Pengfei</td>
      <td>25</td>
      <td>https://cdn.sofifa.org/players/4/19/232256.png</td>
      <td>China PR</td>
      <td>https://cdn.sofifa.org/flags/155.png</td>
      <td>62</td>
      <td>66</td>
      <td>Guizhou Hengfeng FC</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16450</th>
      <td>16450</td>
      <td>193911</td>
      <td>S. Paul</td>
      <td>31</td>
      <td>https://cdn.sofifa.org/players/4/19/193911.png</td>
      <td>India</td>
      <td>https://cdn.sofifa.org/flags/159.png</td>
      <td>57</td>
      <td>57</td>
      <td>NaN</td>
      <td>...</td>
      <td>52.0</td>
      <td>12.0</td>
      <td>15.0</td>
      <td>12.0</td>
      <td>53.0</td>
      <td>48.0</td>
      <td>62.0</td>
      <td>57.0</td>
      <td>60.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16539</th>
      <td>16539</td>
      <td>245167</td>
      <td>L. Lalruatthara</td>
      <td>23</td>
      <td>https://cdn.sofifa.org/players/4/19/245167.png</td>
      <td>India</td>
      <td>https://cdn.sofifa.org/flags/159.png</td>
      <td>57</td>
      <td>63</td>
      <td>NaN</td>
      <td>...</td>
      <td>57.0</td>
      <td>60.0</td>
      <td>61.0</td>
      <td>57.0</td>
      <td>15.0</td>
      <td>12.0</td>
      <td>11.0</td>
      <td>6.0</td>
      <td>12.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16793</th>
      <td>16793</td>
      <td>228192</td>
      <td>E. Lyngdoh</td>
      <td>31</td>
      <td>https://cdn.sofifa.org/players/4/19/228192.png</td>
      <td>India</td>
      <td>https://cdn.sofifa.org/flags/159.png</td>
      <td>56</td>
      <td>56</td>
      <td>NaN</td>
      <td>...</td>
      <td>63.0</td>
      <td>43.0</td>
      <td>48.0</td>
      <td>32.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>9.0</td>
      <td>15.0</td>
      <td>9.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>17129</th>
      <td>17129</td>
      <td>228198</td>
      <td>J. Singh</td>
      <td>26</td>
      <td>https://cdn.sofifa.org/players/4/19/228198.png</td>
      <td>India</td>
      <td>https://cdn.sofifa.org/flags/159.png</td>
      <td>55</td>
      <td>58</td>
      <td>NaN</td>
      <td>...</td>
      <td>42.0</td>
      <td>26.0</td>
      <td>18.0</td>
      <td>13.0</td>
      <td>15.0</td>
      <td>14.0</td>
      <td>8.0</td>
      <td>10.0</td>
      <td>16.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>17339</th>
      <td>17339</td>
      <td>233526</td>
      <td>S. Passi</td>
      <td>23</td>
      <td>https://cdn.sofifa.org/players/4/19/233526.png</td>
      <td>India</td>
      <td>https://cdn.sofifa.org/flags/159.png</td>
      <td>54</td>
      <td>63</td>
      <td>NaN</td>
      <td>...</td>
      <td>45.0</td>
      <td>14.0</td>
      <td>23.0</td>
      <td>21.0</td>
      <td>8.0</td>
      <td>6.0</td>
      <td>13.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>17436</th>
      <td>17436</td>
      <td>236452</td>
      <td>D. Lalhlimpuia</td>
      <td>20</td>
      <td>https://cdn.sofifa.org/players/4/19/236452.png</td>
      <td>India</td>
      <td>https://cdn.sofifa.org/flags/159.png</td>
      <td>54</td>
      <td>67</td>
      <td>NaN</td>
      <td>...</td>
      <td>46.0</td>
      <td>26.0</td>
      <td>17.0</td>
      <td>13.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>12.0</td>
      <td>11.0</td>
      <td>11.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>17539</th>
      <td>17539</td>
      <td>234508</td>
      <td>C. Singh</td>
      <td>21</td>
      <td>https://cdn.sofifa.org/players/4/19/234508.png</td>
      <td>India</td>
      <td>https://cdn.sofifa.org/flags/159.png</td>
      <td>53</td>
      <td>62</td>
      <td>NaN</td>
      <td>...</td>
      <td>41.0</td>
      <td>56.0</td>
      <td>49.0</td>
      <td>53.0</td>
      <td>12.0</td>
      <td>9.0</td>
      <td>6.0</td>
      <td>14.0</td>
      <td>8.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>60 rows × 89 columns</p>
</div>




```python
flt_index = data[data['Jersey Number'].isnull()].index
flt_index
```




    Int64Index([ 5018,  6736,  7922,  9905, 10628, 13236, 13237, 13238, 13239,
                13240, 13241, 13242, 13243, 13244, 13245, 13246, 13247, 13248,
                13249, 13250, 13251, 13252, 13253, 13254, 13255, 13256, 13257,
                13258, 13259, 13260, 13261, 13262, 13263, 13264, 13265, 13266,
                13267, 13268, 13269, 13270, 13271, 13272, 13273, 13274, 13275,
                13276, 13277, 13278, 13279, 13280, 13281, 13282, 13283, 16450,
                16539, 16793, 17129, 17339, 17436, 17539],
               dtype='int64')




```python
data[data.index.isin(flt_index)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>ID</th>
      <th>Name</th>
      <th>Age</th>
      <th>Photo</th>
      <th>Nationality</th>
      <th>Flag</th>
      <th>Overall</th>
      <th>Potential</th>
      <th>Club</th>
      <th>...</th>
      <th>Composure</th>
      <th>Marking</th>
      <th>StandingTackle</th>
      <th>SlidingTackle</th>
      <th>GKDiving</th>
      <th>GKHandling</th>
      <th>GKKicking</th>
      <th>GKPositioning</th>
      <th>GKReflexes</th>
      <th>Release Clause</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5018</th>
      <td>5018</td>
      <td>153160</td>
      <td>R. Raldes</td>
      <td>37</td>
      <td>https://cdn.sofifa.org/players/4/19/153160.png</td>
      <td>Bolivia</td>
      <td>https://cdn.sofifa.org/flags/53.png</td>
      <td>70</td>
      <td>70</td>
      <td>NaN</td>
      <td>...</td>
      <td>64.0</td>
      <td>79.0</td>
      <td>70.0</td>
      <td>70.0</td>
      <td>16.0</td>
      <td>8.0</td>
      <td>11.0</td>
      <td>12.0</td>
      <td>13.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6736</th>
      <td>6736</td>
      <td>175393</td>
      <td>J. Arce</td>
      <td>33</td>
      <td>https://cdn.sofifa.org/players/4/19/175393.png</td>
      <td>Bolivia</td>
      <td>https://cdn.sofifa.org/flags/53.png</td>
      <td>68</td>
      <td>68</td>
      <td>NaN</td>
      <td>...</td>
      <td>67.0</td>
      <td>12.0</td>
      <td>34.0</td>
      <td>33.0</td>
      <td>16.0</td>
      <td>13.0</td>
      <td>15.0</td>
      <td>16.0</td>
      <td>8.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7922</th>
      <td>7922</td>
      <td>195905</td>
      <td>L. Gutiérrez</td>
      <td>33</td>
      <td>https://cdn.sofifa.org/players/4/19/195905.png</td>
      <td>Bolivia</td>
      <td>https://cdn.sofifa.org/flags/53.png</td>
      <td>67</td>
      <td>67</td>
      <td>NaN</td>
      <td>...</td>
      <td>54.0</td>
      <td>72.0</td>
      <td>71.0</td>
      <td>64.0</td>
      <td>11.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>12.0</td>
      <td>12.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9905</th>
      <td>9905</td>
      <td>226044</td>
      <td>R. Vargas</td>
      <td>23</td>
      <td>https://cdn.sofifa.org/players/4/19/226044.png</td>
      <td>Bolivia</td>
      <td>https://cdn.sofifa.org/flags/53.png</td>
      <td>66</td>
      <td>69</td>
      <td>NaN</td>
      <td>...</td>
      <td>64.0</td>
      <td>19.0</td>
      <td>24.0</td>
      <td>23.0</td>
      <td>13.0</td>
      <td>11.0</td>
      <td>11.0</td>
      <td>7.0</td>
      <td>9.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10628</th>
      <td>10628</td>
      <td>216751</td>
      <td>D. Bejarano</td>
      <td>26</td>
      <td>https://cdn.sofifa.org/players/4/19/216751.png</td>
      <td>Bolivia</td>
      <td>https://cdn.sofifa.org/flags/53.png</td>
      <td>65</td>
      <td>66</td>
      <td>NaN</td>
      <td>...</td>
      <td>57.0</td>
      <td>68.0</td>
      <td>69.0</td>
      <td>68.0</td>
      <td>14.0</td>
      <td>12.0</td>
      <td>11.0</td>
      <td>14.0</td>
      <td>12.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13236</th>
      <td>13236</td>
      <td>177971</td>
      <td>J. McNulty</td>
      <td>33</td>
      <td>https://cdn.sofifa.org/players/4/19/177971.png</td>
      <td>Scotland</td>
      <td>https://cdn.sofifa.org/flags/42.png</td>
      <td>62</td>
      <td>62</td>
      <td>Rochdale</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13237</th>
      <td>13237</td>
      <td>195380</td>
      <td>J. Barrera</td>
      <td>29</td>
      <td>https://cdn.sofifa.org/players/4/19/195380.png</td>
      <td>Nicaragua</td>
      <td>https://cdn.sofifa.org/flags/86.png</td>
      <td>62</td>
      <td>62</td>
      <td>Boyacá Chicó FC</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13238</th>
      <td>13238</td>
      <td>139317</td>
      <td>J. Stead</td>
      <td>35</td>
      <td>https://cdn.sofifa.org/players/4/19/139317.png</td>
      <td>England</td>
      <td>https://cdn.sofifa.org/flags/14.png</td>
      <td>62</td>
      <td>62</td>
      <td>Notts County</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13239</th>
      <td>13239</td>
      <td>240437</td>
      <td>A. Semprini</td>
      <td>20</td>
      <td>https://cdn.sofifa.org/players/4/19/240437.png</td>
      <td>Italy</td>
      <td>https://cdn.sofifa.org/flags/27.png</td>
      <td>62</td>
      <td>72</td>
      <td>Brescia</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13240</th>
      <td>13240</td>
      <td>209462</td>
      <td>R. Bingham</td>
      <td>24</td>
      <td>https://cdn.sofifa.org/players/4/19/209462.png</td>
      <td>England</td>
      <td>https://cdn.sofifa.org/flags/14.png</td>
      <td>62</td>
      <td>66</td>
      <td>Hamilton Academical FC</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13241</th>
      <td>13241</td>
      <td>219702</td>
      <td>K. Dankowski</td>
      <td>21</td>
      <td>https://cdn.sofifa.org/players/4/19/219702.png</td>
      <td>Poland</td>
      <td>https://cdn.sofifa.org/flags/37.png</td>
      <td>62</td>
      <td>72</td>
      <td>Śląsk Wrocław</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13242</th>
      <td>13242</td>
      <td>225590</td>
      <td>I. Colman</td>
      <td>23</td>
      <td>https://cdn.sofifa.org/players/4/19/225590.png</td>
      <td>Argentina</td>
      <td>https://cdn.sofifa.org/flags/52.png</td>
      <td>62</td>
      <td>70</td>
      <td>Club Atlético Aldosivi</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13243</th>
      <td>13243</td>
      <td>233782</td>
      <td>M. Feeney</td>
      <td>19</td>
      <td>https://cdn.sofifa.org/players/4/19/233782.png</td>
      <td>England</td>
      <td>https://cdn.sofifa.org/flags/14.png</td>
      <td>62</td>
      <td>78</td>
      <td>Everton</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13244</th>
      <td>13244</td>
      <td>239158</td>
      <td>R. Minor</td>
      <td>30</td>
      <td>https://cdn.sofifa.org/players/4/19/239158.png</td>
      <td>Denmark</td>
      <td>https://cdn.sofifa.org/flags/13.png</td>
      <td>62</td>
      <td>62</td>
      <td>Hobro IK</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13245</th>
      <td>13245</td>
      <td>242998</td>
      <td>Klauss</td>
      <td>21</td>
      <td>https://cdn.sofifa.org/players/4/19/242998.png</td>
      <td>Brazil</td>
      <td>https://cdn.sofifa.org/flags/54.png</td>
      <td>62</td>
      <td>69</td>
      <td>HJK Helsinki</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13246</th>
      <td>13246</td>
      <td>244022</td>
      <td>I. Sissoko</td>
      <td>22</td>
      <td>https://cdn.sofifa.org/players/4/19/244022.png</td>
      <td>France</td>
      <td>https://cdn.sofifa.org/flags/18.png</td>
      <td>62</td>
      <td>68</td>
      <td>AS Béziers</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13247</th>
      <td>13247</td>
      <td>189238</td>
      <td>F. Hart</td>
      <td>28</td>
      <td>https://cdn.sofifa.org/players/4/19/189238.png</td>
      <td>Austria</td>
      <td>https://cdn.sofifa.org/flags/4.png</td>
      <td>62</td>
      <td>62</td>
      <td>SV Mattersburg</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13248</th>
      <td>13248</td>
      <td>211511</td>
      <td>L. McCullough</td>
      <td>24</td>
      <td>https://cdn.sofifa.org/players/4/19/211511.png</td>
      <td>Northern Ireland</td>
      <td>https://cdn.sofifa.org/flags/35.png</td>
      <td>62</td>
      <td>69</td>
      <td>Tranmere Rovers</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13249</th>
      <td>13249</td>
      <td>224055</td>
      <td>Li Yunqiu</td>
      <td>27</td>
      <td>https://cdn.sofifa.org/players/4/19/224055.png</td>
      <td>China PR</td>
      <td>https://cdn.sofifa.org/flags/155.png</td>
      <td>62</td>
      <td>62</td>
      <td>Shanghai Greenland Shenhua FC</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13250</th>
      <td>13250</td>
      <td>244535</td>
      <td>F. Garcia</td>
      <td>29</td>
      <td>https://cdn.sofifa.org/players/4/19/244535.png</td>
      <td>Paraguay</td>
      <td>https://cdn.sofifa.org/flags/58.png</td>
      <td>62</td>
      <td>62</td>
      <td>Itagüí Leones FC</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13251</th>
      <td>13251</td>
      <td>134968</td>
      <td>R. Haemhouts</td>
      <td>34</td>
      <td>https://cdn.sofifa.org/players/4/19/134968.png</td>
      <td>Belgium</td>
      <td>https://cdn.sofifa.org/flags/7.png</td>
      <td>62</td>
      <td>62</td>
      <td>NAC Breda</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13252</th>
      <td>13252</td>
      <td>225336</td>
      <td>E. Binaku</td>
      <td>22</td>
      <td>https://cdn.sofifa.org/players/4/19/225336.png</td>
      <td>Albania</td>
      <td>https://cdn.sofifa.org/flags/1.png</td>
      <td>62</td>
      <td>70</td>
      <td>Malmö FF</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13253</th>
      <td>13253</td>
      <td>171320</td>
      <td>G. Miller</td>
      <td>31</td>
      <td>https://cdn.sofifa.org/players/4/19/171320.png</td>
      <td>Scotland</td>
      <td>https://cdn.sofifa.org/flags/42.png</td>
      <td>62</td>
      <td>62</td>
      <td>Carlisle United</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13254</th>
      <td>13254</td>
      <td>246328</td>
      <td>A. Aidonis</td>
      <td>17</td>
      <td>https://cdn.sofifa.org/players/4/19/246328.png</td>
      <td>Germany</td>
      <td>https://cdn.sofifa.org/flags/21.png</td>
      <td>62</td>
      <td>82</td>
      <td>VfB Stuttgart</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13255</th>
      <td>13255</td>
      <td>196921</td>
      <td>L. Sowah</td>
      <td>25</td>
      <td>https://cdn.sofifa.org/players/4/19/196921.png</td>
      <td>Germany</td>
      <td>https://cdn.sofifa.org/flags/21.png</td>
      <td>62</td>
      <td>65</td>
      <td>Hamilton Academical FC</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13256</th>
      <td>13256</td>
      <td>202809</td>
      <td>R. Deacon</td>
      <td>26</td>
      <td>https://cdn.sofifa.org/players/4/19/202809.png</td>
      <td>England</td>
      <td>https://cdn.sofifa.org/flags/14.png</td>
      <td>62</td>
      <td>62</td>
      <td>Dundee FC</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13257</th>
      <td>13257</td>
      <td>226617</td>
      <td>Jang Hyun Soo</td>
      <td>25</td>
      <td>https://cdn.sofifa.org/players/4/19/226617.png</td>
      <td>Korea Republic</td>
      <td>https://cdn.sofifa.org/flags/167.png</td>
      <td>62</td>
      <td>65</td>
      <td>Suwon Samsung Bluewings</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13258</th>
      <td>13258</td>
      <td>230713</td>
      <td>A. Al Malki</td>
      <td>23</td>
      <td>https://cdn.sofifa.org/players/4/19/230713.png</td>
      <td>Saudi Arabia</td>
      <td>https://cdn.sofifa.org/flags/183.png</td>
      <td>62</td>
      <td>67</td>
      <td>Al Wehda</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13259</th>
      <td>13259</td>
      <td>234809</td>
      <td>E. Guerrero</td>
      <td>27</td>
      <td>https://cdn.sofifa.org/players/4/19/234809.png</td>
      <td>Chile</td>
      <td>https://cdn.sofifa.org/flags/55.png</td>
      <td>62</td>
      <td>65</td>
      <td>CD Palestino</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13260</th>
      <td>13260</td>
      <td>246073</td>
      <td>Hernáiz</td>
      <td>20</td>
      <td>https://cdn.sofifa.org/players/4/19/246073.png</td>
      <td>Spain</td>
      <td>https://cdn.sofifa.org/flags/45.png</td>
      <td>62</td>
      <td>69</td>
      <td>Albacete BP</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13261</th>
      <td>13261</td>
      <td>221498</td>
      <td>H. Al Mansour</td>
      <td>25</td>
      <td>https://cdn.sofifa.org/players/4/19/221498.png</td>
      <td>Saudi Arabia</td>
      <td>https://cdn.sofifa.org/flags/183.png</td>
      <td>62</td>
      <td>64</td>
      <td>Al Nassr</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13262</th>
      <td>13262</td>
      <td>244026</td>
      <td>H. Paul</td>
      <td>24</td>
      <td>https://cdn.sofifa.org/players/4/19/244026.png</td>
      <td>Germany</td>
      <td>https://cdn.sofifa.org/flags/21.png</td>
      <td>62</td>
      <td>66</td>
      <td>TSV 1860 München</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13263</th>
      <td>13263</td>
      <td>244538</td>
      <td>S. Bauer</td>
      <td>25</td>
      <td>https://cdn.sofifa.org/players/4/19/244538.png</td>
      <td>Austria</td>
      <td>https://cdn.sofifa.org/flags/4.png</td>
      <td>62</td>
      <td>66</td>
      <td>FC Admira Wacker Mödling</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13264</th>
      <td>13264</td>
      <td>201019</td>
      <td>M. Chergui</td>
      <td>29</td>
      <td>https://cdn.sofifa.org/players/4/19/201019.png</td>
      <td>France</td>
      <td>https://cdn.sofifa.org/flags/18.png</td>
      <td>62</td>
      <td>62</td>
      <td>Grenoble Foot 38</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13265</th>
      <td>13265</td>
      <td>221499</td>
      <td>D. Gardner</td>
      <td>28</td>
      <td>https://cdn.sofifa.org/players/4/19/221499.png</td>
      <td>England</td>
      <td>https://cdn.sofifa.org/flags/14.png</td>
      <td>62</td>
      <td>62</td>
      <td>Oldham Athletic</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13266</th>
      <td>13266</td>
      <td>237371</td>
      <td>L. Bengtsson</td>
      <td>20</td>
      <td>https://cdn.sofifa.org/players/4/19/237371.png</td>
      <td>Sweden</td>
      <td>https://cdn.sofifa.org/flags/46.png</td>
      <td>62</td>
      <td>73</td>
      <td>Hammarby IF</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13267</th>
      <td>13267</td>
      <td>242491</td>
      <td>F. Jaramillo</td>
      <td>22</td>
      <td>https://cdn.sofifa.org/players/4/19/242491.png</td>
      <td>Colombia</td>
      <td>https://cdn.sofifa.org/flags/56.png</td>
      <td>62</td>
      <td>70</td>
      <td>Itagüí Leones FC</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13268</th>
      <td>13268</td>
      <td>153148</td>
      <td>L. Garguła</td>
      <td>37</td>
      <td>https://cdn.sofifa.org/players/4/19/153148.png</td>
      <td>Poland</td>
      <td>https://cdn.sofifa.org/flags/37.png</td>
      <td>62</td>
      <td>62</td>
      <td>Miedź Legnica</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13269</th>
      <td>13269</td>
      <td>244540</td>
      <td>S. Rivera</td>
      <td>26</td>
      <td>https://cdn.sofifa.org/players/4/19/244540.png</td>
      <td>Colombia</td>
      <td>https://cdn.sofifa.org/flags/56.png</td>
      <td>62</td>
      <td>64</td>
      <td>Jaguares de Córdoba</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13270</th>
      <td>13270</td>
      <td>245564</td>
      <td>Vinicius</td>
      <td>19</td>
      <td>https://cdn.sofifa.org/players/4/19/245564.png</td>
      <td>Brazil</td>
      <td>https://cdn.sofifa.org/flags/54.png</td>
      <td>62</td>
      <td>77</td>
      <td>Bologna</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13271</th>
      <td>13271</td>
      <td>213821</td>
      <td>F. Sepúlveda</td>
      <td>26</td>
      <td>https://cdn.sofifa.org/players/4/19/213821.png</td>
      <td>Chile</td>
      <td>https://cdn.sofifa.org/flags/55.png</td>
      <td>62</td>
      <td>63</td>
      <td>CD Antofagasta</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13272</th>
      <td>13272</td>
      <td>240701</td>
      <td>L. Spence</td>
      <td>22</td>
      <td>https://cdn.sofifa.org/players/4/19/240701.png</td>
      <td>Scotland</td>
      <td>https://cdn.sofifa.org/flags/42.png</td>
      <td>62</td>
      <td>70</td>
      <td>Dundee FC</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13273</th>
      <td>13273</td>
      <td>242237</td>
      <td>B. Lepistu</td>
      <td>25</td>
      <td>https://cdn.sofifa.org/players/4/19/242237.png</td>
      <td>Estonia</td>
      <td>https://cdn.sofifa.org/flags/208.png</td>
      <td>62</td>
      <td>67</td>
      <td>Kristiansund BK</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13274</th>
      <td>13274</td>
      <td>244029</td>
      <td>A. Abruscia</td>
      <td>27</td>
      <td>https://cdn.sofifa.org/players/4/19/244029.png</td>
      <td>Italy</td>
      <td>https://cdn.sofifa.org/flags/27.png</td>
      <td>62</td>
      <td>62</td>
      <td>TSV 1860 München</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13275</th>
      <td>13275</td>
      <td>244541</td>
      <td>E. González</td>
      <td>23</td>
      <td>https://cdn.sofifa.org/players/4/19/244541.png</td>
      <td>Venezuela</td>
      <td>https://cdn.sofifa.org/flags/61.png</td>
      <td>62</td>
      <td>70</td>
      <td>Boyacá Chicó FC</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13276</th>
      <td>13276</td>
      <td>211006</td>
      <td>M. Al Amri</td>
      <td>26</td>
      <td>https://cdn.sofifa.org/players/4/19/211006.png</td>
      <td>Saudi Arabia</td>
      <td>https://cdn.sofifa.org/flags/183.png</td>
      <td>62</td>
      <td>63</td>
      <td>Al Raed</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13277</th>
      <td>13277</td>
      <td>215102</td>
      <td>J. Rebolledo</td>
      <td>26</td>
      <td>https://cdn.sofifa.org/players/4/19/215102.png</td>
      <td>Chile</td>
      <td>https://cdn.sofifa.org/flags/55.png</td>
      <td>62</td>
      <td>62</td>
      <td>Deportes Iquique</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13278</th>
      <td>13278</td>
      <td>246078</td>
      <td>C. Mamengi</td>
      <td>17</td>
      <td>https://cdn.sofifa.org/players/4/19/246078.png</td>
      <td>Netherlands</td>
      <td>https://cdn.sofifa.org/flags/34.png</td>
      <td>62</td>
      <td>79</td>
      <td>FC Utrecht</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13279</th>
      <td>13279</td>
      <td>239679</td>
      <td>P. Mazzocchi</td>
      <td>22</td>
      <td>https://cdn.sofifa.org/players/4/19/239679.png</td>
      <td>Italy</td>
      <td>https://cdn.sofifa.org/flags/27.png</td>
      <td>62</td>
      <td>69</td>
      <td>Perugia</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13280</th>
      <td>13280</td>
      <td>244543</td>
      <td>Y. Ammour</td>
      <td>19</td>
      <td>https://cdn.sofifa.org/players/4/19/244543.png</td>
      <td>France</td>
      <td>https://cdn.sofifa.org/flags/18.png</td>
      <td>62</td>
      <td>77</td>
      <td>Montpellier HSC</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13281</th>
      <td>13281</td>
      <td>212800</td>
      <td>Jwa Joon Hyeop</td>
      <td>27</td>
      <td>https://cdn.sofifa.org/players/4/19/212800.png</td>
      <td>Korea Republic</td>
      <td>https://cdn.sofifa.org/flags/167.png</td>
      <td>62</td>
      <td>62</td>
      <td>Gyeongnam FC</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13282</th>
      <td>13282</td>
      <td>231232</td>
      <td>O. Marrufo</td>
      <td>25</td>
      <td>https://cdn.sofifa.org/players/4/19/231232.png</td>
      <td>Mexico</td>
      <td>https://cdn.sofifa.org/flags/83.png</td>
      <td>62</td>
      <td>65</td>
      <td>Tiburones Rojos de Veracruz</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13283</th>
      <td>13283</td>
      <td>232256</td>
      <td>Han Pengfei</td>
      <td>25</td>
      <td>https://cdn.sofifa.org/players/4/19/232256.png</td>
      <td>China PR</td>
      <td>https://cdn.sofifa.org/flags/155.png</td>
      <td>62</td>
      <td>66</td>
      <td>Guizhou Hengfeng FC</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16450</th>
      <td>16450</td>
      <td>193911</td>
      <td>S. Paul</td>
      <td>31</td>
      <td>https://cdn.sofifa.org/players/4/19/193911.png</td>
      <td>India</td>
      <td>https://cdn.sofifa.org/flags/159.png</td>
      <td>57</td>
      <td>57</td>
      <td>NaN</td>
      <td>...</td>
      <td>52.0</td>
      <td>12.0</td>
      <td>15.0</td>
      <td>12.0</td>
      <td>53.0</td>
      <td>48.0</td>
      <td>62.0</td>
      <td>57.0</td>
      <td>60.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16539</th>
      <td>16539</td>
      <td>245167</td>
      <td>L. Lalruatthara</td>
      <td>23</td>
      <td>https://cdn.sofifa.org/players/4/19/245167.png</td>
      <td>India</td>
      <td>https://cdn.sofifa.org/flags/159.png</td>
      <td>57</td>
      <td>63</td>
      <td>NaN</td>
      <td>...</td>
      <td>57.0</td>
      <td>60.0</td>
      <td>61.0</td>
      <td>57.0</td>
      <td>15.0</td>
      <td>12.0</td>
      <td>11.0</td>
      <td>6.0</td>
      <td>12.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16793</th>
      <td>16793</td>
      <td>228192</td>
      <td>E. Lyngdoh</td>
      <td>31</td>
      <td>https://cdn.sofifa.org/players/4/19/228192.png</td>
      <td>India</td>
      <td>https://cdn.sofifa.org/flags/159.png</td>
      <td>56</td>
      <td>56</td>
      <td>NaN</td>
      <td>...</td>
      <td>63.0</td>
      <td>43.0</td>
      <td>48.0</td>
      <td>32.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>9.0</td>
      <td>15.0</td>
      <td>9.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>17129</th>
      <td>17129</td>
      <td>228198</td>
      <td>J. Singh</td>
      <td>26</td>
      <td>https://cdn.sofifa.org/players/4/19/228198.png</td>
      <td>India</td>
      <td>https://cdn.sofifa.org/flags/159.png</td>
      <td>55</td>
      <td>58</td>
      <td>NaN</td>
      <td>...</td>
      <td>42.0</td>
      <td>26.0</td>
      <td>18.0</td>
      <td>13.0</td>
      <td>15.0</td>
      <td>14.0</td>
      <td>8.0</td>
      <td>10.0</td>
      <td>16.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>17339</th>
      <td>17339</td>
      <td>233526</td>
      <td>S. Passi</td>
      <td>23</td>
      <td>https://cdn.sofifa.org/players/4/19/233526.png</td>
      <td>India</td>
      <td>https://cdn.sofifa.org/flags/159.png</td>
      <td>54</td>
      <td>63</td>
      <td>NaN</td>
      <td>...</td>
      <td>45.0</td>
      <td>14.0</td>
      <td>23.0</td>
      <td>21.0</td>
      <td>8.0</td>
      <td>6.0</td>
      <td>13.0</td>
      <td>9.0</td>
      <td>10.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>17436</th>
      <td>17436</td>
      <td>236452</td>
      <td>D. Lalhlimpuia</td>
      <td>20</td>
      <td>https://cdn.sofifa.org/players/4/19/236452.png</td>
      <td>India</td>
      <td>https://cdn.sofifa.org/flags/159.png</td>
      <td>54</td>
      <td>67</td>
      <td>NaN</td>
      <td>...</td>
      <td>46.0</td>
      <td>26.0</td>
      <td>17.0</td>
      <td>13.0</td>
      <td>7.0</td>
      <td>8.0</td>
      <td>12.0</td>
      <td>11.0</td>
      <td>11.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>17539</th>
      <td>17539</td>
      <td>234508</td>
      <td>C. Singh</td>
      <td>21</td>
      <td>https://cdn.sofifa.org/players/4/19/234508.png</td>
      <td>India</td>
      <td>https://cdn.sofifa.org/flags/159.png</td>
      <td>53</td>
      <td>62</td>
      <td>NaN</td>
      <td>...</td>
      <td>41.0</td>
      <td>56.0</td>
      <td>49.0</td>
      <td>53.0</td>
      <td>12.0</td>
      <td>9.0</td>
      <td>6.0</td>
      <td>14.0</td>
      <td>8.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>60 rows × 89 columns</p>
</div>




```python
data_num[data_num.index.isin(flt_index)]['Jersey Number']
```




    5018    NaN
    6736    NaN
    7922    NaN
    9905    NaN
    10628   NaN
    13236   NaN
    13237   NaN
    13238   NaN
    13239   NaN
    13240   NaN
    13241   NaN
    13242   NaN
    13243   NaN
    13244   NaN
    13245   NaN
    13246   NaN
    13247   NaN
    13248   NaN
    13249   NaN
    13250   NaN
    13251   NaN
    13252   NaN
    13253   NaN
    13254   NaN
    13255   NaN
    13256   NaN
    13257   NaN
    13258   NaN
    13259   NaN
    13260   NaN
    13261   NaN
    13262   NaN
    13263   NaN
    13264   NaN
    13265   NaN
    13266   NaN
    13267   NaN
    13268   NaN
    13269   NaN
    13270   NaN
    13271   NaN
    13272   NaN
    13273   NaN
    13274   NaN
    13275   NaN
    13276   NaN
    13277   NaN
    13278   NaN
    13279   NaN
    13280   NaN
    13281   NaN
    13282   NaN
    13283   NaN
    16450   NaN
    16539   NaN
    16793   NaN
    17129   NaN
    17339   NaN
    17436   NaN
    17539   NaN
    Name: Jersey Number, dtype: float64




```python
data_num_Jersey_Number = data_num[['Jersey Number']]
data_num_Jersey_Number.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Jersey Number</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.impute import SimpleImputer
from sklearn.impute import MissingIndicator
```


```python
indicator = MissingIndicator()
mask_missing_values_only = indicator.fit_transform(data_num_Jersey_Number)
mask_missing_values_only
```




    array([[False],
           [False],
           [False],
           ...,
           [False],
           [False],
           [False]])




```python
strategies=['mean', 'median','most_frequent']
```


```python
def test_num_impute(strategy_param):
    imp_num = SimpleImputer(strategy=strategy_param)
    data_num_imp = imp_num.fit_transform(data_num_Jersey_Number)
    return data_num_imp[mask_missing_values_only]
```


```python
strategies[0], test_num_impute(strategies[0])
```




    ('mean',
     array([19.54609577, 19.54609577, 19.54609577, 19.54609577, 19.54609577,
            19.54609577, 19.54609577, 19.54609577, 19.54609577, 19.54609577,
            19.54609577, 19.54609577, 19.54609577, 19.54609577, 19.54609577,
            19.54609577, 19.54609577, 19.54609577, 19.54609577, 19.54609577,
            19.54609577, 19.54609577, 19.54609577, 19.54609577, 19.54609577,
            19.54609577, 19.54609577, 19.54609577, 19.54609577, 19.54609577,
            19.54609577, 19.54609577, 19.54609577, 19.54609577, 19.54609577,
            19.54609577, 19.54609577, 19.54609577, 19.54609577, 19.54609577,
            19.54609577, 19.54609577, 19.54609577, 19.54609577, 19.54609577,
            19.54609577, 19.54609577, 19.54609577, 19.54609577, 19.54609577,
            19.54609577, 19.54609577, 19.54609577, 19.54609577, 19.54609577,
            19.54609577, 19.54609577, 19.54609577, 19.54609577, 19.54609577]))




```python
strategies[1], test_num_impute(strategies[1])
```




    ('median',
     array([17., 17., 17., 17., 17., 17., 17., 17., 17., 17., 17., 17., 17.,
            17., 17., 17., 17., 17., 17., 17., 17., 17., 17., 17., 17., 17.,
            17., 17., 17., 17., 17., 17., 17., 17., 17., 17., 17., 17., 17.,
            17., 17., 17., 17., 17., 17., 17., 17., 17., 17., 17., 17., 17.,
            17., 17., 17., 17., 17., 17., 17., 17.]))




```python
strategies[2], test_num_impute(strategies[2])
```




    ('most_frequent',
     array([8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8.,
            8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8.,
            8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8., 8.,
            8., 8., 8., 8., 8., 8., 8., 8., 8.]))



### Обработка пропусков в категориальных данных


```python
cat_cols = []
for col in data.columns:
    # Количество пустых значений 
    temp_null_count = data[data[col].isnull()].shape[0]
    dt = str(data[col].dtype)
    if temp_null_count>0 and (dt=='object'):
        cat_cols.append(col)
        temp_perc = round((temp_null_count / total_count) * 100.0, 2)
        print('Колонка {}. Тип данных {}. Количество пустых значений {}, {}%.'.format(col, dt, temp_null_count, temp_perc))
```

    Колонка Club. Тип данных object. Количество пустых значений 241, 1.32%.
    Колонка Preferred Foot. Тип данных object. Количество пустых значений 48, 0.26%.
    Колонка Work Rate. Тип данных object. Количество пустых значений 48, 0.26%.
    Колонка Body Type. Тип данных object. Количество пустых значений 48, 0.26%.
    Колонка Real Face. Тип данных object. Количество пустых значений 48, 0.26%.
    Колонка Position. Тип данных object. Количество пустых значений 60, 0.33%.
    Колонка Joined. Тип данных object. Количество пустых значений 1553, 8.53%.
    Колонка Loaned From. Тип данных object. Количество пустых значений 16943, 93.06%.
    Колонка Contract Valid Until. Тип данных object. Количество пустых значений 289, 1.59%.
    Колонка Height. Тип данных object. Количество пустых значений 48, 0.26%.
    Колонка Weight. Тип данных object. Количество пустых значений 48, 0.26%.
    Колонка LS. Тип данных object. Количество пустых значений 2085, 11.45%.
    Колонка ST. Тип данных object. Количество пустых значений 2085, 11.45%.
    Колонка RS. Тип данных object. Количество пустых значений 2085, 11.45%.
    Колонка LW. Тип данных object. Количество пустых значений 2085, 11.45%.
    Колонка LF. Тип данных object. Количество пустых значений 2085, 11.45%.
    Колонка CF. Тип данных object. Количество пустых значений 2085, 11.45%.
    Колонка RF. Тип данных object. Количество пустых значений 2085, 11.45%.
    Колонка RW. Тип данных object. Количество пустых значений 2085, 11.45%.
    Колонка LAM. Тип данных object. Количество пустых значений 2085, 11.45%.
    Колонка CAM. Тип данных object. Количество пустых значений 2085, 11.45%.
    Колонка RAM. Тип данных object. Количество пустых значений 2085, 11.45%.
    Колонка LM. Тип данных object. Количество пустых значений 2085, 11.45%.
    Колонка LCM. Тип данных object. Количество пустых значений 2085, 11.45%.
    Колонка CM. Тип данных object. Количество пустых значений 2085, 11.45%.
    Колонка RCM. Тип данных object. Количество пустых значений 2085, 11.45%.
    Колонка RM. Тип данных object. Количество пустых значений 2085, 11.45%.
    Колонка LWB. Тип данных object. Количество пустых значений 2085, 11.45%.
    Колонка LDM. Тип данных object. Количество пустых значений 2085, 11.45%.
    Колонка CDM. Тип данных object. Количество пустых значений 2085, 11.45%.
    Колонка RDM. Тип данных object. Количество пустых значений 2085, 11.45%.
    Колонка RWB. Тип данных object. Количество пустых значений 2085, 11.45%.
    Колонка LB. Тип данных object. Количество пустых значений 2085, 11.45%.
    Колонка LCB. Тип данных object. Количество пустых значений 2085, 11.45%.
    Колонка CB. Тип данных object. Количество пустых значений 2085, 11.45%.
    Колонка RCB. Тип данных object. Количество пустых значений 2085, 11.45%.
    Колонка RB. Тип данных object. Количество пустых значений 2085, 11.45%.
    Колонка Release Clause. Тип данных object. Количество пустых значений 1564, 8.59%.
    


```python
cat_temp_data = data[['Preferred Foot']]
cat_temp_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Preferred Foot</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Left</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Right</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Right</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Right</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Right</td>
    </tr>
  </tbody>
</table>
</div>




```python
cat_temp_data['Preferred Foot'].unique()
```




    array(['Left', 'Right', nan], dtype=object)




```python
cat_temp_data[cat_temp_data['Preferred Foot'].isnull()].shape
```




    (48, 1)




```python
imp2 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
data_imp2 = imp2.fit_transform(cat_temp_data)
data_imp2
```




    array([['Left'],
           ['Right'],
           ['Right'],
           ...,
           ['Right'],
           ['Right'],
           ['Right']], dtype=object)




```python
np.unique(data_imp2)
```




    array(['Left', 'Right'], dtype=object)




```python
imp3 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='!!!')
data_imp3 = imp3.fit_transform(cat_temp_data)
data_imp3
```




    array([['Left'],
           ['Right'],
           ['Right'],
           ...,
           ['Right'],
           ['Right'],
           ['Right']], dtype=object)




```python
np.unique(data_imp3)
```




    array(['!!!', 'Left', 'Right'], dtype=object)




```python
data_imp3[data_imp3=='!!!'].size
```




    48



### Преобразование категориальных признаков в числовые


```python
cat_enc = pd.DataFrame({'c1':data_imp2.T[0]})
cat_enc
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>c1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Left</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Right</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Right</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Right</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Right</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>18202</th>
      <td>Right</td>
    </tr>
    <tr>
      <th>18203</th>
      <td>Right</td>
    </tr>
    <tr>
      <th>18204</th>
      <td>Right</td>
    </tr>
    <tr>
      <th>18205</th>
      <td>Right</td>
    </tr>
    <tr>
      <th>18206</th>
      <td>Right</td>
    </tr>
  </tbody>
</table>
<p>18207 rows × 1 columns</p>
</div>



### label encoding


```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
```


```python
le = LabelEncoder()
cat_enc_le = le.fit_transform(cat_enc['c1'])
```


```python
cat_enc['c1'].unique()
```




    array(['Left', 'Right'], dtype=object)




```python
np.unique(cat_enc_le)
```




    array([0, 1])




```python
le.inverse_transform([0, 1])
```




    array(['Left', 'Right'], dtype=object)



### one-hot encoding


```python
ohe = OneHotEncoder()
cat_enc_ohe = ohe.fit_transform(cat_enc[['c1']])
```


```python
cat_enc.shape
```




    (18207, 1)




```python
cat_enc_ohe.shape
```




    (18207, 2)




```python
cat_enc_ohe
```




    <18207x2 sparse matrix of type '<class 'numpy.float64'>'
    	with 18207 stored elements in Compressed Sparse Row format>




```python
cat_enc_ohe.todense()[0:10]
```




    matrix([[1., 0.],
            [0., 1.],
            [0., 1.],
            [0., 1.],
            [0., 1.],
            [0., 1.],
            [0., 1.],
            [0., 1.],
            [0., 1.],
            [0., 1.]])




```python
cat_enc.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>c1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Left</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Right</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Right</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Right</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Right</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Right</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Right</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Right</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Right</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Right</td>
    </tr>
  </tbody>
</table>
</div>



## Масштабирование данных


```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
```

### MinMax масштабирование


```python
sc1 = MinMaxScaler()
sc1_data = sc1.fit_transform(data[['Age']])
```


```python
plt.hist(data['Age'], 50)
plt.show()
```


![png](output_58_0.png)



```python
plt.hist(sc1_data, 50)
plt.show()
```


![png](output_59_0.png)


### Масштабирование данных на основе Z-оценки


```python
sc2 = StandardScaler()
sc2_data = sc2.fit_transform(data[['Age']])
```


```python
plt.hist(sc2_data, 50)
plt.show()
```


![png](output_62_0.png)


### Нормализация данных


```python
sc3 = Normalizer()
sc3_data = sc3.fit_transform(data[['Age']])
```


```python
plt.hist(sc3_data, 50)
plt.show()
```


![png](output_65_0.png)


## Violin plot


```python
sns.violinplot(x=data['Age'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1e268785888>




![png](output_67_1.png)


Для количественных признаков использовался метод импутации. Для категориальных признаков использовались методы label encoding и one-hot encoding. Для дальнейшего построения ММО буду использовать метод Pandas get_dummies для категориальных признаков и импьютацию для числовых, так как pandas get dummies является быстрым вариантом one-hot кодирования, а импьютация меньше влияет на данные в целом и не изменит размер датасета по сравнению с удалением или заполнением нулями. Для масштабирования были использованы MinMax масштабирование и масштабирование данных на основе Z-оценки.


```python

```
