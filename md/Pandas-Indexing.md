# Pandas Multiple or Heirarchical Indexing


```python
import pandas as pd 
import numpy as np 
```


```python
index = [('California', 2000), ('California', 2010),
         ('New York', 2000), ('New York', 2010),
         ('Texas', 2000), ('Texas', 2010)]
populations = [33871648, 37253956,
               18976457, 19378102,
               20851820, 25145561]
pop = pd.Series(populations, index=index)
pop
```




    (California, 2000)    33871648
    (California, 2010)    37253956
    (New York, 2000)      18976457
    (New York, 2010)      19378102
    (Texas, 2000)         20851820
    (Texas, 2010)         25145561
    dtype: int64




```python
index = pd.MultiIndex.from_tuples(index)
```


```python
pop.reindex(index)
```




    California  2000    33871648
                2010    37253956
    New York    2000    18976457
                2010    19378102
    Texas       2000    20851820
                2010    25145561
    dtype: int64




```python
pop = pd.Series(populations, index=index)
```


```python
pop[:, 2000]
```




    California    33871648
    New York      18976457
    Texas         20851820
    dtype: int64




```python
pop_df = pop.unstack()
pop_df
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
      <th>2000</th>
      <th>2010</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>California</th>
      <td>33871648</td>
      <td>37253956</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>18976457</td>
      <td>19378102</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>20851820</td>
      <td>25145561</td>
    </tr>
  </tbody>
</table>
</div>



> Convert Multiply indexed `Series` into a conventionally indexed `DataFrame`.


```python
pop_df.stack()
```




    California  2000    33871648
                2010    37253956
    New York    2000    18976457
                2010    19378102
    Texas       2000    20851820
                2010    25145561
    dtype: int64



>Add a column


```python
pop_df = pd.DataFrame({
    'total': pop,
    'under18': [9284094, 8898092,
                4318033, 4181528,
                6879014, 7432474]
})
pop_df
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
      <th></th>
      <th>total</th>
      <th>under18</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">California</th>
      <th>2000</th>
      <td>33871648</td>
      <td>9284094</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>37253956</td>
      <td>8898092</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">New York</th>
      <th>2000</th>
      <td>18976457</td>
      <td>4318033</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>19378102</td>
      <td>4181528</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Texas</th>
      <th>2000</th>
      <td>20851820</td>
      <td>6879014</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>25145561</td>
      <td>7432474</td>
    </tr>
  </tbody>
</table>
</div>




```python
f_u18 = pop_df['under18'] / pop_df['total']
f_u18
```




    California  2000    0.274096
                2010    0.238850
    New York    2000    0.227547
                2010    0.215786
    Texas       2000    0.329900
                2010    0.295578
    dtype: float64




```python
f_u18.unstack()
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
      <th>2000</th>
      <th>2010</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>California</th>
      <td>0.274096</td>
      <td>0.238850</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>0.227547</td>
      <td>0.215786</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>0.329900</td>
      <td>0.295578</td>
    </tr>
  </tbody>
</table>
</div>



# Multiindex Creation


```python
df = pd.DataFrame(np.random.rand(4,2),
                  index=[['a','a','b','b'],[1,2,1,2]], 
                  columns=['data1', 'data2'])
df
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
      <th></th>
      <th>data1</th>
      <th>data2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">a</th>
      <th>1</th>
      <td>0.653224</td>
      <td>0.249848</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.333277</td>
      <td>0.108866</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">b</th>
      <th>1</th>
      <td>0.250907</td>
      <td>0.088180</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.304413</td>
      <td>0.792930</td>
    </tr>
  </tbody>
</table>
</div>



> A dictionary with tuples as keys


```python
data = {('California', 2010): 37253956,
        ('California', 2020): 39538223,
        ('New York', 2010): 19378102,
        ('New York', 2020): 20201249,
        ('Texas', 2010): 25145561,
        ('Texas', 2020): 29145505}
pd.Series(data)
```




    California  2010    37253956
                2020    39538223
    New York    2010    19378102
                2020    20201249
    Texas       2010    25145561
                2020    29145505
    dtype: int64



>Explicit multiindex constructors


```python
pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b'], [1, 2, 1, 2]])
```




    MultiIndex([('a', 1),
                ('a', 2),
                ('b', 1),
                ('b', 2)],
               )




```python
pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1), ('b', 2)])
```




    MultiIndex([('a', 1),
                ('a', 2),
                ('b', 1),
                ('b', 2)],
               )




```python
pd.MultiIndex.from_product([['a', 'b'], [1, 2]])
```




    MultiIndex([('a', 1),
                ('a', 2),
                ('b', 1),
                ('b', 2)],
               )




```python
pd.MultiIndex(levels=[['a', 'b'], [1, 2]],
              codes=[[0, 0, 1, 1], [0, 1, 0, 1]])
```




    MultiIndex([('a', 1),
                ('a', 2),
                ('b', 1),
                ('b', 2)],
               )



>Level Names


```python
pop
```




    California  2000    33871648
                2010    37253956
    New York    2000    18976457
                2010    19378102
    Texas       2000    20851820
                2010    25145561
    dtype: int64




```python
pop.index.names = ['state', 'year']
pop
```




    state       year
    California  2000    33871648
                2010    37253956
    New York    2000    18976457
                2010    19378102
    Texas       2000    20851820
                2010    25145561
    dtype: int64



## Multiindex for Columns


```python
index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]],
                                   names=['year', 'visit'])
columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'], ['HR', 'Temp']],
                                     names=['subject', 'type'])
```


```python
data = np.round(np.random.randn(4,6), 1)
data[:,::2] *= 10

data += 37
data

health_data = pd.DataFrame(data, index=index, columns=columns)
health_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>subject</th>
      <th colspan="2" halign="left">Bob</th>
      <th colspan="2" halign="left">Guido</th>
      <th colspan="2" halign="left">Sue</th>
    </tr>
    <tr>
      <th></th>
      <th>type</th>
      <th>HR</th>
      <th>Temp</th>
      <th>HR</th>
      <th>Temp</th>
      <th>HR</th>
      <th>Temp</th>
    </tr>
    <tr>
      <th>year</th>
      <th>visit</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">2013</th>
      <th>1</th>
      <td>27.0</td>
      <td>37.9</td>
      <td>29.0</td>
      <td>38.2</td>
      <td>34.0</td>
      <td>35.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>49.0</td>
      <td>36.7</td>
      <td>38.0</td>
      <td>38.6</td>
      <td>29.0</td>
      <td>36.2</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">2014</th>
      <th>1</th>
      <td>48.0</td>
      <td>35.2</td>
      <td>33.0</td>
      <td>39.0</td>
      <td>34.0</td>
      <td>38.9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38.0</td>
      <td>38.1</td>
      <td>51.0</td>
      <td>38.6</td>
      <td>30.0</td>
      <td>38.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
health_data['Guido']
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
      <th>type</th>
      <th>HR</th>
      <th>Temp</th>
    </tr>
    <tr>
      <th>year</th>
      <th>visit</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">2013</th>
      <th>1</th>
      <td>29.0</td>
      <td>38.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38.0</td>
      <td>38.6</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">2014</th>
      <th>1</th>
      <td>33.0</td>
      <td>39.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>51.0</td>
      <td>38.6</td>
    </tr>
  </tbody>
</table>
</div>




```python
health_data.loc[2013]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>subject</th>
      <th colspan="2" halign="left">Bob</th>
      <th colspan="2" halign="left">Guido</th>
      <th colspan="2" halign="left">Sue</th>
    </tr>
    <tr>
      <th>type</th>
      <th>HR</th>
      <th>Temp</th>
      <th>HR</th>
      <th>Temp</th>
      <th>HR</th>
      <th>Temp</th>
    </tr>
    <tr>
      <th>visit</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>27.0</td>
      <td>37.9</td>
      <td>29.0</td>
      <td>38.2</td>
      <td>34.0</td>
      <td>35.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>49.0</td>
      <td>36.7</td>
      <td>38.0</td>
      <td>38.6</td>
      <td>29.0</td>
      <td>36.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
health_data['Guido','HR']
```




    year  visit
    2013  1        29.0
          2        38.0
    2014  1        33.0
          2        51.0
    Name: (Guido, HR), dtype: float64




```python
health_data.iloc[:2,1:2]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>subject</th>
      <th>Bob</th>
    </tr>
    <tr>
      <th></th>
      <th>type</th>
      <th>Temp</th>
    </tr>
    <tr>
      <th>year</th>
      <th>visit</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">2013</th>
      <th>1</th>
      <td>37.9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>36.7</td>
    </tr>
  </tbody>
</table>
</div>




```python
health_data.iloc[:2]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>subject</th>
      <th colspan="2" halign="left">Bob</th>
      <th colspan="2" halign="left">Guido</th>
      <th colspan="2" halign="left">Sue</th>
    </tr>
    <tr>
      <th></th>
      <th>type</th>
      <th>HR</th>
      <th>Temp</th>
      <th>HR</th>
      <th>Temp</th>
      <th>HR</th>
      <th>Temp</th>
    </tr>
    <tr>
      <th>year</th>
      <th>visit</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">2013</th>
      <th>1</th>
      <td>27.0</td>
      <td>37.9</td>
      <td>29.0</td>
      <td>38.2</td>
      <td>34.0</td>
      <td>35.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>49.0</td>
      <td>36.7</td>
      <td>38.0</td>
      <td>38.6</td>
      <td>29.0</td>
      <td>36.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
health_data.iloc[:2,2:3]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>subject</th>
      <th>Guido</th>
    </tr>
    <tr>
      <th></th>
      <th>type</th>
      <th>HR</th>
    </tr>
    <tr>
      <th>year</th>
      <th>visit</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">2013</th>
      <th>1</th>
      <td>29.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Fail 
# health_data.loc[(:, 1), (:, 'HR')]
```


```python
idx = pd.IndexSlice
health_data.loc[idx[:, 1], idx[:, 'HR']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>subject</th>
      <th>Bob</th>
      <th>Guido</th>
      <th>Sue</th>
    </tr>
    <tr>
      <th></th>
      <th>type</th>
      <th>HR</th>
      <th>HR</th>
      <th>HR</th>
    </tr>
    <tr>
      <th>year</th>
      <th>visit</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013</th>
      <th>1</th>
      <td>27.0</td>
      <td>29.0</td>
      <td>34.0</td>
    </tr>
    <tr>
      <th>2014</th>
      <th>1</th>
      <td>48.0</td>
      <td>33.0</td>
      <td>34.0</td>
    </tr>
  </tbody>
</table>
</div>



## Rearranging multi-indexes


```python
index = pd.MultiIndex.from_product([['a', 'c', 'b'], [1, 2]])
data = pd.Series(np.random.rand(6), index=index)
data.index.names = ['char', 'int']
data
```




    char  int
    a     1      0.372084
          2      0.592588
    c     1      0.112209
          2      0.565792
    b     1      0.464795
          2      0.449224
    dtype: float64




```python
data.sort_index()
```




    char  int
    a     1      0.372084
          2      0.592588
    b     1      0.464795
          2      0.449224
    c     1      0.112209
          2      0.565792
    dtype: float64




```python
data.sort_index()['a':'b']
```




    char  int
    a     1      0.372084
          2      0.592588
    b     1      0.464795
          2      0.449224
    dtype: float64



## Stacking and unstacking


```python
pop.unstack(level=0)
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
      <th>state</th>
      <th>California</th>
      <th>New York</th>
      <th>Texas</th>
    </tr>
    <tr>
      <th>year</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000</th>
      <td>33871648</td>
      <td>18976457</td>
      <td>20851820</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>37253956</td>
      <td>19378102</td>
      <td>25145561</td>
    </tr>
  </tbody>
</table>
</div>




```python
pop.unstack(level=1)
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
      <th>year</th>
      <th>2000</th>
      <th>2010</th>
    </tr>
    <tr>
      <th>state</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>California</th>
      <td>33871648</td>
      <td>37253956</td>
    </tr>
    <tr>
      <th>New York</th>
      <td>18976457</td>
      <td>19378102</td>
    </tr>
    <tr>
      <th>Texas</th>
      <td>20851820</td>
      <td>25145561</td>
    </tr>
  </tbody>
</table>
</div>




```python
pop.unstack().stack()
```




    state       year
    California  2000    33871648
                2010    37253956
    New York    2000    18976457
                2010    19378102
    Texas       2000    20851820
                2010    25145561
    dtype: int64



# Index setting and resetting


```python
pop
```




    state       year
    California  2000    33871648
                2010    37253956
    New York    2000    18976457
                2010    19378102
    Texas       2000    20851820
                2010    25145561
    dtype: int64




```python
pop_flat = pop.reset_index(name='population')
pop_flat
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
      <th>state</th>
      <th>year</th>
      <th>population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>California</td>
      <td>2000</td>
      <td>33871648</td>
    </tr>
    <tr>
      <th>1</th>
      <td>California</td>
      <td>2010</td>
      <td>37253956</td>
    </tr>
    <tr>
      <th>2</th>
      <td>New York</td>
      <td>2000</td>
      <td>18976457</td>
    </tr>
    <tr>
      <th>3</th>
      <td>New York</td>
      <td>2010</td>
      <td>19378102</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Texas</td>
      <td>2000</td>
      <td>20851820</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Texas</td>
      <td>2010</td>
      <td>25145561</td>
    </tr>
  </tbody>
</table>
</div>




```python
pop_flat.set_index(['state','year'])
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
      <th></th>
      <th>population</th>
    </tr>
    <tr>
      <th>state</th>
      <th>year</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">California</th>
      <th>2000</th>
      <td>33871648</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>37253956</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">New York</th>
      <th>2000</th>
      <td>18976457</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>19378102</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Texas</th>
      <th>2000</th>
      <td>20851820</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>25145561</td>
    </tr>
  </tbody>
</table>
</div>




```python

```