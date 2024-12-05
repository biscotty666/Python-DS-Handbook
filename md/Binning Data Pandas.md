# Binning Data in Pandas

## `pd.cut`

>Bin values into discrete intervals.
>
>Use `cut` when you need to segment and sort data values into bins. This
function is also useful for going from a continuous variable to a
categorical variable. For example, `cut` could convert ages to groups of
age ranges. Supports binning into an equal number of bins, or a
pre-specified array of bins.


```python
import pandas as pd 
import numpy as np 

arr = np.array([1,7,5,4,6,3])
pd.cut(arr, 3)
```




    [(0.994, 3.0], (5.0, 7.0], (3.0, 5.0], (3.0, 5.0], (5.0, 7.0], (0.994, 3.0]]
    Categories (3, interval[float64, right]): [(0.994, 3.0] < (3.0, 5.0] < (5.0, 7.0]]




```python
binned = pd.cut(arr, 3, retbins=True)
binned
```




    ([(0.994, 3.0], (5.0, 7.0], (3.0, 5.0], (3.0, 5.0], (5.0, 7.0], (0.994, 3.0]]
     Categories (3, interval[float64, right]): [(0.994, 3.0] < (3.0, 5.0] < (5.0, 7.0]],
     array([0.994, 3.   , 5.   , 7.   ]))




```python
binned[1]
```




    array([0.994, 3.   , 5.   , 7.   ])




```python
binned[0]
```




    [(0.994, 3.0], (5.0, 7.0], (3.0, 5.0], (3.0, 5.0], (5.0, 7.0], (0.994, 3.0]]
    Categories (3, interval[float64, right]): [(0.994, 3.0] < (3.0, 5.0] < (5.0, 7.0]]




```python
%matplotlib inline
import matplotlib.pyplot as plt 
plt.style.use('seaborn-v0_8-darkgrid')
```


```python
raw_df = pd.read_excel("data/2018_Sales_Total.xlsx")
raw_df.head()
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
      <th>account number</th>
      <th>name</th>
      <th>sku</th>
      <th>quantity</th>
      <th>unit price</th>
      <th>ext price</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>740150</td>
      <td>Barton LLC</td>
      <td>B1-20000</td>
      <td>39</td>
      <td>86.69</td>
      <td>3380.91</td>
      <td>2018-01-01 07:21:51</td>
    </tr>
    <tr>
      <th>1</th>
      <td>714466</td>
      <td>Trantow-Barrows</td>
      <td>S2-77896</td>
      <td>-1</td>
      <td>63.16</td>
      <td>-63.16</td>
      <td>2018-01-01 10:00:47</td>
    </tr>
    <tr>
      <th>2</th>
      <td>218895</td>
      <td>Kulas Inc</td>
      <td>B1-69924</td>
      <td>23</td>
      <td>90.70</td>
      <td>2086.10</td>
      <td>2018-01-01 13:24:58</td>
    </tr>
    <tr>
      <th>3</th>
      <td>307599</td>
      <td>Kassulke, Ondricka and Metz</td>
      <td>S1-65481</td>
      <td>41</td>
      <td>21.05</td>
      <td>863.05</td>
      <td>2018-01-01 15:05:22</td>
    </tr>
    <tr>
      <th>4</th>
      <td>412290</td>
      <td>Jerde-Hilpert</td>
      <td>S2-34077</td>
      <td>6</td>
      <td>83.21</td>
      <td>499.26</td>
      <td>2018-01-01 23:26:55</td>
    </tr>
  </tbody>
</table>
</div>



## Manual Binning


```python
df = raw_df.groupby(['account number', 'name'])['ext price'].sum().reset_index()
df.head()
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
      <th>account number</th>
      <th>name</th>
      <th>ext price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>141962</td>
      <td>Herman LLC</td>
      <td>82865.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>146832</td>
      <td>Kiehn-Spinka</td>
      <td>99608.77</td>
    </tr>
    <tr>
      <th>2</th>
      <td>163416</td>
      <td>Purdy-Kunde</td>
      <td>77898.21</td>
    </tr>
    <tr>
      <th>3</th>
      <td>218895</td>
      <td>Kulas Inc</td>
      <td>137351.96</td>
    </tr>
    <tr>
      <th>4</th>
      <td>239344</td>
      <td>Stokes LLC</td>
      <td>91535.92</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['ext price'].plot(kind='hist')
```




    <Axes: ylabel='Frequency'>




    
![png](Binning%20Data%20Pandas_files/Binning%20Data%20Pandas_11_1.png)
    


## Binning by quantiles

>Signature:
>pd.qcut(
>    x,
>    q,
>    labels=None,
>    retbins: 'bool' = False,
>    precision: 'int' = 3,
>    duplicates: 'str' = 'raise',
>)
>
>Quantile-based discretization function.>
>
>Discretize variable into equal-sized buckets based on rank or based on sample quantiles. For example 1000 values for 10 quantiles would
>produce a Categorical object indicating quantile membership for each data point. 


```python
df['ext price'].describe()
```




    count        20.000000
    mean     100939.216000
    std       17675.097485
    min       70004.360000
    25%       89137.707500
    50%      100271.535000
    75%      110132.552500
    max      137351.960000
    Name: ext price, dtype: float64




```python
pd.qcut(df['ext price'], q=4)
```




    0       (70004.359, 89137.708]
    1      (89137.708, 100271.535]
    2       (70004.359, 89137.708]
    3      (110132.552, 137351.96]
    4      (89137.708, 100271.535]
    5      (89137.708, 100271.535]
    6       (70004.359, 89137.708]
    7     (100271.535, 110132.552]
    8      (110132.552, 137351.96]
    9      (110132.552, 137351.96]
    10     (89137.708, 100271.535]
    11      (70004.359, 89137.708]
    12      (70004.359, 89137.708]
    13     (89137.708, 100271.535]
    14    (100271.535, 110132.552]
    15     (110132.552, 137351.96]
    16    (100271.535, 110132.552]
    17     (110132.552, 137351.96]
    18    (100271.535, 110132.552]
    19    (100271.535, 110132.552]
    Name: ext price, dtype: category
    Categories (4, interval[float64, right]): [(70004.359, 89137.708] < (89137.708, 100271.535] < (100271.535, 110132.552] < (110132.552, 137351.96]]




```python
df['quantile_ex_1'] = pd.qcut(df['ext price'], q=4)
df['quantile_ex_2'] = pd.qcut(df['ext price'], q=10, precision=0)

df.head()
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
      <th>account number</th>
      <th>name</th>
      <th>ext price</th>
      <th>quantile_ex_1</th>
      <th>quantile_ex_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>141962</td>
      <td>Herman LLC</td>
      <td>82865.00</td>
      <td>(70004.359, 89137.708]</td>
      <td>(82368.0, 87168.0]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>146832</td>
      <td>Kiehn-Spinka</td>
      <td>99608.77</td>
      <td>(89137.708, 100271.535]</td>
      <td>(95908.0, 100272.0]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>163416</td>
      <td>Purdy-Kunde</td>
      <td>77898.21</td>
      <td>(70004.359, 89137.708]</td>
      <td>(70003.0, 82368.0]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>218895</td>
      <td>Kulas Inc</td>
      <td>137351.96</td>
      <td>(110132.552, 137351.96]</td>
      <td>(124627.0, 137352.0]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>239344</td>
      <td>Stokes LLC</td>
      <td>91535.92</td>
      <td>(89137.708, 100271.535]</td>
      <td>(90686.0, 95908.0]</td>
    </tr>
  </tbody>
</table>
</div>



## Counting


```python
df['quantile_ex_1'].value_counts()
```




    quantile_ex_1
    (70004.359, 89137.708]      5
    (89137.708, 100271.535]     5
    (100271.535, 110132.552]    5
    (110132.552, 137351.96]     5
    Name: count, dtype: int64




```python
pd.qcut(df['ext price'], q=4).value_counts()
```




    ext price
    (70004.359, 89137.708]      5
    (89137.708, 100271.535]     5
    (100271.535, 110132.552]    5
    (110132.552, 137351.96]     5
    Name: count, dtype: int64




```python
bin_labels_5 = ['Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond']
```


```python
pd.qcut(df['ext price'], 
        q=[0, .2, .4, .6, .8, 1], 
       labels=bin_labels_5).head()
```




    0     Bronze
    1       Gold
    2     Bronze
    3    Diamond
    4     Silver
    Name: ext price, dtype: category
    Categories (5, object): ['Bronze' < 'Silver' < 'Gold' < 'Platinum' < 'Diamond']




```python
df['medal'] = pd.qcut(df['ext price'], 
        q=[0, .2, .4, .6, .8, 1], 
       labels=bin_labels_5)
df.head()
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
      <th>account number</th>
      <th>name</th>
      <th>ext price</th>
      <th>quantile_ex_1</th>
      <th>quantile_ex_2</th>
      <th>medal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>141962</td>
      <td>Herman LLC</td>
      <td>82865.00</td>
      <td>(70004.359, 89137.708]</td>
      <td>(82368.0, 87168.0]</td>
      <td>Bronze</td>
    </tr>
    <tr>
      <th>1</th>
      <td>146832</td>
      <td>Kiehn-Spinka</td>
      <td>99608.77</td>
      <td>(89137.708, 100271.535]</td>
      <td>(95908.0, 100272.0]</td>
      <td>Gold</td>
    </tr>
    <tr>
      <th>2</th>
      <td>163416</td>
      <td>Purdy-Kunde</td>
      <td>77898.21</td>
      <td>(70004.359, 89137.708]</td>
      <td>(70003.0, 82368.0]</td>
      <td>Bronze</td>
    </tr>
    <tr>
      <th>3</th>
      <td>218895</td>
      <td>Kulas Inc</td>
      <td>137351.96</td>
      <td>(110132.552, 137351.96]</td>
      <td>(124627.0, 137352.0]</td>
      <td>Diamond</td>
    </tr>
    <tr>
      <th>4</th>
      <td>239344</td>
      <td>Stokes LLC</td>
      <td>91535.92</td>
      <td>(89137.708, 100271.535]</td>
      <td>(90686.0, 95908.0]</td>
      <td>Silver</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['medal'].value_counts()
```




    medal
    Bronze      4
    Silver      4
    Gold        4
    Platinum    4
    Diamond     4
    Name: count, dtype: int64



## Return bin ranges
results, bin_edges = pd.qcut(df['ext price'], 
                             q=[0, .2, .4, .6, .8, 1], 
                             labels=bin_labels_5,
                             retbins=True
                        )

```python
results_table = pd.DataFrame(zip(bin_edges, bin_labels_5),
                            columns=['Threshold', 'Tier'])
results_table
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
      <th>Threshold</th>
      <th>Tier</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>70004.360</td>
      <td>Bronze</td>
    </tr>
    <tr>
      <th>1</th>
      <td>87167.958</td>
      <td>Silver</td>
    </tr>
    <tr>
      <th>2</th>
      <td>95908.156</td>
      <td>Gold</td>
    </tr>
    <tr>
      <th>3</th>
      <td>103605.970</td>
      <td>Platinum</td>
    </tr>
    <tr>
      <th>4</th>
      <td>112290.054</td>
      <td>Diamond</td>
    </tr>
  </tbody>
</table>
</div>




```python
results, bins = pd.qcut(df['ext price'], 
                             q=[0, .2, .4, .6, .8, 1], 
                             labels=bin_labels_5,
                             retbins=True
                        )
```


```python
df.describe(include='category')
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
      <th>quantile_ex_1</th>
      <th>quantile_ex_2</th>
      <th>medal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>20</td>
      <td>20</td>
      <td>20</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>4</td>
      <td>10</td>
      <td>5</td>
    </tr>
    <tr>
      <th>top</th>
      <td>(70004.359, 89137.708]</td>
      <td>(70003.0, 82368.0]</td>
      <td>Bronze</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>5</td>
      <td>2</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe(percentiles=[0, 1/3, 2/3, 1])
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
      <th>account number</th>
      <th>ext price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>20.000000</td>
      <td>20.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>476998.750000</td>
      <td>100939.216000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>231499.208970</td>
      <td>17675.097485</td>
    </tr>
    <tr>
      <th>min</th>
      <td>141962.000000</td>
      <td>70004.360000</td>
    </tr>
    <tr>
      <th>0%</th>
      <td>141962.000000</td>
      <td>70004.360000</td>
    </tr>
    <tr>
      <th>33.3%</th>
      <td>332759.333333</td>
      <td>91241.493333</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>476006.500000</td>
      <td>100271.535000</td>
    </tr>
    <tr>
      <th>66.7%</th>
      <td>662511.000000</td>
      <td>104178.580000</td>
    </tr>
    <tr>
      <th>100%</th>
      <td>786968.000000</td>
      <td>137351.960000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>786968.000000</td>
      <td>137351.960000</td>
    </tr>
  </tbody>
</table>
</div>



# `cut`

>For distributing (unevenly) instances with arbitrary boundaries


```python
df = df.drop(columns=['medal', 'quantile_ex_1', 'quantile_ex_2'])
```


```python
pd.cut(df['ext price'], bins=4)
```




    0      (69937.012, 86841.26]
    1      (86841.26, 103678.16]
    2      (69937.012, 86841.26]
    3     (120515.06, 137351.96]
    4      (86841.26, 103678.16]
    5      (86841.26, 103678.16]
    6      (69937.012, 86841.26]
    7     (103678.16, 120515.06]
    8     (103678.16, 120515.06]
    9     (120515.06, 137351.96]
    10     (86841.26, 103678.16]
    11     (69937.012, 86841.26]
    12     (86841.26, 103678.16]
    13     (86841.26, 103678.16]
    14     (86841.26, 103678.16]
    15    (120515.06, 137351.96]
    16     (86841.26, 103678.16]
    17    (103678.16, 120515.06]
    18    (103678.16, 120515.06]
    19     (86841.26, 103678.16]
    Name: ext price, dtype: category
    Categories (4, interval[float64, right]): [(69937.012, 86841.26] < (86841.26, 103678.16] < (103678.16, 120515.06] < (120515.06, 137351.96]]




```python
pd.cut(df['ext price'], bins=4).value_counts()
```




    ext price
    (86841.26, 103678.16]     9
    (69937.012, 86841.26]     4
    (103678.16, 120515.06]    4
    (120515.06, 137351.96]    3
    Name: count, dtype: int64




```python
pd.cut(df['ext price'], bins=np.linspace(40000, 120000, 6)).value_counts()
```




    ext price
    (88000.0, 104000.0]     8
    (104000.0, 120000.0]    4
    (72000.0, 88000.0]      4
    (56000.0, 72000.0]      1
    (40000.0, 56000.0]      0
    Name: count, dtype: int64




```python
pd.interval_range(start=0, freq=10000, end=200000, closed='left')
```




    IntervalIndex([      [0, 10000),   [10000, 20000),   [20000, 30000),
                     [30000, 40000),   [40000, 50000),   [50000, 60000),
                     [60000, 70000),   [70000, 80000),   [80000, 90000),
                    [90000, 100000), [100000, 110000), [110000, 120000),
                   [120000, 130000), [130000, 140000), [140000, 150000),
                   [150000, 160000), [160000, 170000), [170000, 180000),
                   [180000, 190000), [190000, 200000)],
                  dtype='interval[int64, left]')



# Binning shortcut for `value_counts()`


```python
df['ext price'].value_counts(bins=4, sort=False)
```




    (69937.011, 86841.26]     4
    (86841.26, 103678.16]     9
    (103678.16, 120515.06]    4
    (120515.06, 137351.96]    3
    Name: count, dtype: int64




```python

```
