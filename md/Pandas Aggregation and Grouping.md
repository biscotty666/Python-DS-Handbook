# Aggregation and Grouping

>`groupby`


```python
import numpy as np
import pandas as pd

class display(object):
    """Display HTML representation of multiple objects"""
    template = """<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>{0}</p>{1}
    </div>"""
    def __init__(self, *args):
        self.args = args
        
    def _repr_html_(self):
        return '\n'.join(self.template.format(a, eval(a)._repr_html_())
                         for a in self.args)
    
    def __repr__(self):
        return '\n\n'.join(a + '\n' + repr(eval(a))
                           for a in self.args)
```

## Planets data

Contains information on extrasolar planets


```python
import seaborn as sns 
planets = sns.load_dataset('planets')
planets.shape
```




    (1035, 6)




```python
planets.head()
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
      <th>method</th>
      <th>number</th>
      <th>orbital_period</th>
      <th>mass</th>
      <th>distance</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Radial Velocity</td>
      <td>1</td>
      <td>269.300</td>
      <td>7.10</td>
      <td>77.40</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Radial Velocity</td>
      <td>1</td>
      <td>874.774</td>
      <td>2.21</td>
      <td>56.95</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Radial Velocity</td>
      <td>1</td>
      <td>763.000</td>
      <td>2.60</td>
      <td>19.84</td>
      <td>2011</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Radial Velocity</td>
      <td>1</td>
      <td>326.030</td>
      <td>19.40</td>
      <td>110.62</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Radial Velocity</td>
      <td>1</td>
      <td>516.220</td>
      <td>10.50</td>
      <td>119.47</td>
      <td>2009</td>
    </tr>
  </tbody>
</table>
</div>




```python
planets.index
```




    RangeIndex(start=0, stop=1035, step=1)




```python
rng = np.random.default_rng(seed=42)
```

## Simple Series Aggregation


```python
ser = pd.Series(rng.random(5))
ser
```




    0    0.773956
    1    0.438878
    2    0.858598
    3    0.697368
    4    0.094177
    dtype: float64




```python
ser.sum(), ser.mean()
```




    (2.8629777851664118, 0.5725955570332824)



## Simple DataFrame Aggregation


```python
df = pd.DataFrame({'A': rng.random(5),
                   'B': rng.random(5)})
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
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.975622</td>
      <td>0.370798</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.761140</td>
      <td>0.926765</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.786064</td>
      <td>0.643865</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.128114</td>
      <td>0.822762</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.450386</td>
      <td>0.443414</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.mean(), df.mean(axis='columns')
```




    (A    0.620265
     B    0.641521
     dtype: float64,
     0    0.673210
     1    0.843952
     2    0.714965
     3    0.475438
     4    0.446900
     dtype: float64)




```python
df.mean(axis=1)
```




    0    0.673210
    1    0.843952
    2    0.714965
    3    0.475438
    4    0.446900
    dtype: float64



## Describe


```python
planets.dropna().describe()
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
      <th>number</th>
      <th>orbital_period</th>
      <th>mass</th>
      <th>distance</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>498.00000</td>
      <td>498.000000</td>
      <td>498.000000</td>
      <td>498.000000</td>
      <td>498.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.73494</td>
      <td>835.778671</td>
      <td>2.509320</td>
      <td>52.068213</td>
      <td>2007.377510</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.17572</td>
      <td>1469.128259</td>
      <td>3.636274</td>
      <td>46.596041</td>
      <td>4.167284</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.00000</td>
      <td>1.328300</td>
      <td>0.003600</td>
      <td>1.350000</td>
      <td>1989.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.00000</td>
      <td>38.272250</td>
      <td>0.212500</td>
      <td>24.497500</td>
      <td>2005.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.00000</td>
      <td>357.000000</td>
      <td>1.245000</td>
      <td>39.940000</td>
      <td>2009.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.00000</td>
      <td>999.600000</td>
      <td>2.867500</td>
      <td>59.332500</td>
      <td>2011.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>6.00000</td>
      <td>17337.500000</td>
      <td>25.000000</td>
      <td>354.000000</td>
      <td>2014.000000</td>
    </tr>
  </tbody>
</table>
</div>



## `groupby`: Split, Apply, Combine

### Example

<img src="./images/03.08-split-apply-combine.png" />


```python
df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
                   'data': range(6)}, columns=['key', 'data'])
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
      <th>key</th>
      <th>data</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>B</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>C</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.groupby('key')
```




    <pandas.core.groupby.generic.DataFrameGroupBy object at 0x7f2ce8905880>




```python
df.groupby('key').sum()
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
      <th>data</th>
    </tr>
    <tr>
      <th>key</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>3</td>
    </tr>
    <tr>
      <th>B</th>
      <td>5</td>
    </tr>
    <tr>
      <th>C</th>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>



# The `GroupBy` Object

## Column Indexing

> DataFrameGroupBy


```python
planets.groupby('method')
```




    <pandas.core.groupby.generic.DataFrameGroupBy object at 0x7f2ce8b10bf0>



>Series GroupBy


```python
planets.groupby('method')['orbital_period']
```




    <pandas.core.groupby.generic.SeriesGroupBy object at 0x7f2ce8b10d10>




```python
planets.groupby('method')['orbital_period'].median()
```




    method
    Astrometry                         631.180000
    Eclipse Timing Variations         4343.500000
    Imaging                          27500.000000
    Microlensing                      3300.000000
    Orbital Brightness Modulation        0.342887
    Pulsar Timing                       66.541900
    Pulsation Timing Variations       1170.000000
    Radial Velocity                    360.200000
    Transit                              5.714932
    Transit Timing Variations           57.011000
    Name: orbital_period, dtype: float64



## Iteration

>Slower than `apply()`


```python
for (method, group) in planets.groupby('method'): 
    print(f"{method:<30} {group.shape=}")
```

    Astrometry                     group.shape=(2, 6)
    Eclipse Timing Variations      group.shape=(9, 6)
    Imaging                        group.shape=(38, 6)
    Microlensing                   group.shape=(23, 6)
    Orbital Brightness Modulation  group.shape=(3, 6)
    Pulsar Timing                  group.shape=(5, 6)
    Pulsation Timing Variations    group.shape=(1, 6)
    Radial Velocity                group.shape=(553, 6)
    Transit                        group.shape=(397, 6)
    Transit Timing Variations      group.shape=(4, 6)


## Dispatch


```python
planets.groupby('method')['year'].describe()
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>method</th>
      <th></th>
      <th></th>
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
      <th>Astrometry</th>
      <td>2.0</td>
      <td>2011.500000</td>
      <td>2.121320</td>
      <td>2010.0</td>
      <td>2010.75</td>
      <td>2011.5</td>
      <td>2012.25</td>
      <td>2013.0</td>
    </tr>
    <tr>
      <th>Eclipse Timing Variations</th>
      <td>9.0</td>
      <td>2010.000000</td>
      <td>1.414214</td>
      <td>2008.0</td>
      <td>2009.00</td>
      <td>2010.0</td>
      <td>2011.00</td>
      <td>2012.0</td>
    </tr>
    <tr>
      <th>Imaging</th>
      <td>38.0</td>
      <td>2009.131579</td>
      <td>2.781901</td>
      <td>2004.0</td>
      <td>2008.00</td>
      <td>2009.0</td>
      <td>2011.00</td>
      <td>2013.0</td>
    </tr>
    <tr>
      <th>Microlensing</th>
      <td>23.0</td>
      <td>2009.782609</td>
      <td>2.859697</td>
      <td>2004.0</td>
      <td>2008.00</td>
      <td>2010.0</td>
      <td>2012.00</td>
      <td>2013.0</td>
    </tr>
    <tr>
      <th>Orbital Brightness Modulation</th>
      <td>3.0</td>
      <td>2011.666667</td>
      <td>1.154701</td>
      <td>2011.0</td>
      <td>2011.00</td>
      <td>2011.0</td>
      <td>2012.00</td>
      <td>2013.0</td>
    </tr>
    <tr>
      <th>Pulsar Timing</th>
      <td>5.0</td>
      <td>1998.400000</td>
      <td>8.384510</td>
      <td>1992.0</td>
      <td>1992.00</td>
      <td>1994.0</td>
      <td>2003.00</td>
      <td>2011.0</td>
    </tr>
    <tr>
      <th>Pulsation Timing Variations</th>
      <td>1.0</td>
      <td>2007.000000</td>
      <td>NaN</td>
      <td>2007.0</td>
      <td>2007.00</td>
      <td>2007.0</td>
      <td>2007.00</td>
      <td>2007.0</td>
    </tr>
    <tr>
      <th>Radial Velocity</th>
      <td>553.0</td>
      <td>2007.518987</td>
      <td>4.249052</td>
      <td>1989.0</td>
      <td>2005.00</td>
      <td>2009.0</td>
      <td>2011.00</td>
      <td>2014.0</td>
    </tr>
    <tr>
      <th>Transit</th>
      <td>397.0</td>
      <td>2011.236776</td>
      <td>2.077867</td>
      <td>2002.0</td>
      <td>2010.00</td>
      <td>2012.0</td>
      <td>2013.00</td>
      <td>2014.0</td>
    </tr>
    <tr>
      <th>Transit Timing Variations</th>
      <td>4.0</td>
      <td>2012.500000</td>
      <td>1.290994</td>
      <td>2011.0</td>
      <td>2011.75</td>
      <td>2012.5</td>
      <td>2013.25</td>
      <td>2014.0</td>
    </tr>
  </tbody>
</table>
</div>



## Aggregate, Filter, Transform, Apply


```python
rng = np.random.default_rng(seed=0)
```


```python
df = pd.DataFrame({
    'key': ['A', 'B', 'C', 'A', 'B', 'C'],
    'data1': range(6),
    'data2': rng.integers(0, 10, 6)
    }, columns = ['key', 'data1', 'data2'])
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
      <th>key</th>
      <th>data1</th>
      <th>data2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>B</td>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>C</td>
      <td>5</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Aggregation


```python
df.groupby('key').aggregate(['min', np.median, max])
```

    /tmp/ipykernel_108862/968873422.py:1: FutureWarning: The provided callable <function median at 0x7f2da005e0c0> is currently using SeriesGroupBy.median. In a future version of pandas, the provided callable will be used directly. To keep current behavior pass the string "median" instead.
      df.groupby('key').aggregate(['min', np.median, max])
    /tmp/ipykernel_108862/968873422.py:1: FutureWarning: The provided callable <built-in function max> is currently using SeriesGroupBy.max. In a future version of pandas, the provided callable will be used directly. To keep current behavior pass the string "max" instead.
      df.groupby('key').aggregate(['min', np.median, max])





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
      <th colspan="3" halign="left">data1</th>
      <th colspan="3" halign="left">data2</th>
    </tr>
    <tr>
      <th></th>
      <th>min</th>
      <th>median</th>
      <th>max</th>
      <th>min</th>
      <th>median</th>
      <th>max</th>
    </tr>
    <tr>
      <th>key</th>
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
      <th>A</th>
      <td>0</td>
      <td>1.5</td>
      <td>3</td>
      <td>2</td>
      <td>5.0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>B</th>
      <td>1</td>
      <td>2.5</td>
      <td>4</td>
      <td>3</td>
      <td>4.5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>C</th>
      <td>2</td>
      <td>3.5</td>
      <td>5</td>
      <td>0</td>
      <td>2.5</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.groupby('key').aggregate(['min', 'median', 'max'])
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
      <th colspan="3" halign="left">data1</th>
      <th colspan="3" halign="left">data2</th>
    </tr>
    <tr>
      <th></th>
      <th>min</th>
      <th>median</th>
      <th>max</th>
      <th>min</th>
      <th>median</th>
      <th>max</th>
    </tr>
    <tr>
      <th>key</th>
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
      <th>A</th>
      <td>0</td>
      <td>1.5</td>
      <td>3</td>
      <td>2</td>
      <td>5.0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>B</th>
      <td>1</td>
      <td>2.5</td>
      <td>4</td>
      <td>3</td>
      <td>4.5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>C</th>
      <td>2</td>
      <td>3.5</td>
      <td>5</td>
      <td>0</td>
      <td>2.5</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.groupby('key').aggregate({'data1': 'min', 'data2': 'max'})
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
      <th>data1</th>
      <th>data2</th>
    </tr>
    <tr>
      <th>key</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>B</th>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>C</th>
      <td>2</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



## Filtering


```python
def filter_func(x): 
    return x['data2'].std() > 5
```


```python
display('df', "df.groupby('key').std()",
        "df.groupby('key').filter(filter_func)")
```




<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>df</p><div>
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
      <th>key</th>
      <th>data1</th>
      <th>data2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>B</td>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>C</td>
      <td>5</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
    </div>
<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>df.groupby('key').std()</p><div>
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
      <th>data1</th>
      <th>data2</th>
    </tr>
    <tr>
      <th>key</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>2.12132</td>
      <td>4.242641</td>
    </tr>
    <tr>
      <th>B</th>
      <td>2.12132</td>
      <td>2.121320</td>
    </tr>
    <tr>
      <th>C</th>
      <td>2.12132</td>
      <td>3.535534</td>
    </tr>
  </tbody>
</table>
</div>
    </div>
<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>df.groupby('key').filter(filter_func)</p><div>
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
      <th>key</th>
      <th>data1</th>
      <th>data2</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>
    </div>



## Transformation


```python
def center(x): 
    return x - x.mean()
df.groupby('key').transform(center)
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
      <th>data1</th>
      <th>data2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.5</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.5</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.5</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.5</td>
      <td>-3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.5</td>
      <td>-1.5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.5</td>
      <td>-2.5</td>
    </tr>
  </tbody>
</table>
</div>



## Apply


```python
def norm_by_data2(x): 
    x['data1'] /= x['data2'].sum()
    return x

df.groupby('key').apply(norm_by_data2)
```

    /tmp/ipykernel_108862/3919926901.py:5: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
      df.groupby('key').apply(norm_by_data2)





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
      <th>key</th>
      <th>data1</th>
      <th>data2</th>
    </tr>
    <tr>
      <th>key</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">A</th>
      <th>0</th>
      <td>A</td>
      <td>0.000000</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A</td>
      <td>0.300000</td>
      <td>2</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">B</th>
      <th>1</th>
      <td>B</td>
      <td>0.111111</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>B</td>
      <td>0.444444</td>
      <td>3</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">C</th>
      <th>2</th>
      <td>C</td>
      <td>0.400000</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>C</td>
      <td>1.000000</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.groupby('key').apply(norm_by_data2, include_groups=False)
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
    <tr>
      <th>key</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">A</th>
      <th>0</th>
      <td>0.000000</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.300000</td>
      <td>2</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">B</th>
      <th>1</th>
      <td>0.111111</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.444444</td>
      <td>3</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">C</th>
      <th>2</th>
      <td>0.400000</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.000000</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



# Specify the split key

> with list, array, series or index


```python
L = [0, 1, 0, 1, 2, 0]
df.groupby(L).sum()
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
      <th>key</th>
      <th>data1</th>
      <th>data2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACC</td>
      <td>7</td>
      <td>13</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BA</td>
      <td>4</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>B</td>
      <td>4</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.groupby(df['key']).sum()
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
      <th>data1</th>
      <th>data2</th>
    </tr>
    <tr>
      <th>key</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>3</td>
      <td>10</td>
    </tr>
    <tr>
      <th>B</th>
      <td>5</td>
      <td>9</td>
    </tr>
    <tr>
      <th>C</th>
      <td>7</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



>With dictionary or series mapping index


```python
df2 = df.set_index('key')
mapping = {'A': 'vowel', 'B': 'consonant', 'C': 'consonant'}
display('df2', 'df2.groupby(mapping).sum()')
```




<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>df2</p><div>
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
      <th>data1</th>
      <th>data2</th>
    </tr>
    <tr>
      <th>key</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>0</td>
      <td>8</td>
    </tr>
    <tr>
      <th>B</th>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>C</th>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>A</th>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>B</th>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>C</th>
      <td>5</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
    </div>
<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>df2.groupby(mapping).sum()</p><div>
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
      <th>data1</th>
      <th>data2</th>
    </tr>
    <tr>
      <th>key</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>consonant</th>
      <td>12</td>
      <td>14</td>
    </tr>
    <tr>
      <th>vowel</th>
      <td>3</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>
    </div>



## Any Python function


```python
df2.groupby(str.lower).mean()
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
      <th>data1</th>
      <th>data2</th>
    </tr>
    <tr>
      <th>key</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1.5</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>2.5</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>c</th>
      <td>3.5</td>
      <td>2.5</td>
    </tr>
  </tbody>
</table>
</div>



## List of valid keys
>combining keys


```python
df2.groupby([str.lower, mapping]).mean()
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
    <tr>
      <th>key</th>
      <th>key</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <th>vowel</th>
      <td>1.5</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>b</th>
      <th>consonant</th>
      <td>2.5</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>c</th>
      <th>consonant</th>
      <td>3.5</td>
      <td>2.5</td>
    </tr>
  </tbody>
</table>
</div>



# Example

Summarize annual data by decade.


```python
decade = 10 * (planets['year'] // 10)
decade = decade.astype(str) + 's'
decade.name = 'decade'
planets.groupby(['method', decade])['number'].sum().unstack().fillna(0)
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
      <th>decade</th>
      <th>1980s</th>
      <th>1990s</th>
      <th>2000s</th>
      <th>2010s</th>
    </tr>
    <tr>
      <th>method</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Astrometry</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Eclipse Timing Variations</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>Imaging</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>29.0</td>
      <td>21.0</td>
    </tr>
    <tr>
      <th>Microlensing</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>12.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>Orbital Brightness Modulation</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>Pulsar Timing</th>
      <td>0.0</td>
      <td>9.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Pulsation Timing Variations</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Radial Velocity</th>
      <td>1.0</td>
      <td>52.0</td>
      <td>475.0</td>
      <td>424.0</td>
    </tr>
    <tr>
      <th>Transit</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>64.0</td>
      <td>712.0</td>
    </tr>
    <tr>
      <th>Transit Timing Variations</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>9.0</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
