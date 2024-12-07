# Merging Datasets


```python
import pandas as pd
import numpy as np

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


```python
df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'group': ['Accounting', 'Engineering',
                              'Engineering', 'HR']})
df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
                    'hire_date': [2004, 2008, 2012, 2014]})
display('df1', 'df2')
```




<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>df1</p><div>
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
      <th>employee</th>
      <th>group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bob</td>
      <td>Accounting</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jake</td>
      <td>Engineering</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lisa</td>
      <td>Engineering</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sue</td>
      <td>HR</td>
    </tr>
  </tbody>
</table>
</div>
    </div>
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
      <th>employee</th>
      <th>hire_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Lisa</td>
      <td>2004</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bob</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jake</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sue</td>
      <td>2014</td>
    </tr>
  </tbody>
</table>
</div>
    </div>



>One-to-One Joins


```python
df3 = pd.merge(df1, df2)
df3
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
      <th>employee</th>
      <th>group</th>
      <th>hire_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bob</td>
      <td>Accounting</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jake</td>
      <td>Engineering</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lisa</td>
      <td>Engineering</td>
      <td>2004</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sue</td>
      <td>HR</td>
      <td>2014</td>
    </tr>
  </tbody>
</table>
</div>



>Many-to-One


```python
df4 = pd.DataFrame({'group': ['Accounting', 'Engineering', 'HR'],
                    'supervisor': ['Carly', 'Guido', 'Steve']})
display('df3', 'df4', 'pd.merge(df3, df4)')
```




<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>df3</p><div>
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
      <th>employee</th>
      <th>group</th>
      <th>hire_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bob</td>
      <td>Accounting</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jake</td>
      <td>Engineering</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lisa</td>
      <td>Engineering</td>
      <td>2004</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sue</td>
      <td>HR</td>
      <td>2014</td>
    </tr>
  </tbody>
</table>
</div>
    </div>
<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>df4</p><div>
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
      <th>group</th>
      <th>supervisor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Accounting</td>
      <td>Carly</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Engineering</td>
      <td>Guido</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HR</td>
      <td>Steve</td>
    </tr>
  </tbody>
</table>
</div>
    </div>
<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>pd.merge(df3, df4)</p><div>
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
      <th>employee</th>
      <th>group</th>
      <th>hire_date</th>
      <th>supervisor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bob</td>
      <td>Accounting</td>
      <td>2008</td>
      <td>Carly</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jake</td>
      <td>Engineering</td>
      <td>2012</td>
      <td>Guido</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lisa</td>
      <td>Engineering</td>
      <td>2004</td>
      <td>Guido</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sue</td>
      <td>HR</td>
      <td>2014</td>
      <td>Steve</td>
    </tr>
  </tbody>
</table>
</div>
    </div>



>Many-to-Many


```python
df5 = pd.DataFrame({'group': ['Accounting', 'Accounting',
                              'Engineering', 'Engineering', 'HR', 'HR'],
                    'skills': ['math', 'spreadsheets', 'software', 'math',
                               'spreadsheets', 'organization']})
display('df1', 'df5', "pd.merge(df1, df5)")
```




<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>df1</p><div>
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
      <th>employee</th>
      <th>group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bob</td>
      <td>Accounting</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jake</td>
      <td>Engineering</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lisa</td>
      <td>Engineering</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sue</td>
      <td>HR</td>
    </tr>
  </tbody>
</table>
</div>
    </div>
<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>df5</p><div>
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
      <th>group</th>
      <th>skills</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Accounting</td>
      <td>math</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Accounting</td>
      <td>spreadsheets</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Engineering</td>
      <td>software</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Engineering</td>
      <td>math</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HR</td>
      <td>spreadsheets</td>
    </tr>
    <tr>
      <th>5</th>
      <td>HR</td>
      <td>organization</td>
    </tr>
  </tbody>
</table>
</div>
    </div>
<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>pd.merge(df1, df5)</p><div>
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
      <th>employee</th>
      <th>group</th>
      <th>skills</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bob</td>
      <td>Accounting</td>
      <td>math</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bob</td>
      <td>Accounting</td>
      <td>spreadsheets</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jake</td>
      <td>Engineering</td>
      <td>software</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Jake</td>
      <td>Engineering</td>
      <td>math</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Lisa</td>
      <td>Engineering</td>
      <td>software</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Lisa</td>
      <td>Engineering</td>
      <td>math</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Sue</td>
      <td>HR</td>
      <td>spreadsheets</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Sue</td>
      <td>HR</td>
      <td>organization</td>
    </tr>
  </tbody>
</table>
</div>
    </div>



## Specifying the Merge Key

- `on`
- `left_on` and `right_on`
- `left_index` and `right_index`


```python
display('df1', 'df2', "pd.merge(df1, df2, on='employee')")
```




<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>df1</p><div>
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
      <th>employee</th>
      <th>group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bob</td>
      <td>Accounting</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jake</td>
      <td>Engineering</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lisa</td>
      <td>Engineering</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sue</td>
      <td>HR</td>
    </tr>
  </tbody>
</table>
</div>
    </div>
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
      <th>employee</th>
      <th>hire_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Lisa</td>
      <td>2004</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bob</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Jake</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sue</td>
      <td>2014</td>
    </tr>
  </tbody>
</table>
</div>
    </div>
<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>pd.merge(df1, df2, on='employee')</p><div>
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
      <th>employee</th>
      <th>group</th>
      <th>hire_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bob</td>
      <td>Accounting</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jake</td>
      <td>Engineering</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lisa</td>
      <td>Engineering</td>
      <td>2004</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sue</td>
      <td>HR</td>
      <td>2014</td>
    </tr>
  </tbody>
</table>
</div>
    </div>




```python
df3 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'salary': [70000, 80000, 120000, 90000]})
display('df1', 'df3', 'pd.merge(df1, df3, left_on="employee", right_on="name")')
```




<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>df1</p><div>
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
      <th>employee</th>
      <th>group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bob</td>
      <td>Accounting</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jake</td>
      <td>Engineering</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lisa</td>
      <td>Engineering</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sue</td>
      <td>HR</td>
    </tr>
  </tbody>
</table>
</div>
    </div>
<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>df3</p><div>
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
      <th>name</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bob</td>
      <td>70000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jake</td>
      <td>80000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lisa</td>
      <td>120000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sue</td>
      <td>90000</td>
    </tr>
  </tbody>
</table>
</div>
    </div>
<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>pd.merge(df1, df3, left_on="employee", right_on="name")</p><div>
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
      <th>employee</th>
      <th>group</th>
      <th>name</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bob</td>
      <td>Accounting</td>
      <td>Bob</td>
      <td>70000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jake</td>
      <td>Engineering</td>
      <td>Jake</td>
      <td>80000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lisa</td>
      <td>Engineering</td>
      <td>Lisa</td>
      <td>120000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sue</td>
      <td>HR</td>
      <td>Sue</td>
      <td>90000</td>
    </tr>
  </tbody>
</table>
</div>
    </div>



>Drop duplicate columns


```python
pd.merge(df1, df3, left_on="employee", right_on="name").drop('name', axis=1)
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
      <th>employee</th>
      <th>group</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bob</td>
      <td>Accounting</td>
      <td>70000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jake</td>
      <td>Engineering</td>
      <td>80000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lisa</td>
      <td>Engineering</td>
      <td>120000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sue</td>
      <td>HR</td>
      <td>90000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1a = df1.set_index('employee')
df2a = df2.set_index('employee')
display('df1a', 'df2a')
```




<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>df1a</p><div>
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
      <th>group</th>
    </tr>
    <tr>
      <th>employee</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bob</th>
      <td>Accounting</td>
    </tr>
    <tr>
      <th>Jake</th>
      <td>Engineering</td>
    </tr>
    <tr>
      <th>Lisa</th>
      <td>Engineering</td>
    </tr>
    <tr>
      <th>Sue</th>
      <td>HR</td>
    </tr>
  </tbody>
</table>
</div>
    </div>
<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>df2a</p><div>
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
      <th>hire_date</th>
    </tr>
    <tr>
      <th>employee</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Lisa</th>
      <td>2004</td>
    </tr>
    <tr>
      <th>Bob</th>
      <td>2008</td>
    </tr>
    <tr>
      <th>Jake</th>
      <td>2012</td>
    </tr>
    <tr>
      <th>Sue</th>
      <td>2014</td>
    </tr>
  </tbody>
</table>
</div>
    </div>




```python
display('df1a', 'df2a',
        "pd.merge(df1a, df2a, left_index=True, right_index=True)")
```




<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>df1a</p><div>
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
      <th>group</th>
    </tr>
    <tr>
      <th>employee</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bob</th>
      <td>Accounting</td>
    </tr>
    <tr>
      <th>Jake</th>
      <td>Engineering</td>
    </tr>
    <tr>
      <th>Lisa</th>
      <td>Engineering</td>
    </tr>
    <tr>
      <th>Sue</th>
      <td>HR</td>
    </tr>
  </tbody>
</table>
</div>
    </div>
<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>df2a</p><div>
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
      <th>hire_date</th>
    </tr>
    <tr>
      <th>employee</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Lisa</th>
      <td>2004</td>
    </tr>
    <tr>
      <th>Bob</th>
      <td>2008</td>
    </tr>
    <tr>
      <th>Jake</th>
      <td>2012</td>
    </tr>
    <tr>
      <th>Sue</th>
      <td>2014</td>
    </tr>
  </tbody>
</table>
</div>
    </div>
<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>pd.merge(df1a, df2a, left_index=True, right_index=True)</p><div>
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
      <th>group</th>
      <th>hire_date</th>
    </tr>
    <tr>
      <th>employee</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bob</th>
      <td>Accounting</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>Jake</th>
      <td>Engineering</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>Lisa</th>
      <td>Engineering</td>
      <td>2004</td>
    </tr>
    <tr>
      <th>Sue</th>
      <td>HR</td>
      <td>2014</td>
    </tr>
  </tbody>
</table>
</div>
    </div>



>using join


```python
df1a.join(df2a)
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
      <th>group</th>
      <th>hire_date</th>
    </tr>
    <tr>
      <th>employee</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bob</th>
      <td>Accounting</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>Jake</th>
      <td>Engineering</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>Lisa</th>
      <td>Engineering</td>
      <td>2004</td>
    </tr>
    <tr>
      <th>Sue</th>
      <td>HR</td>
      <td>2014</td>
    </tr>
  </tbody>
</table>
</div>



>Mixing keywords


```python
display('df1a', 'df3', "pd.merge(df1a, df3, left_index=True, right_on='name')")
```




<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>df1a</p><div>
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
      <th>group</th>
    </tr>
    <tr>
      <th>employee</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bob</th>
      <td>Accounting</td>
    </tr>
    <tr>
      <th>Jake</th>
      <td>Engineering</td>
    </tr>
    <tr>
      <th>Lisa</th>
      <td>Engineering</td>
    </tr>
    <tr>
      <th>Sue</th>
      <td>HR</td>
    </tr>
  </tbody>
</table>
</div>
    </div>
<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>df3</p><div>
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
      <th>name</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bob</td>
      <td>70000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jake</td>
      <td>80000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lisa</td>
      <td>120000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sue</td>
      <td>90000</td>
    </tr>
  </tbody>
</table>
</div>
    </div>
<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>pd.merge(df1a, df3, left_index=True, right_on='name')</p><div>
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
      <th>group</th>
      <th>name</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Accounting</td>
      <td>Bob</td>
      <td>70000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Engineering</td>
      <td>Jake</td>
      <td>80000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Engineering</td>
      <td>Lisa</td>
      <td>120000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HR</td>
      <td>Sue</td>
      <td>90000</td>
    </tr>
  </tbody>
</table>
</div>
    </div>



# Specifying Set Arthmetic

for values not appearing in both sets


```python
df6 = pd.DataFrame({'name': ['Peter', 'Paul', 'Mary'],
                    'food': ['fish', 'beans', 'bread']},
                   columns=['name', 'food'])
df7 = pd.DataFrame({'name': ['Mary', 'Joseph'],
                    'drink': ['wine', 'beer']},
                   columns=['name', 'drink'])
```

>By default, merge uses inner join


```python
display('df6', 'df7', 'pd.merge(df6, df7)')
```




<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>df6</p><div>
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
      <th>name</th>
      <th>food</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Peter</td>
      <td>fish</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Paul</td>
      <td>beans</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mary</td>
      <td>bread</td>
    </tr>
  </tbody>
</table>
</div>
    </div>
<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>df7</p><div>
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
      <th>name</th>
      <th>drink</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mary</td>
      <td>wine</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Joseph</td>
      <td>beer</td>
    </tr>
  </tbody>
</table>
</div>
    </div>
<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>pd.merge(df6, df7)</p><div>
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
      <th>name</th>
      <th>food</th>
      <th>drink</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mary</td>
      <td>bread</td>
      <td>wine</td>
    </tr>
  </tbody>
</table>
</div>
    </div>



>Specify outer join


```python
display('df6', 'df7', 'pd.merge(df6, df7, how="outer")')
```




<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>df6</p><div>
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
      <th>name</th>
      <th>food</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Peter</td>
      <td>fish</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Paul</td>
      <td>beans</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mary</td>
      <td>bread</td>
    </tr>
  </tbody>
</table>
</div>
    </div>
<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>df7</p><div>
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
      <th>name</th>
      <th>drink</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mary</td>
      <td>wine</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Joseph</td>
      <td>beer</td>
    </tr>
  </tbody>
</table>
</div>
    </div>
<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>pd.merge(df6, df7, how="outer")</p><div>
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
      <th>name</th>
      <th>food</th>
      <th>drink</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Joseph</td>
      <td>NaN</td>
      <td>beer</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mary</td>
      <td>bread</td>
      <td>wine</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Paul</td>
      <td>beans</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Peter</td>
      <td>fish</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>
    </div>



>Left and right joins


```python
display('df6', 'df7', "pd.merge(df6, df7, how='left')")
```




<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>df6</p><div>
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
      <th>name</th>
      <th>food</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Peter</td>
      <td>fish</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Paul</td>
      <td>beans</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mary</td>
      <td>bread</td>
    </tr>
  </tbody>
</table>
</div>
    </div>
<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>df7</p><div>
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
      <th>name</th>
      <th>drink</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mary</td>
      <td>wine</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Joseph</td>
      <td>beer</td>
    </tr>
  </tbody>
</table>
</div>
    </div>
<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>pd.merge(df6, df7, how='left')</p><div>
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
      <th>name</th>
      <th>food</th>
      <th>drink</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Peter</td>
      <td>fish</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Paul</td>
      <td>beans</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mary</td>
      <td>bread</td>
      <td>wine</td>
    </tr>
  </tbody>
</table>
</div>
    </div>



## Overlapping column names

automatically creates column for each set, default suffix can be changed.


```python
df8 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'rank': [1, 2, 3, 4]})
df9 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'rank': [3, 1, 4, 2]})
display('df8', 'df9', 'pd.merge(df8, df9, on="name")')
```




<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>df8</p><div>
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
      <th>name</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bob</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jake</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lisa</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sue</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>
    </div>
<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>df9</p><div>
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
      <th>name</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bob</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jake</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lisa</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sue</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>
    </div>
<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>pd.merge(df8, df9, on="name")</p><div>
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
      <th>name</th>
      <th>rank_x</th>
      <th>rank_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bob</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jake</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lisa</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sue</td>
      <td>4</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>
    </div>




```python
pd.merge(df8, df9, on="name", suffixes=["_L", "_R"])
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
      <th>name</th>
      <th>rank_L</th>
      <th>rank_R</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bob</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jake</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lisa</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sue</td>
      <td>4</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



# US States Data


```python
pop = pd.read_csv("data/state-population.csv")
areas = pd.read_csv("data/state-areas.csv")
abbrevs = pd.read_csv("data/state-abbrevs.csv")

display('pop.head()', 'areas.head()', 'abbrevs.head()')
```




<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>pop.head()</p><div>
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
      <th>state/region</th>
      <th>ages</th>
      <th>year</th>
      <th>population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AL</td>
      <td>under18</td>
      <td>2012</td>
      <td>1117489.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AL</td>
      <td>total</td>
      <td>2012</td>
      <td>4817528.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AL</td>
      <td>under18</td>
      <td>2010</td>
      <td>1130966.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AL</td>
      <td>total</td>
      <td>2010</td>
      <td>4785570.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AL</td>
      <td>under18</td>
      <td>2011</td>
      <td>1125763.0</td>
    </tr>
  </tbody>
</table>
</div>
    </div>
<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>areas.head()</p><div>
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
      <th>area (sq. mi)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>52423</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alaska</td>
      <td>656425</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Arizona</td>
      <td>114006</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arkansas</td>
      <td>53182</td>
    </tr>
    <tr>
      <th>4</th>
      <td>California</td>
      <td>163707</td>
    </tr>
  </tbody>
</table>
</div>
    </div>
<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>abbrevs.head()</p><div>
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
      <th>abbreviation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alaska</td>
      <td>AK</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Arizona</td>
      <td>AZ</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arkansas</td>
      <td>AR</td>
    </tr>
    <tr>
      <th>4</th>
      <td>California</td>
      <td>CA</td>
    </tr>
  </tbody>
</table>
</div>
    </div>



>[!attention]
>Merge with outer if unsure in order to avoid losing data

Given this information, say we want to compute a relatively straightforward result: rank US states and territories by their 2010 population density. We clearly have the data here to find this result, but we'll have to combine the datasets to do so.


```python
merged = pd.merge(pop, abbrevs,
         how='outer',
         left_on='state/region', 
         right_on='abbreviation').drop("abbreviation", axis=1)
merged.head()
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
      <th>state/region</th>
      <th>ages</th>
      <th>year</th>
      <th>population</th>
      <th>state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AK</td>
      <td>total</td>
      <td>1990</td>
      <td>553290.0</td>
      <td>Alaska</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AK</td>
      <td>under18</td>
      <td>1990</td>
      <td>177502.0</td>
      <td>Alaska</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AK</td>
      <td>total</td>
      <td>1992</td>
      <td>588736.0</td>
      <td>Alaska</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AK</td>
      <td>under18</td>
      <td>1991</td>
      <td>182180.0</td>
      <td>Alaska</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AK</td>
      <td>under18</td>
      <td>1992</td>
      <td>184878.0</td>
      <td>Alaska</td>
    </tr>
  </tbody>
</table>
</div>




```python
merged.isnull().any()
```




    state/region    False
    ages            False
    year            False
    population       True
    state            True
    dtype: bool



>Investigate null values


```python
merged[merged["population"].isnull()].head()
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
      <th>state/region</th>
      <th>ages</th>
      <th>year</th>
      <th>population</th>
      <th>state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1872</th>
      <td>PR</td>
      <td>under18</td>
      <td>1990</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1873</th>
      <td>PR</td>
      <td>total</td>
      <td>1990</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1874</th>
      <td>PR</td>
      <td>total</td>
      <td>1991</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1875</th>
      <td>PR</td>
      <td>under18</td>
      <td>1991</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1876</th>
      <td>PR</td>
      <td>total</td>
      <td>1993</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
merged[merged["state"].isnull()].head()
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
      <th>state/region</th>
      <th>ages</th>
      <th>year</th>
      <th>population</th>
      <th>state</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1872</th>
      <td>PR</td>
      <td>under18</td>
      <td>1990</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1873</th>
      <td>PR</td>
      <td>total</td>
      <td>1990</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1874</th>
      <td>PR</td>
      <td>total</td>
      <td>1991</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1875</th>
      <td>PR</td>
      <td>under18</td>
      <td>1991</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1876</th>
      <td>PR</td>
      <td>total</td>
      <td>1993</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
merged.loc[merged['state'].isnull(), 'state/region'].unique()
```




    array(['PR', 'USA'], dtype=object)




```python
merged.loc[merged['state/region'] == 'PR', 'state'] = 'Puerto Rico'
merged.loc[merged['state/region'] == 'USA', 'state'] = 'United States'
merged.isnull().any()
```




    state/region    False
    ages            False
    year            False
    population       True
    state           False
    dtype: bool




```python
final = pd.merge(merged, areas, on='state', how='left')
final.head()
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
      <th>state/region</th>
      <th>ages</th>
      <th>year</th>
      <th>population</th>
      <th>state</th>
      <th>area (sq. mi)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AK</td>
      <td>total</td>
      <td>1990</td>
      <td>553290.0</td>
      <td>Alaska</td>
      <td>656425.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AK</td>
      <td>under18</td>
      <td>1990</td>
      <td>177502.0</td>
      <td>Alaska</td>
      <td>656425.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AK</td>
      <td>total</td>
      <td>1992</td>
      <td>588736.0</td>
      <td>Alaska</td>
      <td>656425.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AK</td>
      <td>under18</td>
      <td>1991</td>
      <td>182180.0</td>
      <td>Alaska</td>
      <td>656425.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AK</td>
      <td>under18</td>
      <td>1992</td>
      <td>184878.0</td>
      <td>Alaska</td>
      <td>656425.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
final.isnull().any()
```




    state/region     False
    ages             False
    year             False
    population        True
    state            False
    area (sq. mi)     True
    dtype: bool




```python
final['state'][final['area (sq. mi)'].isnull()].unique()
```




    array(['United States'], dtype=object)




```python
final[final['area (sq. mi)'].isnull()].unique()
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    /tmp/ipykernel_31379/1305250124.py in ?()
    ----> 1 final[final['area (sq. mi)'].isnull()].unique()
    

    /nix/store/5rsfbp6gh0qf8ir5wgiz45rhnkc9q5y7-python3-3.12.7-env/lib/python3.12/site-packages/pandas/core/generic.py in ?(self, name)
       6295             and name not in self._accessors
       6296             and self._info_axis._can_hold_identifiers_and_holds_name(name)
       6297         ):
       6298             return self[name]
    -> 6299         return object.__getattribute__(self, name)
    

    AttributeError: 'DataFrame' object has no attribute 'unique'



```python
merged.loc[merged['state'].isnull(), 'state/region'].unique()
```




    array([], dtype=object)




```python
merged.loc[merged['state'].isnull()].unique()
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    /tmp/ipykernel_31379/3854662867.py in ?()
    ----> 1 merged.loc[merged['state'].isnull()].unique()
    

    /nix/store/5rsfbp6gh0qf8ir5wgiz45rhnkc9q5y7-python3-3.12.7-env/lib/python3.12/site-packages/pandas/core/generic.py in ?(self, name)
       6295             and name not in self._accessors
       6296             and self._info_axis._can_hold_identifiers_and_holds_name(name)
       6297         ):
       6298             return self[name]
    -> 6299         return object.__getattribute__(self, name)
    

    AttributeError: 'DataFrame' object has no attribute 'unique'



```python
final['state'][final['area (sq. mi)'].isnull()].unique()
```




    array(['United States'], dtype=object)




```python
final.dropna(inplace=True)
final.head()
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
      <th>state/region</th>
      <th>ages</th>
      <th>year</th>
      <th>population</th>
      <th>state</th>
      <th>area (sq. mi)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AK</td>
      <td>total</td>
      <td>1990</td>
      <td>553290.0</td>
      <td>Alaska</td>
      <td>656425.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AK</td>
      <td>under18</td>
      <td>1990</td>
      <td>177502.0</td>
      <td>Alaska</td>
      <td>656425.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AK</td>
      <td>total</td>
      <td>1992</td>
      <td>588736.0</td>
      <td>Alaska</td>
      <td>656425.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AK</td>
      <td>under18</td>
      <td>1991</td>
      <td>182180.0</td>
      <td>Alaska</td>
      <td>656425.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AK</td>
      <td>under18</td>
      <td>1992</td>
      <td>184878.0</td>
      <td>Alaska</td>
      <td>656425.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data2010 = final.query("year == 2010 & ages == 'total'")
data2010.head()
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
      <th>state/region</th>
      <th>ages</th>
      <th>year</th>
      <th>population</th>
      <th>state</th>
      <th>area (sq. mi)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>43</th>
      <td>AK</td>
      <td>total</td>
      <td>2010</td>
      <td>713868.0</td>
      <td>Alaska</td>
      <td>656425.0</td>
    </tr>
    <tr>
      <th>51</th>
      <td>AL</td>
      <td>total</td>
      <td>2010</td>
      <td>4785570.0</td>
      <td>Alabama</td>
      <td>52423.0</td>
    </tr>
    <tr>
      <th>141</th>
      <td>AR</td>
      <td>total</td>
      <td>2010</td>
      <td>2922280.0</td>
      <td>Arkansas</td>
      <td>53182.0</td>
    </tr>
    <tr>
      <th>149</th>
      <td>AZ</td>
      <td>total</td>
      <td>2010</td>
      <td>6408790.0</td>
      <td>Arizona</td>
      <td>114006.0</td>
    </tr>
    <tr>
      <th>197</th>
      <td>CA</td>
      <td>total</td>
      <td>2010</td>
      <td>37333601.0</td>
      <td>California</td>
      <td>163707.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
density = data2010['population'] / data2010['area (sq. mi)']
density.head()
```




    43       1.087509
    51      91.287603
    141     54.948667
    149     56.214497
    197    228.051342
    dtype: float64




```python
data2010.set_index('state', inplace=True)
density = data2010['population'] / data2010['area (sq. mi)']
density.head()
```




    state
    Alaska          1.087509
    Alabama        91.287603
    Arkansas       54.948667
    Arizona        56.214497
    California    228.051342
    dtype: float64




```python
density.sort_values(ascending=False, inplace=True)
density.head()
```




    state
    District of Columbia    8898.897059
    Puerto Rico             1058.665149
    New Jersey              1009.253268
    Rhode Island             681.339159
    Connecticut              645.600649
    dtype: float64




```python
density.tail()
```




    state
    South Dakota    10.583512
    North Dakota     9.537565
    Montana          6.736171
    Wyoming          5.768079
    Alaska           1.087509
    dtype: float64




```python

```
