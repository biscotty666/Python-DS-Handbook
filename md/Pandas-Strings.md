# Panda Vectorized String Operations


```python
import numpy as np 
x = np.array([2, 3, 5, 7, 11, 13])
x * 2
```




    array([ 4,  6, 10, 14, 22, 26])




```python
data = "peter Paul MARY gUIDO".split()
```


```python
[s.capitalize() for s in data]
```




    ['Peter', 'Paul', 'Mary', 'Guido']



### Methods Similar to Python String Methods

Nearly all of Python's built-in string methods are mirrored by a Pandas vectorized string method. Here is a list of Pandas `str` methods that mirror Python string methods:

|           |                |                |                |
|-----------|----------------|----------------|----------------|
|`len()`    | `lower()`      | `translate()`  | `islower()`    | 
|`ljust()`  | `upper()`      | `startswith()` | `isupper()`    | 
|`rjust()`  | `find()`       | `endswith()`   | `isnumeric()`  | 
|`center()` | `rfind()`      | `isalnum()`    | `isdecimal()`  | 
|`zfill()`  | `index()`      | `isalpha()`    | `split()`      | 
|`strip()`  | `rindex()`     | `isdigit()`    | `rsplit()`     | 
|`rstrip()` | `capitalize()` | `isspace()`    | `partition()`  | 
|`lstrip()` | `swapcase()`   | `istitle()`    | `rpartition()` |

Notice that these have various return values. Some, like `lower`, return a series of strings:

### Methods Using Regular Expressions

In addition, there are several methods that accept regular expressions (regexps) to examine the content of each string element, and follow some of the API conventions of Python's built-in `re` module:

| Method    | Description |
|-----------|-------------|
| `match`   | Calls `re.match` on each element, returning a Boolean. |
| `extract` | Calls `re.match` on each element, returning matched groups as strings.|
| `findall` | Calls `re.findall` on each element |
| `replace` | Replaces occurrences of pattern with some other string|
| `contains`| Calls `re.search` on each element, returning a boolean |
| `count`   | Counts occurrences of pattern|
| `split`   | Equivalent to `str.split`, but accepts regexps |
| `rsplit`  | Equivalent to `str.rsplit`, but accepts regexps |


```python
import pandas as pd
monte = pd.Series(
    "Graham Chapman, John Cleese, Terry Gilliam," \
    "Eric Idle, Terry Jones, Michael Palin".split(','))
monte
```




    0    Graham Chapman
    1       John Cleese
    2     Terry Gilliam
    3         Eric Idle
    4       Terry Jones
    5     Michael Palin
    dtype: object




```python
monte = pd.Series(['Graham Chapman', 'John Cleese', 'Terry Gilliam',
                   'Eric Idle', 'Terry Jones', 'Michael Palin'])
```


```python
monte.str.lower()
```




    0    graham chapman
    1       john cleese
    2     terry gilliam
    3         eric idle
    4       terry jones
    5     michael palin
    dtype: object




```python
monte.str.len()
```




    0    14
    1    11
    2    13
    3     9
    4    11
    5    13
    dtype: int64




```python
monte.str.split()
```




    0    [Graham, Chapman]
    1       [John, Cleese]
    2     [Terry, Gilliam]
    3         [Eric, Idle]
    4       [Terry, Jones]
    5     [Michael, Palin]
    dtype: object




```python
monte.str.startswith('T')
```




    0    False
    1    False
    2     True
    3    False
    4     True
    5    False
    dtype: bool




```python
monte.str.extract('([A-Za-z]+)', expand=False)
```




    0     Graham
    1       John
    2      Terry
    3       Eric
    4      Terry
    5    Michael
    dtype: object



### Miscellaneous Methods
Finally, there are some miscellaneous methods that enable other convenient operations:

| Method | Description |
|--------|-------------|
| `get` | Indexes each element |
| `slice` | Slices each element|
| `slice_replace` | Replaces slice in each element with the passed value|
| `cat`      | Concatenates strings|
| `repeat` | Repeats values |
| `normalize` | Returns Unicode form of strings |
| `pad` | Adds whitespace to left, right, or both sides of strings|
| `wrap` | Splits long strings into lines with length less than a given width|
| `join` | Joins strings in each element of the `Series` with the passed separator|
| `get_dummies` | Extracts dummy variables as a `DataFrame` |

## Item access and slicing


```python
monte.str[0:3]
```




    0    Gra
    1    Joh
    2    Ter
    3    Eri
    4    Ter
    5    Mic
    dtype: object




```python
monte.str.split().str[-1]
```




    0    Chapman
    1     Cleese
    2    Gilliam
    3       Idle
    4      Jones
    5      Palin
    dtype: object



## Indicator variables


```python
full_monte = pd.DataFrame({'name': monte, 
                           'info': ['B|C|D', 'B|D', 'A|C', 'B|D', 'B|C', 'B|C|D']})
full_monte
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
      <th>info</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Graham Chapman</td>
      <td>B|C|D</td>
    </tr>
    <tr>
      <th>1</th>
      <td>John Cleese</td>
      <td>B|D</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Terry Gilliam</td>
      <td>A|C</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Eric Idle</td>
      <td>B|D</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Terry Jones</td>
      <td>B|C</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Michael Palin</td>
      <td>B|C|D</td>
    </tr>
  </tbody>
</table>
</div>




```python
full_monte['info'].str.get_dummies('|')
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
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



# Example: Recipe Database


```python
recipes = pd.read_json('data/recipeitems.json', lines=True)
recipes.shape
```




    (173278, 17)




```python
recipes.iloc[0]
```




    _id                                {'$oid': '5160756b96cc62079cc2db15'}
    name                                    Drop Biscuits and Sausage Gravy
    ingredients           Biscuits\n3 cups All-purpose Flour\n2 Tablespo...
    url                   http://thepioneerwoman.com/cooking/2013/03/dro...
    image                 http://static.thepioneerwoman.com/cooking/file...
    ts                                             {'$date': 1365276011104}
    cookTime                                                          PT30M
    source                                                  thepioneerwoman
    recipeYield                                                          12
    datePublished                                                2013-03-11
    prepTime                                                          PT10M
    description           Late Saturday afternoon, after Marlboro Man ha...
    totalTime                                                           NaN
    creator                                                             NaN
    recipeCategory                                                      NaN
    dateModified                                                        NaN
    recipeInstructions                                                  NaN
    Name: 0, dtype: object



## Goal to extract the ingredients from the recipes

### Compare lengths of the string values


```python
recipes.ingredients.str.len().describe()
```




    count    173278.000000
    mean        244.617926
    std         146.705285
    min           0.000000
    25%         147.000000
    50%         221.000000
    75%         314.000000
    max        9067.000000
    Name: ingredients, dtype: float64



### Longest ingredient list


```python
recipes.name[np.argmax(recipes.ingredients.str.len())]
```




    'Carrot Pineapple Spice &amp; Brownie Layer Cake with Whipped Cream &amp; Cream Cheese Frosting and Marzipan Carrots'




```python
recipes.description.str.contains(r'[Bb]reakfast').sum()
```




    3524




```python
recipes.ingredients.str.contains('[Cc]innamon').sum()
```




    10526




```python
recipes.ingredients.str.contains('[Cc]inamon').sum()
```




    11




```python
recipes.ingredients.str.contains('[Cc]in*amon').sum()
```




    10538




```python
recipes.ingredients.str.contains('[Cc]innnamon').sum()
```




    1



## Recipe recommender 

Find recipes containing a given list of ingredients.


```python
spice_list = ['salt', 'pepper', 'oregano', 'sage', 'parsley',
              'rosemary', 'tarragon', 'thyme', 'paprika', 'cumin']
```

Build a dataframe of booleans with spices as columns and recipes as rows.


```python
import re 
spice_df = pd.DataFrame({
    spice: recipes.ingredients.str.contains(spice, re.IGNORECASE)
    for spice in spice_list
})
spice_df.head()
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
      <th>salt</th>
      <th>pepper</th>
      <th>oregano</th>
      <th>sage</th>
      <th>parsley</th>
      <th>rosemary</th>
      <th>tarragon</th>
      <th>thyme</th>
      <th>paprika</th>
      <th>cumin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



Use `query` to find matching recipes


```python
selection = spice_df.query('parsley & paprika & tarragon')
len(selection)
```




    10




```python
type(selection)
```




    pandas.core.frame.DataFrame




```python
selection.index
```




    Index([2069, 74964, 93768, 113926, 137686, 140530, 158475, 158486, 163175,
           165243],
          dtype='int64')




```python
recipes.name[selection.index]
```




    2069      All cremat with a Little Gem, dandelion and wa...
    74964                         Lobster with Thermidor butter
    93768      Burton's Southern Fried Chicken with White Gravy
    113926                     Mijo's Slow Cooker Shredded Beef
    137686                     Asparagus Soup with Poached Eggs
    140530                                 Fried Oyster Po’boys
    158475                Lamb shank tagine with herb tabbouleh
    158486                 Southern fried chicken in buttermilk
    163175            Fried Chicken Sliders with Pickles + Slaw
    165243                        Bar Tartine Cauliflower Salad
    Name: name, dtype: object




```python

```
