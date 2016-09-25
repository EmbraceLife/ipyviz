# pandas_examples


[How to load a file with pd.read_table()?](http://localhost:8888/notebooks/scripts/Load%20file%20with%20pd.read_table.ipynb)

```python
import pandas as pd
orders = pd.read_table('../data/chipotle.tsv')
```


[How to separate columns by '|', when reading/loading data file?](http://localhost:8888/notebooks/scripts/Load%20file%20with%20pd.read_table.ipynb)    
[How to specify no header in data file, when reading/loading data file?](http://localhost:8888/notebooks/scripts/Load%20file%20with%20pd.read_table.ipynb)   
[How to specify header as the first row in data file and rename the columns, when reading/loading data file?](http://localhost:8888/notebooks/scripts/Load%20file%20with%20pd.read_table.ipynb)   
[How to provide column names for data file, when reading/loading data file?](http://localhost:8888/notebooks/scripts/Load%20file%20with%20pd.read_table.ipynb)   

```python
user_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
# a table without header, must use header=None
users = pd.read_table('../data/u.user.txt', sep='|', header=None, names=user_cols)

users = pd.read_table('../data/u.user.txt', sep='|', header=0, names=user_cols)
```


[How to load a csv file with pd.read_table() and pd.read_csv()?](http://localhost:8888/notebooks/scripts/How%20to%20load%20a%20csv%20file%20with%20pd.read_table%20and%20pd.read_csv%3F.ipynb)
```python
import pandas as pd

ufo = pd.read_table('../data/ufo.csv', sep=',')

ufo = pd.read_csv('../data/ufo.csv')

```

[How to extract a series of a dataframe?](http://localhost:8888/notebooks/scripts/How%20to%20access%20a%20series%20of%20a%20dataframe%20with%20dot%20and%20brackets%20with%20strings%3F.ipynb)
[How to create a new column for a dataframe ?](http://localhost:8888/notebooks/scripts/How%20to%20access%20a%20series%20of%20a%20dataframe%20with%20dot%20and%20brackets%20with%20strings%3F.ipynb)

```python
import pandas as pd

ufo = pd.read_csv('../data/ufo.csv')

# access a column or series
ufo.City
ufo['City']

# create a new column
ufo['Location'] = ufo.City + ', ' + ufo.State
```


[use of pd.shape, pd.dtypes, pd.ndim, pd.describe()?](http://localhost:8888/notebooks/scripts/How%20to%20use%20pd.head%2C%20pd.shape%2C%20pd.ndim%2C%20pd.describe%2C%20pd.ntypes.ipynb)
```python
import pandas as pd

movies = pd.read_csv('../data/imdb_1000.csv')

movies.describe()

movies.shape

movies.dtypes

movies.describe(exclude=['int64'])
movies.describe(include=['object', 'float64'])
movies.describe(include=['object'])

movies.ndim
```

[How to rename column names using `pd.rename()` in pandas dataframe?](http://localhost:8888/notebooks/scripts/How%20to%20rename%20column%20names%20in%20pandas%20dataframe.ipynb)    
[How to rename column names using `pd.columns` in pandas dataframe?](http://localhost:8888/notebooks/scripts/How%20to%20rename%20column%20names%20in%20pandas%20dataframe.ipynb)     
[How to rename column names using `pd.columns.str.replace()` in pandas dataframe?](http://localhost:8888/notebooks/scripts/How%20to%20rename%20column%20names%20in%20pandas%20dataframe.ipynb)     

```python
import pandas as pd
ufo = pd.read_csv('../data/ufo.csv')

ufo.columns

ufo.rename(columns={'Colors Reported':'Colors_Reported', 'Shape Reported':'Shape_Reported'}, inplace=True)

ufo_cols = ['city', 'colors reported', 'shape reported', 'state', 'time']
ufo.columns = ufo_cols

ufo = pd.read_csv('../data/ufo.csv', header=0, names=ufo_cols)
ufo.columns = ufo.columns.str.replace(' ', '_')
```



[How to drop or remove rows and columns of a pandas dataframe?](http://localhost:8888/notebooks/scripts/How%20to%20drop%20rows%20and%20columns%20of%20pandas%20dataframe%3F.ipynb)
```python
import pandas as pd
ufo = pd.read_csv('../data/ufo.csv')
ufo.head()

ufo.drop('Colors Reported', axis=1, inplace=True)
ufo.head()

ufo.drop(['City', 'State'], axis=1, inplace=True)
ufo.head()

ufo.drop([0, 1], axis=0, inplace=True)
ufo.head()
```

[How to sort a single series with ascending order?](http://localhost:8888/notebooks/scripts/How%20to%20sort%20series%20or%20dataframe%20based%20on%20series%3F%20.ipynb)    
[How to sort a single series with descending order?](http://localhost:8888/notebooks/scripts/How%20to%20sort%20series%20or%20dataframe%20based%20on%20series%3F%20.ipynb)    
[How to sort a single series and force other columns of the dataframe to follow?](http://localhost:8888/notebooks/scripts/How%20to%20sort%20series%20or%20dataframe%20based%20on%20series%3F%20.ipynb)    
[How to sort two series and force other columns to follow?](http://localhost:8888/notebooks/scripts/How%20to%20sort%20series%20or%20dataframe%20based%20on%20series%3F%20.ipynb)    

```python

import pandas as pd
movies = pd.read_csv('../data/imdb_1000.csv')
movies.head()

movies.title.sort_values().head()
movies.title.sort_values(ascending=False).head()

movies.sort_values('title').head()
movies.sort_values('title', ascending=False).head()

movies.sort_values(['content_rating', 'duration']).head()
```

[How to create an empty list and fill it value by value](http://localhost:8888/notebooks/scripts/How%20to%20filter%20rows%20with%20columns%20and%20conditions%20for%20pandas%20dataframe%3F.ipynb)     
[How to turn list to pd.Series to use functions like .head()](http://localhost:8888/notebooks/scripts/How%20to%20filter%20rows%20with%20columns%20and%20conditions%20for%20pandas%20dataframe%3F.ipynb)     
[list converted to pd.Series without column dimension and pd.DataFrame with  columns dimension ](http://localhost:8888/notebooks/scripts/How%20to%20filter%20rows%20with%20columns%20and%20conditions%20for%20pandas%20dataframe%3F.ipynb)     

```python
import pandas as pd
movies = pd.read_csv('../data/imdb_1000.csv')
movies.head()
movies.shape
booleans = []
for length in movies.duration:
    if length >= 200:
        booleans.append(True)
    else:
        booleans.append(False)

len(booleans)
booleans[0:5]

is_long = pd.Series(booleans)
is_long.head()

pd.Series(booleans).shape
pd.DataFrame(booleans).shape
```

[How to filter the whole dataframe by constrain on a series/column?](http://localhost:8888/notebooks/scripts/How%20to%20filter%20rows%20with%20columns%20and%20conditions%20for%20pandas%20dataframe%3F.ipynb)    
[How to filter the whole dataframe by constrain on a series/column, but only select one column to return?](http://localhost:8888/notebooks/scripts/How%20to%20filter%20rows%20with%20columns%20and%20conditions%20for%20pandas%20dataframe%3F.ipynb)      

```python
import pandas as pd
movies = pd.read_csv('../data/imdb_1000.csv')
movies.head()
movies.shape

movies[movies.duration >= 200]

movies.loc[movies.duration >= 200, 'genre']
```

[How to filter dataframe with 2 or more conditions ?](http://localhost:8888/notebooks/scripts/How%20to%20filter%20on%20multiple%20conditions%20for%20pandas%20dataframe%3F%20.ipynb)

```python

import pandas as pd
movies = pd.read_csv('../data/imdb_1000.csv')
movies.head()
movies[movies.duration >= 200]

movies[(movies.duration >=200) & (movies.genre == 'Drama')]
movies[(movies.duration >=200) | (movies.genre == 'Drama')].head()

movies[(movies.genre == 'Crime') | (movies.genre == 'Drama') | (movies.genre == 'Action')].head(10)

movies[movies.genre.isin(['Crime', 'Drama', 'Action'])].head(10)
```

[How to read a data file into dataframe but with only specific columns?](http://localhost:8888/notebooks/scripts/How%20to%20read%20in%20data%20file%20into%20a%20dataframe%20with%20only%20specific%20columns%20%3F.ipynb)     
```python
import pandas as pd

# read a dataset of UFO reports into a DataFrame, and check the columns
ufo = pd.read_csv('../data/ufo.csv')
ufo.columns

# specify which columns to include by name
ufo = pd.read_csv('../data/ufo.csv', usecols=['City', 'State'])
ufo.columns

# or equivalently, specify columns by position
ufo = pd.read_csv('../data/ufo.csv', usecols=[0, 4])
ufo.columns
```


[How to read a data file into dataframe with specific rows?](http://localhost:8888/notebooks/scripts/How%20to%20read%20in%20data%20file%20into%20dataframe%20with%20only%20specific%20rows%20.ipynb)    

```python
import pandas as pd
ufo = pd.read_csv('../data/ufo.csv', nrows=3)
ufo
```



[How to iterate a single series by rows?](http://localhost:8888/notebooks/scripts/How%20to%20iterate%20series%20and%20dataframe%20by%20rows%3F%20.ipynb)    
[How to iterate a dataframe by rows?](http://localhost:8888/notebooks/scripts/How%20to%20iterate%20series%20and%20dataframe%20by%20rows%3F%20.ipynb)    

```python

import pandas as pd
ufo = pd.read_csv('../data/ufo.csv')

for c in ufo.City:
    print(c)

for index, row in ufo.head().iterrows():
    print(index, row.City, row.State)

for index, row in ufo[0:10].iterrows():
    print(index, row.City, row.State)
    if index > 3:
        print(index)


```

[How to describe different columns of a dataframe?](http://localhost:8888/notebooks/scripts/How%20to%20use%20pd.describe%20and%20How%20to%20select%20only%20numeric%20columns%3F.ipynb)    
```python
# describe all of the numeric columns
drinks.describe()

# pass the string 'all' to describe all columns
drinks.describe(include='all')

# pass a list of data types to only describe certain types
drinks.describe(include=['object', 'float64'])

# pass a list even if you only want to describe a single data type
drinks.describe(include=['object'])
```



[How to select only numeric columns from a dataframe?](http://localhost:8888/notebooks/scripts/How%20to%20use%20pd.describe%20and%20How%20to%20select%20only%20numeric%20columns%3F.ipynb)    
```python
import pandas as pd

drinks = pd.read_csv('../data/drinks.csv')
drinks.dtypes

import numpy as np
drinks.select_dtypes(include=[np.float]).dtypes
drinks.select_dtypes(include=[np.int]).dtypes
drinks.select_dtypes(include=[np.number]).dtypes
```


[How to apply functions like `pd.drop()` and `pd.mean()` to rows and columns of dataframe?](http://localhost:8888/notebooks/scripts/How%20to%20apply%20functions%20to%20rows%20and%20columns%20of%20dataframe%3F.ipynb)
```python
import pandas as pd

drinks = pd.read_csv('../data/drinks.csv')
drinks.head()

drinks.drop('continent', axis=1).head()

drinks.drop(2, axis=0).head()

drinks.mean()

drinks.mean(axis=0)

drinks.mean(axis=1).head()

drinks.mean(axis='index')

drinks.mean(axis='columns').head()
```


[How to use string methods in pandas dataframe?](http://localhost:8888/notebooks/scripts/How%20to%20use%20string%20methods%20in%20pandas%20dataframe%3F.ipynb)
```python
import pandas as pd

orders = pd.read_table('../data/chipotle.tsv')
orders.head()

'hello'.upper()

orders.item_name.str.upper().head()

orders.item_name.str.contains('Chicken').head()

orders[orders.item_name.str.contains('Chicken')].head(10)

orders.choice_description.str.replace('[', '').str.replace(']', '').head()

orders.choice_description.str.replace('[\[\]]', '').head()
```

[How to change data type of pandas series in a dataframe?](http://localhost:8888/notebooks/scripts/How%20to%20change%20data%20type%20of%20pandas%20dataframe%3F%20.ipynb)    
[How to change data type of pandas series of a dataframe when reading data file?](http://localhost:8888/notebooks/scripts/How%20to%20change%20data%20type%20of%20pandas%20dataframe%3F%20.ipynb)    
```python

import pandas as pd

drinks = pd.read_csv('../data/drinks.csv')
drinks.dtypes

drinks['beer_servings'] = drinks.beer_servings.astype(float)
drinks.dtypes

drinks = pd.read_csv('../data/drinks.csv', dtype={'beer_servings':float})
drinks.dtypes

orders = pd.read_table('../data/chipotle.tsv')
orders.dtypes

orders.item_price.str.replace('$', '').astype(float).mean()

orders.item_name.str.contains('Chicken').head()

orders.item_name.str.contains('Chicken').astype(int).head()
```


[How to groupby in pandas dataframe?](http://localhost:8888/notebooks/scripts/How%20and%20when%20to%20use%20groupby%20in%20pandas%20dataframe%3F.ipynb)     
[How to groupby and apply statistics to pandas dataframe?](http://localhost:8888/notebooks/scripts/How%20and%20when%20to%20use%20groupby%20in%20pandas%20dataframe%3F.ipynb)     
```python
import pandas as pd
drinks = pd.read_csv('../data/drinks.csv')

drinks.beer_servings.mean()

drinks[drinks.continent=='Africa'].beer_servings.mean()

drinks.groupby('continent').beer_servings.mean()

drinks.groupby('continent').beer_servings.max()

drinks.groupby('continent').beer_servings.agg(['count', 'mean', 'min', 'max'])

drinks.groupby('continent').mean()

%matplotlib inline

drinks.groupby('continent').mean().plot(kind='bar')
```

[How to get counts of a pd.Series in a dataframe?](http://localhost:8888/notebooks/scripts/often%20used%20pd.series%20methods.ipynb)     
[How to get frequency of a pd.Series in a dataframe?](http://localhost:8888/notebooks/scripts/often%20used%20pd.series%20methods.ipynb)     
[How to get all unique values of a pd.Series in a dataframe?](http://localhost:8888/notebooks/scripts/often%20used%20pd.series%20methods.ipynb)     
[How to get number of unique values of a pd.Series in a dataframe?](http://localhost:8888/notebooks/scripts/often%20used%20pd.series%20methods.ipynb)     
[How to create a crosstable of two pd.Series in a dataframe?](http://localhost:8888/notebooks/scripts/often%20used%20pd.series%20methods.ipynb)     
```python

import pandas as pd
movies = pd.read_csv('../data/imdb_1000.csv')

movies.dtypes

movies.genre.describe()

movies.genre.value_counts()

movies.genre.value_counts(normalize=True)

type(movies.genre.value_counts())

movies.genre.value_counts().head()

movies.genre.unique()

movies.genre.nunique()

pd.crosstab(movies.genre, movies.content_rating)

%matplotlib inline

movies.duration.plot(kind='hist')

movies.genre.value_counts().plot(kind='bar')
```

[How to check each value for NaN in pandas dataframe?](http://localhost:8888/notebooks/scripts/How%20to%20handle%20missing%20values%20in%20pandas%20dataframe%3F.ipynb)     
[How to check each value for NON-NaN in pandas dataframe?](http://localhost:8888/notebooks/scripts/How%20to%20handle%20missing%20values%20in%20pandas%20dataframe%3F.ipynb)     
[How to count each column's total NaN in pandas dataframe?](http://localhost:8888/notebooks/scripts/How%20to%20handle%20missing%20values%20in%20pandas%20dataframe%3F.ipynb)     
[How to drop a row when just one value is NaN in pandas dataframe?](http://localhost:8888/notebooks/scripts/How%20to%20handle%20missing%20values%20in%20pandas%20dataframe%3F.ipynb)     
[How to drop a row when every value is NaN in pandas dataframe?](http://localhost:8888/notebooks/scripts/How%20to%20handle%20missing%20values%20in%20pandas%20dataframe%3F.ipynb)     
[How to drop a row when specified column values are NaN in pandas dataframe?](http://localhost:8888/notebooks/scripts/How%20to%20handle%20missing%20values%20in%20pandas%20dataframe%3F.ipynb)     
[How to count a column's unique values occurrence while/not dropping NA in pandas dataframe?](http://localhost:8888/notebooks/scripts/How%20to%20handle%20missing%20values%20in%20pandas%20dataframe%3F.ipynb)     
[How to fill a column's NaN with a specific value in pandas dataframe?](http://localhost:8888/notebooks/scripts/How%20to%20handle%20missing%20values%20in%20pandas%20dataframe%3F.ipynb)     
```python

import pandas as pd
ufo = pd.read_csv('../data/ufo.csv')

ufo.isnull().tail() # isnull to detect NaN
ufo.notnull().tail()

ufo.isnull().sum()
ufo[ufo.City.isnull()].head()

ufo.shape

ufo.dropna(how='any').shape

ufo.dropna(how='all').shape

ufo.dropna(subset=['City', 'Shape Reported'], how='any').shape

ufo.dropna(subset=['City', 'Shape Reported'], how='all').shape

ufo['Shape Reported'].value_counts().head()
ufo['Shape Reported'].value_counts(dropna=False).head()

ufo['Shape Reported'].fillna(value='VARIOUS', inplace=True)
ufo['Shape Reported'].value_counts().head()

```

[How to make use of pandas dataframe index?](http://localhost:8888/notebooks/scripts/How%20to%20make%20use%20of%20pandas%20dataframe%20index%3F.ipynb)
```python

import pandas as pd
drinks = pd.read_csv('../data/drinks.csv')


print(drinks.index)

drinks[drinks.continent=='South America']
drinks.loc[23, 'beer_servings']

drinks.set_index('country', inplace=True)
print(drinks.index)

drinks.loc['Brazil', 'beer_servings']

drinks.index.name = None    # get rid of indexName "country"
drinks.index.name = 'country'

drinks.describe().loc['25%', 'beer_servings']
```


[How to set display for specific num of column and row?] (http://localhost:8888/notebooks/scripts/more%20about%20pandas%20index.ipynb)    
[How to replace numeric index with a particular column as index for dataframe?] (http://localhost:8888/notebooks/scripts/more%20about%20pandas%20index.ipynb)    
[How to to access index and values from a pd.Series?] (http://localhost:8888/notebooks/scripts/more%20about%20pandas%20index.ipynb)    
[How to access values from index of a pd.Series?] (http://localhost:8888/notebooks/scripts/more%20about%20pandas%20index.ipynb)    
[How to create a pd.Series given values, index and name of Series?] (http://localhost:8888/notebooks/scripts/more%20about%20pandas%20index.ipynb)    
[How to do calculation between pd.Series even they are not equal length?] (http://localhost:8888/notebooks/scripts/more%20about%20pandas%20index.ipynb)    
[How to combine a dataframe with a pd.Series even they are with different length?] (http://localhost:8888/notebooks/scripts/more%20about%20pandas%20index.ipynb)    
```python

import pandas as pd
from IPython.display import display
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)

drinks = pd.read_csv('../data/drinks.csv')
drinks.head()
drinks.index

drinks.set_index('country', inplace=True)
drinks.continent.head()
drinks.continent.value_counts()

drinks.continent.value_counts().index
drinks.continent.value_counts().values
drinks.continent.value_counts()['Africa']


drinks.continent.value_counts().sort_values()
drinks.continent.value_counts().sort_index()

drinks.beer_servings.head()

people = pd.Series([3000000, 85000], index=['Albania', 'Andorra'], name='population')

(drinks.beer_servings * people).head()


pd.concat([drinks, people], axis=1).head()
```

[access and download a stock's fundamental and technical dataset for a period] (https://uqer.io/labs/notebooks/fundamental%20data.nb)
```python
data = DataAPI.MktStockFactorsDateRangeGet(secID=u"",ticker=u"002594",
                                    beginDate=u"20150101",endDate=u"20150701",field=u"ticker,tradeDate,pe,ar,ARTRate",pandas="1")
data.to_csv('byd_fundamental.csv', encoding="GBK")
```

[access and download change of number of shares](https://uqer.io/labs/notebooks/change%20of%20num%20of%20Shares.nb)
```python
DataAPI.EquShareGet(secID=u"",ticker=u"002594",beginDate=u"20110630",endDate=u"20160903",partyID=u"",
  field=u"ticker,secShortName,changeDate,totalShares,sharesA,nonrestfloatA,restShares,nonrestFloatShares",pandas="1")

```


[How to access values by index (numeric) and column (names) in a dataframe?](http://localhost:8888/notebooks/scripts/use%20of%20loc%20and%20ix.ipynb)    
[How to access values by condition on index and column (names) in a dataframe?](http://localhost:8888/notebooks/scripts/use%20of%20loc%20and%20ix.ipynb)    
[How to access values by index (numeric) and column (numeric not names) in a dataframe?](http://localhost:8888/notebooks/scripts/use%20of%20loc%20and%20ix.ipynb)    
[How to access values by index (numeric or names) and column (names or numeric) in a dataframe?](http://localhost:8888/notebooks/scripts/use%20of%20loc%20and%20ix.ipynb)    

```python
import pandas as pd
from IPython.display import display
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)

ufo = pd.read_csv('../data/ufo.csv')

ufo.loc[0, :]
ufo.loc[[0, 1, 2], :]
ufo.loc[0:2, :]
ufo.loc[0:2]
ufo.loc[0:2, 'City']
ufo.loc[0:2, ['City', 'State']]
ufo.loc[0:2, 'City':'State']

ufo.loc[ufo.City=='Oakland', 'State']


ufo.iloc[[0, 1], [0, 3]]
ufo.iloc[0:2, 0:4]
ufo.iloc[0:2, :]

drinks = pd.read_csv('../data/drinks.csv', index_col='country')

drinks.ix['Albania', 0]
drinks.ix[1, 'beer_servings']

drinks.ix['Albania':'Andorra', 0:2]
ufo.ix[0:2, 0:2]
```

[How to back and forward fill NA of a dataframe or a pd.Series?](http://localhost:8888/notebooks/scripts/use%20of%20inplace%20and%20dropna%20with%20method%20bfill%20or%20ffill.ipynb)
```python

import pandas as pd
from IPython.display import display

ufo = pd.read_csv('../data/ufo.csv')
ufo.drop('City', axis=1).head()

ufo.drop('City', axis=1, inplace=True)

ufo.dropna(how='any').shape

ufo = ufo.set_index('Time')


ufo['Colors Reported'].fillna(method='bfill').tail()

ufo.fillna(method='bfill').tail()

ufo.fillna(method='ffill').tail(10)
```


[How to make pandas dataframe smaller and faster?](http://localhost:8888/notebooks/scripts/How%20to%20make%20pandas%20dataframe%20smaller%20and%20faster.ipynb)    
[How much memory does a data file or dataframe require?](http://localhost:8888/notebooks/scripts/How%20to%20make%20pandas%20dataframe%20smaller%20and%20faster.ipynb)    
[How to change dtype object to category?](http://localhost:8888/notebooks/scripts/How%20to%20make%20pandas%20dataframe%20smaller%20and%20faster.ipynb)    
[How to turn category levels to codes?](http://localhost:8888/notebooks/scripts/How%20to%20make%20pandas%20dataframe%20smaller%20and%20faster.ipynb)    
[How to access category values/levels?](http://localhost:8888/notebooks/scripts/How%20to%20make%20pandas%20dataframe%20smaller%20and%20faster.ipynb)    
[How to create and sort a dataframe?](http://localhost:8888/notebooks/scripts/How%20to%20make%20pandas%20dataframe%20smaller%20and%20faster.ipynb)    
[How to convert a category column to a ordered category column?](http://localhost:8888/notebooks/scripts/How%20to%20make%20pandas%20dataframe%20smaller%20and%20faster.ipynb)    
[How to create a filter condition based on ordered category column?](http://localhost:8888/notebooks/scripts/How%20to%20make%20pandas%20dataframe%20smaller%20and%20faster.ipynb)    
```python

import pandas as pd
from IPython.display import display

drinks = pd.read_csv('../data/drinks.csv')

drinks.info()

drinks.info(memory_usage='deep')
drinks.memory_usage(deep=True)


drinks['continent'] = drinks.continent.astype('category')
drinks.dtypes

display(drinks.continent.cat.codes.head())
drinks.continent.cat.codes.unique()

drinks['country'] = drinks.country.astype('category')
drinks.memory_usage(deep=True)

drinks.country.cat.categories

df = pd.DataFrame({'ID':[100, 101, 102, 103], 'quality':['good', 'very good', 'good', 'excellent']})
df

df.sort_values('quality')

df['quality'] = df.quality.astype('category', categories=['good', 'very good', 'excellent'], ordered=True)
df.quality

df.sort_values('quality')

df.loc[df.quality > 'good', :]
```


[How do I use pandas with scikit-learn to create Kaggle submissions with pickle?](http://localhost:8888/notebooks/scripts/How%20do%20I%20use%20pandas%20with%20scikit-learn%20to%20create%20Kaggle%20submissions%3F.ipynb)    
[how to create a feature dataframe and survived series](http://localhost:8888/notebooks/scripts/How%20do%20I%20use%20pandas%20with%20scikit-learn%20to%20create%20Kaggle%20submissions%3F.ipynb)    
[how to access LogisticRegression from sklearn package?](http://localhost:8888/notebooks/scripts/How%20do%20I%20use%20pandas%20with%20scikit-learn%20to%20create%20Kaggle%20submissions%3F.ipynb)    
[how to train LogisticRegression model with feature dataframe and survived column?](http://localhost:8888/notebooks/scripts/How%20do%20I%20use%20pandas%20with%20scikit-learn%20to%20create%20Kaggle%20submissions%3F.ipynb)    
[how to calculate test-survived-column with LogisticRegression model with test-features-dataframe?](http://localhost:8888/notebooks/scripts/How%20do%20I%20use%20pandas%20with%20scikit-learn%20to%20create%20Kaggle%20submissions%3F.ipynb)    
[how to create a dataframe from a dictionary within two series inside?](http://localhost:8888/notebooks/scripts/How%20do%20I%20use%20pandas%20with%20scikit-learn%20to%20create%20Kaggle%20submissions%3F.ipynb)    
[how to save and read csv and pkl files?](http://localhost:8888/notebooks/scripts/How%20do%20I%20use%20pandas%20with%20scikit-learn%20to%20create%20Kaggle%20submissions%3F.ipynb)    


```python

import pandas as pd
from IPython.display import display

train = pd.read_csv('../data/titanic_train.csv')
train.head()

feature_cols = ['Pclass', 'Parch']
X = train.loc[:, feature_cols]
X.shape

y = train.Survived
y.shape

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X, y)

test = pd.read_csv('../data/titanic_test.csv')

X_new = test.loc[:, feature_cols]

new_pred_class = logreg.predict(X_new)

pd.DataFrame({'PassengerId':test.PassengerId, 'Survived':new_pred_class}).head()

pd.DataFrame({'PassengerId':test.PassengerId, 'Survived':new_pred_class}).set_index('PassengerId').head()

pd.DataFrame({'PassengerId':test.PassengerId, 'Survived':new_pred_class}).set_index('PassengerId').to_csv('sub.csv')

train.to_pickle('train.pkl')

pd.read_pickle('train.pkl').head()
```


[pd.isnull, pd.df.isnull, df.loc, df.iloc, df.sample with args n, frac, random_state](http://localhost:8888/notebooks/scripts/pd.isnull%2C%20pd.df.isnull%2C%20df.loc%2C%20df.iloc%2C%20df.sample%20with%20args%20n%2C%20frac%2C%20random_state.ipynb)    
[pd.loc() is inclusive on index range, pd.iloc() is exclusive on index range end point](http://localhost:8888/notebooks/scripts/pd.isnull%2C%20pd.df.isnull%2C%20df.loc%2C%20df.iloc%2C%20df.sample%20with%20args%20n%2C%20frac%2C%20random_state.ipynb)    
[How to randomly select a number of rows of a dataframe?](http://localhost:8888/notebooks/scripts/pd.isnull%2C%20pd.df.isnull%2C%20df.loc%2C%20df.iloc%2C%20df.sample%20with%20args%20n%2C%20frac%2C%20random_state.ipynb)    
[How to randomly select a proportion of rows of a dataframe?](http://localhost:8888/notebooks/scripts/pd.isnull%2C%20pd.df.isnull%2C%20df.loc%2C%20df.iloc%2C%20df.sample%20with%20args%20n%2C%20frac%2C%20random_state.ipynb)    
[How to randomly select a proportion of rows of a dataframe?](http://localhost:8888/notebooks/scripts/pd.isnull%2C%20pd.df.isnull%2C%20df.loc%2C%20df.iloc%2C%20df.sample%20with%20args%20n%2C%20frac%2C%20random_state.ipynb)    
[How to randomly select with a random-seed?](http://localhost:8888/notebooks/scripts/pd.isnull%2C%20pd.df.isnull%2C%20df.loc%2C%20df.iloc%2C%20df.sample%20with%20args%20n%2C%20frac%2C%20random_state.ipynb)    

```python

import pandas as pd
from IPython.display import display

ufo = pd.read_csv('../data/ufo.csv')
ufo.head()

pd.isnull(ufo).head()

ufo.isnull().head()

ufo.loc[0:4, :]
ufo.iloc[0:4, :]

ufo.values[0:4, :]

'python'[0:4]
ufo.loc[0:4, 'City':'State']

ufo.sample(n=3)
ufo.sample(n=3, random_state=42)
train = ufo.sample(frac=0.75, random_state=99)

test = ufo.loc[~ufo.index.isin(train.index), :]
```



[How to create a new Series using pd.SeriesName.map()?](http://localhost:8888/notebooks/scripts/How%20to%20create%20dummy%20variables%20in%20pandas%3F.ipynb)    
[How to convert a single column full of 'female' and 'male' values into female and male two columns with 1 and 0?](http://localhost:8888/notebooks/scripts/How%20to%20create%20dummy%20variables%20in%20pandas%3F.ipynb)    
[How to access a particular column out of the converted 2 or more columns?](http://localhost:8888/notebooks/scripts/How%20to%20create%20dummy%20variables%20in%20pandas%3F.ipynb)    
[How to modify converted column names using prefix argument?](http://localhost:8888/notebooks/scripts/How%20to%20create%20dummy%20variables%20in%20pandas%3F.ipynb)    
[How to convert more than one column into several columns?](http://localhost:8888/notebooks/scripts/How%20to%20create%20dummy%20variables%20in%20pandas%3F.ipynb)    
[How to drop the first column out of the converted several columns?](http://localhost:8888/notebooks/scripts/How%20to%20create%20dummy%20variables%20in%20pandas%3F.ipynb)    
```python
import pandas as pd

train = pd.read_csv('../data/titanic_train.csv')

train['Sex_male'] = train.Sex.map({'female':0, 'male':1})

pd.get_dummies(train.Sex).head()

pd.get_dummies(train.Sex).iloc[:, 1:].head()

pd.get_dummies(train.Sex, prefix='Sex').iloc[:, 0:].head()

pd.get_dummies(train.Embarked, prefix='Embarked').head(10)

pd.get_dummies(train.Embarked, prefix='Embarked').iloc[:, 1:].head(10)

embarked_dummies = pd.get_dummies(train.Embarked, prefix='Embarked').iloc[:, 1:]

train = pd.concat([train, embarked_dummies], axis=1)

train = pd.read_csv('../data/titanic_train.csv')

pd.get_dummies(train, columns=['Sex', 'Embarked']).head()

pd.get_dummies(train, columns=['Sex', 'Embarked'], drop_first=True).head()
```


[How to slice a Series/column of strings in a dataframe?](http://localhost:8888/notebooks/scripts/How%20to%20work%20with%20dates%20and%20times%20in%20pandas%3F.ipynb)     
[How to convert datetime strings to datetime type in a dataframe?](http://localhost:8888/notebooks/scripts/How%20to%20work%20with%20dates%20and%20times%20in%20pandas%3F.ipynb)     
[How to access hour/weekday_name/days/dayofyear values of a column of datetime type?](http://localhost:8888/notebooks/scripts/How%20to%20work%20with%20dates%20and%20times%20in%20pandas%3F.ipynb)     
[How to create a single pd.datetime value and use to compare with other column of datetime values?](http://localhost:8888/notebooks/scripts/How%20to%20work%20with%20dates%20and%20times%20in%20pandas%3F.ipynb)     
[How to do max, min, calculation on two datetime values?](http://localhost:8888/notebooks/scripts/How%20to%20work%20with%20dates%20and%20times%20in%20pandas%3F.ipynb)     
```python

import pandas as pd
from IPython.display import display

ufo = pd.read_csv('../data/ufo.csv')
ufo.head()

ufo.dtypes

ufo.Time.str.slice(-5, -3).astype(int).head()

ufo['Time'] = pd.to_datetime(ufo.Time)
ufo.head()

ufo.dtypes

ufo.Time.dt.hour.head()
ufo.Time.dt.weekday_name.head()
ufo.Time.dt.dayofyear.head()

ts = pd.to_datetime('1/1/1999')

ufo.loc[ufo.Time >= ts, :].head()

ufo.Time.max()

ufo.Time.max() - ufo.Time.min()

(ufo.Time.max() - ufo.Time.min()).days


ufo['Year'] = ufo.Time.dt.year


get_ipython().magic(u'matplotlib inline')
ufo.Year.value_counts().sort_index().plot()
```


[How to read datafile while give it column names and make one column to be index?](http://localhost:8888/notebooks/scripts/How%20to%20find%20and%20remove%20duplicate%20rows%20in%20pandas%3F.ipynb)     
[How to check whether every value of a column of a dataframe is a duplicate to a value previously somewhere in the same column?](http://localhost:8888/notebooks/scripts/How%20to%20find%20and%20remove%20duplicate%20rows%20in%20pandas%3F.ipynb)     
[How to get sum of all duplicates in a column of a dataframe?](http://localhost:8888/notebooks/scripts/How%20to%20find%20and%20remove%20duplicate%20rows%20in%20pandas%3F.ipynb)     
[How to check every row of a dataframe is a duplicate of a row previously somewhere?](http://localhost:8888/notebooks/scripts/How%20to%20find%20and%20remove%20duplicate%20rows%20in%20pandas%3F.ipynb)     
[How to consider only those rows appear in latter to be duplicated rows (keep='first' part as non-duplicate)?](http://localhost:8888/notebooks/scripts/How%20to%20find%20and%20remove%20duplicate%20rows%20in%20pandas%3F.ipynb)     
[How to consider only those rows appear in former to be duplicated rows (keep='last' part as non-duplicate) ?](http://localhost:8888/notebooks/scripts/How%20to%20find%20and%20remove%20duplicate%20rows%20in%20pandas%3F.ipynb)     
[How to keep all duplicated rows ?](http://localhost:8888/notebooks/scripts/How%20to%20find%20and%20remove%20duplicate%20rows%20in%20pandas%3F.ipynb)     
[How to drop the duplicates appear in first part of dataframe?](http://localhost:8888/notebooks/scripts/How%20to%20find%20and%20remove%20duplicate%20rows%20in%20pandas%3F.ipynb)    
[How to drop the duplicates appear in last part of dataframe?](http://localhost:8888/notebooks/scripts/How%20to%20find%20and%20remove%20duplicate%20rows%20in%20pandas%3F.ipynb)    
[How to drop all duplicates of the dataframe?](http://localhost:8888/notebooks/scripts/How%20to%20find%20and%20remove%20duplicate%20rows%20in%20pandas%3F.ipynb)    


```python

import pandas as pd

user_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
users = pd.read_table('../data/u.user.txt', sep='|', header=None, names=user_cols, index_col='user_id')


users.zip_code.duplicated().tail(10)

users.zip_code.duplicated().sum()

users.duplicated().tail()

users.duplicated().sum()

users.loc[users.duplicated(keep='first'), :]

users.loc[users.duplicated(keep='last'), :]

users.loc[users.duplicated(keep=False), :]

users.drop_duplicates(keep='first').shape

users.drop_duplicates(keep='first').loc[60:70,:]

users.drop_duplicates(keep='last').shape

users.drop_duplicates(keep='last').loc[60:70, :]

users.drop_duplicates(keep=False).shape

users.duplicated(subset=['age', 'zip_code']).sum()

print(users.duplicated('age').sum())
users.drop_duplicates('age').shape

print(users.duplicated('zip_code').sum())
users.drop_duplicates('zip_code').shape

users.drop_duplicates(subset=['age', 'zip_code']).shape

```


[How to replace certain values in a dataframe to be np.nan ?](http://localhost:8888/notebooks/scripts/How%20to%20avoid%20a%20SettingWithCopyWarning%20in%20pandas%3F.ipynb)     
[How to make a copy when create a new dataframe from a previous one?](http://localhost:8888/notebooks/scripts/How%20to%20avoid%20a%20SettingWithCopyWarning%20in%20pandas%3F.ipynb)     
```python

import pandas as pd

movies = pd.read_csv('../data/imdb_1000.csv')

movies.content_rating.isnull().sum()

movies[movies.content_rating.isnull()]

movies.content_rating.value_counts()

movies[movies.content_rating=='NOT RATED'].head()

movies[movies.content_rating=='NOT RATED'].content_rating.head()

import numpy as np

movies.loc[movies.content_rating=='NOT RATED', 'content_rating'] = np.nan
movies.loc[[5,6,41,63,66],'content_rating']

top_movies = movies.loc[movies.star_rating >= 9, :]
top_movies.loc[0, 'duration'] = 150
top_movies

top_movies = movies.loc[movies.star_rating >= 9, :].copy()

top_movies.loc[0, 'duration'] = 150
top_movies
```

[How to get, set and reset display options?](http://localhost:8888/notebooks/scripts/How%20to%20change%20display%20options%20in%20pandas%3F.ipynb)     
[How to display with specific precions for numbers?](http://localhost:8888/notebooks/scripts/How%20to%20change%20display%20options%20in%20pandas%3F.ipynb)     
[How to display thousands comma?](http://localhost:8888/notebooks/scripts/How%20to%20change%20display%20options%20in%20pandas%3F.ipynb)     

```python

import pandas as pd

drinks = pd.read_csv('../data/drinks.csv')

pd.get_option('display.max_rows')

pd.set_option('display.max_rows', None)

pd.reset_option('display.max_rows')

pd.get_option('display.max_rows')

pd.get_option('display.max_columns')

train = pd.read_csv('../data/titanic_train.csv')

pd.get_option('display.max_colwidth')

pd.set_option('display.max_colwidth', 1000)

pd.set_option('display.precision', 2)

drinks['x'] = drinks.wine_servings.astype("float64") * 1000
drinks['y'] = drinks.total_litres_of_pure_alcohol * 1000

pd.set_option('display.float_format', '{:,}'.format)

drinks.dtypes

pd.describe_option()
pd.describe_option('rows')
pd.reset_option('all')
```

[How do I create a pandas DataFrame from another object?](http://localhost:8888/notebooks/scripts/How%20do%20I%20create%20a%20pandas%20DataFrame%20from%20another%20object%3F.ipynb)     
[How to create dataframe from dictionary with or without a list of column names?](http://localhost:8888/notebooks/scripts/How%20do%20I%20create%20a%20pandas%20DataFrame%20from%20another%20object%3F.ipynb)     
[How to create dataframe from a list of values and a list of column names?](http://localhost:8888/notebooks/scripts/How%20do%20I%20create%20a%20pandas%20DataFrame%20from%20another%20object%3F.ipynb)     
[How to create dataframe from a random number array with a list of column names?](http://localhost:8888/notebooks/scripts/How%20do%20I%20create%20a%20pandas%20DataFrame%20from%20another%20object%3F.ipynb)     
[How to use np.random.rand(), np.random.randint(), np.arange()?](http://localhost:8888/notebooks/scripts/How%20do%20I%20create%20a%20pandas%20DataFrame%20from%20another%20object%3F.ipynb)     
[How to create a Series with value, index,  and column names ?](http://localhost:8888/notebooks/scripts/How%20do%20I%20create%20a%20pandas%20DataFrame%20from%20another%20object%3F.ipynb)     
```python

import pandas as pd

pd.DataFrame({'id':[100, 101, 102], 'color':['red', 'blue', 'red']})

df = pd.DataFrame({'id':[100, 101, 102], 'color':['red', 'blue', 'red']}, columns=['id', 'color'], index=['a', 'b', 'c'])

pd.DataFrame([[100, 'red'], [101, 'blue'], [102, 'red']], columns=['id', 'color'])

import numpy as np
arr = np.random.rand(4, 2)

pd.DataFrame(arr, columns=['one', 'two'])

pd.DataFrame({'student':np.arange(100, 110, 1), 'test':np.random.randint(60, 101, 10)})

pd.DataFrame({'student':np.arange(100, 110, 1), 'test':np.random.randint(60, 101, 10)}).set_index('student')

s = pd.Series(['round', 'square'], index=['c', 'b'], name='shape')

pd.concat([df, s], axis=1)
pd.concat([s, df], axis=1)
```

[How to use map() and apply() to apply functions to columns of a dataframe?]http://localhost:8888/notebooks/scripts/How%20do%20I%20apply%20a%20function%20to%20a%20pandas%20Series%20or%20DataFrame%3F.ipynb     
[How to let apply() work with np.ceil, custom function, len(), lambda() to apply functions to columns of a dataframe?]http://localhost:8888/notebooks/scripts/How%20do%20I%20apply%20a%20function%20to%20a%20pandas%20Series%20or%20DataFrame%3F.ipynb     
[How to use apply() and np.argmax to apply functions to columns of a dataframe?]http://localhost:8888/notebooks/scripts/How%20do%20I%20apply%20a%20function%20to%20a%20pandas%20Series%20or%20DataFrame%3F.ipynb    
[How to change specific columns' type at once ?]http://localhost:8888/notebooks/scripts/How%20do%20I%20apply%20a%20function%20to%20a%20pandas%20Series%20or%20DataFrame%3F.ipynb     


```python

import pandas as pd

train = pd.read_csv('../data/titanic_train.csv')

train['Sex_num'] = train.Sex.map({'female':0, 'male':1})

train.loc[0:4, ['Sex', 'Sex_num']]

train['Name_length'] = train.Name.apply(len)

train.loc[0:4, ['Name', 'Name_length']]

import numpy as np

train['Fare_ceil'] = train.Fare.apply(np.ceil)
train.loc[0:4, ['Fare', 'Fare_ceil']]

train.Name.str.split(',').head()


def get_element(my_list, position):
    return my_list[position]

train.Name.str.split(',').apply(get_element, position=0).head()

train.Name.str.split(',').apply(lambda x: x[0]).head()

drinks = pd.read_csv('../data/drinks.csv')

drinks.loc[:, 'beer_servings':'wine_servings'].head()

drinks.loc[:, 'beer_servings':'wine_servings'].apply(max, axis=0)

drinks.loc[:, 'beer_servings':'wine_servings'].apply(max, axis=1).head()

drinks.loc[:, 'beer_servings':'wine_servings'].apply(np.argmax, axis=1).head()

drinks.loc[:, 'beer_servings':'wine_servings'].applymap(float).head()

drinks.loc[:, 'beer_servings':'wine_servings'] = drinks.loc[:, 'beer_servings':'wine_servings'].applymap(float)
```


[How to round for the whole dataframe?](http://localhost:8888/notebooks/scripts/use%20of%20round.ipynb)    
[How to round for specific columns of the whole dataframe?](http://localhost:8888/notebooks/scripts/use%20of%20round.ipynb)    
[How to round for specific columns with specific round args for the whole dataframe?](http://localhost:8888/notebooks/scripts/use%20of%20round.ipynb)    
```python

import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.random([3, 3]),
     columns=['A', 'B', 'C'], index=['first', 'second', 'third'])

df.round(2)

df.round({'A': 1, 'C': 2})

decimals = pd.Series([1, 0, 2], index=['A', 'B', 'C'])

df.round(decimals)
```


[How pd.Series work with `endwith()` and `for in` loop?](http://localhost:8888/notebooks/scripts/pandas%20Series%20and%20DataFrame.ipynb)    
[How to create index name and series name when creating a pd.Series?](http://localhost:8888/notebooks/scripts/pandas%20Series%20and%20DataFrame.ipynb)    
[How to create a pd.Series from a dictionary?](http://localhost:8888/notebooks/scripts/pandas%20Series%20and%20DataFrame.ipynb)    
```python

import pandas as pd
import numpy as np

counts = pd.Series([632, 1638, 569, 115])
counts

counts.values

counts.index

bacteria = pd.Series([632, 1638, 569, 115],
    index=['Firmicutes', 'Proteobacteria', 'Actinobacteria', 'Bacteroidetes'])

bacteria['Actinobacteria']

bacteria[[name.endswith('bacteria') for name in bacteria.index]]

[name.endswith('bacteria') for name in bacteria.index]

bacteria[0]

bacteria.name = 'counts'
bacteria.index.name = 'phylum'

np.log(bacteria)

bacteria[bacteria>1000]

bacteria_dict = {'Firmicutes': 632, 'Proteobacteria': 1638, 'Actinobacteria': 569,
                 'Bacteroidetes': 115}
bacteria = pd.Series(bacteria_dict)
bacteria


bacteria2 = pd.Series(bacteria_dict,
                      index=['Cyanobacteria','Firmicutes',
                             'Proteobacteria','Actinobacteria'])

bacteria2

bacteria2.isnull()

bacteria + bacteria2
```

[How to merge and concat on dataframe?](http://localhost:8888/notebooks/scripts/merge%2C%20concat.ipynb)     
[How to do union join, inner join and left join, right join between two dataframe?](http://localhost:8888/notebooks/scripts/merge%2C%20concat.ipynb)    
[How to do row bind and column bind without sharing any column?](http://localhost:8888/notebooks/scripts/merge%2C%20concat.ipynb)
```python
import pandas as pd
import numpy as np

left_frame = pd.DataFrame({'key': range(5),
                           'left_value': ['a', 'b', 'c', 'd', 'e']})
right_frame = pd.DataFrame({'key': range(2, 7),
                           'right_value': ['f', 'g', 'h', 'i', 'j']})
print(left_frame)
print('\n')
print(right_frame)

pd.merge(left_frame, right_frame, on='key', how='inner')

pd.merge(left_frame, right_frame, on='key', how='left')

pd.merge(left_frame, right_frame, on='key', how='right')

pd.merge(left_frame, right_frame, on='key', how='outer')

pd.concat([left_frame, right_frame])

pd.concat([left_frame, right_frame], axis=1)
```

[How to use read_csv with encoding = 'latin-1'?](http://localhost:8888/notebooks/scripts/Greg%20Reda%202.ipynb)    
[How to the first 5 columns to load when reading a datafile?](http://localhost:8888/notebooks/scripts/Greg%20Reda%202.ipynb)    
[How to read the first few lines of any data file?](http://localhost:8888/notebooks/scripts/Greg%20Reda%202.ipynb)      
[How to count values in each column after groupby?](http://localhost:8888/notebooks/scripts/Greg%20Reda%202.ipynb)      
[How to count total values in each group after groupby?](http://localhost:8888/notebooks/scripts/Greg%20Reda%202.ipynb)      
[How to apply sum, mean, median to each group after groupby?](http://localhost:8888/notebooks/scripts/Greg%20Reda%202.ipynb)      
[How to create a ranker column for each group after groupby?](http://localhost:8888/notebooks/scripts/Greg%20Reda%202.ipynb)      
[How to access a group subdataset after groupby?](http://localhost:8888/notebooks/scripts/Greg%20Reda%202.ipynb)      



```python

import pandas as pd
import numpy as np
from IPython.display import display

u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('../data/ml-100k/u.user', sep='|', names=u_cols,
                    encoding='latin-1')

users.head()

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('../data/ml-100k/u.data', sep='\t', names=r_cols,
                      encoding='latin-1')

ratings.head()

m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
movies = pd.read_csv('../data/ml-100k/u.item', sep='|', names=m_cols, usecols=range(5),
                     encoding='latin-1')

movies.head()

get_ipython().system(u'head -n 3 ../data/city-of-chicago-salaries.csv')

headers = ['name', 'title', 'department', 'salary']
chicago = pd.read_csv('../data/city-of-chicago-salaries.csv',
                      header=0,
                      names=headers,
                      converters={'salary': lambda x: float(x.replace('$', ''))})
chicago.head()

chicago.shape

from IPython.display import display

by_dept = chicago.groupby('department')
by_dept

print(by_dept.count().head()) # NOT NULL records within each column
print('\n')
print(by_dept.size().tail()) # total records for each department

print(by_dept.sum()[20:25]) # total salaries of each department
print('\n')
print(by_dept.mean()[20:25]) # average salary of each department
print('\n')
print(by_dept.median()[20:25]) # take that, RDBMS!

by_dept.title.nunique()

by_dept.title.nunique().sort_values(ascending=False)[:5]

def ranker(df):
    """Assigns a rank to each employee based on salary, with 1 being the highest paid.
    Assumes the data is DESC sorted."""
    df['dept_rank'] = np.arange(len(df)) + 1
    return df

chicago.shape

chicago.department.nunique()

len(by_dept)

chicago.sort_values('salary', ascending=False, inplace=True)

chicago.head()

chicago_ranker = chicago.groupby('department').apply(ranker)

chicago_ranker.shape

chicago_ranker.head()

display(chicago_ranker[chicago_ranker.dept_rank == 1].head(14))

chicago[chicago.department == "LAW"][:5]
```


[How to use pd.merge (by default) to join two dataframe sharing one column with different total length?](http://localhost:8888/notebooks/scripts/Greg%20Reda%203.ipynb)    
[How to apply two statistical methods (np.size, np.mean) on to a particular column of each group ?](http://localhost:8888/notebooks/scripts/Greg%20Reda%203.ipynb)    
[How to sort the grouped dataset by mean value column under over arch rating column?](http://localhost:8888/notebooks/scripts/Greg%20Reda%203.ipynb)    
[How to access each big arch column's smaller column?](http://localhost:8888/notebooks/scripts/Greg%20Reda%203.ipynb)    
[How to create range category for another column of the same dataframe?](http://localhost:8888/notebooks/scripts/Greg%20Reda%203.ipynb)    
[How to drop duplicates when two specific columns both have duplicates at the same time and make latter part of duplicates to be dropped?](http://localhost:8888/notebooks/scripts/Greg%20Reda%203.ipynb)    
[How to convert group by group stacked dataframe to unstack table  format?](http://localhost:8888/notebooks/scripts/Greg%20Reda%203.ipynb)    
[How to use pivot table?](http://localhost:8888/notebooks/scripts/Greg%20Reda%203.ipynb)    

```python

import pandas as pd
import numpy as np
from IPython.display import display

u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('../data/ml-100k/u.user', sep='|', names=u_cols,
                    encoding='latin-1')

users.head()


r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('../data/ml-100k/u.data', sep='\t', names=r_cols,
                      encoding='latin-1')
ratings.shape


ratings.movie_id.nunique()

ratings.loc[ratings.movie_id == 1, :].head()

m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
movies = pd.read_csv('../data/ml-100k/u.item', sep='|', names=m_cols, usecols=range(5),
                     encoding='latin-1')

display(movies.shape)
display(movies.movie_id.nunique())

movies.loc[movies.movie_id == 1, :].head()

movie_ratings = pd.merge(movies, ratings)
movie_ratings.shape

movie_ratings.head()

lens = pd.merge(movie_ratings, users)

lens.groupby('title').size()

most_rated = lens.groupby('title').size().sort_values(ascending=False)[:25]
display(most_rated)

lens.title.value_counts()[:25]

movie_stats = lens.groupby('title').agg({'rating': [np.size, np.mean]})
movie_stats.tail()


movie_stats.sort_values([('rating', 'mean')], ascending=False).head()

atleast_100 = movie_stats['rating']['size'] >= 100

movie_stats[atleast_100].sort_values([('rating', 'mean')], ascending=False)[:15]

most_50 = lens.groupby('movie_id').size().sort_values(ascending=False)[:50]

display(most_50)

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt

users.age.plot.hist(bins=30)
plt.title("Distribution of users' ages")
plt.ylabel('count of users')
plt.xlabel('age');



labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
# right = True, meaning include the most right value; false, meaning exclude it
lens['age_group'] = pd.cut(lens.age, range(0, 81, 10), right=False, labels=labels)


# default drop_duplicates() used when both columns are duplicated
lens[['age', 'age_group']].drop_duplicates()[:10]

lens.groupby('age_group').agg({'rating': [np.size, np.mean]})

lens.set_index('movie_id', inplace=True)

by_age = lens.loc[most_50.index].groupby(['title', 'age_group'])
by_age.rating.mean().head(15)

by_age.rating.mean()

by_age.rating.mean().unstack(1).fillna(0)[10:20]

lens.reset_index('movie_id', inplace=True)

pivoted = lens.pivot_table(index=['movie_id', 'title'],
                           columns=['sex'],
                           values='rating',
                           fill_value=0)
pivoted.head()

pivoted['diff'] = pivoted.M - pivoted.F
pivoted.head()
```


[How to draw a line based on a single pd.Series with/without grid?](http://localhost:8888/notebooks/scripts/Plotting-with-Pandas.ipynb)    
[How to do cumsum() on a single pd.Series?](http://localhost:8888/notebooks/scripts/Plotting-with-Pandas.ipynb)    
[How to draw 3-line chart based on a 3-column dataframe?](http://localhost:8888/notebooks/scripts/Plotting-with-Pandas.ipynb)    
[How to create a 3-column dataframe with random distributions like normal, poisson, gamma?](http://localhost:8888/notebooks/scripts/Plotting-with-Pandas.ipynb)    
[How to create 3-lines in 3-charts grouped vertically based on 3-column dataframe?](http://localhost:8888/notebooks/scripts/Plotting-with-Pandas.ipynb)    
[How to create 3-lines in 3-charts grouped vertically based on 3-column dataframe, while give a particular column a different y-axis  y2-axis?](http://localhost:8888/notebooks/scripts/Plotting-with-Pandas.ipynb)    
[How to create 3-lines in 3-charts grouped horizontally based on 3-column dataframe using subplots() and enumerate()?](http://localhost:8888/notebooks/scripts/Plotting-with-Pandas.ipynb)    
[How to read xls data file?](http://localhost:8888/notebooks/scripts/Plotting-with-Pandas.ipynb)    
[How to group a dataframe and display several first and last rows of each group in the dataframe?](http://localhost:8888/notebooks/scripts/Plotting-with-Pandas.ipynb)    
[How to create a bar chart out of a grouped dataframe with sum() or mean() applied to each group?](http://localhost:8888/notebooks/scripts/Plotting-with-Pandas.ipynb)    
[How to create a horizontal barchart for a 2-grouped dataframe applied sum() to each group on a different column?](http://localhost:8888/notebooks/scripts/Plotting-with-Pandas.ipynb)    
[How to create a vertical stacked barchart for a 2-grouped columns crosstab() with a different column from the same dataframe?](http://localhost:8888/notebooks/scripts/Plotting-with-Pandas.ipynb)    
[How to create a histogram based on a pd.Series](http://localhost:8888/notebooks/scripts/Plotting-with-Pandas.ipynb)    
[How to create a histogram with specific number of bins?](http://localhost:8888/notebooks/scripts/Plotting-with-Pandas.ipynb)    
[How to drop na from a pd.Series?](http://localhost:8888/notebooks/scripts/Plotting-with-Pandas.ipynb)    
[How to find proper number of bins using kurtosis()?](http://localhost:8888/notebooks/scripts/Plotting-with-Pandas.ipynb)    
[How to create density curve and set xlim?](http://localhost:8888/notebooks/scripts/Plotting-with-Pandas.ipynb)    
[How to plot histogram and density curve in the same chart?](http://localhost:8888/notebooks/scripts/Plotting-with-Pandas.ipynb)    
[How to create boxplot with a column of dataframe and divided into smaller groups by another column](http://localhost:8888/notebooks/scripts/Plotting-with-Pandas.ipynb)    
[How to plot boxplot and jitter scatterpoints on the same chart?](http://localhost:8888/notebooks/scripts/Plotting-with-Pandas.ipynb)    
[How to create scatter points using pd.Series in a dataframe and set x-y-lim, size and alpha?](http://localhost:8888/notebooks/scripts/Plotting-with-Pandas.ipynb)     
[How to create scatters with a Series of number for colors using c and cmap args?](http://localhost:8888/notebooks/scripts/Plotting-with-Pandas.ipynb)     
[How to create scatter matrix and make diagonal to be density curve?](http://localhost:8888/notebooks/scripts/Plotting-with-Pandas.ipynb)    
[How to create plots with ggplot style?](http://localhost:8888/notebooks/scripts/Plotting-with-Pandas.ipynb)     
[How to create facetgrid() to display density curves which are grouped by two columns?](http://localhost:8888/notebooks/scripts/Plotting-with-Pandas.ipynb)   
[How to create FacetGrid by 2 columns, and plot scatters using another two columns?](http://localhost:8888/notebooks/scripts/Plotting-with-Pandas.ipynb)   
[How regplot can add regression curve to scatter points in FacetGrid groups?](http://localhost:8888/notebooks/scripts/Plotting-with-Pandas.ipynb)   
```python
# 1 refers to sum by columns
# 0 refers to sum by rows
death_counts.sum(1).astype(float)
```


[How to find intersection between two pd.Series, especially with Chinese characters?](https://uqer.io/labs/notebooks/%E5%A6%82%E4%BD%95%E6%89%BE%E5%87%BA%E6%89%80%E6%9C%89%E8%82%A1%E7%A5%A8%E6%89%80%E5%B1%9E%E8%A1%8C%E4%B8%9A%EF%BC%9F.nb)
```python
# industry2 and industry3 are pd.Series
pd.Series(list(set(industry2) & set(industry3)))
```

[How to display all the rows and all the texts inside a cell?](https://uqer.io/labs/notebooks/%E5%A6%82%E4%BD%95%E6%9F%A5%E6%89%BE%E4%B8%8A%E5%B8%82%E5%85%AC%E5%8F%B8%E7%9A%84%E7%BB%8F%E8%90%A5%E8%8C%83%E5%9B%B4%EF%BC%9F.nb)     
```python
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_colwidth', 1000)
```
