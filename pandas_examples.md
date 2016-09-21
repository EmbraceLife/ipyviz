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
drinks.loc[23, 'beer_servings']
# set an existing column as the index
drinks.set_index('country', inplace=True)
# country name can now be used for selection
drinks.loc['Brazil', 'beer_servings']

# index name is optional
drinks.index.name = None    # get rid of indexName "country"
# restore the index name, and move the index back to a column
drinks.index.name = 'country'

# many DataFrame methods output a DataFrame
drinks.describe()
# you can interact with any DataFrame using its index and columns
drinks.describe().loc['25%', 'beer_servings']
```


[value_counts, index, values, sort_values, sort_index, pd.concat] (http://localhost:8888/notebooks/scripts/more%20about%20pandas%20index.ipynb)

```python
drinks.continent.value_counts().index
drinks.continent.value_counts().values
drinks.continent.value_counts()['Africa']
drinks.continent.value_counts().sort_values()
drinks.continent.value_counts().sort_index()
people = pd.Series([3000000, 85000], index=['Albania', 'Andorra'], name='population')
# people has 2 rows originally, but it will match the length of beer_servings by adding NaNs
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


[use of loc and ix](http://localhost:8888/notebooks/scripts/use%20of%20loc%20and%20ix.ipynb)
```python
ufo[['City', 'State']].head(3)
ufo.loc[ufo.City=='Oakland', 'State']
ufo[ufo.City=='Oakland'].State

ufo.loc[0, :]
ufo.loc[[0, 1, 2], :]
ufo.loc[0:2, :]
ufo.loc[0:2]
ufo.loc[0:2, 'City']
ufo.loc[0:2, ['City', 'State']]
ufo.loc[0:2, 'City':'State']

ufo.iloc[[0, 1], [0, 3]]
ufo.iloc[0:2, 0:4]
ufo.iloc[0:2, :]

drinks.ix['Albania', 0]
drinks.ix[1, 'beer_servings']
drinks.ix['Albania':'Andorra', 0:2]
ufo.ix[0:2, 0:2]
```

[How to back and forward fill NA of a dataframe?](http://localhost:8888/notebooks/scripts/use%20of%20inplace%20and%20dropna%20with%20method%20bfill%20or%20ffill.ipynb)
```python
# drop a row if any value is missing from that row (doesn't affect the DataFrame since inplace=False)
ufo.dropna(how='any').shape

ufo = ufo.set_index('Time')

# fill missing values using "backward fill" strategy (doesn't affect the DataFrame since inplace=False)
ufo.fillna(method='bfill').tail()

# fill missing values using "backward fill" strategy (doesn't affect the DataFrame since inplace=False)
ufo.fillna(method='bfill').tail()
```


[How to make pandas dataframe smaller and faster?](http://localhost:8888/notebooks/scripts/How%20to%20make%20pandas%20dataframe%20smaller%20and%20faster.ipynb)
```python
# use the 'category' data type (new in pandas 0.15) to store the 'continent' strings as integers
drinks['continent'] = drinks.continent.astype('category')
# strings are now encoded (0 means 'Africa', 1 means 'Asia', 2 means 'Europe', etc.)
drinks.continent.cat.codes.head()
drinks.country.cat.categories
df['quality'] = df.quality.astype('category', categories=['good', 'very good', 'excellent'], ordered=True)

# sort the DataFrame by the 'quality' Series (logical order)
df.sort_values('quality')

# comparison operators work with ordered categories
df.loc[df.quality > 'good', :]
```


[How do I use pandas with scikit-learn to create Kaggle submissions with pickle?](http://localhost:8888/notebooks/scripts/How%20do%20I%20use%20pandas%20with%20scikit-learn%20to%20create%20Kaggle%20submissions%3F.ipynb)
```python
# using two columns to train model
feature_cols = ['Pclass', 'Parch']
X = train.loc[:, feature_cols]

# get response column
y = train.Survived

# fit a classification model to the training data
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X, y)

# create a feature matrix from the testing data that matches the training data
X_new = test.loc[:, feature_cols]
# use the fitted model to make predictions for the testing set observations
new_pred_class = logreg.predict(X_new)

# ensure that PassengerID is the first column by setting it as the index
pd.DataFrame({'PassengerId':test.PassengerId, 'Survived':new_pred_class}).set_index('PassengerId').head()

# save a DataFrame to disk ("pickle it")
train.to_pickle('train.pkl')

# read a pickled object from disk ("unpickle it")
pd.read_pickle('train.pkl').head()
```


[pd.isnull, pd.df.isnull, df.loc, df.iloc, df.sample with args n, frac, random_state](http://localhost:8888/notebooks/scripts/pd.isnull%2C%20pd.df.isnull%2C%20df.loc%2C%20df.iloc%2C%20df.sample%20with%20args%20n%2C%20frac%2C%20random_state.ipynb)
```python
# use 'isnull' as a top-level function
pd.isnull(ufo).head()

# sample 3 rows from the DataFrame without replacement (new in pandas 0.16.1)
ufo.sample(n=3)

# use the 'random_state' parameter for reproducibility
ufo.sample(n=3, random_state=42)

# sample 75% of the DataFrame's rows without replacement
train = ufo.sample(frac=0.75, random_state=99)

# store the remaining 25% of the rows in another DataFrame
test = ufo.loc[~ufo.index.isin(train.index), :]
```



[How to create dummy variables in pandas?](http://localhost:8888/notebooks/scripts/How%20to%20create%20dummy%20variables%20in%20pandas%3F.ipynb)
```python
# create the 'Sex_male' dummy variable using the 'map' method
train['Sex_male'] = train.Sex.map({'female':0, 'male':1})

# alternative: use 'get_dummies' to create one column for every possible value
pd.get_dummies(train.Sex).head()

# drop the first dummy variable ('female') using the 'iloc' method
pd.get_dummies(train.Sex).iloc[:, 1:].head()

# add a prefix to identify the source of the dummy variables
pd.get_dummies(train.Sex, prefix='Sex')
pd.get_dummies(train.Sex, prefix='Sex').iloc[:, 1:].head()

# save the DataFrame of dummy variables and concatenate them to the original DataFrame
embarked_dummies = pd.get_dummies(train.Embarked, prefix='Embarked').iloc[:, 1:]
train = pd.concat([train, embarked_dummies], axis=1)

# pass the DataFrame to 'get_dummies' and specify which columns to dummy (it drops the original columns)
pd.get_dummies(train, columns=['Sex', 'Embarked']).head()

# use the 'drop_first' parameter (new in pandas 0.18) to drop the first dummy variable for each feature
pd.get_dummies(train, columns=['Sex', 'Embarked'], drop_first=True).head()
```


[How to work with dates and times in pandas?](http://localhost:8888/notebooks/scripts/How%20to%20work%20with%20dates%20and%20times%20in%20pandas%3F.ipynb)
```python
# convert 'Time' to datetime format
ufo['Time'] = pd.to_datetime(ufo.Time)

# convenient Series attributes are now available
ufo.Time.dt.hour.head()

ufo.Time.dt.weekday_name.head()

ufo.Time.dt.dayofyear.head()

# convert a single string to datetime format (outputs a timestamp object)
ts = pd.to_datetime('1/1/1999')

# compare a datetime Series with a timestamp
ufo.loc[ufo.Time >= ts, :].head()

# perform mathematical operations with timestamps (outputs a timedelta object)
ufo.Time.max() - ufo.Time.min()

# timedelta objects also have attributes you can access
(ufo.Time.max() - ufo.Time.min()).days

# count the number of UFO reports per year
ufo['Year'] = ufo.Time.dt.year
display(ufo.Year)
# after value_counts, index becomes years
ufo.Year.value_counts().sort_index().head()
```

[How to find and remove duplicate rows in pandas?](http://localhost:8888/notebooks/scripts/How%20to%20find%20and%20remove%20duplicate%20rows%20in%20pandas%3F.ipynb)
```python
# read a dataset of movie reviewers into a DataFrame
user_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
users = pd.read_table('../data/u.user.txt', sep='|', header=None, names=user_cols, index_col='user_id')

# detect duplicate zip codes: True if an item is identical to a previous item
users.zip_code.duplicated().tail()

# count the duplicate items (True becomes 1, False becomes 0)
users.zip_code.duplicated().sum()

# detect duplicate DataFrame rows: True if an entire row is identical to a previous row
users.duplicated().tail()

# examine the duplicate rows (ignoring the first occurrence)
users.loc[users.duplicated(keep='first'), :]

# examine the duplicate rows (ignoring the last occurrence)
users.loc[users.duplicated(keep='last'), :]

# examine the duplicate rows (including all duplicates)
users.loc[users.duplicated(keep=False), :]

# drop duplicated rows but keep the first occurrences
users.drop_duplicates(keep='first').loc[60:70,:]      

# drop all duplicates first and last
users.drop_duplicates(keep=False).shape

# only consider a subset of columns when identifying duplicates
users.duplicated(subset=['age', 'zip_code']).sum()

users.drop_duplicates(subset=['age', 'zip_code']).shape      
```


[How to avoid a SettingWithCopyWarning in pandas?](http://localhost:8888/notebooks/scripts/How%20to%20avoid%20a%20SettingWithCopyWarning%20in%20pandas%3F.ipynb)
```python
# explicitly create a copy of 'movies'
top_movies = movies.loc[movies.star_rating >= 9, :].copy()

# pandas now knows that you are updating a copy instead of a view (does not cause a SettingWithCopyWarning)
top_movies.loc[0, 'duration'] = 150
```

[How to change display options in pandas?](http://localhost:8888/notebooks/scripts/How%20to%20change%20display%20options%20in%20pandas%3F.ipynb)
```python
# overwrite the current setting so that more characters will be displayed
pd.set_option('display.max_colwidth', 1000)

# overwrite the 'precision' setting to display 2 digits after the decimal point of 'Fare'
pd.set_option('display.precision', 2)
train.head()

drinks['x'] = drinks.wine_servings.astype("float64") * 1000
drinks['y'] = drinks.total_litres_of_pure_alcohol * 1000

# use a Python format string to specify a comma as the thousands separator
pd.set_option('display.float_format', '{:,}'.format)
```

[How do I create a pandas DataFrame from another object?](http://localhost:8888/notebooks/scripts/How%20do%20I%20create%20a%20pandas%20DataFrame%20from%20another%20object%3F.ipynb)
```python
# create a DataFrame from a dictionary (keys become column names, values become data)
pd.DataFrame({'id':[100, 101, 102], 'color':['red', 'blue', 'red']})

# optionally specify the order of columns and define the index
df = pd.DataFrame({'id':[100, 101, 102], 'color':['red', 'blue', 'red']}, columns=['id', 'color'], index=['a', 'b', 'c'])

# create a DataFrame from a list of lists (each inner list becomes a row)
pd.DataFrame([[100, 'red'], [101, 'blue'], [102, 'red']], columns=['id', 'color'])

# create a NumPy array (with shape 4 by 2) and fill it with random numbers between 0 and 1
import numpy as np
arr = np.random.rand(4, 2)
# create a DataFrame from the NumPy array
pd.DataFrame(arr, columns=['one', 'two'])

# create a DataFrame of student IDs (100 through 109) and test scores (random integers between 60 and 100)
pd.DataFrame({'student':np.arange(100, 110, 1), 'test':np.random.randint(60, 101, 10)})

# 'set_index' can be chained with the DataFrame constructor to select an index
pd.DataFrame({'student':np.arange(100, 110, 1), 'test':np.random.randint(60, 101, 10)}).set_index('student')

# create a new Series using the Series constructor
s = pd.Series(['round', 'square'], index=['c', 'b'], name='shape')

# concatenate the DataFrame and the Series (use axis=1 to concatenate columns)
pd.concat([df, s], axis=1)
```

[How do I apply a function to a pandas Series or DataFrame?](http://localhost:8888/notebooks/scripts/How%20do%20I%20apply%20a%20function%20to%20a%20pandas%20Series%20or%20DataFrame%3F.ipynb)
```python
# map 'female' to 0 and 'male' to 1
train['Sex_num'] = train.Sex.map({'female':0, 'male':1})
train.loc[0:4, ['Sex', 'Sex_num']]

# calculate the length of each string in the 'Name' Series
train['Name_length'] = train.Name.apply(len)
train.loc[0:4, ['Name', 'Name_length']]

# round up each element in the 'Fare' Series to the next integer
import numpy as np
train['Fare_ceil'] = train.Fare.apply(np.ceil)
train.loc[0:4, ['Fare', 'Fare_ceil']]

# use a string method to split the 'Name' Series at commas (returns a Series of lists)
train.Name.str.split(',').head()

# define a function that returns an element from a list based on position
def get_element(my_list, position):
    return my_list[position]

# apply the 'get_element' function and pass 'position' as a keyword argument
train.Name.str.split(',').apply(get_element, position=0).head()

# alternatively, use a lambda function
train.Name.str.split(',').apply(lambda x: x[0]).head()

# apply the 'max' function along axis 0 to calculate the maximum value in each column
drinks.loc[:, 'beer_servings':'wine_servings'].apply(max, axis=0)

# apply the 'max' function along axis 1 to calculate the maximum value in each row
drinks.loc[:, 'beer_servings':'wine_servings'].apply(max, axis=1).head()

# use 'np.argmax' to calculate which column has the maximum value for each row
drinks.loc[:, 'beer_servings':'wine_servings'].apply(np.argmax, axis=1).head()

# convert every DataFrame element into a float
drinks.loc[:, 'beer_servings':'wine_servings'].applymap(float).head()

# overwrite the existing DataFrame columns
drinks.loc[:, 'beer_servings':'wine_servings'] = drinks.loc[:, 'beer_servings':'wine_servings'].applymap(float)
drinks.head()
```


[How to make use of round in pandas for dataframe?](http://localhost:8888/notebooks/scripts/use%20of%20round.ipynb)
