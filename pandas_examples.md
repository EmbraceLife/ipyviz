# pandas_examples


[How to load a file with pd.read_table()?](http://localhost:8888/notebooks/scripts/Load%20file%20with%20pd.read_table.ipynb)
- pd.read_table()

[How to load a csv file with pd.read_table() and pd.read_csv()?](http://localhost:8888/notebooks/scripts/How%20to%20load%20a%20csv%20file%20with%20pd.read_table%20and%20pd.read_csv%3F.ipynb)
- pd.read_table()
- pd.read_csv()

[How to extract a series of a dataframe?](http://localhost:8888/notebooks/scripts/How%20to%20access%20a%20series%20of%20a%20dataframe%20with%20dot%20and%20brackets%20with%20strings%3F.ipynb)
- pd.df.colname
- pd.df["colname"]

[How to understand pd.attributes and pd.methods?](http://localhost:8888/notebooks/scripts/How%20to%20use%20pd.head%2C%20pd.shape%2C%20pd.ndim%2C%20pd.describe%2C%20pd.ntypes.ipynb)
- pd.head()
- pd.shape
- pd.ndim
- pd.ntypes
- pd.describe()

[How to rename column names in pandas dataframe?](http://localhost:8888/notebooks/scripts/How%20to%20rename%20column%20names%20in%20pandas%20dataframe.ipynb)
- how to rename columns with pd.rename(columns = dict)
- get data file from up-directory "..data/chipotle.csv"
- how to rename columns with pd.read_csv()
- how to rename columns with str.replace()
- how to rename columns with pd.df.columns = list of names

[How to drop or remove rows and columns of a pandas dataframe?](http://localhost:8888/notebooks/scripts/How%20to%20drop%20rows%20and%20columns%20of%20pandas%20dataframe%3F.ipynb)
- pd.drop()

[How to sort series or dataframe based on series?](http://localhost:8888/notebooks/scripts/How%20to%20sort%20series%20or%20dataframe%20based%20on%20series%3F%20.ipynb)
-pd.sort_values

[How to filter rows with columns conditions for pandas dataframe?](http://localhost:8888/notebooks/scripts/How%20to%20filter%20rows%20with%20columns%20and%20conditions%20for%20pandas%20dataframe%3F.ipynb)
- `for in` loop
- fill in empty list with `list.append`
- convert a series to a list with `pd.Series()`
- get the length of a list with `len()`
- how to filter rows by `df[df.colname1 > number].colname2`?
- how to filter with `df.loc[df.colname1 > number, 'colname2']`?

[How to filter on multiple conditions for pandas dataframe?](http://localhost:8888/notebooks/scripts/How%20to%20filter%20on%20multiple%20conditions%20for%20pandas%20dataframe%3F%20.ipynb)
- | and &
- `df.colname.isin(list)`

[How to read a data file into dataframe but with only specific columns?](http://localhost:8888/notebooks/scripts/How%20to%20read%20in%20data%20file%20into%20a%20dataframe%20with%20only%20specific%20columns%20%3F.ipynb)

[How to read a data file into dataframe with specific rows?](http://localhost:8888/notebooks/scripts/How%20to%20read%20in%20data%20file%20into%20dataframe%20with%20only%20specific%20rows%20.ipynb)
[How to iterate series or dataframe by rows?](http://localhost:8888/notebooks/scripts/How%20to%20iterate%20series%20and%20dataframe%20by%20rows%3F%20.ipynb)

[How to describe different columns of a dataframe?](http://localhost:8888/notebooks/scripts/How%20to%20use%20pd.describe%20and%20How%20to%20select%20only%20numeric%20columns%3F.ipynb)
[How to select only numeric columns from a dataframe?](http://localhost:8888/notebooks/scripts/How%20to%20use%20pd.describe%20and%20How%20to%20select%20only%20numeric%20columns%3F.ipynb)
- pd.describe()
- pd.select_ntypes(include = [np.number])


[How to apply functions to rows and columns of dataframe?](http://localhost:8888/notebooks/scripts/How%20to%20apply%20functions%20to%20rows%20and%20columns%20of%20dataframe%3F.ipynb)
- apply pd.drop()
- apply pd.mean() statistical functions

[How to use string methods in pandas dataframe?](http://localhost:8888/notebooks/scripts/How%20to%20use%20string%20methods%20in%20pandas%20dataframe%3F.ipynb)
- pd.dataframe works with str.upper()
- str.replace()
- str.contains()

[How to change data type of pandas dataframe?](http://localhost:8888/notebooks/scripts/How%20to%20change%20data%20type%20of%20pandas%20dataframe%3F%20.ipynb)
- pd.dtypes
- pd.astype()

[How and when to use groupby in pandas dataframe?](http://localhost:8888/notebooks/scripts/How%20and%20when%20to%20use%20groupby%20in%20pandas%20dataframe%3F.ipynb)
- group rows into groups based on categorical column

[often used pd.series methods](http://localhost:8888/notebooks/scripts/often%20used%20pd.series%20methods.ipynb)
- pd.Series.value_counts()
- pd.Series.value_counts(normalize=True)
- pd.Series.unique()
- pd.Series.nunique()
- pd.crosstab(pd.series, pd.series)

[How to handle missing values in pandas dataframe?](http://localhost:8888/notebooks/scripts/How%20to%20handle%20missing%20values%20in%20pandas%20dataframe%3F.ipynb)
- pd.df.isnull()
- pd.df.notnull()
- pd.df[pd.df.isnull()]
- pd.df.dropna(how = "all/any")
- pd.df.dropna(how = "all/any", subset = ["colname1", "colname2"])
- pd.df.value_counts(dropna=False)
- pd.df.fillna(value = "string", inplace=True)

[How to make use of pandas dataframe index?](http://localhost:8888/notebooks/scripts/How%20to%20make%20use%20of%20pandas%20dataframe%20index%3F.ipynb)
- pd.df.index: get row label or index of rows
- pd.df.index.name = "string" to set index name
- pd.df.set_index('colName', inplace=True)
- pd.df.columns: get column names
- pd.df.loc[rowIndex, 'columnName']
- pd.df.describe().loc['25%', 'beer_servings']

[more about pandas index](http://localhost:8888/notebooks/scripts/more%20about%20pandas%20index.ipynb)
- power of `pd.df.set_index('country', inplace=True)`
- pd.df.colname1.value_counts()
- pd.df.colname1.value_counts().index
- pd.df.colname1.value_counts().values
- pd.df.colname1.value_counts()['index1']
- pd.df.colname1.value_counts().sort_values()
- pd.df.colname1.value_counts().sort_index()
- pd.series * pd.series regulated by index
- pd.concat([series1, series2], axis=1).head()


[use of loc and ix](http://localhost:8888/notebooks/scripts/use%20of%20loc%20and%20ix.ipynb)
- pd.df.loc()
- pd.df.ix()

[use of inplace and fillna with method bfill or ffill](http://localhost:8888/notebooks/scripts/use%20of%20inplace%20and%20dropna%20with%20method%20bfill%20or%20ffill.ipynb)
- `fillna(method='bfill')`: fill NA backward
- `fillna(method='ffill')`: fill NA forward


[How to make pandas dataframe smaller and faster?](http://localhost:8888/notebooks/scripts/How%20to%20make%20pandas%20dataframe%20smaller%20and%20faster.ipynb)
- pd.df.info()
- pd.df.info(memory_usage = 'deep')
- pd.df.memory_usage(deep = True)
- pd.df.colname1.astype('category'): from string to category type
- pd.df.colname1.cat.codes.head(): check index and codes
- pd.df.colname1.cat.category.head(): check index and category
- df.colname1.astype('category', categories=['good', 'very good', 'excellent'], ordered=True)

[How do I use pandas with scikit-learn to create Kaggle submissions?](http://localhost:8888/notebooks/scripts/How%20do%20I%20use%20pandas%20with%20scikit-learn%20to%20create%20Kaggle%20submissions%3F.ipynb)
- pd.DataFrame(): create a df from a dict
- from sklearn.linear_model import LogisticRegression
- logreg = LogisticRegression()
- logreg.fit(X, y)
- new_pred_class = logreg.predict(X_new)
- pd.df.set_index().to_csv()
- pd.df.to_pickle()
- pd.read_pickle()


[pd.isnull, pd.df.isnull, df.loc, df.iloc, df.sample with args n, frac, random_state](http://localhost:8888/notebooks/scripts/pd.isnull%2C%20pd.df.isnull%2C%20df.loc%2C%20df.iloc%2C%20df.sample%20with%20args%20n%2C%20frac%2C%20random_state.ipynb)

[How to create dummy variables in pandas?](http://localhost:8888/notebooks/scripts/How%20to%20create%20dummy%20variables%20in%20pandas%3F.ipynb)

[How to work with dates and times in pandas?](http://localhost:8888/notebooks/scripts/How%20to%20work%20with%20dates%20and%20times%20in%20pandas%3F.ipynb)
- pd.df.str.slice(start, end).astype(int)
- pd.to_datetime()
- df.Time.dt.hour.head()
- df.Time.dt.weekday_name.head()
- df.Time.dt.dayofyear.head()
- df.Time.max() - df.Time.min()
- (ufo.Time.max() - ufo.Time.min()).days

[How to find and remove duplicate rows in pandas?](http://localhost:8888/notebooks/scripts/How%20to%20find%20and%20remove%20duplicate%20rows%20in%20pandas%3F.ipynb)
- pd.series.duplicated()
- pd.series.duplicated().sum()
- pd.df.duplicated()
- pd.df.duplicated().sum()
- df.loc[df.duplicated(keep=False), :]
- df.loc[df.duplicated(keep='last'), :]
- df.loc[df.duplicated(keep='first'), :]
- df.drop_duplicates(subset, keep)

[How to avoid a SettingWithCopyWarning in pandas?](http://localhost:8888/notebooks/scripts/How%20to%20avoid%20a%20SettingWithCopyWarning%20in%20pandas%3F.ipynb)
- df.loc[df.colname1 >= 9, :].copy()

[How to change display options in pandas?](http://localhost:8888/notebooks/scripts/How%20to%20change%20display%20options%20in%20pandas%3F.ipynb)
- pd.get_option('display.max_rows')
- pd.set_option('display.max_rows', None)
- pd.reset_option('display.max_rows')
- pd.describe_option('rows')
- pd.set_option('display.float_format', '{:,}'.format)

[How do I create a pandas DataFrame from another object?](http://localhost:8888/notebooks/scripts/How%20do%20I%20create%20a%20pandas%20DataFrame%20from%20another%20object%3F.ipynb)
- create pandas df from dict, list, np.array
- create pandas df from dict(np.array)
- create pandas df from pd.concat([df, series], axis=1)

[How do I apply a function to a pandas Series or DataFrame?](http://localhost:8888/notebooks/scripts/How%20do%20I%20apply%20a%20function%20to%20a%20pandas%20Series%20or%20DataFrame%3F.ipynb)
