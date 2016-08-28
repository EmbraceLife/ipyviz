# ipyviz

## Questions and Examples

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
