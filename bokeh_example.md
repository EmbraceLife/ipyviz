# bokeh examples

[How to create a cosin curve line of circles](http://localhost:8888/notebooks/scripts/a%20cosin%20curve%20line%20of%20circles.ipynb)
- numpy: cosin and linspace
- bokeh.plotting: circle, show, figure, output_notebook

[create quad, circle, triangle in a single figure with same dataset](http://localhost:8888/notebooks/scripts/create%20quad%2C%20circle%2C%20triangle%20in%20a%20single%20figure%20with%20same%20dataset.ipynb)
- figure.quad
- figure.circle
- figure.triangle
- from IPython.display import display

[Link 3 independent figures by brush or by row when selecting](http://localhost:8888/notebooks/scripts/Link%203%20independent%20figures%20by%20brush%20or%20by%20row%20when%20selecting.ipynb)
- from bokeh.plotting import figure, output_notebook, show
- from bokeh.sampledata.autompg import autompg
- from bokeh.models import ColumnDataSource
- from bokeh.layouts import gridplot
- source = ColumnDataSource(autompg.to_dict("list"))


[scatter plot with x, y, color](http://localhost:8888/notebooks/scripts/scatter%20plot%20with%20x%2C%20y%2C%20color.ipynb)
- any two columns to form x, y
- any other column to make color or marker
- scatter = Scatter(df, x='mpg', y='hp', color='cyl', marker='origin',
                  title="Auto MPG", xlabel="Miles Per Gallon",
                  ylabel="Horsepower")

```python
import pandas as pd
from bokeh.charts import output_notebook, show

output_notebook()


from bokeh.charts import Scatter
from bokeh.sampledata.autompg import autompg as df

p = Scatter(df, x='mpg', y='hp', color='cyl', title="HP vs MPG (shaded by CYL)",
            xlabel="Miles Per Gallon", ylabel="Horsepower", legend="top_right")

show(p)

from bokeh.sampledata.autompg import autompg as df
from bokeh.charts import Scatter, output_notebook, show
output_notebook()

df.columns
df.origin.unique()

scatter = Scatter(df, x='mpg', y='hp', color='cyl', marker='origin',
                  title="Auto MPG", xlabel="Miles Per Gallon",
                  ylabel="Horsepower")

show(scatter)

```




[a stacked bar](http://localhost:8888/notebooks/scripts/a%20stacked%20bar.ipynb)  
- from bokeh.charts.utils import df_from_json         
- from IPython.display import display    
- pd.set_option('display.max_rows', 10)
- df = df_from_json(data)

```python
from bokeh.charts import Bar
from bokeh.charts.attributes import cat, color
from bokeh.charts.operations import blend
from bokeh.charts.utils import df_from_json
from bokeh.sampledata.olympics2014 import data

type(data)

df = df_from_json(data)

# filter by countries with at least one medal and sort by total medals
df = df[df['total'] > 0]
df = df.sort_values(by="total", ascending=False) # descending

df.head()

bar = Bar(df,
          values=blend('bronze', 'silver', 'gold', name='medals', labels_name='medal'),
          label=cat(columns='abbr', sort=False),
          stack=cat(columns='medal', sort=False),
          color=color(columns='medal', palette=['SaddleBrown', 'Silver', 'Goldenrod'],
                      sort=False),
          legend='top_right',
          title="Medals per Country, Sorted by Total Medals",
          tooltips=[('medal', '@medal'), ('country', '@abbr')]
         )

show(bar)

```

[bokeh chart: a histogram](http://localhost:8888/notebooks/scripts/bokeh%20chart%20a%20histogram.ipynb)   
- hist = Histogram(df, values='weight', color = 'origin',
                 title="HP Distribution by Cylinder Count", legend='top_right')

```python


from bokeh.charts import Histogram, show, output_notebook
from bokeh.sampledata.autompg import autompg as df
output_notebook()

print(df.columns)
print(df.head())
df.sort_values('cyl', inplace=True, ascending=False)
df.head()

hist = Histogram(df, values='hp', color = 'origin',
                 title="HP Distribution by Cylinder Count", legend='top_right')

show(hist)

```


[bokeh chart: a boxplot](http://localhost:8888/notebooks/scripts/bokeh%20chart%20a%20boxplot.ipynb)
```python
from bokeh.charts import BoxPlot, output_notebook, show
from bokeh.sampledata.autompg import autompg as df
output_notebook()

p = BoxPlot(df, values='mpg', label='origin', color='origin',
            title="MPG Summary (grouped and shaded by CYL)")

show(p)

```



[bokeh chart a heat map](http://localhost:8888/notebooks/scripts/bokeh%20chart%20a%20heat%20map.ipynb)                 
- hm = HeatMap(autompg, x=bins('hp'), y=bins('displ'))
```python

from bokeh.charts import HeatMap, bins, output_notebook, show
from bokeh.sampledata.autompg import autompg

output_notebook()

autompg.head()

hm = HeatMap(autompg, x=bins('hp'), y=bins('displ'))

show(hm)

```




[bokeh chart: 2 step line](http://localhost:8888/notebooks/scripts/bokeh%20chart%202%20step%20line%20in%20one%20figure.ipynb)
```python
# Step

from bokeh.charts import Step, output_notebook, show
import pandas as pd

range(1999, 2016)

# build a dataset where multiple columns measure the same thing
data = dict(stamp=[
                .33, .33, .34, .37, .37, .37, .37, .39, .41, .42,
                .44, .44, .44, .45, .46, .49, .49],
            postcard=[
                .20, .20, .21, .23, .23, .23, .23, .24, .26, .27,
                .28, .28, .29, .32, .33, .34, .35],
            year = range(1999, 2016),
            )

# create a line chart where each column of measures receives a unique color and dash style
line = Step(data, y=['stamp', 'postcard'],
            x = 'year',
            dash=['stamp', 'postcard'],
            color=['stamp', 'postcard'],
            title="U.S. Postage Rates (1999-2015)", ylabel='Rate per ounce', legend=True)

show(line)
```
