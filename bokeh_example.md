# bokeh examples

[How to create a cosin curve line of circles](http://localhost:8888/notebooks/scripts/a%20cosin%20curve%20line%20of%20circles.ipynb)
- numpy: cosin and linspace
- bokeh.plotting: circle, show, figure, output_notebook

[create quad, circle, triangle in a single figure with same dataset](http://localhost:8888/notebooks/scripts/create%20quad%2C%20circle%2C%20triangle%20in%20a%20single%20figure%20with%20same%20dataset.ipynb)
- figure.quad
- figure.circle
- figure.triangle
- from IPython.display import display
- pd.set_option('display.max_rows', 10)
- - pd.set_option('display.max_columns', 10)

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

[copy a data and convert string to number for dataframe](http://localhost:8888/notebooks/scripts/Charts.ipynb)
```python

import pandas as pd

from bokeh.io import output_notebook, show
output_notebook()

from bokeh.sampledata.autompg import autompg
autompg.head(n=10)

from bokeh.sampledata.glucose import data

glucose = data.copy()[0:2000]
glucose.isig = pd.to_numeric(glucose.isig, errors=False)
glucose.head()
# glucose.dtypes
```


[How to set default configure for figures?](http://localhost:8888/notebooks/scripts/Charts.ipynb)
```python
from bokeh.charts import defaults

defaults.plot_height=300
defaults.plot_width=800
defaults.tools='pan, wheel_zoom, reset'
```

[How to create a donut chart?](http://localhost:8888/notebooks/scripts/Charts.ipynb)
```python
show(Donut(autompg.cyl.astype('str'), palette=Spectral8, plot_width=400, plot_height=400, responsive=False,))
```

[How to manipulate a string in pd.dataframe?]()
```python
autompg['make'] = autompg.name.str.split(' ').str[0]

autompg['detail'] = autompg.name.str.split(' ').str[1]
```

[How to use ColumnDataSource?](http://localhost:8888/notebooks/scripts/How%20to%20use%20ColumnDataSource.ipynb)
- ColumnDataSource takes both dict and dataframe
- turn from dict to dataframe
- column_data_source_dict = ColumnDataSource(data_dict)
- column_data_source_dict.data

[check version, IFrame, histogram, utils-gapminder](http://localhost:8888/notebooks/scripts/tutorial/00%20-%20intro.ipynb)
- How to check library versions
- histogram and bins
- create a library on your own as utils  
- run gapminder

[How to create gap minder on jupyter notebook](http://localhost:8888/notebooks/scripts/gapminder%20code%20in%20notebook.ipynb)


[example.notebook.charts](http://localhost:8888/notebooks/scripts/01%20-%20charts.ipynb)
- bar, stacked bar, grouped bar
- scatter with color and marker
- pd.set_option
- histogram with random.normal and random.lognormal
- pd.DataFrame(dict)
- pd.concat([normal, lognormal]): row-bind two dataframe
- two histograms on the same figure or plot
- 3 boxplots with diff colors on the same figure


[notebook plotting](http://localhost:8888/notebooks/scripts/02%20-%20plotting.ipynb)
- from bokeh.io import output_notebook, show
- create scatter plot with circle
- numpy.linspace
- size, color can be vectorized
- different markers to apply: cicle, square, triangle, inverted, cross, diamend...
- image and rgba
- plot multiple glyphs on the same figure
- create a new list based on an old list

[notebook styling](http://localhost:8888/notebooks/scripts/03%20-%20styling.ipynb)
- figure.outline_line_width  
- figure.outline_line_alpha
- figure.outline_line_color
- r.glyph.size = 50
- r.glyph.fill_alpha = 0.2
- r.glyph.line_color = "firebrick"
- r.glyph.line_dash = [5, 5]
- r.glyph.line_width = 2
- tools = 'tap'
- selection_color="firebrick"
- nonselection_fill_alpha=0.2,
- nonselection_fill_color="grey",
- nonselection_line_color="firebrick",
- nonselection_line_alpha=1.0
- turn index to a series: type(subset.index.to_series())
- How to set no tools and hide tools
- p.line with x, y as two series
- add hover tools: p.add_tools(HoverTool(tooltips=None, renderers=[cr], mode='vline'))
- p.xaxis.major_label_orientation: change x-axis label's orientation
- change minor tick line size and position: p.axis.minor_tick_in = -3
- p.axis.minor_tick_out = 6
- set x and y axis labels: p.yaxis.axis_label = "Pressure"
- change x, y axis path width and color: p.xaxis.axis_line_width = 3
p.xaxis.axis_line_color = "red"
- change major tick/label: p.yaxis.major_label_text_color = "orange"
p.yaxis.major_label_orientation = "vertical"
- set x, y axis grid line: p.xgrid.grid_line_color/alpha/dash =
- set x, y axis grid band: p.ygrid.band_fill_alpha = 0.1
p.ygrid.band_fill_color = "navy"
- make a legend
- set legend location
- fill_color: None and "white" is quite different


[notebook interactions](http://localhost:8888/notebooks/scripts/04%20-%20interactions.ipynb)
- calc new list from old list
- calc new series or dataframe from old series
- create 3 plots in grids, without any connection or links
- link panning: `s2 = figure(x_range=s1.x_range, y_range=s1.y_range, **plot_options)`
- link brush: `left.circle('x', 'y0', source=source)`
- ColumnDataSource data both dict or df: `source = ColumnDataSource(data=dict(x=x, y0=y0, y1=y1))`
- get_custom_hover() using html/css to create tooltips
- source.data['y'] = A * np.sin(w * x + phi)
- push_notebook()
- widgetbox() and Slider()
- customJS callbacks: `from bokeh.models import TapTool, CustomJS, ColumnDataSource`
- How to use customJS: `callback = CustomJS(code="alert('hello world')")`
- How to use TapTool: `tap = TapTool(callback=callback)`

[stock lines in a single figure](http://localhost:8888/notebooks/scripts/stock%20lines%20in%20a%20single%20figure.ipynb)
- access and download stock data from Uqer.io   
- parse date format
- rename columns and drop columns and create new columns


[Notebook models](http://localhost:8888/notebooks/scripts/06%20-%20models.ipynb)
- How to access keys from values in a dictionary?
- How to manipulate dataframe's each column using `map`, `apply`, `lambda args:`
- How to use apply() and map() and tuple() together: `sprint["SelectedName"] = sprint[["Name", "Medal", "Year"]].apply(tuple, axis=1).map(lambda args: selected_name(*args))`
- args vs `*args` : how to bring in multiple args
- How to set x-y-axis, path line, major tick, color, size, position
- How to add tooltips?

[Notebook annotations](http://localhost:8888/notebooks/scripts/07%20-%20annotations.ipynb)
- add horizontal or vertical span lines
- add box annotations inside a figure
- make a label (text) annotation
- create a group of labels or labeset
- create arrows


[Notebook server](http://localhost:8888/notebooks/scripts/08%20-%20server.ipynb)
- running .py files on terminal with `bokeh serve --show app.py`
- add button and input widget
- update plotting using button and input to change certain values
- streaming data

[Notebook Geo](http://localhost:8888/notebooks/scripts/09%20-%20geo.ipynb)

[Notebook datashader](http://localhost:8888/notebooks/scripts/10%20-%20datashader.ipynb)

[bolinger](http://localhost:8888/notebooks/scripts/bollinger.ipynb)
- bokeh.plotting.patch()


[burtin]()
- StringIO(stringData): convert stringdata to dataframe


[How to run bokeh app project?]()
- go to 'examples/app' folder
- source activate tryme
- bokeh serve --show gapminder (folder name)


[How to write slider with python instead of javascript](http://localhost:8888/notebooks/scripts/How%20to%20write%20slider%20with%20python%20not%20javascript.ipynb)
