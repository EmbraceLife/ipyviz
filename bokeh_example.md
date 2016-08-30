# bokeh examples

[How to create a cosin curve line of circles](http://localhost:8888/notebooks/scripts/a%20cosin%20curve%20line%20of%20circles.ipynb)
- numpy: cosin and linspace
- bokeh.plotting: circle, show, figure, output_notebook

[create quad, circle, triangle in a single figure with same dataset](http://localhost:8888/notebooks/scripts/create%20quad%2C%20circle%2C%20triangle%20in%20a%20single%20figure%20with%20same%20dataset.ipynb)
- figure.quad
- figure.circle
- figure.triangle

[Link 3 independent figures by brush or by row when selecting](http://localhost:8888/notebooks/scripts/Link%203%20independent%20figures%20by%20brush%20or%20by%20row%20when%20selecting.ipynb)
- from bokeh.plotting import figure, output_notebook, show
- from bokeh.sampledata.autompg import autompg
- from bokeh.models import ColumnDataSource
- from bokeh.layouts import gridplot
- source = ColumnDataSource(autompg.to_dict("list"))
