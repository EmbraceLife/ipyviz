# ipyviz

## Questions and Examples
[pandas examples](https://github.com/EmbraceLife/ipyviz/blob/master/pandas_examples.md)     

[bokeh examples](https://github.com/EmbraceLife/ipyviz/blob/master/bokeh_example.md)

## Housework
[How to update all packages in conda environment?](http://conda.pydata.org/docs/test-drive.html)    
1. to update conda using `conda update conda`   
2. to create an environment installing specific packages `conda create --name tryme numpy pandas bokeh`    
3. check all environments created `conda info --envs`      
4. make an exact copy of an environment `conda create --name trybokeh --clone tryme`    
5. delete an environment `conda remove --name trybokeh --all`     
6. check version of a package `python --version`   
7. check packages installed in an environment `conda list`    
8. check a specific package is available for installation `conda search beautifulsoup4`    
9. install and update a package `conda install bokeh`    
10. remove a pacakge `conda remove -name trybokeh bokeh`    
