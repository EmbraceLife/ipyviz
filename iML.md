# iML

[ML in Human in vitro fertilization]()       
1. eggs from ovaries, fertilized with sperms, select best embryos to uterus    
2. select the best by 60 traits of embryos, num of possibilities is 2^60    

[ML in deciding which cow to turn to meet each year]()    
1. 700 attributes to decide which cows to keep or kill for meat    

[ML help uncover two types of patterns]()    
1. unstructured patterns     
2. structured patterns     

[Contact Lens Data]()    
<img src="https://github.com/EmbraceLife/ipyviz/blob/master/images/contactLensData.png" height="300" width="600">    


[contact lens data can be structured in rules]()     
1. there are maximum 3 × 2 × 2 × 2 = 24 rules in total
2. only 14 possible rules in the table
3. rules examples:
> If tear production rate = reduced then recommendation = none Otherwise, if age = young and astigmatic = no then recommendation = soft    
----  


[Weather data structured in rules or decision list]()    
<img src="https://github.com/EmbraceLife/ipyviz/blob/master/images/weatherData.png" height="300" width="600">
1. This creates 36 possible combinations (3 × 3 × 2 × 2 = 36), of which 14 are present in the set of input examples.   
2. **decision list**: a set of rules presented in sequence as a whole      
3. rules out of context of decision list, is incorrect    
<img src ='https://github.com/EmbraceLife/ipyviz/blob/master/images/decisionList.png' height='300' width='600'>
----


[Weather numberic and categoric mix data]()    
<img src="https://github.com/EmbraceLife/ipyviz/blob/master/images/weatherMix.png" height="300" width="600">



## probability

[What does p(H) mean for flipping a fair coin?](https://youtu.be/uzkc-qNVoOk?t=52s)    
1. ends at nearly 5:00
2. possibilities meet constraint / all equally likely possibilities
3. do experiments large number of times and take percentage    

[What does p(1), p(2 or 5), p(3 and 4), p(even) mean for rolling a dice just one time?](https://youtu.be/uzkc-qNVoOk?t=5m9s)    
1. p(1) = 1/6
2. p(2 or 5) = (1+1) / 6
3. p(3 and 4) = 0 / 6 = 0
4. p(even) = (1+1+1)/6 = 1/2   

[what is p(pulling a yellow marble) out of a bag of 3 yellow marble, 2 red, 2 green, 1 blue? ](https://www.khanacademy.org/math/probability/probability-geometry/probability-basics/v/simple-probability)     
1. all possible outcome : 8
2. all outcomes meet condition: 3

[What is p(a point of big circle is also in the small circle) given the small circle is inside big circle?](https://youtu.be/mLE-SlOZToc?t=6m)    
1. all possible outcome: the big circle's area  
2. all outcome meet condition: the small circle's area  

[what is p(pulling a non-blue marble) out of a bag of 9 red, 2 blue, 3 green?](https://www.khanacademy.org/math/probability/probability-geometry/probability-basics/v/probability-1-module-examples)


[exercise on definition of probability above](https://www.khanacademy.org/math/probability/probability-geometry/probability-basics/e/probability_1?utm_source=Annotation&utm_medium=Annotation&utm_campaign=Annotation)

[what is p(17th game score >= 30 points), given the last 16 games and their points?](https://youtu.be/RdehfQJ8i_0?t=2m44s)     
1. Based on past experience, we can make reasonable estimates of the likelihood of future events.
2. total possible outcomes : 16
3. all possible outcomes meet condition: numbe of games with points over 30


## Anomaly detection
Anomaly detection is the search for items or events which do not conform to an expected pattern. These detected patterns are called anomalies and translate to critical and actionable information in various application domains. It is also referred as outliers.

## Association rule
Association rule learning searches for relationships among variables. For example a supermarket might gather data about how the customer purchasing the various products. With the help of association rule, the supermarket can identify which products are frequently bought together and this information can be used for marketing purposes. This is sometimes known as market basket analysis.

## Clustering   
Clustering discovers the groups and structures in the data in some way or another similar way, without using known structures in the data.

## Classification
Classification generalizes known structure to apply to new data. Take an example; an e-mail program might attempt to classify an e-mail as "legitimate" or as "spam" mail.

## Regression
Regression attempts to find a function which models the data with the least error.

## Summarization   
Summarization provides a more compact representation of the data set, which includes visualization and report generation.

[Decision tree basics](http://localhost:8888/notebooks/scripts/decision%20tree%20basics.ipynb)

[How to plot decision tree on regression?](http://localhost:8888/notebooks/scripts/plot_tree_regression.ipynb)
