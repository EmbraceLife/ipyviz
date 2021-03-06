# iML

### ML in Human in vitro fertilization       
1. eggs from ovaries, fertilized with sperms, select best embryos to uterus    
2. select the best by 60 traits of embryos, num of possibilities is 2^60    

### ML in deciding which cow to turn to meet each year  
1. 700 attributes to decide which cows to keep or kill for meat    

### ML help uncover two types of patterns      
1. unstructured patterns     
2. structured patterns     

### Contact Lens Data    
![](https://github.com/EmbraceLife/ipyviz/blob/master/images/contactLensData.png)


### contact lens data can be structured in rules   
1. there are maximum 3 × 2 × 2 × 2 = 24 rules in total
2. only 14 possible rules in the table
3. rules examples:

> If tear production rate = reduced then recommendation = none Otherwise, if age = young and astigmatic = no then recommendation = soft    

### List of rules for contact lens data     
1. classification rules
2. association rules  

![](https://github.com/EmbraceLife/ipyviz/blob/master/images/contactLensRules.png)   

### Decision tree for contact lens data     
1. more concise and easily visualized with clarity
2. but incorrect with two example data compared to list of rules above
![](https://github.com/EmbraceLife/ipyviz/blob/master/images/decisionTreeContactLens.png)


### Weather data example      
![](https://github.com/EmbraceLife/ipyviz/blob/master/images/weatherData.png)

### Weather data can be structured in decision list     
1. This creates 36 possible combinations (3 × 3 × 2 × 2 = 36), of which 14 are present in the set of input examples.   
2. **decision list**: a set of rules presented in sequence as a whole      
3. rules out of context of decision list, is incorrect    


### decision List example     
![](https://github.com/EmbraceLife/ipyviz/blob/master/images/decisionList.png)

### Weather mix data example    
![](https://github.com/EmbraceLife/ipyviz/blob/master/images/weatherMix.png)


### iris dataset   
![](https://github.com/EmbraceLife/ipyviz/blob/master/images/irisData.png)    

### set of rules learnt from iris dataset
1. this set looks cumbersome and wary
2. it should be more compact
![](https://github.com/EmbraceLife/ipyviz/blob/master/images/irisRules.png)

### CPU performance dataset   
1. both attributes and outcome are numeric  
2. The classic way of dealing with continuous prediction is to write the outcome as a linear sum of the attribute values with appropriate weights  
> PRP = −55.9 + 0.0489 MYCT + 0.0153 MMIN + 0.0056 MMAX + 0.6410 CACH − 0.2700 CHMIN + 1.480 CHMAX   

3. the process of determining the weights is called regression
4. the basic regression method is incapable of discovering nonlinear relation- ships

### CPU performance dataset
![](https://github.com/EmbraceLife/ipyviz/blob/master/images/cpuPerformanceData.png)


### labor negotiation dataset   
1. there are missing and unknown data
2. is real life dataset
3. data table is reshaped for better viewing
![](https://github.com/EmbraceLife/ipyviz/blob/master/images/laborNegotiationData.png)

### labor negotiation decision trees
![](https://github.com/EmbraceLife/ipyviz/blob/master/images/laborNegotiationTrees.png)    
**tree on left**       
1. simple and intuitively making sense  
2. but predict incorrectly on some data    
3. could have better predictive result on new data

**tree on right**     
1. more complex and difficult to make sense     
2. more accurate but may not perform well on new data     
3. may be more likely to overfit    


### identification of rules for diagnosing soybean diseases
1. 680 rows: 680 diseased plants     
2. 35 attributes/features of a plant     
3. 19 disease categories    
4. table of data: sample of one plant    
![](https://github.com/EmbraceLife/ipyviz/blob/master/images/soybeanDiseaseDiagnose.png)   
5. two rules discovered by data examples    
> If leaf condition = normal and
> stem condition = abnormal and
> stem cankers = below soil line and canker lesion color = brown
> then diagnosis is rhizoctonia root rot

> If leaf malformation = absent
> and stem condition = abnormal and
> stem cankers = below soil line and canker lesion color = brown
> then diagnosis is rhizoctonia root rot

6. these two rules may converge into one or are two distinct rules   
7. ML-discovered-rules could outnumber and outperform experts-discovered-rules

## Real application of ML   
### Web Mining  
1. search engine use pageRank to reorder web pages   
2. ML on training dataset which includes queries and human judgements     
3. mine user's queries   
4. users reveal their inner thoughts by search queries, blogs and facebook and etc   

### Decisions involving judgements
1. Bankers ask customers to fill questionnaires, use some statistical methods to sort out 90% of customers for clear cut of making a loan or not  
2. remaining 10% of border line customers are decided by professional loaners   
3. but over 50% of loans are defaults or wrong    
4. ML outperform loan professionals by making 2/3 right with a small set of classification rules
5. ML's set of classification rules offer new rules/insights to professionals on what or how to decide

### screening images   
1. ML can replace professional eye checking work for oil slick images on the ocean    
2. problem1: example data is scarce;   
3. problem2: actual slick image is too small in each example, while background images vary greatly    
4. problem3: not understood    
> 'Finally, the performance task was to serve as a filter, and the user had to be provided with a convenient means of varying the false-alarm rate'    

### Load Forecast   
1. Electricity Supply Industry make great economies/save a lot of money if they can predict in good advance and precision of future demand for electricity    
2. current automating assistant model has 3 components: base load for the year, load periodicity over the year, and the effect of holidays    
3. three fundamental periodicity:
> diurnal, where usage has an early morning minimum and midday and afternoon maxima; weekly, where demand is lower at weekends; and seasonal, where increased demand during winter and summer for heating and cooling, respectively, creates a yearly cycle; and major and minor holidays    

4. How to deal with different periodicity:
> reconstructing a year’s load as a sequence of typical days, fitting the holidays in their correct position, and denormalizing the load to account for overall growth    

5. add correction to the static model above:
> take weather conditions into account by locating the previous day most similar to the current circumstances and using the historical information from that day as a predictor.
> To guard against outliers, the eight most similar days are located and their additive corrections averaged.
> A database was constructed of temperature, humidity, wind speed, and cloud cover at three local weather centers for each hour of the 15-year historical record, along with the difference between the actual load and that predicted by the static model.
> A linear regression analysis was performed to determine the relative effects of these parameters on load, and the coefficients were used to weight the distance function used to locate the most similar days.    

6. as a result, this ML/automated assistant can do the same work much faster and
> Human operators can analyze the forecast’s sensitivity to simulated changes in weather and bring up for examination the “most similar” days that the system used for weather adjustment.    

### Diagnostics
1. Diagnosis is one of the principal application areas of expert systems. Although the handcrafted rules used in expert systems often perform well, machine learning can be useful in situations in which producing rules manually is too labor intensive.    
2. Preventative maintenance of electromechanical devices such as motors and generators can forestall failures that disrupt industrial processes.    
3. A particular chemical plant uses more than 1000 different devices, diagnosed by a human expert with 20 years or more of experience.     
4. The derived attributes were run through an induction algorithm to produce a set of diagnostic rules. Initially, the expert was not satisfied with the rules because he could not relate them to his own knowledge and experience. **For him, mere statistical evidence was not, by itself, an adequate explanation. Further background knowledge had to be used before satisfactory rules were generated**. Although the resulting rules were quite complex, the expert liked them because he could justify them in light of his mechanical knowledge.


### Marketing and Sales
1. Banks: problem of  fickle customer loyalty and the challenge of detecting customers who are likely to defect so that they can be wooed back into the fold by giving them special treatment.    
2. successful data mining allows them to be precisely targeted to those customers who are likely to yield maximum benefit. Market basket analysis is the use of association techniques to find groups of items that tend to occur together in transactions, typically supermarket checkout data.    
3. There is enormous added value in being able to identify individual customer’s sales histories. Discount or “loyalty” cards let retailers identify all the purchases that each individual customer makes. This personal data is far more valuable than the cash value of the discount.     

### ML and statistics
1. some algorithms are statistics or probabilities     
2. statistics help test models, prepare data, select features and visualize data






### Sophisticated manufacturing processes often involve tweaking control parameters   
1. control parameters for separating gas and crude oil
2. control parameters for manufacturing nuclear fuel pellets
3. control parameters for printing process




---

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
