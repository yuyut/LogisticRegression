# LogisticRegression

This task is to predict whether the person’s annual income is greater than 50K from the given personal information. 
Dataset: A total of 32561 training data and 16,281 test data (8140 in private test set and 8141 in public test set)

### Data Normalization:
Without normalization, a specific feature may influence the model the most, which is not we want. ** Here we normalize the feature to Normal distribution.** 

### Logistic regression classification model.

From the algorithm description, we implement:

- _sigmoid: 
to compute the sigmoid of the input. Use np.clip to avoid overflow. The smallest representable positive number is

- get_prob: 
given weight and bias, find out the model predict the probability to output 1

- infer: 
if the probability > 0.5, then output 1, or else output 0.

- _cross_entropy: 
compute the cross-entropy between the model output and the true label.

- _compute_loss : 
to compute the loss function  𝐿(𝑤)  with input  𝑋 ,  𝑌  and  𝑤 

- _gradient : 
With math derivation, the gradient of the cross entropy is  ∑𝑛−(𝑦̂ 𝑛−𝑓𝑤,𝑏(𝑥𝑛))𝑥𝑛𝑖

- Gradient Desent 
We construct a loss function  𝐿(𝑤)  with a parameter  𝑤 : We want to find out $w^* = \underset{w}{\operatorname{argmin}} L(w)$
Pick an inital value  𝑤0 
compute  $\frac{dL}{dw}|_{w=w_0}$ 
update  $w_{i+1} ← w_i - \eta \frac{dL}{dw}|_{w=w_i}$

We can show that the both the training loss and validation loss decays during training:

![](https://i.imgur.com/4VIDGFG.png)


![](https://i.imgur.com/TbgZQSx.png)


### attributes:
age: continuous.

workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.

fnlwgt: continuous.

education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.

education-num: continuous.

marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.

occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.

relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.

race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.

sex: Female, Male.

capital-gain: continuous.

capital-loss: continuous.

hours-per-week: continuous.

native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
