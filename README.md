
# Drinking water free of micropollutants

Micropollutants in water are small man-made contaminants that are widespread in natural water resources that receive water streams that have served human purposes (e.g. rivers, lakes, groundwater). 
Micropollutants can be classified by their origin, namely pharmaceuticasl, personal care products, hormones, pesticides, herbicides, and many other synthetic/organic compounds released in nature. 
Each chemical substance has some physico-chemical characteristic that will determine its removal through a water treatment process. 

## Project Set Up and Installation

## Dataset

### Overview

In water treatment, some special membranes are used to remove micropollutants (MP). MPs are ubiquitous contaminants present in the environment (soil, water) in concentrations of µg/L or ng/L. 
The dataset contains physico-chemical parameters used to characterize MPs. The dataset includes lab scale and pilot scale results of membrane water treatment, where operational variables and 
membrane types can influence the way a MP is removed. The dataset is external and comes from the book "Rejection of Emerging Organic Contaminants by Nanofiltration and Reverse Osmosis Membranes: 
Effects of Fouling, Modelling and Water Reuse".

Metadata:

Name – name of substance
ID – name abbreviation
MW – molecular weight
log_Kow – octanol water partition coefficient
log_D – octanol water partition coefficient
dipole – dipole moment
MV – molecular volume
length – molecular length
width – molecular width
depth – molecular depth
eq_width – equivalent width
membrane – membrane type
MWCO – molecular weight cutoff
SR – salt rejection
rejection – removal in % of MP by membrane

### Task

Any micropollutant is a chemical substance that can be characterised by a set of physico-chemical properties. What is more cumbersome to achieve is an estimation of the removal attained by a given 
membrane type during water treatment. The purpose of the ML project is to create a regression model that can help us to predict the rejection of a MP by a given membrane type.

### Access

The data is uploaded to the datastore of the workspace first as csv file. Then, the data is registered as a dataset that can be called from the worksapce for further use.

## Automated ML

Setting of the automl were chosen based on the task. The data contains non-categorical and categorical data, so featurization is specified as true. With featurization tasks like
normalization, handling missing data, or converting text to numeric are handled before model implementation. Since the target (rejection) is a real, we want to evaluate 
the accuracy of the model with the coefficient of determination R². The data is small, so 15 minutes as experiment timeout is good enough. The number of cross validations as 5 is also reasonable.

### Results

The regressors LightGBM and XGBoostRegressor did a quite decent job with R² 0.87 and 0.89 , respectively. This models did better because the data normalization with MaxAbsScaler helped 
in delivering the right values to the models. It is also expected that VotingEnsemble (R² = 0.90) and StackEnsemble ((R² = 0.89)) improved to a certain degree the quality of predictions, 
since those models are the result of agregating the results of different models.
The parameters of the Voting Ensemble model, are those corresponding to each of the models used during aggregation. It will be difficult to improve the performance of the model unless
the automl settings of experiment timeout is increase.

RunDetails AutoML screenshot 
![RunDetails AutoML](https://github.com/viyq/mle-CAPR-p3/blob/0503b4526088a845177b11dac994cad8e7bbc5c8/screenshots/RunDetails%20AutoML%20screenshot.jpg)

Best model with RunID screenshot
![Best model with RunID](https://github.com/viyq/mle-CAPR-p3/blob/9f689e80c30d740ae2005ef2868d6999a847dd1b/screenshots/best%20model%20with%20RunID.jpg)

## Hyperparameter Tuning

The selected model is the MLPRegressor of the package scikit-learn. The model uses neural networks to predict a response. The model was selected considering that the problem can be 
relatively solved by a multilinear regression, however the power of the feedfoward neural network of the MLPregressor can deliver an optimized model.
The data is first prepared with a onehotencoder to account for categorical data, and then is normalized with a minmaxscaler.
Hyperparameter sampling will create the parameters out of a random parameter sampling object. One example of parameters used for sampling are shown below

params = {
    "hidden_layer_sizes": 100,
    "solver": 'lbfgs',
    "activation": 'relu',
    "alpha": 0.0001,
    "tol": 1e-2,
    "max_iter": 500,
}

which is chosen from ranges 

{"hidden_layer_sizes": choice(100,200),
	"solver": choice(['lbfgs', 'sgd']),
	"activation": choice(['relu', 'logistic']),
	"alpha": choice([0.0001,0.001]),
	"tol": choice(0.005,0.01),
	"max_iter": choice(500,1000)}


### Results

The best hyperparameter model gave an R² = 0.82 with hyperparameters:
    "solver": 'lbfgs',
    "activation": 'logistic',
    "alpha": 0.001,
    "hidden_layer_sizes": 100,
    "tol": 0.005,
    "max_iter": 500,

A better model could have been achieved by allowing a greater number of runs and narrowing the sampling of parameters.

Hyperparameter RunDetails 
![Hyperparameter RunDetails](https://github.com/viyq/mle-CAPR-p3/blob/9f689e80c30d740ae2005ef2868d6999a847dd1b/screenshots/hyperparam%20RunDetails.jpg)

Hyperparam best model ID param
![Hyperparam best model ID param](https://github.com/viyq/mle-CAPR-p3/blob/9f689e80c30d740ae2005ef2868d6999a847dd1b/screenshots/hyperparam%20best%20model%20ID%20param.jpg)

## Model Deployment

The best model deployed is VotingEnsemble with R² = 0.90 . The model can be queried by sending a request to the URL having the azure container of the deployed model.

data = {
    "data":
    [
        {
            "Name": 'Caffeine',
            "ID": 'CFN',
            "MW": 194.19,
            "log_Kow": -0.07, 
            "log_D": -0.45, 
            "dipole": 3.71, 
            "MV": 133.3, 
            "length": 0.98, 
            "width": 0.87, 
            "depth": 0.56, 
            "eq_width": 0.70, 
            "membrane": 'ESPA', 
            "MWCO": 200, 
            "SR": 0.99,
        },
        {
            "Name": 'Bisphenol A',
            "ID": 'BPA',
            "MW": 228.29,
            "log_Kow": 3.32, 
            "log_D": 3.86, 
            "dipole": 2.13, 
            "MV": 199.5, 
            "length": 1.25, 
            "width": 0.83, 
            "depth": 0.75, 
            "eq_width": 0.79, 
            "membrane": 'De-HL', 
            "MWCO": 175, 
            "SR": 0.97,
        },
    ],
}

body = str.encode(json.dumps(data))
url = ...
api_key = 'xxxxx' # Replace this with the API key for the web service
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}
req = urllib.request.Request(url, body, headers)

will print predictions [87.39, 92.73]

## Screen Recording

- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

Link https://youtu.be/k1iLi14fdoU

## Standout Suggestions

- The notebook includes two cells for converting and saving the best model to the ONNX format.
- The webservice has been implemented with enabled applications insights for allowing logging of the requests: 
time of the request, error print.

Screenshots 

Application insights logs
![Application insights logs](https://github.com/viyq/mle-CAPR-p3/blob/9f689e80c30d740ae2005ef2868d6999a847dd1b/screenshots/application%20insights%20logs.jpg)

Service deployment logs
![Service deployment logs](https://github.com/viyq/mle-CAPR-p3/blob/9f689e80c30d740ae2005ef2868d6999a847dd1b/screenshots/service%20deployment%20logs.jpg)

Saved onnx model
![Saved onnx model](https://github.com/viyq/mle-CAPR-p3/blob/9f689e80c30d740ae2005ef2868d6999a847dd1b/screenshots/saved%20onnx%20model.jpg)

