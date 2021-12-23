
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
Effects of Fouling, Modelling and Water Reuse". [data link] (http://desalination-delft.nl/wp-content/uploads/2018/06/Yangali-2010-Yangali.pdf) 

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
membrane type during water treatment. The purpose of the ML project is to create a "regression model" that can help us to predict the rejection of a MP by a given membrane type.

In this project the dataset will be used to generate two best regression models with two different ML approaches. With Azure automated ML (AutoML) the dataset will be presented as it is, without any 
previous data pipeline (e.g. normalization, onehotencoder). In AutoML data preparation will happen automatically. The regression model selection will be also automatic in the AutoML engine, then 
the best model giving the best primary metric will be selected and deployed. The second approach will use Hyperparameter tuning (Hyperdrive) after an specific model has been selected.

After getting the best ML model, the model will be first registered and then deployed as a Webservice for consumption (endpoint). The model deployment will be realized with an inference config call to have 
the model as a Webservice, with Enable Application Insights to track logs when the models is consumed. Finally the model webservice (endpoint) will be tested by sending data to request model predictions.

### Access

The data is uploaded to the datastore of the workspace first as csv file. Then, the data is registered as a dataset that can be called from the worksapce for further use.

## Automated ML

Settings of the automl were chosen based on the task. The data contains non-categorical and categorical data, so featurization is specified as true. With featurization tasks like
normalization, handling missing data, or converting text to numeric are handled before model implementation. Since the target (rejection) is a real, we want to evaluate 
the accuracy of the model with the coefficient of determination R². The data is small, so 15 minutes as experiment timeout is good enough. The number of cross validations as 5 is also reasonable.

### Results

The regressors LightGBM and XGBoostRegressor did a quite decent job with R² 0.87 and 0.89 , respectively. This models did better because the data normalization with MaxAbsScaler helped 
in delivering the right values to the models. It is also expected that VotingEnsemble (R² = 0.90) and StackEnsemble ((R² = 0.89)) improved to a certain degree the quality of predictions, 
since those models are the result of agregating the results of different models.
The parameters of the Voting Ensemble model, are those corresponding to each of the models used during aggregation. 

RunDetails AutoML screenshot 
![RunDetails AutoML](https://github.com/viyq/mle-CAPR-p3/blob/0503b4526088a845177b11dac994cad8e7bbc5c8/screenshots/RunDetails%20AutoML%20screenshot.jpg)

Best model with RunID screenshot
![Best model with RunID](https://github.com/viyq/mle-CAPR-p3/blob/9f689e80c30d740ae2005ef2868d6999a847dd1b/screenshots/best%20model%20with%20RunID.jpg)

## Hyperparameter Tuning

The selected model is the MLPRegressor of the package scikit-learn. The model uses neural networks to predict a response. The model was selected considering that the problem can be 
relatively solved by a multilinear regression, however the power of the feedfoward neural network of the MLPregressor can deliver an optimized model.
The data is first prepared with a onehotencoder to account for categorical data, and then is normalized with a minmaxscaler.

Hyperparameter sampling will create the parameters out of a random parameter sampling object. The train.py file contains information about the type of each parameter and
what is intended with its use. More information about the MLPRegressor cane be found in the link [Sciki-learn MLPRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor)

parser.add_argument('--hidden_layer_sizes', type=int, default=100, help="The ith element represents the number of neurons in the ith hidden layer")
parser.add_argument('--solver', type=str, default='lbfgs', help="The solver for weight optimization.")
parser.add_argument('--activation', type=str, default='relu', help="Activation function for the hidden layer.")
parser.add_argument('--alpha', type=float, default=0.0001, help="L2 penalty (regularization term) parameter")
parser.add_argument('--tol', type=float, default=0.005, help="Tolerance for the early stopping")
parser.add_argument('--max_iter', type=int, default=300, help="Maximum number of iterations to converge")

One example of parameters used for sampling is shown below

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

Screenshots

Hyperparameter RunDetails 
![Hyperparameter RunDetails](https://github.com/viyq/mle-CAPR-p3/blob/9f689e80c30d740ae2005ef2868d6999a847dd1b/screenshots/hyperparam%20RunDetails.jpg)

Hyperparam best model ID param
![Hyperparam best model ID param](https://github.com/viyq/mle-CAPR-p3/blob/9f689e80c30d740ae2005ef2868d6999a847dd1b/screenshots/hyperparam%20best%20model%20ID%20param.jpg)

### Future improvement suggestions

For the AutoML notebook, it may be difficult to improve the performance of the VotingEnsemble model unless the automl settings of experiment timeout 
is increased. The minimum time of 0.25 h was used due to time restrictions in the lab.

For the Hyperparameter select, a better model could have been achieved by allowing a greater number of runs and narrowing the sampling of parameters.
Alternatively, a new Voting Ensemble model could be considered with selected best models from AutoML including the MLPRegressor.

## Model Deployment

Models can be deployed locally or in the cloud. An Azure Container Instance (ACI) web service will be used for deploying the AzureML trained best model. 
The model is already registered in the workspace, which means the model itself is available. But still, a few other requirements are needed:
- Inference configuration with an entry script 'score.py'. The script has instructions for using the model, e.g. make a prediction with the data, and return a response.
- Inference configuration with an environment '.yml' file that describes dependencies required by the model and the entry script.
- The deployment configuration, which will point into a target compute configuration (e.g. CPU cores, memory) needed to run the web service.

When deploying, the environmnet will be registered, an image will be created and the deployment will be computed. 
Once the model has been deployed the ACI service creation operation is finished. An endpoint is available now in the Workspace. The status and 
needed API key instruction will be provided in the endpoint to make use of the model.

Endpoint with healthy deployed model
![Healthy deployed model](https://github.com/viyq/mle-CAPR-p3/blob/163067ac3426b061d4b178b7ca501a042ce035e1/screenshots/endpoint%20health.jpg)

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

