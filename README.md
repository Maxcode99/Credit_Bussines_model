# Credit_Bussines_model
An idea of a Business enterprise using machine learning and python 

# To see Project

## Windows ##

Creating venv
python -m venv .venv

to activate .venv <br> 
.venv/Scripts/activate

to install libraries <br>
pip install -r requirements.txt

## Mac Os ##

Creating venv
python3 -m venv .venv

to activate .venv <br> 
source .venv/bin/activate

to install libraries <br>
pip install -r requirements.txt


# Repository

/data

Place where the data is going to be analyzed  <br> 

/models <br> 

Place where the models are implement. There are three different models in the folder.  <br> 

model_1 = Uses a Gaussian naive bayes model to make classifications <br> 
model_2 = Uses a Random forest model to make classifications <br> 
model_3 = Uses a Support Vector Machine model to make classifications  <br> 

/performance <br> 

Shows the performance of each model, and how they developed <br> 

/preprocessing <br>

Place dedicated to transform data into a more usable form, in order to improve our models performance

/saved_models <br> 

Place where the pickle models are going to be saved in orders to use them later for later predictions, or furthermore <br> 


## Program Structure <br>

In /models: Overview of Each Module's Aspects <br>
__init__ Method <br>
This is where everything begins. In this method, the module reads the required files and prepares them to be used in subsequent steps. <br>

get_model Method <br>
The get_model method serves two purposes: <br>

Loading the Model: If a model already exists, it loads it directly. <br>
Creating the Model: If no model exists, it creates one by searching for the optimal hyperparameters, focusing on maximizing accuracy. <br>

get_hyperparameters Method <br>
This method identifies the best hyperparameters for the model, prioritizing accuracy as the primary metric. The process involves multiple iterations <br>
to refine the hyperparameters. If no model exists, this process can be time-consuming <br>

get_performance Method <br>
The get_performance method calculates key metrics, including accuracy, precision, recall, and F1-score, during both the training and testing phases. <br>
Additionally, it provides the AUC-ROC curve to evaluate the model's performance and development <br>

get_confusion_matrix Method <br>
This method generates the confusion matrix for both the training and testing phases <br>

In /performance: Performance Evaluation <br>
Individual Performances <br>
This section allows us to view the performance of each individual model used to create the final individual models. <br>

Stacked Performance <br>
Here, we can assess the performance of the models used to create the stacked model. <br>


In /preprocessing: Data Transformation <br>
The preprocessing module is responsible for transforming raw data into a format that can be effectively utilized by the models.  <br>
This ensures compatibility with each model in the pipeline. <br>





