Binary Prediction of Smoker Status using Bio-Signals

Smoking has been proven to negatively affect health in a multitude of ways. Evidence-based treatment for assistance in smoking cessation had been proposed and promoted. however, only less than one third of the participants could achieve the goal of abstinence. Many physicians found counseling for smoking cessation ineffective and time-consuming, and did not routinely do so in daily practice. To overcome this problem, several factors had been proposed to identify smokers who had a better chance of quitting. Providing a prediction model might be a favorable way to understand the chance of quitting smoking for each individual smoker.
This model will be used by a physicians.

Dataset Description -
age : 5-years gap
height(cm)
weight(kg)
waist(cm) : Waist circumference length
eyesight(left)
eyesight(right)
hearing(left)
hearing(right)
systolic : Blood pressure
relaxation : Blood pressure
fasting blood sugar
Cholesterol : total
triglyceride
HDL : cholesterol type
LDL : cholesterol type
hemoglobin
Urine protein
serum creatinine
AST : glutamic oxaloacetic transaminase type
ALT : glutamic oxaloacetic transaminase type
Gtp : γ-GTP
dental caries
smoking

Data folder includes 2 files:
train.csv - for training
test.csv - patients data for prediction

Data preparation and model selection
During data prepaperations I've used correlation matrix to find important features: 'hemoglobin', 'height_cm', 'weight_kg', 'triglyceride', 'gtp', 'serum_creatinine', 'waist_cm'. Futher models thraining showed better results with full features using.
I've trained the foolowing models:
 - LogisticRegression with 'liblinear' and 'lbfgs', parameter C tunning in range [1.0, 0.01, 0.1, 0.5, 10] and StandardScaler
 - SGDClassifier with alpha tunning in range [0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10] and StandardScaler
 - RandomForestRegressor max_depth tunning in range [5, 10, 15, 20, 25] and n_estimators tunning in range [10, 30, 50, 70, 90, 110, 130, 150, 170, 190]. The model showed overfitting with max_depth > 10.
The best model is RandomForestRegressor with max_depth=10 and n_estimators=110

How to run the model
1. Install docker on Ubuntu or Windows WSL Ubuntu:
sudo apt-get install docker.io
2. Build docker file:
docker build -t smoking-prediction .
3. Run docker image:
docker run -it -p 9797:9797 smoking-prediction:latest
4. Install pipenv localy:
pip install pipenv
5. Install python packages localy:
pipenv install --system --deploy
6. Run webclient.py to get random patient from data/test.csv with prediction:
python3 webclient.py

