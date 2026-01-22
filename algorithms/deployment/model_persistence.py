import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error
import joblib

df = pd.read_csv('/Users/tess/Desktop/MLE2025/ML-Masterclass/UNZIP_FOR_NOTEBOOKS_FINAL (1)/DATA/Advertising.csv')

X = df.drop('sales',axis=1)
y = df['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

# Further split 30% of test into validation and hold-out (15% and 15% each)
X_validation, X_holdout_test, y_validation, y_holdout_test = train_test_split(X_test, y_test, test_size=0.5, random_state=101)

print("---n_estimators=10,random_state=101---")
model = RandomForestRegressor(n_estimators=10,random_state=101)

model.fit(X_train,y_train)

validation_predictions = model.predict(X_validation)

MAE = mean_absolute_error(y_validation,validation_predictions)
print("Mean Absolute Error:", MAE)

RMSE = mean_squared_error(y_validation,validation_predictions)**0.5 #RMSE
print("Root Mean Squared Error:", RMSE)

        ### Hyperparameter Tuning
print("---n_estimators=35,random_state=101---\n")
model = RandomForestRegressor(n_estimators=35,random_state=101)
model.fit(X_train,y_train)

validation_predictions = model.predict(X_validation)

MAE = mean_absolute_error(y_validation,validation_predictions)
print("Mean Absolute Error:", MAE)

RMSE =mean_squared_error(y_validation,validation_predictions)**0.5 #RMSE
print("Root Mean Squared Error:", RMSE)

        ## Final Hold Out Test Performance for Reporting
print("---n_estimators=35,random_state=101---\n")
model = RandomForestRegressor(n_estimators=35,random_state=101)
model.fit(X_train,y_train)

test_predictions = model.predict(X_holdout_test)
MAE = mean_absolute_error(y_holdout_test,test_predictions)
print("Mean Absolute Error:", MAE)
RMSE = mean_squared_error(y_holdout_test,test_predictions)**0.5
print("Root Mean Squared Error:", RMSE)

        ## Full Training with Best Model on All Data, not just Train and Validation data
print("============= Final Model Training on Full Data =============") 
print("---n_estimators=35,random_state=101---\n")
final_model = RandomForestRegressor(n_estimators=35,random_state=101)
final_model.fit(X,y)

joblib.dump(final_model,'final_model.pkl')

print("X.columns:", list(X.columns))

joblib.dump(list(X.columns),'column_names.pkl')

        ## Loading Model (Model Persistence)

col_names = joblib.load('column_names.pkl')

print("col_names", col_names)

loaded_model = joblib.load('final_model.pkl')
body = [{ "TV": 230.1, "radio": 37.8, "newspaper": 69.2 },
 { "TV": 123.1, "radio": 37.8, "newspaper": 76.2 },
 { "TV": 343.1, "radio": 22.9, "newspaper": 54.2 },
 { "TV": 144.1, "radio": 37.8, "newspaper": 15 }]

input_df = pd.DataFrame(body)
prediction = loaded_model.predict(input_df)

print("Prediction from loaded model:", prediction.tolist())
