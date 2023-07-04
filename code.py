import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn import model_selection
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, add_dummy_feature

from google.colab import drive
drive.mount('/content/drive')

data = pd.read_csv('/content/drive/My Drive/BMGT347/Final Project/characteristics_anom.csv')

data = data.set_index(['permno'])
data.head()

X_var_names = data.columns[2:]
y_var_name = data.columns[1]
print('X variables: ', X_var_names.values)
print('y variable: ', y_var_name)

data['date'] = pd.to_datetime(data['date'], format='%m/%Y')

# date of significant macroeconomic events
black_mon_drop = '1987-10'
fri_13_drop = '1989-10'
dotcom_drop_start = '2000-03'
dotcom_drop_end = '2002-10'
finCri_08_start = '2008-09'
finCri_08_end = '2009-03'

# drop the rows where major economic or external events caused significant market disruption
data = data.loc[~((data['date'] == black_mon_drop) | (data['date'] == fri_13_drop) |
              ((data['date'] >= dotcom_drop_start) & (data['date'] <= dotcom_drop_end)) |
              ((data['date'] >= finCri_08_start) & (data['date'] <= finCri_08_end))), :]

print(data.describe())

# split data into training and testing sets
train=data.loc[data['date']<"01/2005"]
test=data.loc[data['date']>="01/2005"]

X_train = train[X_var_names]
y_train = train[y_var_name]

X_test = test[X_var_names]
y_test = test[y_var_name]

print(test.head())

# incorporate all 43 predictors within the formula
formula='re~'+("+").join(X_var_names.tolist())

# fit ols model on training data
ols=smf.ols(formula=formula,data=train).fit()

# use model to make prediction on training data
ols_pred_train=ols.predict(X_train)

# evaluate performance of the model using R-Squared and RMSE metrics
print("OLS Training R-Squared:",r2_score(y_train,ols_pred_train))
print("OLS Training MSE:",mean_squared_error(y_train,ols_pred_train))

# use model to make prediction on test data
ols_preds = ols.predict(X_test)

# evaluate model performance using R-squared and RMSE metrics
print("OLS R-squared:", r2_score(y_test, ols_preds))
print("OLS MSE:", mean_squared_error(y_test, ols_preds))

# Random Forests
# fit Random Forest model on training data
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators = 100, max_depth = 7, random_state = 0).fit(X_train, y_train)

# generate prediction for training set data
rf_preds_train = forest.predict(X_train)
# evaluate model performance in training set
print("RF Training MSE:", mean_squared_error(y_train,rf_preds_train))
print("RF Training R-Squared:", r2_score(y_train,rf_preds_train))

# use the model to make predictions on the test data
rf_preds = forest.predict(X_test)

# evaluate model performance using R-squared and RMSE metrics
print("RF R-squared:", r2_score(y_test, rf_preds))
print("RF MSE:", mean_squared_error(y_test, rf_preds))

# feature importance
# print feature importances of model to find variables most predictive of excess returns
coeff_forest = pd.DataFrame()
feature_importances = forest.feature_importances_
coeff_forest['Feature importance'] = feature_importances
coeff_forest.index = X_var_names.values.tolist()
coeff_forest = coeff_forest.sort_values(by = 'Feature importance',ascending=False)
coeff_forest

# fit Lasso regression model on training data using cross-validation to select regularization parameter
alphas = 10**np.linspace(3,-5,2000)*0.5

lasso = LassoCV(alphas=alphas, cv=10, random_state=30).fit(X_train, y_train)

# use model to make predictions on training data
lasso_preds_train=lasso.predict(X_train)

# evaluate performance of training data using R-Squared and RMSE metrics
print("Lasso Training R-Squared:",r2_score(y_train,lasso_preds_train))
print("Lasso Training MSE:",mean_squared_error(y_train,lasso_preds_train))

# use model to make predictions on testing data
lasso_preds = lasso.predict(X_test)

# evaluate model performance using R-squared and RMSE metrics
print("Lasso R-squared:", r2_score(y_test, lasso_preds))
print("Lasso MSE:", mean_squared_error(y_test, lasso_preds))

# print coefficients of model to find variables most predictive of excess returns
coeff_lasso = pd.DataFrame()
coefficients = lasso.coef_
coeff_lasso['Coefficient'] = coefficients
coeff_lasso['abs_coeff'] = abs(coeff_lasso['Coefficient'])
coeff_lasso.index = X_var_names.values.tolist()
coeff_lasso = coeff_lasso.sort_values(by = 'abs_coeff',ascending=False)
pd.DataFrame(coeff_lasso.iloc[:,0])

# cross validation within training sample to find best method

# Linear Regression
Linear_regression = LinearRegression()
print('Linear Cross Validation Score: ', cross_val_score(Linear_regression, X_train, y_train, cv = 10, scoring = 'r2' ).mean())

# Lasso
Lasso = LassoCV()
print('Lasso Cross Validation Score: ', cross_val_score(Lasso, X_train, y_train, cv = 10, scoring = 'r2').mean())

# Random Forest
Random_forest = RandomForestRegressor(n_estimators = 100, max_depth = 7, random_state = 0)
print('Random Forest Cross Validation Score: ', cross_val_score(Random_forest, X_train, y_train, cv = 10, scoring = 'r2').mean())