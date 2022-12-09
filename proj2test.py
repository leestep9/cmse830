import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('ds_salaries.csv')

st.title('Data Science Salaries Regression Modeling Tutorial')
st.markdown('Stephen Lee')
st.markdown('Libraries: pandas, numpy, seaborn, plotly, plotly.express, plotly graph_objects, sklearn')

st.header('Info on columns')
st.markdown('Data Science Job Salaries Dataset contains 11 columns, each are:')
st.markdown('1. work_year: The year the salary was paid.')
st.markdown('2. experience_level: The experience level in the job during the year')
st.markdown('3. employment_type: The type of employment for the role')
st.markdown('4. job_title: The role worked in during the year.')
st.markdown('5. salary: The total gross salary amount paid.')
st.markdown('6. salary_currency: The currency of the salary paid as an ISO 4217 currency code.')
st.markdown('7. salaryinusd: The salary in USD')
st.markdown('8. employee_residence: Employees primary country of residence in during the work year as an ISO 3166 country code.')
st.markdown('9. remote_ratio: The overall amount of work done remotely')
st.markdown("10. company_location: The country of the employer's main office or contracting branch")
st.markdown('11. company_size: The median number of people that worked for the company during the year')



df = df.replace({'Machine Learning Scientist': 'Data Scientist',
                 'Product Data Analyst': 'Data Analyst',
                 'Lead Data Scientist': 'Data Scientist',
                 'Business Data Analyst': 'Data Analyst',
                 'Lead Data Analyst': 'Data Analyst',
                 'Data Science Consultant': 'Data Scientist',
                 'BI Data Analyst': 'Data Analyst',
                 'Director of Data Science': 'Data Scientist',
                 'Research Scientist': 'Data Scientist',
                 'Machine Learning Manager': 'Data Scientist',
                 'AI Scientist': 'Data Scientist',
                 'Principal Data Scientist': 'Data Scientist',
                 'Data Science Manager': 'Data Scientist',
                 'Head of Data': 'Data Scientist',
                 'Applied Data Scientist': 'Data Scientist',
                 'Marketing Data Analyst': 'Data Analyst',
                 'Financial Data Analyst': 'Data Analyst',
                 'Machine Learning Developer': 'Data Scientist',
                 'Applied Machine Learning Scientist': 'Data Scientist',
                 'Data Analytics Manager': 'Data Analyst',
                 'Head of Data Science': 'Data Scientist',
                 'Data Specialist': 'Data Scientist',
                 'Data Architect': 'Data Engineer',
                 'Principal Data Analyst': 'Data Analyst',
                 'Staff Data Scientist': 'Data Scientist',
                 'Big Data Architect': 'Data Engineer',
                 'Analytics Engineer': 'Data Engineer',
                 'ETL Developer': 'Data Engineer',
                 'Head of Machine Learning': 'Data Engineer',
                 'NLP Engineer': 'Data Engineer',
                 'Lead Machine Learning Engineer': 'Data Engineer',
                 'Data Analytics Lead': 'Data Analyst',
                 'Big Data Engineer': 'Data Engineer',
                 'Machine Learning Engineer': 'Data Engineer',
                 'Lead Data Engineer': 'Data Engineer',
                 'Machine Learning Infrastructure Engineer': 'Data Engineer',
                 'ML Engineer': 'Data Engineer',
                 'Computer Vision Engineer': 'Data Engineer',
                 'Data Analytics Engineer': 'Data Engineer',
                 'Cloud Data Engineer': 'Data Engineer',
                 'Computer Vision Software Engineer': 'Data Engineer',
                 'Director of Data Engineering': 'Data Engineer',
                 'Data Science Engineer': 'Data Engineer',
                 'Principal Data Engineer': 'Data Engineer',
                 '3D Computer Vision Researcher': 'Data Scientist',
                 'Data Engineering Manager': 'Data Engineer',
                 'Finance Data Analyst': 'Data Analyst'})
df = df.replace({'DE':'International', 'JP':'International', 'GB':'International', 'HN':'International', 'HU':'International',
                 'NZ':'International', 'FR':'International', 'IN':'International', 'PK':'International', 'CN':'International',
                 'GR':'International', 'AE':'International', 'NL':'International', 'MX':'International', 'CA':'International',
                 'AT':'International', 'NG':'International', 'ES':'International', 'PT':'International', 'DK':'International',
                 'IT':'International','HR':'International', 'LU':'International', 'PL':'International', 'SG':'International',
                 'RO':'International', 'IQ':'International', 'BR':'International', 'BE':'International', 'UA':'International',
                 'IL':'International', 'RU':'International','MT':'International', 'CL':'International', 'IR':'International',
                 'CO':'International', 'MD':'International', 'KE':'International', 'SI':'International', 'CH':'International',
                 'VN':'International', 'AS':'International', 'TR':'International','CZ':'International', 'DZ':'International',
                 'EE':'International', 'MY':'International', 'AU':'International', 'IE':'International','PH':'International',
                 'BG':'International', 'HK':'International', 'RS':'International', 'PR':'International','JE':'International',
                 'AR':'International','TN':'International', 'BO':'International'})
df = df.drop(columns=['Unnamed: 0', 'salary' , 'salary_currency'])

X = df.copy()

# Show the table data
if st.checkbox('Show the dataset as table data'):
    st.dataframe(df)

st.header('Why do we transform categorical data into quantative')
st.markdown('The simple answer, because it makes things easier, the long answer is that for a computer to perform linear regression we can turn the variables into dummy variables. An example of dummy variables , is that suppose we have created two dummy variables between experience level, i.e. junior,mid. We always create one dummy variable less than the number of categories available. Now if the level is mid the it is ‘1’ otherwise it is ‘0’ , same goes for the other variables. Now if you fit a regression model , the coefficient for mid level tells us the average difference between mid and senior experience level. Now similarly the coefficient of mid level gives us the average difference between mid level and senior level. ')

left_column, right_column = st.columns(2)
bool_dummy = left_column.radio('Transform categorical to quantative', ('Yes','No'))
if bool_dummy == 'Yes':
  df_std = df.copy()
  df_std = pd.get_dummies(df_std, columns=['work_year',
 'experience_level',
 'employment_type',
 'job_title',
 'employee_residence',
 'remote_ratio',
 'company_location',
 'company_size'], drop_first=True)


st.header('Why do we standardize the data?')
st.markdown('Standardizing the independent variables is a simple method to reduce multicollinearity that is produced by higher-order terms. This typically makes it to get better results with our linear regressoin model')

st.header('Check out the new data set after it has been preprocessed')
if st.checkbox('Show the new dataset'):
    st.dataframe(df_std)


st.header('Why do we split the data?')
st.markdown('We split the data into what we want to predict and other variables that we want to use to predict it. We do this so we can get a ')


#test size
test_size = st.slider('Select the test split size you want' , 0.0, 1.0, .2,step=.01)

random_size = st.slider('Select the test split seed you want' , 10, 150, 1,step=1)

df_nosal = df_std.drop(columns=['salary_in_usd'])
X = df_nosal
Y = df_std['salary_in_usd']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size =test_size, random_state = random_size)

st.header('What is MinMaxScaler?')
st.markdown('It scales features using statistics that are robust to outliers. This method removes the median and scales the data in the range between 1st quartile and 3rd quartile. i.e., in between 25th quantile and 75th quantile range. This range is also called an Interquartile range. The median and the interquartile range are then stored so that it could be used upon future data using the transform method. If outliers are present in the dataset, then the median and the interquartile range provide better results and outperform the sample mean and variance')
left_column, right_column = st.columns(2)
bool_scal = left_column.radio('Apply MinMaxScaler?', ('No','Yes'))
if bool_scal == 'Yes':
  # use minMax scaler
  min_max_scaler = MinMaxScaler()
  X_train = min_max_scaler.fit_transform(X_train)
  X_test = min_max_scaler.transform(X_test)



regressor = LinearRegression()
regressor.fit(X_train, Y_train)

    
Y_pred_train = regressor.predict(X_train)
Y_pred_val = regressor.predict(X_test)

st.header('What is RMSE?')
st.markdown('Root Mean Square Error (RMSE) is the standard deviation of the residuals (prediction errors). Residuals are a measure of how far from the regression line data points are; RMSE is a measure of how to spread out these residuals are. In other words, it tells you how concentrated the data is around the line of best fit.')
Y_pred = regressor.predict(X_test)
rmse = mean_squared_error(y_true=Y_test,y_pred=Y_pred,squared=False)
st.write(f'RMSE: {rmse:.2f}')

score = regressor.score(X_test, Y_test)
st.write(f'Accuracy: {score:.2f}')

st.header('Not a bad score but also not a great score...')
st.markdown('Now to do some feature engineering, I am going to see what happens when I remove more columns that I think are not useful such as employee location, company location')

left_column, right_column = st.columns(2)
bool_col = left_column.radio('Remove some columns?', ('No','Yes'))
if bool_col == 'Yes':
  df_std = df.copy()
  df_std = df.drop(columns=['employee_residence', 'company_location','work_year'])
  df_std = pd.get_dummies(df_std, columns=['experience_level',
  'employment_type',
  'job_title',
  'remote_ratio',
  'company_size'], drop_first=True)

  df_nosal = df_std.drop(columns=['salary_in_usd'])
  X = df_nosal
  Y = df_std['salary_in_usd']
  X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size =test_size, random_state = random_size)


  regressor = LinearRegression()
  regressor.fit(X_train, Y_train)
  Y_pred_train = regressor.predict(X_train)
  Y_pred_val = regressor.predict(X_test)

  rmse = mean_squared_error(y_true=Y_test,y_pred=Y_pred,squared=False)
  st.write(f'RMSE: {rmse:.2f}')

  score = regressor.score(X_test,Y_test)
  st.write(f'Accuracy: {score:.2f}')

"""
### Plot the result
"""
left_column, right_column = st.columns(2)
show_train = left_column.radio(
                'Show the training dataset:', 
                ('Yes','No')
                )
show_val = right_column.radio(
                'Show the validation dataset:', 
                ('Yes','No')
                )

# default axis range
y_max_train = max([max(Y_train), max(Y_pred_train)])
y_max_val = max([max(Y_test), max(Y_pred_val)])
y_max = int(max([y_max_train, y_max_val])) 
# interactive axis range
left_column, right_column = st.columns(2)
x_min = left_column.number_input('x_min:',value=0,step=1)
x_max = right_column.number_input('x_max:',value=y_max,step=1)
left_column, right_column = st.columns(2)
y_min = left_column.number_input('y_min:',value=0,step=1)
y_max = right_column.number_input('y_max:',value=y_max,step=1)


fig = plt.figure(figsize=(3, 3))
if show_train == 'Yes':
    plt.scatter(Y_train, Y_pred_train,lw=0.1,color="r",label="training data")
if show_val == 'Yes':
    plt.scatter(Y_test, Y_pred_val,lw=0.1,color="b",label="validation data")
plt.xlabel("salay in usd",fontsize=8)
plt.ylabel("salary in usd prediction",fontsize=8)
plt.xlim(int(x_min), int(x_max)+5)
plt.ylim(int(y_min), int(y_max)+5)
plt.legend(fontsize=6)
plt.tick_params(labelsize=6)
st.pyplot(fig)


st.header('Time to try another model')
st.markdown("We're going to be using some SVR and see how that fairs.")

st.header('What is SVR')
st.markdown("Support Vector Regression is a supervised learning algorithm that is used to predict discrete values. Support Vector Regression uses the same principle as the SVMs. The basic idea behind SVR is to find the best fit line. In SVR, the best fit line is the hyperplane that has the maximum number of points.")

left_column, right_column = st.columns(2)
bool_reset = left_column.radio('Reset Data', ('No','Yes'))
if bool_reset == 'Yes':
  X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size =test_size, random_state = random_size)
  X_train

left_column, right_column = st.columns(2)
bool_scal_again = left_column.radio('Apply MinMaxScaler ?', ('No','Yes'))
if bool_scal_again == 'Yes':
  # use minMax scaler
  min_max_scaler = MinMaxScaler()
  X_train = min_max_scaler.fit_transform(X_train)
  X_test = min_max_scaler.transform(X_test)

left_column, right_column = st.columns(2)
bool_svr = left_column.radio('Apply SVR?', ('No','Yes'))
if bool_svr == 'Yes':
  regressor = SVR(kernel = 'rbf')
  regressor.fit(X_train, Y_train)
  score = regressor.score(X_test,Y_test)
  score
  Y_pred_train = regressor.predict(X_train)
  Y_pred_val = regressor.predict(X_test)

# default axis range
y_max_train = max([max(Y_train), max(Y_pred_train)])
y_max_val = max([max(Y_test), max(Y_pred_val)])
y_max = int(max([y_max_train, y_max_val])) 
# interactive axis range
left_column, right_column = st.columns(2)
x_min = left_column.number_input('x_min: ',value=0,step=1)
x_max = right_column.number_input('x_max: ',value=y_max,step=1)
left_column, right_column = st.columns(2)
y_min = left_column.number_input('y_min: ',value=0,step=1)
y_max = right_column.number_input('y_max: ',value=y_max,step=1)


fig = plt.figure(figsize=(3, 3))
if show_train == 'Yes':
    plt.scatter(Y_train, Y_pred_train,lw=0.1,color="r",label="training data")
if show_val == 'Yes':
    plt.scatter(Y_test, Y_pred_val,lw=0.1,color="b",label="validation data")
plt.xlabel("salay in usd",fontsize=8)
plt.ylabel("salary in usd prediction",fontsize=8)
plt.xlim(int(x_min), int(x_max)+5)
plt.ylim(int(y_min), int(y_max)+5)
plt.legend(fontsize=6)
plt.tick_params(labelsize=6)
st.pyplot(fig)
 
st.header('Logistic Regression')
st.markdown('It is a predictive algorithm using independent variables to predict the dependent variable, just like Linear Regression, but with a difference that the dependent variable should be categorical variable.')

left_column, right_column = st.columns(2)
bool_reset_again = left_column.radio('Reset Data again?', ('No','Yes'))
if bool_reset_again == 'Yes':
  X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size =test_size, random_state = random_size)

left_column, right_column = st.columns(2)
bool_scal_again_again = left_column.radio('Apply MinMaxScaler again?', ('No','Yes'))
if bool_scal_again_again == 'Yes':
  # use minMax scaler
  min_max_scaler = MinMaxScaler()
  X_train = min_max_scaler.fit_transform(X_train)
  X_test = min_max_scaler.transform(X_test)


left_column, right_column = st.columns(2)
bool_log = left_column.radio('Apply Logistic Regression', ('No','Yes'))
if bool_log == 'Yes':
  regressor = LogisticRegression()
  regressor.fit(X_train,Y_train)

  score = regressor.score(X_test,Y_test)
  st.write(f'Accuracy: {score:.2f}')
  Y_pred_train = regressor.predict(X_train)
  Y_pred_val = regressor.predict(X_test)

# default axis range
y_max_train = max([max(Y_train), max(Y_pred_train)])
y_max_val = max([max(Y_test), max(Y_pred_val)])
y_max = int(max([y_max_train, y_max_val])) 
# interactive axis range
left_column, right_column = st.columns(2)
x_min = left_column.number_input('x_min:  ',value=0,step=1)
x_max = right_column.number_input('x_max:  ',value=y_max,step=1)
left_column, right_column = st.columns(2)
y_min = left_column.number_input('y_min:  ',value=0,step=1)
y_max = right_column.number_input('y_max:  ',value=y_max,step=1)


fig = plt.figure(figsize=(3, 3))
if show_train == 'Yes':
    plt.scatter(Y_train, Y_pred_train,lw=0.1,color="r",label="training data")
if show_val == 'Yes':
    plt.scatter(Y_test, Y_pred_val,lw=0.1,color="b",label="validation data")
plt.xlabel("salay in usd",fontsize=8)
plt.ylabel("salary in usd prediction",fontsize=8)
plt.xlim(int(x_min), int(x_max)+5)
plt.ylim(int(y_min), int(y_max)+5)
plt.legend(fontsize=6)
plt.tick_params(labelsize=6)
st.pyplot(fig)


st.header('Conclusion and Notes')
st.markdown('None of the models performed too great ')
st.markdown('LinearRegression was the best model so far but it would be good to try others')
st.markdown('It might be good to try to precrocess the data a different way')