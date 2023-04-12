import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import KBinsDiscretizer, Binarizer
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import numpy as np
np.set_printoptions(suppress=True)

def to_string(s):
    f = s.split(" ")
    f = f[0]
    return f
pd.set_option("display.max.columns",80)
pd.set_option("display.width",240)
data = pd.read_csv("fighters.csv")
knockouts = data['ko_rate'].str.replace('%','')
H  =  list(data['height'])
height = []
for i in H:
    j = i.split(" ")
    height.append(j[0])

R = list(data['reach'])
reach = []

for i in R:
    j = i.split(" ")
    reach.append(j[0])

ans = pd.DataFrame()
ans['Name'] = data['name']
ans['Wins'] = data['wins']
ans['Losses'] = data['looses']
ans['Draws'] = data['draws']
ans['Ko_rate'] = data['ko_rate'].str.replace('%','') 
data['age'] = pd.to_numeric(data['age'], errors='coerce').fillna(0)
ans['Age'] = data['age']
ans['Height'] = height
ans['Height'] = pd.to_numeric(ans['Height'], errors='coerce').fillna(0)
ans['Country'] = data['country']



#print(data['country'].unique())
#print(ans['Height'].unique())
boxing = ans


data2 = pd.read_csv("tsunami.csv")

disaster = pd.DataFrame()
data2 = data2.drop(0)
disaster['Year'] = data2['Year'] 


disaster['Validity'] = data2['Tsunami Event Validity']
data2['Latitude'] = pd.to_numeric(data2['Latitude'], errors='coerce').fillna(0)
disaster['Latitude'] = data2['Latitude']
data2['Longitude'] = pd.to_numeric(data2['Longitude'], errors='coerce').fillna(0)
disaster['Longitude'] = data2['Longitude']
data2['Total Deaths'] = pd.to_numeric(data2['Total Deaths'], errors='coerce').fillna(0)
disaster['Deaths'] = data2['Total Deaths']
data2['Maximum Water Height (m)'] = pd.to_numeric(data2['Maximum Water Height (m)'], errors='coerce').fillna(0)
disaster['Max Height'] = data2['Maximum Water Height (m)']
#data['Total Damage ($Mil)'] = pd.to_numeric(data['Total Damage ($Mil)'], errors='coerce').fillna(0)
#disaster['Cost'] = data['Total Damage ($Mil)']
data2['Total Houses Destroyed'] = pd.to_numeric(data2['Total Houses Destroyed'], errors='coerce').fillna(0)
disaster['Houses Destroyed'] = data2['Total Houses Destroyed']
disaster['Country'] = data2['Country']
disaster['Location'] = data2['Location Name']
disaster = disaster.reset_index()

##################################################################################################################
#print(boxing)

#FIXING COUNTRIES
countries = boxing.groupby(['Country'])['Country'].count().nlargest(20)
fix_countries = []
for c in boxing['Country']:
	if c not in countries:
		fix_countries.append('Other')
	elif c == 'Venezuela, Bolivarian Republic of':
		fix_countries.append('Venezuela')
	elif c == 'Congo, the Democratic Republic of the':
		 fix_countries.append('Congo')
	else:
		fix_countries.append(c)
boxing['Country'] = fix_countries

#ONE HOTING COUNTRIES IN BOXING DATAFRAME
#boxing_features = boxing['Country'].to_numpy().reshape(len(boxing['Country']),1)
#boxing_ohe = OneHotEncoder(categories=[np.array(boxing['Country'].unique())])
#ohe_boxing = boxing_ohe.fit_transform(boxing_features) 
#ohe_boxing = pd.DataFrame(ohe_boxing.todense())

#ohe_boxing = ohe_boxing.rename(columns={0:'Other', 1:'Argentina', 2:'Venezuela', 3:'Mexico',
# 4:'Germany', 5:'United States',6:'Colombia', 7:'Puerto Rico', 8:'United Kingdom',
# 9:'Serbia', 10:'Russian Federation', 11:'France',12:'Canada',13:'Australia',14:'Japan',
# 15:'South Africa' ,16:'Brazil', 17:'Italy' , 18:'Dominican Republic'
# ,16:'Congo' ,17:'Philippines'})
#boxing = boxing.reset_index()
#boxing = boxing.drop(columns='Country')
#boxing = boxing.join(ohe_boxing)

#print(boxing)
#NORMALIZING BOXING WINS
norm = MinMaxScaler(feature_range=(0,1))
norm_wins = norm.fit_transform(boxing['Wins'].to_numpy().reshape(len(boxing['Wins']),1))
boxing['Wins'] = norm_wins


#NORMALIZING BOXING LOSSES
norm_losses =  norm.fit_transform(boxing['Losses'].to_numpy().reshape(len(boxing['Losses']),1))
boxing['Losses'] = norm_losses
#print(boxing['Losses'])

#NORMALIZING BOXING DRAWS
norm_draws = norm.fit_transform(boxing['Draws'].to_numpy().reshape(len(boxing['Draws']),1))
boxing['Draws'] = norm_draws
#print(boxing['Draws'])


norm_ko =  norm.fit_transform(boxing['Ko_rate'].to_numpy().reshape(len(boxing['Ko_rate']),1))
boxing['Ko_rate'] = norm_ko
#print(boxing['Ko_rate'])
#STANDARDIZE AGE AND HEIGHT
standardizer = StandardScaler(with_mean=True, with_std=True)
boxStand = boxing['Age'].to_numpy().reshape(len(boxing['Age']),1)
standardAge = standardizer.fit_transform(boxStand)

boxStand =  boxing['Height'].to_numpy().reshape(len(boxing['Height']),1)
standardHeight = standardizer.fit_transform(boxStand)
boxing['Height'] = standardHeight
boxing['Age'] = standardAge
boxing = boxing.set_index(boxing['Name'])
boxing = boxing.drop(boxing['Name'])

#BOXING LINEAR MODEL FOR WINS
x = np.c_[norm_losses,norm_draws,standardAge,norm_ko]
y = norm_wins

#print(x)
lr = LinearRegression()
Xtrain, Xtest, ytrain, ytest = train_test_split(x,y,shuffle=True, test_size=.2)

lr.fit(Xtrain,ytrain)
score =  lr.score(Xtest, ytest)
print(f"Boxing: Linear Regression Test Score for Wins: {score}")  


#print(f"Boxing: Linear Regression Confusion Matrix: {confusion_matrix(ytest, lr.predict(Xtest))}") 



#BOXING LOGISTIC REGRESSION FOR COUNTRIES
le = LabelEncoder()
x = np.c_[norm_losses,norm_draws,standardAge,norm_ko]
y = le.fit_transform(fix_countries)

Xtrain, Xtest, ytrain, ytest = train_test_split(x,y,train_size=.8,shuffle=True)
lr = LogisticRegression()
lr.fit(Xtrain, ytrain)
score = lr.score(Xtest, ytest)
print(f"Boxing: Logistic Regression Test Score for predicting Country: {score}")



###############TSUNAMI############################

## NORMALIZING VALIDITY
norm = MinMaxScaler(feature_range=(0,1))
norm_valid = norm.fit_transform(disaster['Validity'].to_numpy().reshape(len(disaster['Validity']),1))
#NORMALIZING HOUSESDESTROYED and DEATHS
norm_houses = norm.fit_transform(disaster['Houses Destroyed'].to_numpy().reshape(len(disaster['Houses Destroyed']),1))
norm_deaths =  norm.fit_transform(disaster['Deaths'].to_numpy().reshape(len(disaster['Deaths']),1))
##STANDARDIZE WAVE HEIGHT
height_stand = disaster['Max Height'].to_numpy().reshape(len(disaster['Max Height']),1)
stand_wave = standardizer.fit_transform(height_stand)
#Years
norm_year = norm.fit_transform(disaster['Year'].to_numpy().reshape(len(disaster['Year']),1))
norm_height = norm.fit_transform(disaster['Max Height'].to_numpy().reshape(len(disaster['Max Height']),1))

stand_deaths =  disaster['Deaths'].to_numpy().reshape(len(disaster['Deaths']),1)
stand_d =  standardizer.fit_transform(stand_deaths)

stand_houses = disaster['Houses Destroyed'].to_numpy().reshape(len(disaster['Houses Destroyed']),1)
stand_house = standardizer.fit_transform(stand_houses)




#LINEAR REGRESSION FOR YEARS
lr = LinearRegression() 
X = np.c_[norm_houses]
y = norm_deaths

Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,shuffle=True, test_size=.2)
lr.fit(Xtrain,ytrain)
#score = lr.score(Xtest,ytest)
print(f"Tsunami: Linear Regression Test score for predicting Deaths: {lr.score(Xtest,ytest)}")  

#LOGISTIC REGRESSION FOR COUNTRIES
lr = LogisticRegression()
cities = le.fit_transform(disaster['Location'])
country = le.fit_transform(disaster['Country'])
x = np.c_[norm_deaths,norm_houses,norm_height]
y = cities 

Xtrain, Xtest, ytrain, ytest = train_test_split(x,y,train_size=.8,shuffle=True)
lr.fit(Xtrain,ytrain)
score = lr.score(Xtest,ytest)
print(f"Tsunami: Logistic Regression Test score for predicting Country: {score}") 


