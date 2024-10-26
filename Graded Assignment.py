#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pandas as pd
from sklearn.impute import KNNImputer


# In[25]:


BI=pd.read_csv(r"C:\Users\LENOVO\Downloads\Bank Personal Loan Modelling (1).csv")


# In[26]:


BI


# In[27]:


BI.info()


# #Section 1

# In[28]:


BI.isna().sum()


# In[29]:


SCORES=[85]*10
MEAN_SCORE=np.mean(SCORES)
VARIANCE_SCORE=np.var(SCORES)

print('mean score:',MEAN_SCORE)
print('variance score:',VARIANCE_SCORE)

if VARIANCE_SCORE==0:
    print("THE SCORE BY ALL THE LEARNER ARE SAME,AND THE SCORE IS:",MEAN_SCORE)
else:
          print('SCORES VARY ')


# In[30]:


2.#let take house size as it can be seen from the quesiton that mean is greater than median so it is right skew it means there are more big houses on the right
house_size=(1200,1300,1500,1700,1800,2000,2200,2500,3000,3500,4000,4500,5000)
mean_house=np.mean(house_size)
median_house=np.median(house_size)

print('mean of house:',mean_house)
print('median of house:',median_house)

if mean_house>median_house:
    print('it is right skew ,which means there are more house which is bigger in size')
else:
    print('it is left skew,there is no bigger house')
    
#Creating Graph for better understanding
plt.figure(figsize=(10, 5))
sns.histplot(house_size,kde=True)
plt.axvline(mean_house,color='Red',linestyle='--',label=f'Mean{mean_house}')
plt.axvline(median_house,color='Blue',linestyle='-',label=f'Median{median_house}')
plt.legend()
plt.title('HOUSE SIZE DISTRIBUTION')
plt.xlabel('HOUSE SIZE')
plt.ylabel('FREQUENCY')
plt.show()            


# In[31]:


3.#noting down the  question for better clarity
group1_mean = 500000  
group1_variance = 125000  
group2_mean = 40000
group2_variance = 10000 
#first of all calculating the variance
Group1_std=group1_variance**0.5
Group2_std=group2_variance**0.5
#Finding out the coefficient
Group1_cf=(Group1_std/group1_mean)*100
Group2_cf=(Group2_std/group2_mean)*100
print('cf of Group1:',Group1_cf,'%')
print('cf of Group2:',Group2_cf,'%')

if Group1_cf>Group2_cf:
    print('Group 1 has higher relative variability')
else:
    print('Group 2 has higher relative variability')      
          


# In[32]:


#4.first adding the data frame into the notebook
Data={"Age_Interval": ["5-15", "15-25", "25-35", "35-45", "45-55", "55-65"],
    "Patients": [6, 11, 21, 23, 14, 5]}
df=pd.DataFrame(Data)
df


# In[33]:


#A.
df['CF']=df['Patients'].cumsum()


# In[34]:


df


# In[35]:


# A part find out the highest freqency class interval
Highest_Frequency=df.loc[df['Patients'].idxmax(),'Age_Interval']
print('class interval having the highest frequency:',Highest_Frequency)


# In[36]:


#B part find out which age is least affected
Lowest_Affected_Age=df.loc[df['Patients'].idxmin(),'Age_Interval']
print('THE LEAST AFFECTED AGE GROUP:',Lowest_Affected_Age)


# In[37]:


#C part find out How many patients aged 45 years and above were admitted?
Number_of_Patients=df.loc[df['Age_Interval'].isin(['45-55','55-65']),'Patients'].sum()
print('Patients aged 45 or Above:',Number_of_Patients)


# In[38]:


#D part to find out Which is the modal class interval in the above dataset
Modal_Class_interval = Highest_Frequency
print('Modal Class Interval:',Modal_Class_interval )


# In[39]:


#E part to find out Median of Age
median_threshold = 40
median_class_interval=df.loc[df['CF']>=median_threshold,'Age_Interval'].iloc[0]
print("Median class interval:", median_class_interval)


# In[40]:


#5 
years = [2015, 2016, 2017, 2018, 2019, 2020]
returns = [0.36, 0.23, -0.48, -0.30, 0.15, 0.31]  # Convert percentages to decimals
asset_prices = [5000, 6400, 7890, 9023, 4567, 3890]

# Calculate Arithmetic Mean
arithmetic_mean = sum(returns) / len(returns)

# Calculate Geometric Mean
product = 1
for r in returns:
    product *= (1 + r)
geometric_mean = (product ** (1 / len(returns))) - 1

arithmetic_mean_percentage = arithmetic_mean * 100
geometric_mean_percentage = geometric_mean * 100

print("Arithmetic Return:", arithmetic_mean_percentage, "%")
print("Geometric Return:", geometric_mean_percentage, "%")



# In[41]:


#6
true_average_height = 175 
std_dev = 7                 
population_size = 8000000000 
sample_size = 1000
sample_heights = np.random.normal(loc=true_average_height, scale=std_dev, size=sample_size)
sample_average_height = np.mean(sample_heights)
print("Sample Average Height: {:.2f} cm".format(sample_average_height))
print("True Average Height: {:.2f} cm".format(true_average_height))


# In[42]:


#7
X = [4.5,6.2,7.3,9.1,10.4,11]
mean_X = np.mean(X)
std_dev_X = np.std(X)
z_scores = [(x - mean_X) / std_dev_X for x in X]

print("Mean:", mean_X)
print("SD:", std_dev_X)
print("Z-scores:", z_scores)


# In[43]:


#8
summary_stats = df.describe(include='all')
print("Statistical Summary:\n", summary_stats)


# In[44]:


#9
central_dispersion_stats = df.describe().loc[['mean', '50%', 'std', 'min', 'max']]
central_dispersion_stats.rename(index={'50%': 'median'}, inplace=True)
print("Central Tendency and Dispersion:\n", central_dispersion_stats)


# In[48]:


BI=BI.rename(columns={'Age': 'age'})


# In[49]:


BI


# In[53]:


##10.Linear Relationship between Age and Experience
# Calculate correlation
correlation, _ = stats.pearsonr(BI['age'], BI['Experience'])
print("Correlation between Age and Experience:", correlation)

# Plot relationship
plt.figure(figsize=(8, 5))
sns.scatterplot(x="age",y="Experience",data=BI)
plt.title('Relationship between Age and Experience')
plt.xlabel('Age')
plt.ylabel('Experience')
plt.show()


# In[50]:


BI


# In[54]:


#11
family_mode = BI['Family'].mode()[0]
print("Most Frequent Family Size:", family_mode)


# In[64]:


income_cv = (BI['Income'].std() / BI['Income'].mean()) * 100
print("Percentage Variation in Income:", income_cv)


# In[ ]:


#13 NOT ABLE TO DO MODE IS ZERO


# In[58]:


#14
plt.figure(figsize=(8, 5))
sns.kdeplot(BI[BI['CreditCard'] == 1]['CCAvg'], shade=True)
plt.title("Density Curve of CCAvg for Customers with Credit Cards")
plt.xlabel("CCAvg")
plt.show()


# In[60]:


#15
plt.figure(figsize=(10, 6))
sns.boxplot(data=BI[['Income', 'CCAvg', 'Mortgage']])
plt.title("Outliers in Quantitative Variables")
plt.show()


# In[61]:


#16
income_deciles = BI['Income'].quantile([0.1 * i for i in range(1, 10)])
print("Decile Values of Income:\n", income_deciles)


# In[65]:


#17
iqr_values = BI[['Income', 'CCAvg', 'Mortgage']].quantile(0.75) - BI[['Income', 'CCAvg', 'Mortgage']].quantile(0.25)
print("IQR of Quantitative Variables:\n", iqr_values)


# In[63]:


#18
correlation_income_ccavg, _ = stats.pearsonr(BI['Income'],BI['CCAvg'])
print("Correlation between Income and CCAvg:", correlation_income_ccavg)


# In[67]:


#19
online_count = BI['Online'].sum()
online_income = BI[BI['Online'] == 1]['Income'].mean()
print("Number of customers using online banking:", online_count)
print("Average Income of online banking users:", online_income)


# In[68]:


#20
income_mean = BI['Income'].mean()
income_std = BI['Income'].std()
z_scores_income = (BI['Income'] - income_mean) / income_std

# Find observations outside ±3σ
outliers = BI[abs(z_scores_income) > 3]
print("Number of Outliers (Income) beyond ±3σ:", len(outliers))


# In[ ]:




