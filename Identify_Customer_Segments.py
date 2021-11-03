#!/usr/bin/env python
# coding: utf-8

# # Project: Identify Customer Segments
# 
# In this project, you will apply unsupervised learning techniques to identify segments of the population that form the core customer base for a mail-order sales company in Germany. These segments can then be used to direct marketing campaigns towards audiences that will have the highest expected rate of returns. The data that you will use has been provided by our partners at Bertelsmann Arvato Analytics, and represents a real-life data science task.
# 
# This notebook will help you complete this task by providing a framework within which you will perform your analysis steps. In each step of the project, you will see some text describing the subtask that you will perform, followed by one or more code cells for you to complete your work. **Feel free to add additional code and markdown cells as you go along so that you can explore everything in precise chunks.** The code cells provided in the base template will outline only the major tasks, and will usually not be enough to cover all of the minor tasks that comprise it.
# 
# It should be noted that while there will be precise guidelines on how you should handle certain tasks in the project, there will also be places where an exact specification is not provided. **There will be times in the project where you will need to make and justify your own decisions on how to treat the data.** These are places where there may not be only one way to handle the data. In real-life tasks, there may be many valid ways to approach an analysis task. One of the most important things you can do is clearly document your approach so that other scientists can understand the decisions you've made.
# 
# At the end of most sections, there will be a Markdown cell labeled **Discussion**. In these cells, you will report your findings for the completed section, as well as document the decisions that you made in your approach to each subtask. **Your project will be evaluated not just on the code used to complete the tasks outlined, but also your communication about your observations and conclusions at each stage.**

# In[120]:


# import libraries here; add more as necessary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.ticker import PercentFormatter
from sklearn.cluster import KMeans

# magic word for producing visualizations in notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Step 0: Load the Data
# 
# There are four files associated with this project (not including this one):
# 
# - `Udacity_AZDIAS_Subset.csv`: Demographics data for the general population of Germany; 891211 persons (rows) x 85 features (columns).
# - `Udacity_CUSTOMERS_Subset.csv`: Demographics data for customers of a mail-order company; 191652 persons (rows) x 85 features (columns).
# - `Data_Dictionary.md`: Detailed information file about the features in the provided datasets.
# - `AZDIAS_Feature_Summary.csv`: Summary of feature attributes for demographics data; 85 features (rows) x 4 columns
# 
# Each row of the demographics files represents a single person, but also includes information outside of individuals, including information about their household, building, and neighborhood. You will use this information to cluster the general population into groups with similar demographic properties. Then, you will see how the people in the customers dataset fit into those created clusters. The hope here is that certain clusters are over-represented in the customers data, as compared to the general population; those over-represented clusters will be assumed to be part of the core userbase. This information can then be used for further applications, such as targeting for a marketing campaign.
# 
# To start off with, load in the demographics data for the general population into a pandas DataFrame, and do the same for the feature attributes summary. Note for all of the `.csv` data files in this project: they're semicolon (`;`) delimited, so you'll need an additional argument in your [`read_csv()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html) call to read in the data properly. Also, considering the size of the main dataset, it may take some time for it to load completely.
# 
# Once the dataset is loaded, it's recommended that you take a little bit of time just browsing the general structure of the dataset and feature summary file. You'll be getting deep into the innards of the cleaning in the first major step of the project, so gaining some general familiarity can help you get your bearings.

# In[2]:


# Load in the general demographics data.
azdias = pd.read_csv('Udacity_AZDIAS_Subset.csv', sep=';')

# Load in the feature summary file.
feat_info = pd.read_csv('AZDIAS_Feature_Summary.csv', sep=';')


# In[3]:


# Check the structure of the data after it's loaded (e.g. print the number of
# rows and columns, print the first few rows).
azdias.shape
azdias.head(10)


# In[4]:


#Print the top5 rows of the demographics
azdias.head(5)


# In[5]:


azdias.info()


# In[6]:


azdias.describe()


# In[7]:


print (azdias.isnull().sum())


# In[8]:


feat_info


# In[9]:


feat_info.shape


# In[10]:


feat_info.head


# > **Tip**: Add additional cells to keep everything in reasonably-sized chunks! Keyboard shortcut `esc --> a` (press escape to enter command mode, then press the 'A' key) adds a new cell before the active cell, and `esc --> b` adds a new cell after the active cell. If you need to convert an active cell to a markdown cell, use `esc --> m` and to convert to a code cell, use `esc --> y`. 
# 
# ## Step 1: Preprocessing
# 
# ### Step 1.1: Assess Missing Data
# 
# The feature summary file contains a summary of properties for each demographics data column. You will use this file to help you make cleaning decisions during this stage of the project. First of all, you should assess the demographics data in terms of missing data. Pay attention to the following points as you perform your analysis, and take notes on what you observe. Make sure that you fill in the **Discussion** cell with your findings and decisions at the end of each step that has one!
# 
# #### Step 1.1.1: Convert Missing Value Codes to NaNs
# The fourth column of the feature attributes summary (loaded in above as `feat_info`) documents the codes from the data dictionary that indicate missing or unknown data. While the file encodes this as a list (e.g. `[-1,0]`), this will get read in as a string object. You'll need to do a little bit of parsing to make use of it to identify and clean the data. Convert data that matches a 'missing' or 'unknown' value code into a numpy NaN value. You might want to see how much data takes on a 'missing' or 'unknown' code, and how much data is naturally missing, as a point of interest.
# 
# **As one more reminder, you are encouraged to add additional cells to break up your analysis into manageable chunks.**

# In[11]:


# Identify missing or unknown data values and convert them to NaNs.
def replace_missing_values_with_nan(df, mapper, mapper_column_name, mapper_column_definition, non_numerical_indicators=[]):
    # For each mapper_df row definition
    for index, row in mapper.iterrows():
        # Omit rows with no NaNs or with lists without values
        if len(row[mapper_column_definition]) > 0 and row[mapper_column_definition] != ['']:
            try:
                # For each column definition
                # build a list of nan_indicators
                replace_mask = []
                for nan_indicator in row[mapper_column_definition]:
                    
                    if nan_indicator in non_numerical_indicators:
                        replace_mask.append(nan_indicator)
                    else:
                        replace_mask.append(int(nan_indicator))
                
                #print(replace_mask)
                # Replace masked values with NaN
                df[row[mapper_column_name]] = df[row[mapper_column_name]].replace(replace_mask, np.nan)
                
            except Exception as e:
                print('Exception: {} -> {} --> {}'.format(row[mapper_column_name], str(e), nan_indicator))
                continue
                
    return df


# In[12]:


# Identify correlations between numeric features
def correlated_columns_to_drop(df, min_corr_level=0.95):
    # Create correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than min_corr_level
    to_drop = [column for column in upper.columns if any(upper[column] > min_corr_level)]

    return to_drop


# In[13]:


#To check the columns with no or missing data
null_data = azdias.isnull().sum()[azdias.isnull().sum() != 0]

data_dict = {'count': null_data.values, 'pct':np.round(null_data.values *100/891221,2)}

azdias_init_null = pd.DataFrame(data=data_dict, index=null_data.index)
azdias_init_null.sort_values(by='count', ascending=False, inplace=True)
azdias_init_null


# In[14]:


azdias_init_null.shape


# In[15]:


azdias.tail()


# #### Step 1.1.2: Assess Missing Data in Each Column
# 
# How much missing data is present in each column? There are a few columns that are outliers in terms of the proportion of values that are missing. You will want to use matplotlib's [`hist()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html) function to visualize the distribution of missing value counts to find these columns. Identify and document these columns. While some of these columns might have justifications for keeping or re-encoding the data, for this project you should just remove them from the dataframe. (Feel free to make remarks about these outlier columns in the discussion, however!)
# 
# For the remaining features, are there any patterns in which columns have, or share, missing data?

# In[16]:


# Perform an assessment of how much missing data there is in each column of the
# dataset.

azdias.isnull().sum().sort_values(ascending=False)

def extract_columns_with_nan(df):

    # Extract columns with NaNs
    columns_with_nans = df.isnull().sum()[df.isnull().sum() > 0]

    # Calculate percentage of NaN for each column
    dic_nan = {'sum': columns_with_nans.values, 'percent': np.round(columns_with_nans.values * 100 / df.shape[0], 2)}

    # Build a dataframe including only columns with NaNs
    df_nan = pd.DataFrame(data=dic_nan, index=columns_with_nans.index)

    # Sort by percentage
    df_nan.sort_values(by='percent', ascending=False, inplace=True)

    return df_nan


# By using matplotlib's histogram function, Investigating the amount of missing data in each column.

# In[17]:


# Investigate patterns in the amount of missing data in each column.
azdias.isnull().sum().hist()


# In[18]:


drop_columns = list(azdias.isnull().sum()[azdias.isnull().sum()>300000].index)


# In[19]:


# Remove the outlier columns from the dataset. (You'll perform other data
# engineering tasks such as re-encoding and imputation later.)
azdias.drop(drop_columns, axis='columns', inplace=True)


# Plotting the table with plt.hist() to check the data

# In[20]:


plt.hist(azdias[['CAMEO_INTL_2015']]);


# In[21]:


plt.hist(azdias[['AGER_TYP']]);


# In[22]:


azdias.isnull().describe()


# #### Discussion 1.1.2: Assess Missing Data in Each Column
# There are 6 columns have missing data around 30%.
# Almost all columns have missing data around 0.5% to 87%. The missing outliers have been removed.

# #### Step 1.1.3: Assess Missing Data in Each Row
# 
# Now, you'll perform a similar assessment for the rows of the dataset. How much data is missing in each row? As with the columns, you should see some groups of points that have a very different numbers of missing values. Divide the data into two subsets: one for data points that are above some threshold for missing values, and a second subset for points below that threshold.
# 
# In order to know what to do with the outlier rows, we should see if the distribution of data values on columns that are not missing data (or are missing very little data) are similar or different between the two groups. Select at least five of these columns and compare the distribution of values.
# - You can use seaborn's [`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html) function to create a bar chart of code frequencies and matplotlib's [`subplot()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html) function to put bar charts for the two subplots side by side.
# - To reduce repeated code, you might want to write a function that can perform this comparison, taking as one of its arguments a column to be compared.
# 
# Depending on what you observe in your comparison, this will have implications on how you approach your conclusions later in the analysis. If the distributions of non-missing features look similar between the data with many missing values and the data with few or no missing values, then we could argue that simply dropping those points from the analysis won't present a major issue. On the other hand, if the data with many missing values looks very different from the data with few or no missing values, then we should make a note on those data as special. We'll revisit these data later on. **Either way, you should continue your analysis for now using just the subset of the data with few or no missing values.**

# In[23]:


# How much data is missing in each row of the dataset?
azdias.isnull().sum(axis=1).sort_values(ascending=False).hist()
plt.xlabel('Percentage of missing value (%)')
plt.ylabel('Counts')
plt.title('Missing value counts in rows using Histogram')
plt.show()


# In[24]:


missing_data_by_row = azdias.isnull().sum(axis=1)
missing_data_by_row.describe()


# In[25]:


missing_data_by_row[missing_data_by_row == 45]


# In[26]:


# Ratio of rows without missing data?
count_rows = missing_data_by_row.count()
count_rows_no_missing_data = missing_data_by_row[missing_data_by_row == 0].count()
count_rows_with_missing_data = missing_data_by_row[missing_data_by_row != 0].count()
percent_rows_with_missing_data = count_rows_with_missing_data * 100 / count_rows
percent_rows_no_missing_data = 100 - percent_rows_with_missing_data

print('{}\t -> Number of all rows (100%).'.format(count_rows))
print('{}\t -> Number of rows with missing data ({:0.3f}%).'.format(count_rows_with_missing_data, percent_rows_with_missing_data))
print('{}\t -> Number of rows with no missing data ({:0.3f}%).'.format(count_rows_no_missing_data, percent_rows_no_missing_data))


# In[27]:


plt.figure(figsize=(18, 8))
plt.xticks(np.arange(0, 50, 1))
plt.xlabel('Number of NaNs per row')
plt.ylabel('Percent of rows')
# Observations in the Y axis
plt.hist(missing_data_by_row, weights=np.ones(len(missing_data_by_row)) / len(missing_data_by_row), bins=np.arange(0, 50, 1))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.show();


# The data is highly right-skewed with an increase towards the end of the tail where we can see rows with more than 30 missing values. Looking at the figure above and the distribution of records with no and some missing data, 1 looks like a good candidate for the split.

# In[28]:


# Select randomly 5 columns for the comparison
np.random.seed(23456)
sample_columns = np.random.choice(azdias.columns, 5).tolist()
print(sample_columns)


# In[29]:


# Add information about number of NaN for each row to the master dataset
azdias = azdias.assign(NAN_NUM=pd.Series(missing_data_by_row.values))
azdias.head(10)


# In[30]:


df_num = azdias.select_dtypes(include = ['float64', 'int64'])
df_num.shape


# In[31]:


# Write code to divide the data into two subsets based on the number of missing
# values in each row
# rows with less than 5 missing values and rows with more than and equal 5 missing values
few_missing = azdias[azdias.isnull().sum(axis=1) < 5].reset_index(drop=True)

more_missing = azdias[azdias.isnull().sum(axis=1) >= 5].reset_index(drop=True)


# In[32]:


# Compare the distribution of values for at least five columns where there are
# no or few missing values, between the two subsets.
col_names_few = few_missing.columns

def print_countplot(cols,num):
    
    fig, axs = plt.subplots(num,2, figsize=(15, 15), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace =2 , wspace=.2)
    axs = axs.ravel()

    for i in range(num):
    
        sns.countplot(few_missing[cols[i]], ax=axs[i*2])
        axs[i*2].set_title('few_missing')
        
        sns.countplot(more_missing[cols[i]], ax=axs[i*2+1])
        axs[i*2+1].set_title('more_missing')


# In[33]:


print_countplot(col_names_few,9)


# #### Discussion 1.1.3: Assess Missing Data in Each Row
# 
# (Double-click this cell and replace this text with your own text, reporting your observations regarding missing data in rows. Are the data with lots of missing values are qualitatively different from data with few or no missing values?)
# 
# Looking at "Missing value counts in rows using histogram" in rows we can see that most rows have less than 4-5% missing values which is about 3-4 columns from total number of columns after dropping the outlier columns.
# 
# The data with lots of missing values has different distribution with data with few or no missing values.

# ### Step 1.2: Select and Re-Encode Features
# 
# Checking for missing data isn't the only way in which you can prepare a dataset for analysis. Since the unsupervised learning techniques to be used will only work on data that is encoded numerically, you need to make a few encoding changes or additional assumptions to be able to make progress. In addition, while almost all of the values in the dataset are encoded using numbers, not all of them represent numeric values. Check the third column of the feature summary (`feat_info`) for a summary of types of measurement.
# - For numeric and interval data, these features can be kept without changes.
# - Most of the variables in the dataset are ordinal in nature. While ordinal values may technically be non-linear in spacing, make the simplifying assumption that the ordinal variables can be treated as being interval in nature (that is, kept without any changes).
# - Special handling may be necessary for the remaining two variable types: categorical, and 'mixed'.
# 
# In the first two parts of this sub-step, you will perform an investigation of the categorical and mixed-type features and make a decision on each of them, whether you will keep, drop, or re-encode each. Then, in the last part, you will create a new data frame with only the selected and engineered columns.
# 
# Data wrangling is often the trickiest part of the data analysis process, and there's a lot of it to be done here. But stick with it: once you're done with this step, you'll be ready to get to the machine learning parts of the project!

# In[34]:


# How many features are there of each data type?
feat_info.groupby(by='type').count()


# In[35]:


feat_info.type.value_counts()


# #### Step 1.2.1: Re-Encode Categorical Features
# 
# For categorical data, you would ordinarily need to encode the levels as dummy variables. Depending on the number of categories, perform one of the following:
# - For binary (two-level) categoricals that take numeric values, you can keep them without needing to do anything.
# - There is one binary variable that takes on non-numeric values. For this one, you need to re-encode the values as numbers or create a dummy variable.
# - For multi-level categoricals (three or more values), you can choose to encode the values using multiple dummy variables (e.g. via [OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)), or (to keep things straightforward) just drop them from the analysis. As always, document your choices in the Discussion section.

# In[36]:


# Assess categorical variables: which are binary, which are multi-level, and
# which one needs to be re-encoded?
for attribute in feat_info [feat_info['type']=='categorical'].attribute:
    if attribute in azdias.columns:
        if azdias[attribute].nunique() == 2:
            print(attribute, azdias[attribute].unique())


# In[37]:


azdias['OST_WEST_KZ'].unique()


# In[38]:


drop_columns


# In[39]:


few_missing.columns


# In[40]:


few_missing.shape


# In[41]:


# Re-encode categorical variable(s) to be kept in the analysis.
few_missing = pd.get_dummies(few_missing, columns=['OST_WEST_KZ'])


# In[42]:


few_missing.columns


# #### Discussion 1.2.1: Re-Encode Categorical Features
# 
# (Double-click this cell and replace this text with your own text, reporting your findings and decisions regarding categorical features. Which ones did you keep, which did you drop, and what engineering steps did you perform?)
# 
# In this section I remove non-binary categorical features. I kept all binary-level variables I dropped all multi-level variables.
# I encoded OST_WEST_KZ.

# #### Step 1.2.2: Engineer Mixed-Type Features
# 
# There are a handful of features that are marked as "mixed" in the feature summary that require special treatment in order to be included in the analysis. There are two in particular that deserve attention; the handling of the rest are up to your own choices:
# - "PRAEGENDE_JUGENDJAHRE" combines information on three dimensions: generation by decade, movement (mainstream vs. avantgarde), and nation (east vs. west). While there aren't enough levels to disentangle east from west, you should create two new variables to capture the other two dimensions: an interval-type variable for decade, and a binary variable for movement.
# - "CAMEO_INTL_2015" combines information on two axes: wealth and life stage. Break up the two-digit codes by their 'tens'-place and 'ones'-place digits into two new ordinal variables (which, for the purposes of this project, is equivalent to just treating them as their raw numeric values).
# - If you decide to keep or engineer new features around the other mixed-type features, make sure you note your steps in the Discussion section.
# 
# Be sure to check `Data_Dictionary.md` for the details needed to finish these tasks.

# In[43]:


# Select mixed features names
mixed_features = feat_info[feat_info.type == 'mixed'].attribute
print(mixed_features)


# In[44]:


azdias[mixed_features].sample(15).T


# In[45]:


# Investigate "PRAEGENDE_JUGENDJAHRE" and engineer two new variables.
generations = {0: [1, 2], # 40s
               1: [3, 4], # 50s
               2: [5, 6, 7], # 60s
               3: [8, 9], # 70s
               4: [10, 11, 12, 13], # 80s
               5:[14, 15]} # 90s

def classify_generation(value):
    try:
        for key, values in generations.items():
            if value in values:
                return key
    # In case value is NaN
    except ValueError:
        return np.nan
    
# Movement 
mainstream = [1, 3, 5, 8, 10, 12, 14]

def classify_movement(value):
    try:
        if value in mainstream:
            return 1
        else:
            return 0
    # In case value is NaN
    except ValueError:
        return np.nan


# In[46]:


# Engineer generation column
azdias['PRAEGENDE_JUGENDJAHRE_GEN'] = azdias['PRAEGENDE_JUGENDJAHRE'].apply(classify_generation)
#azdias.loc[:,'PRAEGENDE_JUGENDJAHRE_GEN'] = azdias['PRAEGENDE_JUGENDJAHRE'].apply(classify_generation)

# Engineer movement column
azdias['PRAEGENDE_JUGENDJAHRE_MOV'] = azdias['PRAEGENDE_JUGENDJAHRE'].apply(classify_movement)
#azdias_new.loc[:,'PRAEGENDE_JUGENDJAHRE_MOV'] = azdias_new['PRAEGENDE_JUGENDJAHRE'].apply(classify_movement)


# In[47]:


azdias.sample(15).T


# In[48]:


# Investigate "CAMEO_INTL_2015" and engineer two new variables.
# first digit -> wealth
# second digit -> life stage

# Wealth 
def classify_wealth(value):
    # In case value is NaN
    if pd.isnull(value):
        return np.nan
    else:
        # Return first digit
        return int(str(value)[0])

# Life stage
def classify_lifestage(value):
    # In case value is NaN
    if pd.isnull(value):
        return np.nan
    else:
        # Return second digit
        return int(str(value)[1])


# In[49]:


# Remove unneeded and the rest of mixed columns (as per instruction above)
azdias = azdias.drop(['PRAEGENDE_JUGENDJAHRE',
                      'CAMEO_INTL_2015',
                      'PLZ8_BAUMAX',
                      'LP_LEBENSPHASE_FEIN',
                      'WOHNLAGE',
                      'REGIOTYP',
                      'KKK'], axis=1)


# #### Discussion 1.2.2: Engineer Mixed-Type Features
# 
# (Double-click this cell and replace this text with your own text, reporting your findings and decisions regarding mixed-value features. Which ones did you keep, which did you drop, and what engineering steps did you perform?)
# 
# 
# I have decoded PRAEGENDE_JUGENDJAHRE and CAMEO_INTL_2015 ,The new features are ordinal, and therefore don't need to be one-hot-encoded. There are mixed features too, but I have removed them as per the guidelines given above.

# #### Step 1.2.3: Complete Feature Selection
# 
# In order to finish this step up, you need to make sure that your data frame now only has the columns that you want to keep. To summarize, the dataframe should consist of the following:
# - All numeric, interval, and ordinal type columns from the original dataset.
# - Binary categorical features (all numerically-encoded).
# - Engineered features from other multi-level categorical features and mixed features.
# 
# Make sure that for any new columns that you have engineered, that you've excluded the original columns from the final dataset. Otherwise, their values will interfere with the analysis later on the project. For example, you should not keep "PRAEGENDE_JUGENDJAHRE", since its values won't be useful for the algorithm: only the values derived from it in the engineered features you created should be retained. As a reminder, your data should only be from **the subset with few or no missing values**.

# In[50]:


azdias.head(20).T


# ### Step 1.3: Create a Cleaning Function
# 
# Even though you've finished cleaning up the general population demographics data, it's important to look ahead to the future and realize that you'll need to perform the same cleaning steps on the customer demographics data. In this substep, complete the function below to execute the main feature selection, encoding, and re-engineering steps you performed above. Then, when it comes to looking at the customer data in Step 3, you can just run this function on that DataFrame to get the trimmed dataset in a single step.

# In[51]:


drop_columns


# In[52]:


def clean_data(azdias, feat_info, split_value):
    """
    Perform feature trimming, re-encoding, and engineering for demographics
    data
    
    INPUT: Demographics DataFrame, feature info DataFrame, and split_value
    OUTPUT: Trimmed and cleaned demographics DataFrame and feature info DataFrame
    """
    
    # Remove brackets and split strings into lists
    # in feat_info.missing_or_unknown
    if type(feat_info.missing_or_unknown[0]) == str:
        feat_info.missing_or_unknown = feat_info.missing_or_unknown.str.strip('[').str.strip(']').str.split(',')
    
    # Replace with NaNs all values in all columns of azdias as per mapping in feat_info
    azdias = replace_missing_values_with_nan(azdias, feat_info, 'attribute', 'missing_or_unknown', non_numerical_indicators=['X', 'XX'])

    azdias_nan = extract_columns_with_nan(azdias)

    # Drop the outlier columns
    columns_to_drop = azdias_nan[azdias_nan.percent > 30.].index.tolist()
    azdias.drop(columns_to_drop, axis=1, inplace=True)

    # Drop the corresponding columns from feat_info
    feat_info = feat_info[~feat_info.attribute.isin(columns_to_drop)]

    # In order to merge to dataframes by index we need to have indexes
    # as attributes in both dataframes
    feat_info.set_index('attribute', inplace=True, drop=False)

    # Add NaN information columns to feat_info
    # by index / rows, for better columns understanding
    #feat_info = feat_info.join(azdias_nan)
    
    # Examin numerical features
    df_num = azdias.select_dtypes(include = ['float64', 'int64'])
    
    # Remove highly correlated numerical features
    columns_to_drop = correlated_columns_to_drop(df_num, 0.95)
    azdias.drop(columns_to_drop, axis=1, inplace=True)
    feat_info = feat_info[~feat_info.attribute.isin(columns_to_drop)]
    
    missing_data_by_row = azdias.isnull().sum(axis=1)
    
    # Add information about number of NaN for each row to the master dataset
    azdias = azdias.assign(NAN_NUM=pd.Series(missing_data_by_row.values))
    
    high_nan_rows = azdias[azdias.NAN_NUM >= split_value].copy()
    low_nan_rows = azdias[azdias.NAN_NUM < split_value].copy()

    # Remove all rows with missing values above split_value
    azdias = azdias[azdias.NAN_NUM < split_value]
    
    # Select categorical features names
    categorical_features = feat_info[feat_info.type == 'categorical'].attribute

    # Split categorical variables into binary or multi buckets
    categorical_binary = []
    categorical_multi = []
    for feature in categorical_features:
        if azdias[feature].nunique() > 2:
            categorical_multi.append(feature)
        else:
            categorical_binary.append(feature)

    # Standardize binary columns into 0 or 1
    azdias['VERS_TYP'].replace([2.0, 1.0], [1, 0], inplace=True)
    azdias['OST_WEST_KZ'].replace(['W', 'O'], [1, 0], inplace=True)
    azdias['ANREDE_KZ'].replace([2, 1], [1, 0], inplace=True)

    # Do one-hot-encoding and remove reference columns
    azdias = pd.get_dummies(azdias, columns=categorical_multi)

    # Select mixed features names
    mixed_features = feat_info[feat_info.type == 'mixed'].attribute

    # Engineer features
    azdias['PRAEGENDE_JUGENDJAHRE_GEN'] = azdias['PRAEGENDE_JUGENDJAHRE'].apply(classify_generation)
    azdias['PRAEGENDE_JUGENDJAHRE_MOV'] = azdias['PRAEGENDE_JUGENDJAHRE'].apply(classify_movement)
    azdias['CAMEO_INTL_2015_WEALTH'] = azdias['CAMEO_INTL_2015'].apply(classify_wealth)
    azdias['CAMEO_INTL_2015_LIFESTAGE'] = azdias['CAMEO_INTL_2015'].apply(classify_lifestage)
    
    # Remove unneeded and the rest of mixed columns (as per instruction above)
    azdias = azdias.drop(['PRAEGENDE_JUGENDJAHRE',
                          'CAMEO_INTL_2015',
                          'PLZ8_BAUMAX',
                          'LP_LEBENSPHASE_FEIN',
                          'WOHNLAGE'], axis=1)
    
    if 'REGIOTYP' in azdias.columns:
        azdias = azdias.drop(['REGIOTYP'], axis=1)
        
    if 'KKK' in azdias.columns:
        azdias = azdias.drop(['KKK'], axis=1)
        
    if 'KBA05_BAUMAX' in azdias.columns:
        azdias = azdias.drop(['KBA05_BAUMAX'], axis=1)
        
    if 'TITEL_KZ' in azdias.columns:
        azdias = azdias.drop(['TITEL_KZ'], axis=1)
        
    if 'AGER_TYP' in azdias.columns:
        azdias = azdias.drop(['AGER_TYP'], axis=1)
        
    if 'GEBURTSJAHR' in azdias.columns:
        azdias = azdias.drop(['GEBURTSJAHR'], axis=1)
        
    if 'ALTER_HH' in azdias.columns:
        azdias = azdias.drop(['ALTER_HH'], axis=1)
        
    if 'GEBAEUDETYP_5.0' in azdias.columns:
        azdias = azdias.drop(['GEBAEUDETYP_5.0'], axis=1)

    # Remove highly correlated numerical features
    columns_to_drop = correlated_columns_to_drop(azdias, 0.95)
    azdias.drop(columns_to_drop, axis=1, inplace=True)
    feat_info = feat_info[~feat_info.attribute.isin(columns_to_drop)]

    # Remove NAN_NUM column
    azdias = azdias.drop(['NAN_NUM'], axis=1)
    
    return azdias, feat_info, high_nan_rows, low_nan_rows


# In[53]:


#Before cleaning
azdias.info()


# In[54]:


#Cleaning

# Load in the general demographics data.
azdias_clean = pd.read_csv('Udacity_AZDIAS_Subset.csv', sep=';')

# Load in the feature summary file.
feat_info_clean = pd.read_csv('AZDIAS_Feature_Summary.csv', sep=';')
split_value=1
azdias_clean, feat_info_clean, high_nan_rows_clean, low_nan_rows_clean = clean_data(azdias_clean, feat_info_clean, split_value)


# In[55]:


# After cleaning
azdias_clean.info()


# ## Step 2: Feature Transformation
# 
# ### Step 2.1: Apply Feature Scaling
# 
# Before we apply dimensionality reduction techniques to the data, we need to perform feature scaling so that the principal component vectors are not influenced by the natural differences in scale for features. Starting from this part of the project, you'll want to keep an eye on the [API reference page for sklearn](http://scikit-learn.org/stable/modules/classes.html) to help you navigate to all of the classes and functions that you'll need. In this substep, you'll need to check the following:
# 
# - sklearn requires that data not have missing values in order for its estimators to work properly. So, before applying the scaler to your data, make sure that you've cleaned the DataFrame of the remaining missing values. This can be as simple as just removing all data points with missing data, or applying an [Imputer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html) to replace all missing values. You might also try a more complicated procedure where you temporarily remove missing values in order to compute the scaling parameters before re-introducing those missing values and applying imputation. Think about how much missing data you have and what possible effects each approach might have on your analysis, and justify your decision in the discussion section below.
# - For the actual scaling function, a [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) instance is suggested, scaling each feature to mean 0 and standard deviation 1.
# - For these classes, you can make use of the `.fit_transform()` method to both fit a procedure to the data as well as apply the transformation to the data at the same time. Don't forget to keep the fit sklearn objects handy, since you'll be applying them to the customer demographics data towards the end of the project.

# In[56]:


azdias = pd.read_csv('Udacity_AZDIAS_Subset.csv', sep=';')


# In[57]:


# Number of rows with missing data comparing to number of all rows
count_all_rows = azdias_clean.shape[0]
azdias_clean_nans = azdias_clean[azdias_clean.isnull().any(axis=1)]
count_rows_with_nans = azdias_clean_nans.shape[0]
percent_nans = count_rows_with_nans * 100 / count_all_rows
print('All rows {}, complete rows {}, incomplete rows {} ({:0.2f}%)'.format(count_all_rows,
                                                                      count_all_rows - count_rows_with_nans,
                                                                      count_rows_with_nans,
                                                                      percent_nans))


# In[63]:


# Apply feature scaling to the general population demographics data.
normalizer = StandardScaler()
azdias_clean_normalized = normalizer.fit_transform(azdias_clean)


# ### Discussion 2.1: Apply Feature Scaling
# 
# (Double-click this cell and replace this text with your own text, reporting your decisions regarding feature scaling.)
# 
#  I decided to use imputer to fill in all missing data and use standardscaler to feature scaling
# 
# 

# ### Step 2.2: Perform Dimensionality Reduction
# 
# On your scaled data, you are now ready to apply dimensionality reduction techniques.
# 
# - Use sklearn's [PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) class to apply principal component analysis on the data, thus finding the vectors of maximal variance in the data. To start, you should not set any parameters (so all components are computed) or set a number of components that is at least half the number of features (so there's enough features to see the general trend in variability).
# - Check out the ratio of variance explained by each principal component as well as the cumulative variance explained. Try plotting the cumulative or sequential values using matplotlib's [`plot()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html) function. Based on what you find, select a value for the number of transformed features you'll retain for the clustering part of the project.
# - Once you've made a choice for the number of components to keep, make sure you re-fit a PCA instance to perform the decided-on transformation.

# In[64]:


# Apply PCA to the data.
pca = PCA()
azdias_pca = pca.fit_transform(azdias_clean_normalized)


# In[65]:


# Investigate the variance accounted for by each principal component.
def scree_plot(pca, components_no=None, show_labels=False):

    vals = pca.explained_variance_ratio_
    if components_no:
        vals = vals[:components_no]
    num_components = len(vals)
    ind = np.arange(num_components)
 
    plt.figure(figsize=(18, 8))
    ax = plt.subplot(111)
    cumvals = np.cumsum(vals)
    ax.bar(ind, vals)
    ax.plot(ind, cumvals)
    
    if show_labels:
        for i in range(num_components):
            ax.annotate(r"%s%%" % ((str(vals[i]*100)[:4])), (ind[i]+0.2, vals[i]), va="bottom", ha="center", fontsize=12)
 
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)
 
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    plt.title('Explained Variance Per Principal Component')


# In[66]:


scree_plot(pca, show_labels=False)


# In[67]:


scree_plot(pca, components_no=80, show_labels=False)


# In[68]:


pca.explained_variance_ratio_[:5].sum()


# The first 5 components capture more than 23% of the variance

# In[71]:


for p in np.arange(10, 81, 10):
    print('{} components explain {} of variance.'.format(p, pca.explained_variance_ratio_[:p].sum()))


# In[74]:


# Number of original features
azdias_clean_normalized.shape[1]


# In[76]:


# Re-apply PCA to the data while selecting for number of components to retain.
pca = PCA(n_components=80)
azdias_pca = pca.fit_transform(azdias_clean_normalized)


# In[77]:


scree_plot(pca, show_labels=False)


# ### Discussion 2.2: Perform Dimensionality Reduction
# 
# (Double-click this cell and replace this text with your own text, reporting your findings and decisions regarding dimensionality reduction. How many principal components / transformed features are you retaining for the next step of the analysis?)
# 
# According to Principal Component Analysis (PCA), we should be fine reducing the dimensions of our dataset by half, from 164 features to 80 PCA components. This reduction gives 83 % percentage of data variability.

# ### Step 2.3: Interpret Principal Components
# 
# Now that we have our transformed principal components, it's a nice idea to check out the weight of each variable on the first few components to see if they can be interpreted in some fashion.
# 
# As a reminder, each principal component is a unit vector that points in the direction of highest variance (after accounting for the variance captured by earlier principal components). The further a weight is from zero, the more the principal component is in the direction of the corresponding feature. If two features have large weights of the same sign (both positive or both negative), then increases in one tend expect to be associated with increases in the other. To contrast, features with different signs can be expected to show a negative correlation: increases in one variable should result in a decrease in the other.
# 
# - To investigate the features, you should map each weight to their corresponding feature name, then sort the features according to weight. The most interesting features for each principal component, then, will be those at the beginning and end of the sorted list. Use the data dictionary document to help you understand these most prominent features, their relationships, and what a positive or negative value on the principal component might indicate.
# - You should investigate and interpret feature associations from the first three principal components in this substep. To help facilitate this, you should write a function that you can call at any time to print the sorted list of feature weights, for the *i*-th principal component. This might come in handy in the next step of the project, when you interpret the tendencies of the discovered clusters.

# In[78]:


# Map weights for the first principal component to corresponding feature names
# and then print the linked values, sorted by weight.
# HINT: Try defining a function here or in a new cell that you can reuse in the
# other cells.
def pca_results(df, pca, component_no, features_no, show_plot=True):
    # Construct a dataframe with features_no features capturing most variability
    pca_comp = pd.DataFrame(np.round(pca.components_, 4), columns=df.keys()).iloc[component_no - 1]
    pca_comp.sort_values(ascending=False, inplace=True)
    pca_comp = pd.concat([pca_comp.head(features_no), pca_comp.tail(features_no)])
    if show_plot:
        # Print the result
        pca_comp.plot(kind='bar', 
                  title='Most {} weighted features for PCA component {}'.format(features_no*2, component_no),
                  figsize=(12, 6))
        plt.show()
    
    return pca_comp


# In[79]:


# Check first 3 components
#weights for second principal component along with their feature names
#weights for third principal component along with their feature names
for i in np.arange(0, 3, 1):
    res = pca_results(azdias_clean, pca, i, 5)
    print(res)


# ### Discussion 2.3: Interpret Principal Components
# 
# (Double-click this cell and replace this text with your own text, reporting your observations from detailed investigation of the first few principal components generated. Can we interpret positive and negative values from them in a meaningful way?)
# 
# Principal Component 1:
# 
# The first principal component is 
# 
# POSITIVELY ALIGNED WITH (TOP-DOWN):
# 
# *LP_STATUS_FEIN_4.0 (Social status: Villagers)
# *CAMEO_DEU_2015_5D (Wealth: Stock Market Junkies)
# *CAMEO_DEU_2015_4D (Life stage typology: Empty Nest)
# *CAMEO_DEU_2015_7E (Life stage typology: Urban Parents)
# *SOHO_KZ (Small office / home office)
# 
# AND NEGATIVELY ALIGNED WITH (BOTTOM-UP):
# 
# *ZABEOTYP_6 (Energy consumption typology: Indifferent)
# *LP_STATUS_FEIN_8.0 (Social status: New houseowners)
# *CAMEO_DEU_2015_5F (Life stage typology: Active Retirement)
# *ZABEOTYP_2 (Energy consumption typology: Smart)
# *GFK_URLAUBERTYP_7.0 (Vacation habits: Golden ager)
# 
# It appears that the first principal component is mostly influenced by social, financial and later stage of life.
# 
# Principal Component 2:
# 
# The second principal component is 
# 
# POSITIVELY ALIGNED WITH (TOP-DOWN):
# 
# *PLZ8_ANTG3 (Number of 6-10 family houses in the PLZ8 region)
# *CAMEO_INTL_2015_WEALTH (Wealth)
# *PLZ8_ANTG4 (Number of 10+ family houses in the PLZ8 region)
# *HH_EINKOMMEN_SCORE (Estimated household net income)
# *FINANZ_SPARER (Financial typology / hoarding level)
# 
# AND NEGATIVELY ALIGNED WITH (BOTTOM-UP):
# 
# *KBA05_GBZ (Number of buildings in the microcell)
# *PLZ8_ANTG1 (Number of 1-2 family houses in the PLZ8 region)
# *KBA05_ANTG1 (Number of 1-2 family houses in the microcell)
# *MOBI_REGIO (Movement patterns)
# *FINANZ_MINIMALIST (Financial typology: Low financial interest)
# 
# It appears that the second principal component is influenced by household, surroundings, moving patterns and income.
# 
# Principal Component 3:
# 
# The third principal component is
# 
# POSITIVELY ALIGNED WITH (TOP-DOWN):
# 
# *ALTERSKATEGORIE_GROB (Estimated age based on given name analysis)
# *ZABEOTYP_3 (Fair supplied energy consumption)
# *FINANZ_VORSORGER (Financial typology / prevention level)
# *SEMIO_ERL (Event-oriented personality type)
# *RETOURTYP_BK_S (product return type of personality)
# 
# AND NEGATIVELY ALIGNED WITH (BOTTOM-UP):
# 
# *SEMIO_PFLICHT (dutiful personality type)
# *FINANZ_SPARER (Financial typology / hoarding level)
# *FINANZ_UNAUFFAELLIGER (Financial typology / noticeability level)
# *SEMIO_REL (Religious personality type)
# *PRAEGENDE_JUGENDJAHRE_GEN (generation to which a person belongs)
# 
# It appears that the third principal component is influenced by age, money related habits (product returns, events, savings level), openess and religiousness.

# ## Step 3: Clustering
# 
# ### Step 3.1: Apply Clustering to General Population
# 
# You've assessed and cleaned the demographics data, then scaled and transformed them. Now, it's time to see how the data clusters in the principal components space. In this substep, you will apply k-means clustering to the dataset and use the average within-cluster distances from each point to their assigned cluster's centroid to decide on a number of clusters to keep.
# 
# - Use sklearn's [KMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans) class to perform k-means clustering on the PCA-transformed data.
# - Then, compute the average difference from each point to its assigned cluster's center. **Hint**: The KMeans object's `.score()` method might be useful here, but note that in sklearn, scores tend to be defined so that larger is better. Try applying it to a small, toy dataset, or use an internet search to help your understanding.
# - Perform the above two steps for a number of different cluster counts. You can then see how the average distance decreases with an increasing number of clusters. However, each additional cluster provides a smaller net benefit. Use this fact to select a final number of clusters in which to group the data. **Warning**: because of the large size of the dataset, it can take a long time for the algorithm to resolve. The more clusters to fit, the longer the algorithm will take. You should test for cluster counts through at least 10 clusters to get the full picture, but you shouldn't need to test for a number of clusters above about 30.
# - Once you've selected a final number of clusters to use, re-fit a KMeans instance to perform the clustering operation. Make sure that you also obtain the cluster assignments for the general demographics data, since you'll be using them in the final Step 3.3.

# In[80]:


#Reduce the computation size
azdias_pca_sample = azdias_pca[np.random.choice(azdias_pca.shape[0], int(azdias_pca.shape[0]*0.25), replace=False)]
azdias_pca_sample


# In[83]:


# Over a number of different cluster counts...
    # run k-means clustering on the data and...
    # compute the average within-cluster distances.
    # Sum of Squared Errors(SSE)
sse = [] 
k_range = np.arange(10, 31)
for k in k_range:
    kmeans = KMeans(k, random_state=1234, max_iter=30, n_jobs=-1).fit(azdias_pca_sample)
    score = np.abs(kmeans.score(azdias_pca_sample))
    sse.append(score)
    print('Clustering done for {} k, with SSE {}'.format(k, score))


# In[84]:


# Investigate the change in within-cluster distance across number of clusters.
# HINT: Use matplotlib's plot function to visualize this relationship.
plt.figure(figsize=(18, 8))
plt.xticks(np.arange(0, k_range[-1]+1, step=1))
plt.xlabel('K')
plt.ylabel('SSE')
plt.title('SSE vs. K')
plt.plot(k_range, sse, linestyle='-', marker='o');


# In[85]:


# Re-fit the k-means model with the selected number of clusters and obtain
# cluster predictions for the general population demographics data.
k = 24
kmeans = KMeans(k, random_state=1234, max_iter=30, n_jobs=-1).fit(azdias_pca)
population_clusters = kmeans.predict(azdias_pca)


# ### Discussion 3.1: Apply Clustering to General Population
# 
# (Double-click this cell and replace this text with your own text, reporting your findings and decisions regarding clustering. Into how many clusters have you decided to segment the population?)
# 
# To find the perfect number of clusters, I used elbow method here and randomly selecteed the rows . There is no clearly distinguishable area in the plot and at K=24 the line flattens so I assume K=24.

# ### Step 3.2: Apply All Steps to the Customer Data
# 
# Now that you have clusters and cluster centers for the general population, it's time to see how the customer data maps on to those clusters. Take care to not confuse this for re-fitting all of the models to the customer data. Instead, you're going to use the fits from the general population to clean, transform, and cluster the customer data. In the last step of the project, you will interpret how the general population fits apply to the customer data.
# 
# - Don't forget when loading in the customers data, that it is semicolon (`;`) delimited.
# - Apply the same feature wrangling, selection, and engineering steps to the customer demographics using the `clean_data()` function you created earlier. (You can assume that the customer demographics data has similar meaning behind missing data patterns as the general demographics data.)
# - Use the sklearn objects from the general demographics data, and apply their transformations to the customers data. That is, you should not be using a `.fit()` or `.fit_transform()` method to re-fit the old objects, nor should you be creating new sklearn objects! Carry the data through the feature scaling, PCA, and clustering steps, obtaining cluster assignments for all of the data in the customer demographics data.

# In[86]:


# Load in the customer demographics data.
customers = pd.read_csv('Udacity_CUSTOMERS_Subset.csv', sep=';')


# In[87]:


# Apply preprocessing, feature transformation, and clustering from the general
# demographics onto the customer data, obtaining cluster predictions for the
# customer demographics data.
customers_clean, feat_info_clean, cust_high_nan_rows_clean, cust_low_nan_rows_clean = clean_data(customers, feat_info, split_value)


# In[88]:


customers_clean.info()


# In[89]:


customers_clean = customers_clean.dropna()


# In[91]:


# Apply feature scaling 
customers_clean_std = normalizer.transform(customers_clean)


# In[92]:


#Applying PCA
customers_pca = pca.transform(customers_clean_std)


# In[93]:


#Clustering the customers demographics data
kmeans = KMeans(k, random_state=1234, max_iter=30, n_jobs=-1).fit(customers_pca)
customer_clusters = kmeans.predict(customers_pca)


# ### Step 3.3: Compare Customer Data to Demographics Data
# 
# At this point, you have clustered data based on demographics of the general population of Germany, and seen how the customer data for a mail-order sales company maps onto those demographic clusters. In this final substep, you will compare the two cluster distributions to see where the strongest customer base for the company is.
# 
# Consider the proportion of persons in each cluster for the general population, and the proportions for the customers. If we think the company's customer base to be universal, then the cluster assignment proportions should be fairly similar between the two. If there are only particular segments of the population that are interested in the company's products, then we should see a mismatch from one to the other. If there is a higher proportion of persons in a cluster for the customer data compared to the general population (e.g. 5% of persons are assigned to a cluster for the general population, but 15% of the customer data is closest to that cluster's centroid) then that suggests the people in that cluster to be a target audience for the company. On the other hand, the proportion of the data in a cluster being larger in the general population than the customer data (e.g. only 2% of customers closest to a population centroid that captures 6% of the data) suggests that group of persons to be outside of the target demographics.
# 
# Take a look at the following points in this step:
# 
# - Compute the proportion of data points in each cluster for the general population and the customer data. Visualizations will be useful here: both for the individual dataset proportions, but also to visualize the ratios in cluster representation between groups. Seaborn's [`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html) or [`barplot()`](https://seaborn.pydata.org/generated/seaborn.barplot.html) function could be handy.
#   - Recall the analysis you performed in step 1.1.3 of the project, where you separated out certain data points from the dataset if they had more than a specified threshold of missing values. If you found that this group was qualitatively different from the main bulk of the data, you should treat this as an additional data cluster in this analysis. Make sure that you account for the number of data points in this subset, for both the general population and customer datasets, when making your computations!
# - Which cluster or clusters are overrepresented in the customer dataset compared to the general population? Select at least one such cluster and infer what kind of people might be represented by that cluster. Use the principal component interpretations from step 2.3 or look at additional components to help you make this inference. Alternatively, you can use the `.inverse_transform()` method of the PCA and StandardScaler objects to transform centroids back to the original data space and interpret the retrieved values directly.
# - Perform a similar investigation for the underrepresented clusters. Which cluster or clusters are underrepresented in the customer dataset compared to the general population, and what kinds of people are typified by these clusters?

# In[101]:


# Compare the proportion of data in each cluster for the customer data to the
# proportion of data in each cluster for the general population.
population_clusters = pd.Series(population_clusters)
popc = population_clusters.value_counts().sort_index()
popc = pd.Series(popc)
customer_clusters = pd.Series(customer_clusters)
custc = customer_clusters.value_counts().sort_index()
custc = pd.Series(custc)
# Missing rows from population dataset
popm = more_missing.shape[0]
# Missing rows from customer dataset
custm = cust_high_nan_rows_clean.shape[0]


# In[102]:


# What kinds of people are part of a cluster that is overrepresented in the
# customer data compared to the general population?
# Create a dataset summarizing clustering information
pop_cust = pd.concat([popc, custc], axis=1).reset_index()
pop_cust.columns = ['cluster', 'population', 'customers']

# Add missing rows cluster
pop_cust.loc[-1] = ['-1', popm, custm]
# Recalculate index
pop_cust.index = pop_cust.index + 1
# Sort by cluster (index)
pop_cust.sort_index(inplace=True)

# Calculate proprotions
pop_cust['cust_prop'] = pop_cust['customers'] / pop_cust['customers'].sum()
pop_cust['pop_prop'] = pop_cust['population'] / pop_cust['population'].sum()
pop_cust


# In[103]:


#Plotting bar graph
pop_cust.plot(x='cluster', y=['pop_prop', 'cust_prop'], kind='bar', figsize=(18, 8))
plt.title('Population vs. Customer Data Comparison')
plt.xlabel('Cluster Number (-1 represents missing rows data)')
plt.ylabel('Proportion of persons in cluster')
plt.show()


# In[105]:


# What kinds of people are part of a cluster that is underrepresented in the
# customer data compared to the general population?
cc11 = kmeans.cluster_centers_[11]
cc11 = pd.Series(cc11)
cc11.sort_values(ascending=False, inplace=True)
cc11.head(5)


# In[107]:


# Transform cluster 11 to original values
cc11 = normalizer.inverse_transform(pca.inverse_transform(cc11))
cc11 = pd.Series(cc11).round()
cc11.index = customers_clean.columns
cc11


# In[108]:


pca_results(customers_clean, pca, 74, 5, False)


# In[109]:


pca_results(customers_clean, pca, 62, 5, False)


# In[115]:


#Considering Cluster1
cc1 = kmeans.cluster_centers_[1]
cc1 = pd.Series(cc1)
cc1.sort_values(ascending=False, inplace=True)
cc1.head(5)


# In[117]:


cc1 = normalizer.inverse_transform(pca.inverse_transform(cc1))
cc1 = pd.Series(cc1).round()
cc1.index = customers_clean.columns
cc1


# In[118]:


pca_results(customers_clean, pca, 74, 5, False)


# In[119]:


pca_results(customers_clean, pca, 57, 5, False)


# ### Discussion 3.3: Compare Customer Data to Demographics Data
# 
# (Double-click this cell and replace this text with your own text, reporting findings and conclusions from the clustering analysis. Can we describe segments of the population that are relatively popular with the mail-order company, or relatively unpopular with the company?)
# Arvato Company seems to work well with older people, people living in less-dense households,more traditional and conservative people who focusses on investing and saving where the orders usually gets a hike in sales.
# The company doesn't focusses on Teen and young people. Cluster 1 refers to general population.
# 

# > Congratulations on making it this far in the project! Before you finish, make sure to check through the entire notebook from top to bottom to make sure that your analysis follows a logical flow and all of your findings are documented in **Discussion** cells. Once you've checked over all of your work, you should export the notebook as an HTML document to submit for evaluation. You can do this from the menu, navigating to **File -> Download as -> HTML (.html)**. You will submit both that document and this notebook for your project submission.

# In[ ]:




