import numpy as np
import pandas as pd
import timeit 
import textwrap
from datetime import datetime
from sqlalchemy import create_engine

class PandasIntro:
    def __init__(self):
        pass

    def dataframe00(self):
        np.random.seed(101)
        mydata = np.random.randint(0,101,(4,3))
        print("4 by 3 randomly filled matrix: \n", mydata)

        df = pd.DataFrame(data=mydata)
        print("matrix converted to dataframe: \n", df)

        myindex = ['CA','NY','AZ','TX']
        df = pd.DataFrame(data=mydata,index=myindex)
        print("matrix to dataframe with added custom index/row: \n", df)


        mycolumns = ['Jan','Feb','Mar']
        df = pd.DataFrame(data=mydata,index=myindex,columns=mycolumns)
        print("===========matrix to dataframe with added custom index/row and feature/column: \n", df)
        print("===========dataframe general info: \n")
        print(df.info())
        print("===========df statistical summary: \n", df.describe())
        print("===========transposed: ", df.describe().transpose())
        print("===========df columns: ", df.columns)
        print("===========df index/rows ", df.index)
        print("===========df first 3: ", df.head(3))
        print("===========df last 3: ", df.tail(3))
        print("===========df rows size: ", len(df))
        print("===========Fetch only 'Jan' column: \n", df['Jan'])
        print("===========Fetch only 'Jan' and 'Feb' column: \n", df[['Jan', 'Feb']])
        print("===========Grab only the first row: \n", df.iloc[0])
        print("===========Grab only the second row: \n", df.iloc[1])
        print("===========Data Type of the 'Jan' Column: \n", type(df['Jan'])) #  <class 'pandas.core.series.Series'

    def dataframe01(self):
        df = pd.read_csv('/Users/tess/Desktop/MLE2025/projects/UNZIP_FOR_NOTEBOOKS_FINAL/DATA/tips.csv')
        # Adding new column 'tip_percentage' and 'price_per_person'
        df['tip_percentage'] = 100* df['tip'] / df['total_bill']
        df['price_per_person'] = df['total_bill'] / df['size']
        # Rounding using Numpy
        df['price_per_person'] = np.round(df['price_per_person'],2)
        print("===========After adding new column 'tip_percentage': \n")
        print(df.head())
        # Removing Columns
        df = df.drop("tip_percentage",axis=1)
        print("===========After dropping 'tip_percentage': \n")
        print(df.head())

        print("===========Our INDEX: ", df.index)
        print("===========Setting New Index: ")
        df = df.set_index('Payment ID')
        print(df.head())
        print("===========Resetting to the default INDEX: ")
        df = df.reset_index()
        print(df.head())
        df = df.set_index('Payment ID')
        print("===========Grab first row via INDEX: ", df.iloc[0])
        print("===========Grab first row via NAME: ", df.loc['Sun2959'])
        print("===========Grab multiple rows via INDEX RANGE: ", df.iloc[0:4])
        print("===========Grab multiple rows via comma separated NAMES: ", df.loc[['Sun2959','Sun5260']])
        print("===========Removing row with index='Sun2959':", df.drop('Sun2959',axis=0).head())
        # df.drop(0,axis=0).head() # if we have NUMBER INDEX


        """
        Insert a New Row
        Pretty rare to add a single row like this. Usually you use pd.concat() to add many rows at once. 
        You could use the .append() method with a list of pd.Series() objects, but you won't see us do this with realistic real-world data.

        """
        one_row = df.iloc[0]
        print("===========Data to be inserted at the last row: ", one_row)
        print("===========It's type is: ", type(one_row))
        print("===========Tail as of now: ", df.tail())
        print("===========Tail after Inserting: ", df._append(one_row).tail())

    def dataframe02(self):
        df = pd.read_csv('/Users/tess/Desktop/MLE2025/projects/UNZIP_FOR_NOTEBOOKS_FINAL/DATA/tips.csv')
        print("===========Conditional Filtering:")
        print("==========='total_bill' > 30: \n", df[df['total_bill'] > 30])
        print("===========Male 'sex': \n", df[df['sex'] == 'Male'])
        print("===========total_bill' and 'sex' Male': \n", df[(df['total_bill'] > 30) & (df['sex']=='Male')])
        print("==========='tota_bill' and 'sex'Female: \n", df[(df['total_bill'] > 30) & ~(df['sex']=='Male')])
        print("==========='total_bill' and 'sex' Female 2: \n", df[(df['total_bill'] > 30) & (df['sex']!='Male')])
        print("===========weekends: \n", df[(df['day'] =='Sun') | (df['day']=='Sat')])

        ## Conditional Operator isin()

        print("===========isin weekends: \n", df['day'].isin(['Sat','Sun']))

    def dataframe03(self):
        df = pd.read_csv('./UNZIP_FOR_NOTEBOOKS_FINAL/DATA/tips.csv')

        #### APPLY
        def last_four(num):
            return str(num)[-4:]

        def yelp(price):
            if price < 10:
                return '$'
            elif price >= 10 and price < 30:
                return '$$'
            else:
                return '$$$'

        def quality(total_bill,tip):
            if tip/total_bill  > 0.25:
                return "Generous"
            else:
                return "Other"      

        df['last_four'] = df['CC Number'].apply(last_four)
        print("===========After adding 'last_four' column extracted from the 'CC Number': \n", df.head())

        df['Expensive'] = df['total_bill'].apply(yelp)
        print("===========After adding 'Expensive' column: \n", df.tail())

        df['total_bill'].apply(lambda bill:bill*0.18)
        print("===========After adjusting 'total_bill' via lambda function: \n", df.head())

        df['Tip Quality'] = df[['total_bill','tip']].apply(lambda df: quality(df['total_bill'],df['tip']),axis=1)
        print("===========After adding 'Tip Quality' column using 'total_bill' and 'tip' columns via lambda call to a 'quality' function: \n", df.tail())

        df['Tip Quality'] = np.vectorize(quality)(df['total_bill'], df['tip'])
        print("===========After adding 'Tip Quality' column using 'total_bill' and 'tip' columns via *np.vectorize* call to a 'quality' function: \n", df.tail())


        #########################################################################
        self.speed_apply_vs_vectorize()
        #########################################################################

        #### SORTING 
        df.sort_values('tip')
        print("===========After sorting by 'tip' column: \n", df.tail())

        df.sort_values(['tip','size'])
        print("===========After sorting by 'tip' and then by 'size' column: \n", df.tail())

        #### CORRELATION BETWEEN FEATURES OR COLUMNS
        # print("===========Correlations between features: \n", df.corr()) # ERROR since not all columns are NUMERICAL
        df_numeric = df.select_dtypes(include=['number'])
        print("===========1. Correlations between only NUMERICAL features: \n", df_numeric.corr())
        # df_encoded = df.copy()
        # df_encoded['sex'] = df_encoded['sex'].map({'Male': 0, 'Female': 1}) # But we have multiple NON NUMERICAL features
        # print("===========2. Correlations between ALL features after replacing (Male to 0) and (Female to 1): \n", df_encoded.corr())
        df_encoded = pd.get_dummies(df, drop_first=True)
        print("===========2. Correlations between ALL features after get_dummies(): \n", df_encoded.corr())


        print("===========Sub Correlation: Corr between 'total_bill' and 'tip' features: \n", df[['total_bill','tip']].corr())
        print("===========(MAX, maxIndex, MIN, minIndex)=", df['total_bill'].max(), df['total_bill'].idxmax(), df['total_bill'].min(), df['total_bill'].idxmin())
        print("===========Row at Index=170: ", df.iloc[170])

        #### COUNTS
        print("===========How many Male and how many Female our data has: \n", df['sex'].value_counts())

        #### REPLACE
        df['Tip Quality'] = df['Tip Quality'].replace(to_replace='Other',value='Okay')
        print("===========After replacing 'Tip Quality' OTHER by UNKNOWN: \n:", df)

        #### UNIQUE
        print("==========='size' unique: ", df['size'].unique())
        print("==========='size' number of unique: ", df['size'].nunique())
        print("==========='time' unique: ", df['time'].unique())
        print("==========='time'. number of unique: ", df['time'].nunique())

        #### MAP

        my_map = {'Dinner':'D','Lunch':'L'}
        df['time'] = df['time'].map(my_map)
        print("===========After mapping (Dinner to D) and (Lunch to L): \n", df.tail())

        #### DUPLICATED AND DROP DUPLICATES
        simple_df = pd.DataFrame([1,2,2],['a','b','c'])
        print("===========Is duplicated: \n", simple_df.duplicated())
        simple_df.drop_duplicates()
        print("===========After drop duplicates: \n", simple_df)

        #### BETWEEN
        print("===========Is 'total_bill' between 10 and 20: \n", df['total_bill'].between(10,20,inclusive='both'))
        print("===========Show dataset for 'total_bill' btn 10 and 20: \n", df[df['total_bill'].between(10,20,inclusive='both')])

        #### SAMPLE
        print("===========Sample of 5 rows: \n", df.sample(5))
        print("===========Sample of 10 percent: \n", df.sample(frac=0.1))
        print("===========Show 10 top dataset with largest 'tip': \n", df.nlargest(10,'tip'))

    
    def dataframe04_missing_data(self):
        """
        ****** What Null/NA/nan objects look like: ******
        Source: https://github.com/pandas-dev/pandas/issues/28095

        A new pd.NA value (singleton) is introduced to represent scalar missing values. 
        Up to now, pandas used several values to represent missing data: 
            np.nan is used for this for float data, 
            np.nan or None for object-dtype data and 
            pd.NaT for datetime-like data. 
        The goal of pd.NA is to provide a “missing” indicator that can be used consistently across data types. 
        pd.NA is currently used by the nullable integer and boolean data types and the new string data type

        """
        print("(np.nan, pd.NA, pd.NaT)=", (np.nan, pd.NA, pd.NaT)) # (np.nan, pd.NA, pd.NaT)= (nan, <NA>, NaT)
        """
        Note! Typical comparisons should be avoided with Missing Values
        This is generally because the logic here is, since we don't know these values, we can't know if they are equal to each other.
        """
        # [(np.nan == np.nan), (np.nan in [np.nan]), (np.nan is np.nan), (pd.NA == pd.NA)]= [False, True, True, <NA>]
        print("===========[(np.nan == np.nan), (np.nan in [np.nan]), (np.nan is np.nan), (pd.NA == pd.NA)]=", [(np.nan == np.nan), (np.nan in [np.nan]), (np.nan is np.nan), (pd.NA == pd.NA)])
        df = pd.read_csv('/Users/tess/Desktop/MLE2025/projects/UNZIP_FOR_NOTEBOOKS_FINAL/DATA/movie_scores.csv')
        print("===========df: \n", df)
        print("===========df.isnull(): \n", df.isnull())
        print("===========df.notnull(): \n", df.notnull())
        print("===========Having 'first_name': \n", df[df['first_name'].notnull()])
        print("===========Null 'pre_movie_score' and Null 'sex': \n", df[(df['pre_movie_score'].isnull()) & df['sex'].notnull()])

        #### dropna
        print("===========df.dropna(): \n", df.dropna())
        print("===========df.dropna(thresh=1): \n", df.dropna(thresh=1)) # drop rows where number of missing values < 1
        print("===========df.dropna(axis=1): \n", df.dropna(axis=1))
        print("===========df.dropna(thresh=4,axis=1): \n", df.dropna(thresh=4,axis=1))

        #### fillna
        print("===========df.fillna('NEW VALUE!'): \n", df.fillna("NEW VALUE!"))
        df['first_name'] = df['first_name'].fillna("Empty")
        print("===========After fillna 'first_name' with 'Empty': \n", df.head())

        df['pre_movie_score'].fillna(df['pre_movie_score'].mean())
        print("===========After fillna 'pre_movie_score' by 'mean': \n", df.head())

        print("===========Data Types of the df: \n", df.dtypes)
        df.fillna(df.mean(numeric_only=True)) # filling all NA by MEAN ???
        print("===========After filling all NA by MEAN: \n", df.head())

                                # Filling with Interpolation
        # The method ser.interpolate() is used on a pandas Series to fill in missing (NaN) values by estimating them based on neighboring data.

        # airline_tix = {'first':100,'business':np.nan,'economy-plus':50,'economy':30}

        # ser = pd.Series(airline_tix)
        # ser.interpolate()
        # ser.interpolate(method='spline')
        # df = pd.DataFrame(ser,columns=['Price'])
        # df.interpolate()
        # df = df.reset_index()
        # df.interpolate(method='spline',order=2)

    def dataframe05_groupby(self):
        df = pd.read_csv('/Users/tess/Desktop/MLE2025/projects/UNZIP_FOR_NOTEBOOKS_FINAL/DATA/mpg.csv')
        # Creates a groupby object waiting for an aggregate method
        # df.groupby('model_year')
        #### Adding an aggregate method call. To use a grouped object, you need to tell pandas how you want to aggregate the data.
        # mean(), sum(), size(), count(), std(), var(), sem(), describe(), first(), last(), nth(), min(), max()

        print("===========Columns: \n", df.columns)
        print("===========Data Types of the df: \n", df.dtypes)

        print("===========groupby the 'model_year, and take the mean():")
        avg_year = df.groupby('model_year').mean(numeric_only=True)
        print("===========Now, model_year becomes the index! It is NOT a column name,it is now the name of the index: \n", avg_year)
        print("===========New statistical summary: \n", avg_year.describe())
        print("===========New statistical summary transposed: \n", avg_year.describe().transpose())

        print("===========So, the Index: \n", avg_year.index)
        print("===========And the Columns: \n", avg_year.columns)
        print("===========Average 'mpg' by 'model_year': \n", avg_year['mpg'])
        print("===========Or directly, Average 'mpg' by 'model_year': \n", df.groupby('model_year').mean(numeric_only=True)['mpg'])

                                # MultiIndex

                                ## The MultiIndex Object
        ## Groupby Multiple Columns
        # First group by 'model_year', and then further sub-group by 'cylinders' and take the MEAN, Multiple Nested Group
        # year_cyl_avg = df.groupby(['model_year','cylinders']).mean() # ERROR since not all columns are NUMBER
        year_cyl_avg = df.groupby(['model_year', 'cylinders'])[df.select_dtypes(include='number').columns].mean()

        print("===========First group by 'model_year', and then further sub-group by 'cylinders' and take the MEAN, Multiple Nested Group: \n", year_cyl_avg.head())
        print("===========Here's the new tupled index: \n", year_cyl_avg.index)
        print("===========tupled index as list of levels: \n", year_cyl_avg.index.levels)
        print("===========tupled index as list of NAMED levels: \n", year_cyl_avg.index.names)
        print("===========Indexing SINGLE using the first tuple, 'model_year'=70: \n", year_cyl_avg.loc[70])
        print("===========Indexing MULTIPLE using comma separated multiple model_years like 70 and 72: \n", year_cyl_avg.loc[[70,72]])
        print("===========Indexing Down the Nested level using both tuple, 'model_year'=70 and 'cylinders'=8: \n", year_cyl_avg.loc[(70,8)])
        print("===========Using .XS(): Indexing Down the Nested level using both tuple, 'model_year'=70 and 'cylinders'=8: \n", year_cyl_avg.xs((70, 8)))

        # Grab Based on Cross-section with .xs()
        print("-------------------: ", year_cyl_avg.xs(key=70,axis=0,level='model_year'))
        print("*******************: Mean column values for 4 cylinders per year", year_cyl_avg.xs(key=4,axis=0,level='cylinders'))


        ### Careful note!

        # Keep in mind, its usually much easier to filter out values **before** running a groupby() call, so you should attempt to filter out any values/categories you don't want to use. 
        # For example, its much easier to remove **4** cylinder cars before the groupby() call, very difficult to this sort of thing after a group by.
        df[df['cylinders'].isin([6,8])].groupby(['model_year','cylinders'])[df.select_dtypes(include='number').columns].mean()
        print("*******************: After removing 4 cylinder cars, and groupby 'model_year' and 'cylinders' for MEAN: \n", df.head())

        ### Sorting MultiIndex
        print("******************* Sort by 'model_year': \n", year_cyl_avg.sort_index(level='model_year',ascending=False))
        print("******************* Sort by 'cylinders': \n", year_cyl_avg.sort_index(level='cylinders',ascending=False))

        ### Advanced: agg() method
        # The agg() method allows you to customize what aggregate functions you want per category
        # These strings need to match up with built-in method names
        print("******************* agg by 'median' and 'mean': \n", df.select_dtypes(include='number').agg(['median','mean']))
        print("******************* Specify aggregate methods per column: \n", df.select_dtypes(include='number').agg({'mpg':['median','mean'],'weight':['mean','std']}))
        print("******************* agg() with groupby(): \n", df.groupby('model_year').agg({'mpg':['median','mean'],'weight':['mean','std']}))


    def pandas06_concatenation(self):

        data_one = {'A': ['A0', 'A1', 'A2', 'A3'],'B': ['B0', 'B1', 'B2', 'B3']}
        one = pd.DataFrame(data_one)
        print("df 'one': \n", one)
        data_two = {'C': ['C0', 'C1', 'C2', 'C3'], 'D': ['D0', 'D1', 'D2', 'D3']}
        two = pd.DataFrame(data_two)
        print("df 'two': \n", two)

        # Axis = 0
        # Concatenate along rows
        axis0 = pd.concat([one,two],axis=0)
        print("===========After Concatenating along rows: \n", axis0)

        # Axis = 1
        # Concatenate along columns
        axis1 = pd.concat([one,two],axis=1)
        print("===========After Concatenating along columns: \n", axis1)

        ### Axis 0 , but columns match up
        two.columns = one.columns
        print("==========='two' taking 'one' column namings: \n", pd.concat([one,two]))

        ### Merge
        registrations = pd.DataFrame({'reg_id':[1,2,3,4],'name':['Andrew','Bobo','Claire','David']})
        print("===========registrations: \n", registrations)
        logins = pd.DataFrame({'log_id':[1,2,3,4],'name':['Xavier','Andrew','Yolanda','Bobo']})
        print("===========logins: \n", logins)

                        # Inner,Left, Right, and Outer Joins
        ## Inner Join
            # **Match up where the key is present in BOTH tables. There should be no NaNs due to the join, since by definition to be part of the Inner Join they need info in both tables.**
            # **Only Andrew and Bobo both registered and logged in.**
        # Notice pd.merge doesn't take in a list like concat
        print("===========After Inner Join: \n", pd.merge(registrations,logins,how='inner',on='name'))
        # Pandas smart enough to figure out key column (on parameter) if only one column name matches up
        # pd.merge(registrations,logins,how='inner')

        ## Left Join
            # **Match up AND include all rows from Left Table.**
            # **Show everyone who registered on Left Table, if they don't have login info, then fill with NaN.**
        print("===========After Left Join: \n", pd.merge(registrations,logins,how='left'))

        ## Right Join
            # Match up AND include all rows from Right Table. 
            # Show everyone who logged in on the Right Table, if they don't have registration info, then fill with NaN.
        print("===========After Right Join: \n", pd.merge(registrations,logins,how='right'))

        ## Outer Join
            # Match up on all info found in either Left or Right Table. Show everyone that's in the Log in table and the registrations table. Fill any missing info with NaN
        print("===========After the Outer Join: \n", pd.merge(registrations,logins,how='outer'))

        ## Join on Index or Column
            # Use combinations of left_on,right_on,left_index,right_index to merge a column or index on each other

        registrations = registrations.set_index("name")
        print("===========After setting 'name' as index for registrations: \n", registrations)
        print("===========Merge1: \n", pd.merge(registrations,logins,left_index=True,right_on='name'))
        print("===========Merge2: \n", pd.merge(logins,registrations,right_index=True,left_on='name'))


        ### Dealing with differing key column names in joined tables
        registrations = registrations.reset_index()
        print("===========registrations after resetting index: \n", registrations)
        registrations.columns = ['reg_name','reg_id']
        print("===========registrations after changing column names to 'reg_name' and 'reg_id': \n", registrations)

        print("===========Merge3: \n", pd.merge(registrations,logins,left_on='reg_name',right_on='name'))
        print("===========Merge4: \n", pd.merge(registrations,logins,left_on='reg_name',right_on='name').drop('reg_name',axis=1))

        ### Pandas automatically tags duplicate columns
        registrations.columns = ['name','id']
        logins.columns = ['id','name']
        # _x is for left
        # _y is for right
        print("===========Merge5: \n", pd.merge(registrations,logins,on='name'))
        print("===========Merge6: \n", pd.merge(registrations,logins,on='name',suffixes=('_reg','_log')))


    def pands07_text_methods(self):
        mystring = 'hello'
        print("===========After Capitalizing: ", mystring.capitalize())
        print("===========Checking if isdigit(): ", mystring.isdigit())

        names = pd.Series(['andrew','bobo','claire','david','4'])
        print("===========Pandas Series 'names': \n", names)
        print("===========Pandas Series 'names' capitalized: \n", names.str.capitalize())
        print("===========Pandas Series 'names' isdigit(): \n", names.str.isdigit())

        ## Splitting , Grabbing, and Expanding
        tech_finance = ['GOOG,APPL,AMZN','JPM,BAC,GS']
        print("===========length of 'tech_finance': ", len(tech_finance))
        tickers = pd.Series(tech_finance)
        print("===========Converting 'tech_finance' to Pandas Series: \n", tickers)
        print("===========After Pandas Series str.split:\n ", tickers.str.split(','))
        print("===========Only the index=0 of the above: \n", tickers.str.split(',').str[0])
        print("===========Via str.split and expand=True: \n", tickers.str.split(',',expand=True))

        ## Cleaning or Editing Strings
        messy_names = pd.Series(["andrew  ","bo;bo","  claire  "])
        print("===========messy_names: ", messy_names)
        print("===========Cleaner1 messy_names: \n", messy_names.str.replace(";",""))
        print("===========Cleaner2 messy_names: \n", messy_names.str.strip())
        print("===========Cleaner3 messy_names: \n", messy_names.str.replace(";","").str.strip())
        print("===========Cleaner4 messy_names: \n", messy_names.str.replace(";","").str.strip().str.capitalize())

        ## Alternative with Custom apply() call
        def cleanup(name):
            name = name.replace(";","")
            name = name.strip()
            name = name.capitalize()
            return name
        print("===========After apply(cleanup): \n", messy_names.apply(cleanup))

        ### Let's test which one is faster
        self.speed_strMethods_vs_apply_vs_vectorize()
        ################

    def speed_apply_vs_vectorize(self):
        # code snippet to be executed only once 
        setup = textwrap.dedent("""
        import numpy as np
        import pandas as pd
        df = pd.read_csv('/Users/tess/Desktop/MLE2025/projects/UNZIP_FOR_NOTEBOOKS_FINAL/DATA/tips.csv')
        def quality(total_bill,tip):
            if tip/total_bill  > 0.25:
                return "Generous"
            else:
                return "Other"
        """)
          
        # code snippet whose execution time is to be measured 
        stmt_one = textwrap.dedent(""" 
        df['Tip Quality'] = df[['total_bill','tip']].apply(lambda df: quality(df['total_bill'],df['tip']),axis=1)
        """)

        stmt_two = textwrap.dedent("""
        df['Tip Quality'] = np.vectorize(quality)(df['total_bill'], df['tip'])
        """)
        print("===========Let's execute the above two operations 1000 times, number, to measure the relative operation speed:")
        print("===========Time taken using 'apply' call to 'quality' function via lambda: \n", timeit.timeit(setup = setup, stmt = stmt_one, number = 1000))
        print("===========Time taken using np.vectorize(quality): \n", timeit.timeit(setup = setup, stmt = stmt_two, number = 1000))

        print("Wow! Vectorization is much faster! Keep **np.vectorize()** in mind for the future.")

    def speed_strMethods_vs_apply_vs_vectorize(self):
        # code snippet to be executed only once 
        setup = textwrap.dedent("""
        import pandas as pd
        import numpy as np
        messy_names = pd.Series(["andrew  ","bo;bo","  claire  "])
        def cleanup(name):
            name = name.replace(";","")
            name = name.strip()
            name = name.capitalize()
            return name
        """)
          
        # code snippet whose execution time is to be measured 
        stmt_pandas_str = textwrap.dedent(""" 
        messy_names.str.replace(";","").str.strip().str.capitalize()
        """)

        stmt_pandas_apply = textwrap.dedent("""
        messy_names.apply(cleanup)
        """)

        stmt_pandas_vectorize= textwrap.dedent("""
        np.vectorize(cleanup)(messy_names)
        """)
                     
                    
        print("===========Let's execute the above two operations 1000 times, number, to measure the relative operation speed:")
        print("===========Time taken using 'str.replace.str.strip().str.capitalize()': \n", timeit.timeit(setup = setup, stmt = stmt_pandas_str, number = 10000))
        print("===========Time taken using .apply(cleanup): \n", timeit.timeit(setup = setup, stmt = stmt_pandas_apply, number = 10000))
        print("===========Time taken using np.vectorize(cleanup)(messy_names): \n", timeit.timeit(setup = setup, stmt = stmt_pandas_vectorize, number = 10000))

        print("Wow! Vectorization is much faster, followed by ''.apply(cleanup)'! Keep **np.vectorize()** in mind for the future.")

    def pandas08_time_methods(self):
        # To illustrate the order of arguments
        my_year = 2017
        my_month = 1
        my_day = 2
        my_hour = 13
        my_minute = 30
        my_second = 15

        # January 2nd, 2017
        my_date = datetime(my_year,my_month,my_day)
        print("===========my_date: ", my_date)

        # January 2nd, 2017 at 13:30:15
        my_date_time = datetime(my_year,my_month,my_day,my_hour,my_minute,my_second)
        print("===========my_date_time: ", my_date_time)

        print("===========(my_date.day, my_date_time.hour)= ", (my_date.day, my_date_time.hour))


        # Converting to datetime
        # Often when data sets are stored, the time component may be a string. Pandas easily converts strings to datetime objects.
        myser = pd.Series(['Nov 3, 2000', '2000-01-01', None])

        print("===========Pandas converting Series to datetime: \n", pd.to_datetime(myser, format="mixed", dayfirst=False))
        obvi_euro_date = '31-12-2000'
        print("===========Converting '31-12-2000' to datetime: ", pd.to_datetime(obvi_euro_date))

        # 10th of Dec OR 12th of October?
        # We may need to tell pandas via 'dayfirst'
        euro_date = '10-12-2000'
        print("===========Converting to datetime: ", pd.to_datetime(euro_date))
        print("===========Converting: ", pd.to_datetime(euro_date,dayfirst=True))

        ### Custom Time String Formatting
        # Sometimes dates can have a non standard format, luckily you can always specify to pandas the format. 
        # You should also note this could speed up the conversion, so it may be worth doing even if pandas can parse on its own.


        style_date = '12--Dec--2000'
        print("Pandas converting any style of date by providing the specific format: ", pd.to_datetime(style_date, format='%d--%b--%Y'))
        strange_date = '12th of Dec 2000'
        print("Pandas convertng custom string of date to datetime object: ", pd.to_datetime(strange_date))


        sales = pd.read_csv('./UNZIP_FOR_NOTEBOOKS_FINAL/DATA/RetailSales_BeerWineLiquor.csv')
        print("===========sales1: ", sales)
        print("-----------1: ", sales.info())

        sales.iloc[0]['DATE'] # '1992-01-01'
        type(sales.iloc[0]['DATE']) # str

        sales['DATE'] = pd.to_datetime(sales['DATE'])
        print("===========sales2: ", sales)
        sales.iloc[0]['DATE'] # Timestamp('1992-01-01 00:00:00')
        type(sales.iloc[0]['DATE']) # pandas._libs.tslibs.timestamps.Timestamp



        # Parse Column at Index 0 as Datetime, so that Pandas will automatically parse data as datetime
        sales = pd.read_csv('./UNZIP_FOR_NOTEBOOKS_FINAL/DATA/RetailSales_BeerWineLiquor.csv',parse_dates=[0])
        print("===========sales3: ", sales)
        print("-----------3: ", sales.info())

        ## Resample
        # A common operation with time series data is resampling based on the time series index.

        sales = sales.set_index("DATE")
        print("===========sales4: After setting DATE as index: ", sales)

        # Yearly Means
        print("===========Yearly Means: ", sales.resample(rule='YE').mean())

        # .dt Method Calls
        # Once a column or index is ina  datetime format, you can call a variety of methods off of the .dt library inside pandas:
        sales = sales.reset_index()
        print("===========sales5: After resetting index: ", sales)
        print("===========.dt.month: ", sales['DATE'].dt.month)
        print("===========.dt.is_leap_year: ", sales['DATE'].dt.is_leap_year)


    def pandas09_csv(self):
        df = pd.read_csv('/Users/tess/Desktop/MLE2025/projects/UNZIP_FOR_NOTEBOOKS_FINAL/DATA/example.csv') # READING
        # #df = pd.read_csv('/Users/tess/Desktop/MLE2025/projects/UNZIP_FOR_NOTEBOOKS_FINAL/DATA/example.csv', index_col=0) # to use the 0th column as an index


        df.to_csv('/Users/tess/Desktop/MLE2025/projects/UNZIP_FOR_NOTEBOOKS_FINAL/DATA/new_file.csv',index=False) # WRITING
        # #df.to_csv('/Users/tess/Desktop/MLE2025/projects/UNZIP_FOR_NOTEBOOKS_FINAL/DATA/new_file.csv',index=True) # if want to save the index

    def pandas09_html(self):
        #                         ###HTML Input
        # Pandas read_html function will read tables off of a webpage and return a list of DataFrame objects. 
        # NOTE: This only works with well defined objects in the html on the page, this can not magically read in tables that are images on a page.

        tables = pd.read_html('https://en.wikipedia.org/wiki/World_population')
        table2 = tables[2]    
        print("data type: ", type(table2))
        print(table2)

        # If you are working on a website and want to quickly output the .html file, you can use to_html
        table2.to_html('simple_table2.html',index=False)

    def pandas09_excel(self):

        excel_sheet_df = pd.read_excel('./UNZIP_FOR_NOTEBOOKS_FINAL/03-Pandas/my_excel_file.xlsx',sheet_name='First_Sheet') # specific sheet_name
        print(excel_sheet_df)

        lst_of_sheet_names = pd.ExcelFile('./UNZIP_FOR_NOTEBOOKS_FINAL/03-Pandas/my_excel_file.xlsx').sheet_names
        print("Available List of Excel Sheet Names: ", lst_of_sheet_names ) # ['First_Sheet']

        excel_sheets_dfs = pd.read_excel('./UNZIP_FOR_NOTEBOOKS_FINAL/03-Pandas/my_excel_file.xlsx',sheet_name=None) # Grab all sheets
        print("the key-value, sheet_name as key, and value as excel_sheet_df: ", excel_sheets)
        print("From the dict, First_Sheet: \n", excel_sheets['First_Sheet'])

        mydf = excel_sheets['First_Sheet']
        mydf.to_excel('./UNZIP_FOR_NOTEBOOKS_FINAL/03-Pandas/example2.xlsx',sheet_name='First_Sheet',index=False) # WRITE


    def pandas09_sql(self):

        
        # Pandas can READ and WRITE to various SQL engines through the use of a driver SQL engines 
        # through the use of a driver and the sqlalchemy python library.

        # Step 1: Identify what SQL Engine to connect to: PostgreSQL, MySQL, MS SQL Server etc.
        # Step 2: Need to get specific libraries for your specific SQL Engine. 
                  # PostgreSQL >> psycopg2, MySQL >> pymysql, MS SQL Server >> pyodbc
                  # Here, let's use SQLite, which comes built in with Python and we can easily create
                  # a temporary database inside your RAM.
        # Step 3: Use the sqlalchemy library to connect ot your SQL database with the driver.

        # """

        temp_db = create_engine('sqlite:///:memory:') # creates a temporary SQLite DB inside your computer's RAM
        print("temp_db: ", temp_db)
        tables = pd.read_html('https://en.wikipedia.org/wiki/World_population')

        pop_df = tables[6]
        pop_df.to_sql(name='populations_new_table3',con=temp_db)

        # ### Read from SQL Database
        # # Read in an entire table
        new_df = pd.read_sql(sql='populations_new_table3',con=temp_db)
        print("new_df: \n", new_df)

        # # Read in with a SQL Query
        new_df2 = pd.read_sql_query(sql="SELECT Country FROM populations_new_table3",con=temp_db)
        print("new_df2: \n", new_df2)

        where = pd.read_sql("SELECT name FROM sqlite_master WHERE type='populations_new_table3';", con=temp_db)
        print("where: ", where)


        # # Note: It is difficult to generalize pandas and SQL, due to a wide array of issues, including permissions,security, online access, varying SQL engines, etc... 


    def pandas09_pandas_pivot(self):
        """
        The pivot() method reshapes data based on column values and reassignment of the index. 
        Keep in mind, it doesn't always make sense to pivot data. 
        Pivot methods are mainly for data analysis,visualization, and exploration.
    
        ** What type of question does a pivot help answer?**
        Imagine we wanted to know, how many licenses of each product type did Google purchase? 
        Currently the way the data is formatted is hard to read. 
        Let's pivot it so this is clearer, we will take a subset of the data for the question at hand.



        """
        df = pd.read_csv('./UNZIP_FOR_NOTEBOOKS_FINAL/03-Pandas/Sales_Funnel_CRM.csv')
        print("Full df we have: \n", df)
        # Let's take a subset, otherwise we'll get an error due to duplicate rows and data
        subset = df[['Company','Product','Licenses']]
        print("Subset df we're interested in: \n", subset)
        pivoted_subset = pd.pivot(data=subset,index='Company',columns='Product',values='Licenses')
        print("Pivoted Subset: \n", pivoted_subset)

    
        ## The pivot_table() method
        # Similar to the pivot() method, the pivot_table() can add aggregation functions to a pivot call.
        # Notice Account Number sum() doesn't make sense to keep/use
        pivot_table1 = pd.pivot_table(df,index="Company",aggfunc='sum')
        print("pivot table 1, but summing Account Number is not right: \n", pivot_table1)
        # Either grab the columns
        pivot_table2 = pd.pivot_table(df,index="Company",aggfunc='sum')[['Licenses','Sale Price']]
        print("pivot table 2, explicitly describing columns 'Licenses', and 'Sale Price' to be summed: \n", pivot_table2)
        # Or state them as wanted values
        pivot_table3 = pd.pivot_table(df,index="Company",aggfunc='sum',values=['Licenses','Sale Price'])
        print("pivot table 3, 'Licenses' and 'Sale Price' via 'values': \n", pivot_table3)

        grouped_and_summed = df.groupby('Company').sum()[['Licenses','Sale Price']]
        print("pivot table 3 using groupby.sum: \n", grouped_and_summed)

        multi_index_pivot_table = pd.pivot_table(df,index=["Account Manager","Contact"],values=['Sale Price'],aggfunc='sum')
        print("multi index pivoted table: \n", multi_index_pivot_table)

        # Columns are optional - they provide an additional way to segment the actual values you care about. The aggregation functions are applied to the values you list.
        pd.pivot_table(df,index=["Account Manager","Contact"],values=["Sale Price"],columns=["Product"],aggfunc=["sum"])
        pd.pivot_table(df,index=["Account Manager","Contact"],values=["Sale Price"],columns=["Product"],aggfunc=["sum"],fill_value=0)
        # Can add multiple agg functions
        pd.pivot_table(df,index=["Account Manager","Contact"],values=["Sale Price"],columns=["Product"],aggfunc=["sum","mean"],fill_value=0)
        # Can add on multiple columns
        pd.pivot_table(df,index=["Account Manager","Contact"],values=["Sale Price","Licenses"],columns=["Product"],aggfunc=["sum"],fill_value=0)

        # Can add on multiple columns
        pd.pivot_table(df,index=["Account Manager","Contact","Product"],values=["Sale Price","Licenses"],aggfunc=["sum"],fill_value=0)
        # get Final "ALL" with margins = True
        # Can add on multiple columns
        pd.pivot_table(df,index=["Account Manager","Contact","Product"],values=["Sale Price","Licenses"],aggfunc=["sum"],fill_value=0,margins=True)
        pd.pivot_table(df,index=["Account Manager","Status"],values=["Sale Price"],aggfunc=["sum"],fill_value=0,margins=True)





if __name__ == "__main__":
    pandas = PandasIntro()
    pandas.dataframe00()
    pandas.dataframe01()
    pandas.dataframe02()
    pandas.dataframe03()
    pandas.dataframe04_missing_data()
    pandas.dataframe05_groupby()
    pandas.pandas06_concatenation()
    pandas.pands07_text_methods()
    pandas.pandas08_time_methods()
    pandas.pandas09_csv()
    pandas.pandas09_html()
    pandas.pandas09_sql()
    pandas.pandas09_pandas_pivot()


