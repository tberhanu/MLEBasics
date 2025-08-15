import numpy as np
import pandas as pd
import textwrap
from datetime import datetime
from sqlalchemy import create_engine
from utils import Utils

class PandasIntro:
    def __init__(self):
        pass

    def pandas_basics(self):
        # df.columns, df.corr(), df.corr()['col_name']
        # df['col_name'].value_counts()
        # df['col_name'].sort_values()
        # df.plot(), df.plot(kind='bar')
        # dtypes, info, describe, transpose, type, len, head, tail
        np.random.seed(101)
        mydata = np.random.randint(0,101,(4,3))
        df1 = pd.DataFrame(data=mydata)
        myindex = ['CA','NY','AZ','TX']
        df2 = pd.DataFrame(data=mydata,index=myindex)
        mycolumns = ['Jan','Feb','Mar']
        df3 = pd.DataFrame(data=mydata,index=myindex,columns=mycolumns)
        df_types = df3.dtypes
        df_info = df3.info()
        stats = df3.describe()
        stats2 = df3.describe().transpose()
        colmns = df3.columns
        indices = df3.index
        first_last = (df3.head(1), df3.tail(1))
        num_of_rows = len(df3)
        rows_cols = df3.shape
        row_zero = df3.iloc[0]
        col_type = type(df3['Jan'])
        # indexing, np.round, drop, set_index, reset_index, _append
        df = pd.read_csv('./UNZIP_FOR_NOTEBOOKS_FINAL/DATA/tips.csv')
        
        df['tip_percentage'] = 100* df['tip'] / df['total_bill'] # Making New Features
        df['price_per_person'] = df['total_bill'] / df['size']
        df['price_per_person'] = np.round(df['price_per_person'],2) # Rounding using Numpy

        df = df.drop("tip_percentage",axis=1) # Dropping Column
        df = df.set_index('Payment ID') # Setting 'Feature' as Index
        df = df.reset_index() # Resetting Index
        df = df.set_index('Payment ID') # Setting 'Feature' as Index
        first_row = df.iloc[0] # Regular Indexing
        first_row2 = df.loc['Sun2959'] # Indexing via non-numerical Index
        indexing_multiple_data = df.iloc[[0, 1]]
        indexing_multiple_data2 = df.loc[['Sun2959','Sun5260']]
        remove_row = df.drop('Sun2959',axis=0) # Dropping Data Row
        add_row = df._append(df.iloc[0]) # Appending to Tail. Usually you use 'pd.concat()' to add many rows at once

        # conditional filters
        expensive_df = df[df['total_bill'] > 30] 
        expensive_male_df = df[(df['total_bill'] > 30) & (df['sex']=='Male')]
        expensive_female_df = df[(df['total_bill'] > 30) & ~(df['sex']=='Male')]
        expensive_female_df2 = df[(df['total_bill'] > 30) & (df['sex']!='Male')]
        weekends_df = df[(df['day'] =='Sun') | (df['day']=='Sat')]
        weekends_df2 = df['day'].isin(['Sat','Sun'])

    def df03_apply_vectorize(self):
        df = pd.read_csv('./UNZIP_FOR_NOTEBOOKS_FINAL/DATA/tips.csv')

        df['last_four'] = df['CC Number'].apply(Utils.last_four)
        df['Expensive'] = df['total_bill'].apply(Utils.yelp)
        df['total_bill'].apply(lambda bill:bill*0.18)
        df['Tip Quality'] = df[['total_bill','tip']].apply(lambda df: Utils.quality(df['total_bill'],df['tip']),axis=1)
        df['Tip Quality'] = np.vectorize(Utils.quality)(df['total_bill'], df['tip'])

        #########################################################################
        Utils.speed_apply_vs_vectorize()
        #########################################################################

    def df03_sort_select_corr_encode_counts_nuniques_dups_btn_sample(self):
        df = pd.read_csv('./UNZIP_FOR_NOTEBOOKS_FINAL/DATA/tips.csv')
        df['Tip Quality'] = np.vectorize(Utils.quality)(df['total_bill'], df['tip'])
        df['Tip Quality'] = df['Tip Quality'].replace(to_replace='Other',value='Okay')


        #### SORTING
        df2 = df.sort_values('tip')

        #### SELECTING NUMERICAL FEATURES ONLY
        df_numeric = df.select_dtypes(include=['number'])
        df_numeric2 = df.select_dtypes(include='number')

        #### CORRELATION BETWEEN FEATURES OR COLUMNS whose data type is NUMERIC
        correlation_matrix = df_numeric.corr() # Based on only numeric features/columns
        sub_corr_btn_features = df[['total_bill','tip']].corr() # Sub Correlation: Corr between 'total_bill' and 'tip' features

        #### 0-1 Encoding
        df_encoded = pd.get_dummies(df, drop_first=True) 

        ### MIN, MAX
        getMax, getMaxIndex = df['total_bill'].max(), df['total_bill'].idxmax()
        getMin, getMinIndex = df['total_bill'].min(), df['total_bill'].idxmin()

        #### COUNTS
        counts_per_categorical_feature = df['sex'].value_counts()

        #### REPLACE
        df['Tip Quality'] = df['Tip Quality'].replace(to_replace='Other',value='Okay')

        #### UNIQUE and NON-UNIQUE
        uniques = df['size'].unique()
        non_uniques = df['size'].nunique()

        #### MAP, REPLACING VALUES
        my_map = {'Dinner':'D','Lunch':'L'}
        df['time'] = df['time'].map(my_map)

        #### DUPLICATED AND DROP DUPLICATES
        simple_df = pd.DataFrame([1,2,2],['a','b','c'])
        duplicated_rows = simple_df.duplicated()
        dedup_df = simple_df.drop_duplicates()

        #### BETWEEN
        is_between_10_and_20 = df['total_bill'].between(10,20,inclusive='both')

        #### SAMPLE
        random_five_sample = df.sample(5)
        random_10percent_sample = df.sample(frac=0.1)
        nlargest = df.nlargest(10, 'tip')
        nsmallest = df.nsmallest(10, 'tip')
        
    def dataframe04_missing_data(self):
        
        print("(np.nan, pd.NA, pd.NaT)=", (np.nan, pd.NA, pd.NaT)) # (np.nan, pd.NA, pd.NaT)= (nan, <NA>, NaT)
        # [(np.nan == np.nan), (np.nan in [np.nan]), (np.nan is np.nan), (pd.NA == pd.NA)]= [False, True, True, <NA>]
        print("[(np.nan == np.nan), (np.nan in [np.nan]), (np.nan is np.nan), (pd.NA == pd.NA)]=", [(np.nan == np.nan), (np.nan in [np.nan]), (np.nan is np.nan), (pd.NA == pd.NA)])
        df = pd.read_csv('./UNZIP_FOR_NOTEBOOKS_FINAL/DATA/movie_scores.csv')
        # Specific Feature Value whether None or Not None
        boolean_df1 = df["first_name"].isnull()
        boolean_df2 = df["pre_movie_score"].isnull()
        boolean_df3 = df['first_name'].notnull()
        filtered = df[(df['pre_movie_score'].isnull()) & df['sex'].notnull()]

        #### dropna
        # Removes any row that contains at least one missing value:
        remove_any_na_rows = df.dropna() # by default, axis=0 and how='any'
        # Removes any column that contains at least one missing value:
        remove_na_cols = df.dropna(axis=1) # by default, how='any'
        # Remove ROW only if ALL values in that row are missing:
        remove_all_na_rows = df.dropna(how='all')
        # Not to be removed, ROW need to contain at least 2 non-missing values:
        remove_thresh_na_rows = df.dropna(thresh=2)
        # Not to be removed, COLUMN need to contain at least 4 non-missing values:
        remove_thresh_na_cols = df.dropna(thresh=4,axis=1)

        #### fillna
        df2 = df.fillna("NEW VALUE!") # fill all None value with the 'NEW VALUE'
        df['first_name'] = df['first_name'].fillna("Empty") # 'Empty' for None valued 'first_name'

        df['pre_movie_score'] = df['pre_movie_score'].fillna(df['pre_movie_score'].mean()) # fillna by MEAN Value of 'pre_movie_score' column

        df3 = df.fillna(df.mean(numeric_only=True)) # fill all None by the Column Mean for all numerical features
        # import pdb; pdb.set_trace()

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

    def df05_groupby_isin_and_select_dtypes(self):
        df = pd.read_csv('./UNZIP_FOR_NOTEBOOKS_FINAL/DATA/mpg.csv')
        
        group1 = df.groupby('model_year') # Creates a groupby object waiting for an aggregate method like mean(), sum(), size(), 
                                          # count(), std(), var(), sem(), describe(), first(), last(), nth(), min(), max()
        avg_year = group1.mean(numeric_only=True)


        ## The MultiIndex Object: Groupby Multiple Columns
        # First group by 'model_year', and then further sub-group by 'cylinders' and take the MEAN, Multiple Nested Group
        # year_cyl_avg = df.groupby(['model_year','cylinders']).mean() # ERROR since not all columns are NUMBER
        nested_group = df.groupby(['model_year', 'cylinders'])
        numerical_features = df.select_dtypes(include='number').columns
        numerical_nested_group_cols = nested_group[numerical_features]
        year_cyl_avg = numerical_nested_group_cols.mean()
        # import pdb; pdb.set_trace()
        index_names = year_cyl_avg.index.names # FrozenList(['model_year', 'cylinders'])
        tpl_index = year_cyl_avg.index # [(70, 4), (70, 6), (70, 8), (71, 4), .... (model_year, cylinders)]
        index_levels = year_cyl_avg.index.levels # FrozenList([[70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82], [3, 4, 5, 6, 8]])
        year_70s = year_cyl_avg.loc[70]
        year_71s = year_cyl_avg.xs(key=71,axis=0,level='model_year')
        year_70_cylinders_4 = year_cyl_avg.loc[(70, 4)]
        year_70_cylinders_8 = year_cyl_avg.xs((70, 8))
        year_70_and_72 = year_cyl_avg.loc[[70,72]]

        ### Sorting MultiIndex
        sort_by_model_year = year_cyl_avg.sort_index(level='model_year',ascending=False)
        sort_by_cylinders = year_cyl_avg.sort_index(level='cylinders',ascending=False)

        ### Careful note! Filter out values before running a groupby() call
        filtered_by_cyl = df[df['cylinders'].isin([6,8])]
        nested_group2 = filtered_by_cyl.groupby(['model_year','cylinders'])
        numerical_nested_group_cols2 = nested_group2[numerical_features]
        year_cyl_avg2 = numerical_nested_group_cols2.mean()

        

        ### Advanced: agg() method
        # The agg() method allows you to customize what aggregate functions you want per category
        # These strings need to match up with built-in method names
        agg_by_median_mean = df.select_dtypes(include='number').agg(['median','mean'])
        agg_by_median_mean_std = df.select_dtypes(include='number').agg({'mpg':['median','mean'],'weight':['mean','std']})
        agg_by_median_mean_std2 = df.groupby('model_year').agg({'mpg':['median','mean'],'weight':['mean','std']})


        
        # import pdb; pdb.set_trace()

    def pandas06_concatenation(self):

        data_one = {'A': ['A0', 'A1', 'A2', 'A3'],'B': ['B0', 'B1', 'B2', 'B3']}
        df1 = pd.DataFrame(data_one)
        data_two = {'C': ['C0', 'C1', 'C2', 'C3'], 'D': ['D0', 'D1', 'D2', 'D3']}
        df2 = pd.DataFrame(data_two)

        # Axis = 0
        # Concatenate along rows
        axis0 = pd.concat([df1,df2],axis=0)

        # Axis = 1
        # Concatenate along columns
        axis1 = pd.concat([df1,df2],axis=1)

        ### Axis 0 , but columns match up
        df2.columns = df1.columns
        axis2 = pd.concat([df1,df2]) # 'two' taking 'one' column namings

        ### Merge
        registrations = pd.DataFrame({'reg_id':[1,2,3,4],'name':['Andrew','Bobo','Claire','David']})
        logins = pd.DataFrame({'log_id':[1,2,3,4],'name':['Xavier','Andrew','Yolanda','Bobo']})

        ## Inner Join
        inner_joined = pd.merge(registrations,logins,how='inner',on='name')

        ## Left Join: Match up AND include all rows from Left Table.
        left_joined = pd.merge(registrations,logins,how='left')

        ## Right Join: Match up AND include all rows from Right Table. 
        right_joined = pd.merge(registrations,logins,how='right')

        ## Outer Join: Match up on all info found in either Left or Right Table. 
        outer_joined = pd.merge(registrations,logins,how='outer')


        ## Join on Index or Column
        # Use combinations of left_on,right_on,left_index,right_index to merge a column or index on each other
        # import pdb; pdb.set_trace()

        registrations = registrations.set_index("name")
        merge1 = pd.merge(registrations,logins,left_index=True,right_on='name')
        merge2 = pd.merge(logins,registrations,right_index=True,left_on='name')

        ### Dealing with differing key column names in joined tables
        registrations = registrations.reset_index()
        registrations.columns = ['reg_name','reg_id']
        

        merge3 = pd.merge(registrations,logins,left_on='reg_name',right_on='name')
        merge4 = pd.merge(registrations,logins,left_on='reg_name',right_on='name').drop('reg_name',axis=1)
        
        ### Pandas automatically tags duplicate columns
        registrations.columns = ['name','id']
        logins.columns = ['id','name']
        # _x is for left
        # _y is for right
        merge5 = pd.merge(registrations,logins,on='name')
        merge6 = pd.merge(registrations,logins,on='name',suffixes=('_reg','_log'))


    def pands07_text_methods(self):
        mystring = 'hello'
        str_capitalized = mystring.capitalize()
        is_str_digit = mystring.isdigit()

        myseries = pd.Series(['andrew','bobo','claire','david','4'])
        series_capitalized = myseries.str.capitalize()
        is_series_digit = myseries.str.isdigit()

        ## Splitting , Grabbing, and Expanding
        tech_finance = ['GOOG,APPL,AMZN','JPM,BAC,GS']
        myseries2 = pd.Series(tech_finance)
        splitted = myseries2.str.split(',')
        splitted2 = myseries2.str.split(',').str[0]
        splitted3 = myseries2.str.split(',',expand=True)

        ## Cleaning or Editing Strings
        messy_names = pd.Series(["andrew  ","bo;bo","  claire  "])
        names = messy_names.str.replace(";","")
        names2 = messy_names.str.strip()
        names3 = messy_names.str.replace(";","").str.strip()
        names4 = messy_names.str.replace(";","").str.strip().str.capitalize()

        
        names5 = messy_names.apply(Utils.cleanup)
        # import pdb; pdb.set_trace()
        ### Let's test which one is faster
        ##############################################
        Utils.speed_strMethods_vs_apply_vs_vectorize()
        ##############################################

    def pandas08_time_methods(self):
        # To illustrate the order of arguments
        my_year, my_month, my_day, my_hour, my_minute, my_second = 2017, 1, 2, 13, 30, 15
        my_date = datetime(my_year,my_month,my_day) # January 2nd, 2017
        my_date_time = datetime(my_year,my_month,my_day,my_hour,my_minute,my_second) # January 2nd, 2017 at 13:30:15
        mydate, myhour = my_date.day, my_date_time.hour


        # Converting to datetime
        # Often when data sets are stored, the time component may be a string. Pandas easily converts strings to datetime objects.
        myser = pd.Series(['Nov 3, 2000', '2000-01-01', None])

        mydate = pd.to_datetime(myser, format="mixed", dayfirst=False)
        obvi_euro_date = '31-12-2000'
        # my_euro_date = pd.to_datetime(obvi_euro_date) # OK, but warning
        my_euro_date = pd.to_datetime(obvi_euro_date, dayfirst=True) # OK, but warning


        # 10th of Dec OR 12th of October?
        # We may need to tell pandas via 'dayfirst'
        euro_date = '10-12-2000'
        mydate3 = pd.to_datetime(euro_date)
        mydate4 = pd.to_datetime(euro_date,dayfirst=True)

        ### Custom Time String Formatting
        # Sometimes dates can have a non standard format, luckily you can always specify to pandas the format. 
        # You should also note this could speed up the conversion, so it may be worth doing even if pandas can parse on its own.


        style_date = '12--Dec--2000'
        mydate5 = pd.to_datetime(style_date, format='%d--%b--%Y')
        strange_date = '12th of Dec 2000'
        mydate6 = pd.to_datetime(strange_date)


        sales = pd.read_csv('./UNZIP_FOR_NOTEBOOKS_FINAL/DATA/RetailSales_BeerWineLiquor.csv')

        sales.iloc[0]['DATE'] # '1992-01-01'
        type(sales.iloc[0]['DATE']) # str

        sales['DATE'] = pd.to_datetime(sales['DATE'])
        sales.iloc[0]['DATE'] # Timestamp('1992-01-01 00:00:00')
        type(sales.iloc[0]['DATE']) # pandas._libs.tslibs.timestamps.Timestamp



        # Parse Column at Index 0 as Datetime, so that Pandas will automatically parse data as datetime
        sales = pd.read_csv('./UNZIP_FOR_NOTEBOOKS_FINAL/DATA/RetailSales_BeerWineLiquor.csv',parse_dates=[0])

        ## Resample
        # A common operation with time series data is resampling based on the time series index.

        sales = sales.set_index("DATE")

        # Yearly Means
        yearly_means = sales.resample(rule='YE').mean()

        # .dt Method Calls
        # Once a column or index is ina  datetime format, you can call a variety of methods off of the .dt library inside pandas:
        sales = sales.reset_index()
        the_month = sales['DATE'].dt.month
        is_leap_year = sales['DATE'].dt.is_leap_year


    def pandas09_csv_html_excel(self):
        df = pd.read_csv('/Users/tess/Desktop/MLE2025/projects/UNZIP_FOR_NOTEBOOKS_FINAL/DATA/example.csv') # READING
        # #df = pd.read_csv('/Users/tess/Desktop/MLE2025/projects/UNZIP_FOR_NOTEBOOKS_FINAL/DATA/example.csv', index_col=0) # to use the 0th column as an index

        df.to_csv('/Users/tess/Desktop/MLE2025/projects/UNZIP_FOR_NOTEBOOKS_FINAL/DATA/new_file.csv',index=False) # WRITING
        # #df.to_csv('/Users/tess/Desktop/MLE2025/projects/UNZIP_FOR_NOTEBOOKS_FINAL/DATA/new_file.csv',index=True) # if want to save the index

        ### HTML Input
        # Pandas read_html function will read tables off of a webpage and return a list of DataFrame objects. 
        # NOTE: This only works with well defined objects in the html on the page, this can not magically read in tables that are images on a page.

        tables = pd.read_html('https://en.wikipedia.org/wiki/World_population')
        table2 = tables[2]    

        # If you are working on a website and want to quickly output the .html file, you can use to_html
        table2.to_html('simple_table2.html',index=False)


        ### EXCEL
        excel_sheet_df = pd.read_excel('./UNZIP_FOR_NOTEBOOKS_FINAL/03-Pandas/my_excel_file.xlsx',sheet_name='First_Sheet') # specific sheet_name

        lst_of_sheet_names = pd.ExcelFile('./UNZIP_FOR_NOTEBOOKS_FINAL/03-Pandas/my_excel_file.xlsx').sheet_names

        excel_sheets_dfs = pd.read_excel('./UNZIP_FOR_NOTEBOOKS_FINAL/03-Pandas/my_excel_file.xlsx',sheet_name=None) # Grab all sheets
        single_sheet = excel_sheets_dfs['First_Sheet']
    
        single_sheet.to_excel('./UNZIP_FOR_NOTEBOOKS_FINAL/03-Pandas/example2.xlsx',sheet_name='First_Sheet',index=False) # WRITE

    def pandas09_sql(self):
        ### SQL
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
        tables = pd.read_html('https://en.wikipedia.org/wiki/World_population')

        pop_df = tables[6]
        pop_df.to_sql(name='populations_new_table3',con=temp_db)

        # ### Read from SQL Database
        # # Read in an entire table
        new_df = pd.read_sql(sql='populations_new_table3',con=temp_db)

        # # Read in with a SQL Query
        new_df2 = pd.read_sql_query(sql="SELECT Country FROM populations_new_table3",con=temp_db)

        # whereee = pd.read_sql("SELECT name FROM sqlite_master WHERE type='populations_new_table3';", con=temp_db) # ?????


        # # Note: It is difficult to generalize pandas and SQL, due to a wide array of issues, including permissions,security, online access, varying SQL engines, etc... 

    def pandas09_pivot_table(self):
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
        # Let's take a subset, otherwise we'll get an error due to duplicate rows and data
        subset = df[['Company','Product','Licenses']]
        pivoted_subset = pd.pivot(data=subset,index='Company',columns='Product',values='Licenses')

    
        ## The pivot_table() method
        # Similar to the pivot() method, the pivot_table() can add aggregation functions to a pivot call.
        # Notice Account Number sum() doesn't make sense to keep/use
        pivot_table1 = pd.pivot_table(df,index="Company",aggfunc='sum')
        # Either grab the columns
        pivot_table2 = pd.pivot_table(df,index="Company",aggfunc='sum')[['Licenses','Sale Price']]
        # Or state them as wanted values
        pivot_table3 = pd.pivot_table(df,index="Company",aggfunc='sum',values=['Licenses','Sale Price'])

        grouped_and_summed = df.groupby('Company').sum()[['Licenses','Sale Price']]

        multi_index_pivot_table = pd.pivot_table(df,index=["Account Manager","Contact"],values=['Sale Price'],aggfunc='sum')

        # Columns are optional - they provide an additional way to segment the actual values you care about. The aggregation functions are applied to the values you list.
        pivot_table4 = pd.pivot_table(df,index=["Account Manager","Contact"],values=["Sale Price"],columns=["Product"],aggfunc=["sum"])
        pivot_table5 = pd.pivot_table(df,index=["Account Manager","Contact"],values=["Sale Price"],columns=["Product"],aggfunc=["sum"],fill_value=0)
        # Can add multiple agg functions
        pivot_table6 = pd.pivot_table(df,index=["Account Manager","Contact"],values=["Sale Price"],columns=["Product"],aggfunc=["sum","mean"],fill_value=0)
        # Can add on multiple columns
        pivot_table7 = pd.pivot_table(df,index=["Account Manager","Contact"],values=["Sale Price","Licenses"],columns=["Product"],aggfunc=["sum"],fill_value=0)

        # Can add on multiple columns
        pivot_table8 = pd.pivot_table(df,index=["Account Manager","Contact","Product"],values=["Sale Price","Licenses"],aggfunc=["sum"],fill_value=0)
        # get Final "ALL" with margins = True
        # Can add on multiple columns
        pivot_table8 = pd.pivot_table(df,index=["Account Manager","Contact","Product"],values=["Sale Price","Licenses"],aggfunc=["sum"],fill_value=0,margins=True)
        pivot_table9 = pd.pivot_table(df,index=["Account Manager","Status"],values=["Sale Price"],aggfunc=["sum"],fill_value=0,margins=True)


    
 

if __name__ == "__main__":
    pandas = PandasIntro()
    pandas.pandas_basics()
    pandas.df03_apply_vectorize()
    pandas.df03_sort_select_corr_encode_counts_nuniques_dups_btn_sample()
    pandas.dataframe04_missing_data()
    pandas.df05_groupby_isin_and_select_dtypes()
    pandas.pandas06_concatenation()
    pandas.pands07_text_methods()
    pandas.pandas08_time_methods()
    pandas.pandas09_csv_html_excel()
    pandas.pandas09_sql()
    pandas.pandas09_pivot_table()





