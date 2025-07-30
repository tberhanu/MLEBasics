import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class PandasBasics:
    def __init__(self):
        pass 

    def pandas_df_groupby(self):
        data = {'Category': ['A', 'B', 'A', 'C', 'B', 'A'], 'Value': [10, 15, 20, 25, 30, 35]}
        df = pd.DataFrame(data)
        groups = df.groupby("Category")
        for group in groups:
            print("--------------\n")
            print("group: ", group)
        values = df.groupby("Category")["Value"]
        for value in values:
            print("==============\n")
            print("value: ", value)
        # Group by 'Category' and calculate the sum of 'Value' for each category
        grouped_data = df.groupby('Category')['Value'].sum()
        print("group by category value sum: ", grouped_data)

    def pandas_series_index(self):
        # Create a Series with a default integer index
        s1 = pd.Series([10, 20, 30])
        print("Series with default index:")
        print(s1)
        print("Index of s1:", s1.index)

        # Create a Series with a custom string index
        cities = ['New York', 'London', 'Paris']
        populations = [8.4, 8.9, 2.1]
        s2 = pd.Series(populations, index=cities)
        print("\nSeries with custom index:")
        print(s2)
        print("Index of s2:", s2.index)

        # Accessing elements using index labels
        print("\nPopulation of London:", s2['London'])

    def create_dummy_variables(self):
        person_state =  pd.Series(['Dead','Alive','Dead','Alive','Dead','Dead'])
        print("Original Feature Unique Values for 'person_state': \n", person_state)
        print(f"Dummy Variables for feature 'person_state': \n", pd.get_dummies(person_state))
        print(f"Dummy Variables after dropping one column: \n", pd.get_dummies(person_state,drop_first=True))

if __name__ == "__main__":
    pandasBasics = PandasBasics()
    pandasBasics.pandas_df_groupby()
    pandasBasics.pandas_series_index()
    pandasBasics.create_dummy_variables()