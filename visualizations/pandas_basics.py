import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D 


class PandasBasics:
    def __init__(self):
        pass 

    def pandas_df_groupby(self):
        data = {'Category': ['A', 'B', 'A', 'C', 'B', 'A'], 'Value': [10, 15, 20, 25, 30, 35]}
        df = pd.DataFrame(data)
        groups = df.groupby("Category")
        values = df.groupby("Category")["Value"]
    
        # Group by 'Category' and calculate the sum of 'Value' for each category
        grouped_data = df.groupby('Category')['Value'].sum()

    def pandas_series_index(self):
        # Create a Series with a default integer index
        s1 = pd.Series([10, 20, 30])
        indices = s1.index

        # Create a Series with a custom string index
        cities = ['New York', 'London', 'Paris']
        populations = [8.4, 8.9, 2.1]
        s2 = pd.Series(populations, index=cities)
        indices = s2.index

        # Accessing elements using index labels
        elements = s2['London']

    def create_dummy_variables(self):
        person_state =  pd.Series(['Dead','Alive','Dead','Alive','Dead','Dead'])
        dummies = pd.get_dummies(person_state)
        dummies2 = pd.get_dummies(person_state,drop_first=True)

    def scatter_plot_3d_4d(self):

        """
        If only 3d plot needed, just use c=fourth_variable as a COLOR by using one of the three Variables or just directly like for RED >> c='r'
        Also, you may not need to add a colorbar, 'cbar'.
        """
        # Generate some example data
        n_points = 100
        x = np.random.rand(n_points) * 10
        y = np.random.rand(n_points) * 10
        z = np.random.rand(n_points) * 10
        fourth_variable = np.random.rand(n_points) * 50  # This will determine the color
        # print("fourth_variable: ", fourth_variable)

        # Create a figure and a 3D subplot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Create the 3D scatter plot, coloring by the fourth variable
        scatter_plot = ax.scatter(x, y, z, c=fourth_variable, cmap='viridis', s=50) # 'viridis' is a good default colormap

        # Set labels for the axes
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')

        # Add a colorbar
        cbar = fig.colorbar(scatter_plot, ax=ax, pad=0.1) # Add padding to avoid overlapping with plot
        cbar.set_label('Fourth Variable Value')

        plt.title('3D Scatter Plot with Color Representing a Fourth Variable')
        plt.show()

if __name__ == "__main__":
    pandasBasics = PandasBasics()
    pandasBasics.pandas_df_groupby()
    pandasBasics.pandas_series_index()
    pandasBasics.create_dummy_variables()
    pandasBasics.scatter_plot_3d_4d()


    