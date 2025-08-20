import textwrap
import timeit 
import matplotlib.pyplot as plt

class Utils:

    @staticmethod
    def drow_symbols(self):

        # Sample data
        x = [4, 2, 0, 6, 8, 7, 3]
        y = [7, 9, 3, 1, 5, 0, 6]
        z = ['⬜', '▭', '◇', '⬛', '▢', '▣', '▪']

        # Create scatter plot with symbols as labels
        fig, ax = plt.subplots()
        ax.scatter(x, y)

        # Add symbols at each point
        for i in range(len(x)):
            ax.text(x[i], y[i], z[i], fontsize=14, ha='center', va='center')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Plot of Symbols at (x, y)')
        plt.grid(True)
        plt.show()

    @staticmethod
    def last_four(num):
        return str(num)[-4:]

    @staticmethod
    def yelp(price):
        if price < 10:
            return '$'
        elif price >= 10 and price < 30:
            return '$$'
        else:
            return '$$$'

    @staticmethod
    def quality(total_bill,tip):
        if tip/total_bill  > 0.25:
            return "Generous"
        else:
            return "Other"   

    @staticmethod
    def cleanup(name):
            name = name.replace(";","")
            name = name.strip()
            name = name.capitalize()
            return name

    @staticmethod
    def speed_apply_vs_vectorize():
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
        print("Let's execute the above two operations 1000 times, number, to measure the relative operation speed:")
        print("Time taken using 'apply' call to 'quality' function via lambda: \n", timeit.timeit(setup = setup, stmt = stmt_one, number = 1000))
        print("Time taken using np.vectorize(quality): \n", timeit.timeit(setup = setup, stmt = stmt_two, number = 1000))

        print("Wow! Vectorization is much faster! Keep **np.vectorize()** in mind for the future.")

    @staticmethod
    def speed_strMethods_vs_apply_vs_vectorize():
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
                     
                    
        print("Let's execute the above two operations 1000 times, number, to measure the relative operation speed:")
        print("Time taken using 'str.replace.str.strip().str.capitalize()': \n", timeit.timeit(setup = setup, stmt = stmt_pandas_str, number = 10000))
        print("Time taken using .apply(cleanup): \n", timeit.timeit(setup = setup, stmt = stmt_pandas_apply, number = 10000))
        print("Time taken using np.vectorize(cleanup)(messy_names): \n", timeit.timeit(setup = setup, stmt = stmt_pandas_vectorize, number = 10000))

        print("Wow! Vectorization is much faster, followed by ''.apply(cleanup)'! Keep **np.vectorize()** in mind for the future.")




