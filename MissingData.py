import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class MissingData:
    """
    ********** Removing Features or Removing Rows **********
    If only a few rows relative to the size of your dataset are missing some values, then it might just be a good idea to drop those rows. What does this cost you in terms of performace? 
    It essentialy removes potential training/testing data, but if its only a few rows, its unlikely to change performance.

    Sometimes it is a good idea to remove a feature entirely if it has too many null values. 
    However, you should carefully consider why it has so many null values, in certain situations null could just be used as a separate category.

    Take for example a feature column for the number of cars that can fit into a garage. Perhaps if there is no garage then there is a null value, instead of a zero. 
    It probably makes more sense to quickly fill the null values in this case with a zero instead of a null. Only you can decide based off your domain expertise and knowledge of the data set!

    Working based on Rows Missing Data
    Filling in Data or Dropping Data?
    Let's explore how to choose to remove or fill in missing data for rows that are missing some data. 
    Let's choose some threshold where we decide it is ok to drop a row if its missing some data (instead of attempting to fill in that missing data point).
    We will choose 1% as our threshold. This means if less than 1% of the rows are missing this feature, we will consider just dropping that row, instead of dealing with the feature itself. 
    There is no right answer here, just use common sense and your domain knowledge of the dataset, 
    obviously you don't want to drop a very high threshold like 50% , you should also explore correlation to the dataset, maybe it makes sense to drop the feature instead.

    """
    def __init__(self, visualize=True):
        self.df = pd.read_csv("/Users/tess/Desktop/MLE2025/projects/UNZIP_FOR_NOTEBOOKS_FINAL/DATA/ALTERED/Ames_outliers_removed.csv")
        self.visualize = visualize

    def get_data_detailed_info(self):
        with open('/Users/tess/Desktop/MLE2025/projects/UNZIP_FOR_NOTEBOOKS_FINAL/DATA/Ames_Housing_Feature_Description.txt','r') as f: 
            print(f.read())

    def percent_missing(self):
        percent_nan = 100* self.df.isnull().sum() / len(self.df) # type(percent_nan) is Pandas Series
        percent_nan = percent_nan[percent_nan>0].sort_values()

        return percent_nan

    def set_missing_threshold_and_fillna_rows(self, ylim_percentage=1):
        if self.visualize:
            ylim_percentage, plt_title = ylim_percentage, "1. Less than 1% NaN"
            self.visualize_barplot(plt_title, ylim_percentage)

           
        # Let's see rows with such NaN values
        null1 = self.df[self.df['Total Bsmt SF'].isnull()]
        null2 = self.df[self.df['Bsmt Half Bath'].isnull()]

        bsmt_num_cols = ['BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF','Total Bsmt SF', 'Bsmt Full Bath', 'Bsmt Half Bath']
        bsmt_num_cols_types = [self.df.dtypes[col] for col in bsmt_num_cols]
        self.df[bsmt_num_cols] = self.df[bsmt_num_cols].fillna(0)

        bsmt_str_cols =  ['Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2']
        bsmt_str_cols_types = [self.df.dtypes[col] for col in bsmt_str_cols]
        self.df[bsmt_str_cols] = self.df[bsmt_str_cols].fillna('None')

        if self.visualize:
            ylim_percentage, plt_title = None, "2. After filling 'Bsmt*'' NaN with 0 and None"
            self.visualize_barplot(plt_title, ylim_percentage)


    def dropna_rows(self):
        nan_features = ['Electrical','Garage Cars'] # Assuming we decide to drop all Rows with NaN value for these features
        self.df = self.df.dropna(axis=0, subset=nan_features)
        if self.visualize:
            ylim_percentage, plt_title = 1, "3. After dropna Electrical & Garage Cars rows"
            self.visualize_barplot(plt_title, ylim_percentage)
        

    def visualize_barplot(self, plt_title, ylim_percentage=None):
        percent_nan = self.percent_missing()
        sns.barplot(x=percent_nan.index,y=percent_nan)
        plt.xticks(rotation=90)
        if ylim_percentage:
            plt.ylim(0, ylim_percentage)
        plt.title(plt_title)
        plt.show()

    def visualize_baxplot(self, plt_title):
        plt.figure(figsize=(8,12)) #, dpi=200)
        sns.boxplot(x='Lot Frontage',y='Neighborhood',data=self.df,orient='h')
        plt.title(plt_title)
        plt.show()

    def fillna_mas_vnr_feature(self):
        """
        Based on the Description Text File, Mas Vnr Type and Mas Vnr Area being missing (NaN) is likely to mean the house simply just doesn't 
        have a masonry veneer, in which case, we will fill in this data as we did before.

        """
        self.df["Mas Vnr Type"] = self.df["Mas Vnr Type"].fillna("None")
        self.df["Mas Vnr Area"] = self.df["Mas Vnr Area"].fillna(0)
        if self.visualize:
            plt_title = "4. After fillna mas vnr feature"
            self.visualize_barplot(plt_title)

    def fillna_for_no_garage(self):
        """
        Filling In Missing Column Data
        Our previous approaches were based more on rows missing data. 
        Now we will take an approach based on the column features themselves, since larger percentages of the data appears to be missing.

        """
        garage_related_features = self.df[['Garage Type', 'Garage Finish', 'Garage Qual', 'Garage Cond']]

        gar_str_cols = ['Garage Type', 'Garage Finish', 'Garage Qual', 'Garage Cond']
        self.df[gar_str_cols] = self.df[gar_str_cols].fillna('None')

        valueCounts = self.df['Garage Yr Blt'].value_counts()
        self.df['Garage Yr Blt'] = self.df['Garage Yr Blt'].fillna(0)
        if self.visualize:
            plt_title = "5. After fillna for no garage"
            self.visualize_barplot(plt_title)

    def removing_features(self):
        subset_df = self.df[['Lot Frontage', 'Fireplace Qu', 'Fence', 'Alley', 'Misc Feature','Pool QC']]

        self.df = self.df.drop(['Pool QC','Misc Feature','Alley','Fence'],axis=1)
        if self.visualize:
            plt_title = "6. After removing some features or columns"
            self.visualize_barplot(plt_title)

    def fillna_fireplace_qu(self):
        valueCounts = self.df['Fireplace Qu'].value_counts()
        self.df['Fireplace Qu'] = self.df['Fireplace Qu'].fillna("None")
        if self.visualize:
            plt_title = "7. After fillna 'Fireplace Qu'"
            self.visualize_barplot(plt_title)

    def missing_data_imputation(self):
        """
        To impute missing data, we need to decide what other filled in (no NaN values) feature most probably relates and
        is correlated with the missing feature data. 
        In this particular case we will use:

            'Neighborhood': Physical locations within Ames city limits
            'LotFrontage': Linear feet of street connected to property

        We will operate under the assumption that the Lot Frontage is related to what neighborhood a house is in.

        """

        nei_unique_values = self.df['Neighborhood'].unique()
        isNull =  self.df[self.df['Lot Frontage'].isnull()]
        isNull2 = self.df[self.df['Neighborhood'].isnull()]

        if self.visualize:
            # BOXPLOT TO ANALYZE IF 'Neighborhood' and 'Lot Frontage' are related.
            plt_title = "8. BOXPLOT Relationshp: Lot Frontage vs Neighborhood"
            self.visualize_baxplot(plt_title)

        # Roughly, with some domain knowledge, let's assume we noticed there is a relationship,
        # and if we miss the 'Lot Frontage', then we can fill it up with the average value of 'Lot Frontage' at that 'Neighborhood'
        lotFrontage_of_nei = self.df.groupby('Neighborhood')['Lot Frontage']
        lotFrontage_of_nei_mean = self.df.groupby('Neighborhood')['Lot Frontage'].mean() # notice that some are NaN
        
        self.df['Lot Frontage'] = self.df.groupby('Neighborhood')['Lot Frontage'].transform(lambda val: val.fillna(val.mean()))
        if self.visualize:
            plt_title = "9. Data Imputation: 'Neighborhood's Mean' filled 'Lot Frontage's NaN'"
            self.visualize_barplot(plt_title)

        # We still have few NaN values for 'Lot Frontage' because some 'Neighborhood' mean value of 'Lot Frontage' is also NaN, so need to take care of that too.
        # For example 'GrnHill' and 'Landmrk' neighborhood have NaN for Lot Frontage Mean because their Lot Frontage is NaN as seen below'
        GrnHill_Landmrk = self.df[(self.df['Neighborhood'] == 'GrnHill') | (self.df['Neighborhood'] == 'Landmrk')]
        self.df['Lot Frontage'] = self.df['Lot Frontage'].fillna(0)
        missing_counts = self.df.isnull().sum()
        percent_nan = self.percent_missing()
        try:
            self.visualize_barplot("10. Empty Space because No NaNs!!!")
        except Exception as e:
            print("As Expected: Error: 'min() arg is an empty sequence: ", e)
        else:
            print("No Exception Raised.")
        finally:
            print("Anyways, let's save our work, regardless of NaN leftover presence or not.")
            self.df.to_csv("./UNZIP_FOR_NOTEBOOKS_FINAL/DATA/ALTERED/Ames_NO_Missing_Data.csv",index=False)





if __name__ == "__main__":
    missingData = MissingData(False) # visualize=True
    missingData.set_missing_threshold_and_fillna_rows(1)
    missingData.dropna_rows()
    missingData.fillna_mas_vnr_feature()
    missingData.fillna_for_no_garage()
    missingData.removing_features()
    missingData.fillna_fireplace_qu()
    missingData.missing_data_imputation()

    





