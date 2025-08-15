import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



class SeabornPlots:
	def __init__(self):

		"""
		1. Google Image Searching "Choosing a plot visualization" to see many useful flowcharts:
			Line Charts, Bar Charts, Scatter Plots, Area Charts, Pie Charts, Histograms, Heat Maps, Bubble Charts, Waterfall Charts, Treemaps, Maps (Geographic).

		- Scatter Plots: (x, y)=2D, with added legend for hue='column_z'=>3D, with style and coloring
		- Distribution Plots: Rug Plot, Histogram, and KDE Plot
		- Categorical Plots 
		- Comparison Plots
		- Seaborn Grids
		- Matrix Plots like Heat Maps

		"""
		self.df = pd.read_csv("./UNZIP_FOR_NOTEBOOKS_FINAL/05-Seaborn/dm_office_sales.csv")
		self.df2 = pd.read_csv("./UNZIP_FOR_NOTEBOOKS_FINAL/05-Seaborn/StudentsPerformance.csv")
		self.df3 = pd.read_csv('./UNZIP_FOR_NOTEBOOKS_FINAL/05-Seaborn/country_table.csv')

		# print(self.df)
		# print(self.df.info())

	def scatterplot(self):
		sns.scatterplot(x='salary',y='sales',data=self.df)
		plt.title("1.scatterplot 2D")
		plt.show()

		## Connecting to Figure in Matplotlib
		# **Note how matplotlib is still connected to seaborn underneath (even without importing matplotlib.pyplot), since seaborn itself is directly making a Figure call with matplotlib. 
		# We can import matplotlib.pyplot and make calls to directly effect the seaborn figure.**
		
		plt.figure(figsize=(12,8)) # this is a Matplotlib thing, but still affect the Seaborn
		sns.scatterplot(x='salary',y='sales',data=self.df)
		plt.title("2.scatterplot with plt.figure")
		plt.show()

		plt.figure(figsize=(12,8))
		sns.scatterplot(x='salary',y='sales',data=self.df,hue='division')
		plt.title("3.scatterplot with hue=division")
		plt.show()

		plt.figure(figsize=(12,8))
		sns.scatterplot(x='salary',y='sales',data=self.df,hue='work experience')
		plt.title("4.scatterplot with hue=work experience")
		plt.show()

		plt.figure(figsize=(12,8))
		sns.scatterplot(x='salary',y='sales',data=self.df,hue='work experience',palette='viridis')
		plt.title("5.scatterplot with hue and palette")

		plt.figure(figsize=(12,8))
		sns.scatterplot(x='salary',y='sales',data=self.df,size='work experience')
		plt.title("6.scatterplot with size=work experience")
		plt.show()

		plt.figure(figsize=(12,8))
		sns.scatterplot(x='salary',y='sales',data=self.df,s=200)
		plt.title("7.scatterplot with s=20")
		plt.show()

		plt.figure(figsize=(12,8))
		sns.scatterplot(x='salary',y='sales',data=self.df,s=200,linewidth=0,alpha=0.2)
		plt.title("8.scatterplot with s, linewidth and alpha")
		plt.show()

		plt.figure(figsize=(12,8))
		sns.scatterplot(x='salary',y='sales',data=self.df,style='level of education')
		plt.title("9.scatterplot with style=level of education")
		plt.show()

		plt.figure(figsize=(12,8))
		# Sometimes its nice to do BOTH hue and style off the same column
		sns.scatterplot(x='salary',y='sales',data=self.df,style='level of education',hue='level of education',s=100)
		plt.title("10.scatterplot with both style and hue")

		# Call savefig in the same cell
		plt.savefig('my_scatterplot.jpg')
		plt.show()



	def rugplot_1D(self):
		## Rugplot

		# Very simple plot that puts down one mark per data point. 
		# This plot needs the single array passed in directly. 
		# We won't use it too much since its not very clarifying for large data sets.
		# The rugplot itself is not very informative for larger data sets distribution around the mean since so many ticks makes it hard to distinguish one tick from another.
		sns.rugplot(x='salary',data=self.df)
		plt.title("rugplot")
		plt.show()

		sns.rugplot(x='salary',data=self.df,height=0.5)
		plt.title("rugplot height=0.5")
		plt.show()

	def displot_and_histplot(self):

		#The displot is a plot type that can show you the distribution of a single feature. 
		# It is a HISTOGRAM with the option of adding a "KDE" plot (Kernel Density Estimation) on top of the histogram.
		sns.displot(data=self.df,x='salary',kde=True)
		plt.title("displot kde=True")
		plt.show()

		sns.histplot(data=self.df,x='salary')
		plt.title("histplot")
		plt.show()
		sns.histplot(data=self.df,x='salary',bins=10)
		plt.title("histplot bins=10")
		plt.show()
		sns.histplot(data=self.df,x='salary',bins=100)
		plt.title("histplot bins=100")
		plt.show()

		### grid_and_styles
		sns.set(style='darkgrid')
		sns.histplot(data=self.df,x='salary',bins=100)
		plt.title("1. histplot: style='darkgrid'")
		plt.show()

		sns.set(style='white')
		sns.histplot(data=self.df,x='salary',bins=100)
		plt.title("2. histplot: style='white'")
		plt.show()

		### Adding in keywords from matplotlib
		# Seaborn plots, NOT all, can accept keyword arguments directly from the matplotlib code that seaborn uses. 
		sns.displot(data=self.df,x='salary',bins=20,kde=False,color='red',edgecolor='black',lw=4,ls='--')
		plt.title("3. displot: color-edgecolor-lw-ls")
		plt.show()

	def kdeplot(self):
		## KDE: The Kernel Density Estimation Plot
		# The KDE plot maps an estimate of a probability *density* function of a random variable. 
		# Kernel density estimation is a fundamental data smoothing problem where inferences about the population are made, based on a finite data sample.

		np.random.seed(42)
		sample_ages = np.random.randint(0,100,200)
		sample_ages = pd.DataFrame(sample_ages,columns=["age"])
		sns.rugplot(data=sample_ages,x='age')
		plt.title("4. 1D rugplot")
		plt.show()
		plt.figure(figsize=(12,8))
		sns.displot(data=sample_ages,x='age',bins=10,rug=True,kde=True)
		plt.title("5. displot, bins=10, rug and kde")
		plt.show()
		sns.kdeplot(data=sample_ages,x='age')
		plt.title("6. kdeplot")
		plt.show()
		### Cut Off KDE
		# We could cut off the KDE if we know our data has hard limits (no one can be a negative age and no one in the population can be older than 100 for some reason)
		# plt.figure(figsize=(12,8))
		sns.kdeplot(data=sample_ages,x='age',clip=[0,100])
		plt.title("7. kdeplot, clip[0, 100]")
		plt.show()
		### Bandwidth
		# KDE is constructed through the summation of the kernel (most commonly Gaussian), we can effect the bandwith of this kernel to make the KDE more "sensitive" to the data. 
		# Notice how with a smaller bandwith, the kernels don't stretch so wide, meaning we don't need the cut-off anymore. 
		# This is analagous to increasing the number of bins in a histogram (making the actual bins narrower).
		sns.kdeplot(data=sample_ages,x='age',bw_adjust=0.1)
		plt.title("8. kde, bw_adjust=0.1")
		plt.show()
		sns.kdeplot(data=sample_ages,x='age',bw_adjust=0.5)
		plt.title("9. kde, bw_adjust=0.5")
		plt.show()
		sns.kdeplot(data=sample_ages,x='age',bw_adjust=1)
		plt.title("10. kde, bw_adjust=1")
		plt.show()
		# Basic Styling on KDE
		sns.kdeplot(data=sample_ages,x='age',bw_adjust=0.5,shade=True,color='red')
		plt.title("11. kde, bw_adjust=0.5, shade=True, color='red")
		plt.show()
		# 2D KDE
		random_data = pd.DataFrame(np.random.normal(0,1,size=(100,2)),columns=['x','y'])
		sns.kdeplot(data=random_data,x='x',y='y')
		plt.title("12. 2D KDE")
		plt.show()

		### Bonus Code for Visualizations: SKIPPED
	def categorical_countplot(self):
		## Countplot()=>(similar to a .groupby(x_axis).count() call in pandas)
		# A simple plot, it merely shows the total count of rows per category. 
		print("========Just using Pandas: \n", self.df['division'].value_counts())
		plt.figure(figsize=(10,4),dpi=100)
		sns.countplot(x='division',data=self.df)
		plt.title("1. 1D countplot")
		plt.show()

		plt.figure(figsize=(10,4),dpi=100)
		sns.countplot(x='level of education',data=self.df,hue='training level')
		plt.title("2. countplot, hue=training level")
		plt.show()
		plt.figure(figsize=(10,4),dpi=100)
		sns.countplot(x='level of education',data=self.df,hue='training level',palette='Set1')
		plt.title("3. countplot nested group via hue")
		plt.show()
		plt.figure(figsize=(10,4),dpi=100)
		# Paired would be a good choice if there was a distinct jump from 0 and 1 to 2 and 3
		sns.countplot(x='level of education',data=self.df,hue='training level',palette='Paired')
		plt.title("4. countplot nested group via hue, palette")
		plt.show()

	def categorical_barplot(self):
		# barplot()
		# By default barplot() will show the mean if 'y'. The stick,'sd', is for standard deviation
		plt.figure(figsize=(10,6),dpi=100)
		sns.barplot(x='level of education',y='salary',data=self.df,estimator=np.mean,errorbar='sd')
		plt.title("5. barplot, mean, sd")
		plt.show()

		plt.figure(figsize=(12,6))
		sns.barplot(x='level of education',y='salary',data=self.df,estimator=np.mean,errorbar='sd',hue='division')
		plt.title("6. barplot, nested group via hue")
		plt.show()

		plt.figure(figsize=(12,6),dpi=100)
		sns.barplot(x='level of education',y='salary',data=self.df,estimator=np.mean,errorbar='sd',hue='division')
		plt.legend(bbox_to_anchor=(1.05, 1))
		plt.title("7. barplot, with legend location")
		plt.show()

	def categorical_boxplot(self):
		## Boxplot(aka. Whisker Plot)
		# A boxplot display distribution through the use of quartiles and an IQR(=Inter Quartile Range) for outliers.
		# Anything outside 1.5 * IQR to the left, and 1.5 * IQR to the right are Outliers.


				# ****Distribution of CATEGORICAL FEATURE Values to the CONTINUOUS FEATURE Values****
		# Here, distribution of ALL DATA to the Math Score.
		plt.figure(figsize=(12,6))
		sns.boxplot(x='math score',data=self.df2)
		plt.title("1. boxplot 'math score' without category")
		plt.show()

		# Here, distribution of Parental Education to the Math Score
		# It's expected the higher Parental Education, the Better Math Score of the Student
		plt.figure(figsize=(12,6))
		sns.boxplot(x='parental level of education',y='math score',data=self.df2)
		plt.title("2. boxplot with category")
		plt.show()

		# We keep asking: On the previous distribution, does GENDER make a difference ?
		# Here, distribution of Parental Education and GENDER to the Math Score
		plt.figure(figsize=(12,6))
		sns.boxplot(x='parental level of education',y='math score',data=self.df2,hue='gender')
		# Optional move the legend outside
		plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title="colored-title")
		# plt.legend(loc=(1.05, 1), title="some-title") # works fine too 
		plt.title("3. boxplot with hue='gender' and legend bbox_to_anchor")
		plt.show()


		plt.figure(figsize=(12,6))
		sns.boxplot(x='parental level of education',y='math score',data=self.df2,hue='gender',width=0.3)
		plt.title("4. boxplot with hue and width")
		plt.show()
		### Boxplot Styling Parameters
		#### Orientation
		# NOTICE HOW WE HAVE TO SWITCH X AND Y FOR THE ORIENTATION TO MAKE SENSE!
		sns.boxplot(x='math score',y='parental level of education',data=self.df2,orient='h')
		plt.title("5. boxplot with orient='h")
		plt.show()

	def categorical_violinplot(self):
		## Violinplot
		# A violin plot plays a similar role as a box and whisker plot. 
		# It shows the distribution of quantitative data across several levels of one (or more) categorical variables such that those distributions can be compared. 
		# Unlike a box plot, in which all of the plot components correspond to actual datapoints, the violin plot features a kernel density estimation of the underlying distribution.


		plt.figure(figsize=(12,6))
		sns.violinplot(x='parental level of education',y='math score',data=self.df2)
		plt.title("6. violinplot")
		plt.show()
		plt.figure(figsize=(12,6))
		sns.violinplot(x='parental level of education',y='math score',data=self.df2,hue='gender')
		plt.title("7. violinplot with 'hue'")
		plt.show()


		### Violinplot Parameters
		#### split
		# When using hue nesting with a variable that takes two levels, setting split to True will draw half of a violin for each level. This can make it easier to directly compare the distributions.

		plt.figure(figsize=(12,6))
		sns.violinplot(x='parental level of education',y='math score',data=self.df2,hue='gender',split=True)
		plt.title("8. violinplot with 'hue' and 'split'")
		plt.show()
		#### inner

		# Representation of the datapoints in the violin interior. If 'box', draw a miniature boxplot. 
		# If 'quartiles', draw the quartiles of the distribution. If 'point' or 'stick', show each underlying datapoint. Using 'None' will draw unadorned violins.
		plt.figure(figsize=(12,6))
		sns.violinplot(x='parental level of education',y='math score',data=self.df2,inner=None)
		plt.title("9. violinplot with inner=None")
		plt.show()
		plt.figure(figsize=(12,6))
		sns.violinplot(x='parental level of education',y='math score',data=self.df2,inner='box')
		plt.title("10. violinplot with inner='box'")
		plt.show()
		plt.figure(figsize=(12,6))
		sns.violinplot(x='parental level of education',y='math score',data=self.df2,inner='quartile')
		plt.title("11. violinplot with inner='quartile'")
		plt.show()
		plt.figure(figsize=(12,6))
		sns.violinplot(x='parental level of education',y='math score',data=self.df2,inner='stick')
		plt.title("12. violinplot with inner='stick'")
		plt.show()
		#### orientation
		# Simply switch the continuous variable to y and the categorical to x
		sns.violinplot(x='math score',y='parental level of education',data=self.df2,bw_adjust=0.1)
		plt.title("13. violinplot diff orientation")
		plt.show()
		#### bandwidth
		# Similar to bandwidth argument for kdeplot
		plt.figure(figsize=(12,6))
		sns.violinplot(x='parental level of education',y='math score',data=self.df2,bw_adjust=0.1)
		plt.title("14. violinplot with bw=0.1")
		plt.show()
		#### Advanced Plots
		# We can use a boxenplot and swarmplot to achieve the same effect as the boxplot and violinplot, but with slightly more information included. 
		# Be careful when using these plots, as they often require you to educate the viewer with how the plot is actually constructed. 
		# Only use these if you are sure your audience will understand the visualization.

	def categorical_swarmplot(self):
		### swarmplot
		sns.swarmplot(x='math score',data=self.df2)
		plt.title("15. 1D swarmplot")
		plt.show()
		sns.swarmplot(x='math score',data=self.df2,size=2)
		plt.title("16. 1D swarmplot with size=2")
		plt.show()
		sns.swarmplot(x='math score',y='race/ethnicity',data=self.df2,size=3)
		plt.title("17. 2D swarmplot with size=3")
		plt.show()
		sns.swarmplot(x='race/ethnicity',y='math score',data=self.df2,size=3)
		plt.title("18. 2D swarmplot diff orientation")
		plt.show()
		plt.figure(figsize=(12,6))
		sns.swarmplot(x='race/ethnicity',y='math score',data=self.df2,hue='gender')
		plt.title("19. 3D swarmplot with hue='gender'")
		plt.show()
		plt.figure(figsize=(12,6))
		sns.swarmplot(x='race/ethnicity',y='math score',data=self.df2,hue='gender',dodge=True)
		plt.title("20. 3D swarmplot with hue='gender' and dodge=True")
		plt.show()

	def categorical_boxenplot(self):
		#### boxenplot (letter-value plot)
		# This style of plot was originally named a “letter value” plot because it shows a large number of quantiles that are defined as “letter values”. 
		# It is similar to a box plot in plotting a nonparametric representation of a distribution in which all features correspond to actual observations. 
		# By plotting more quantiles, it provides more information about the shape of the distribution, particularly in the tails.

		sns.boxenplot(x='math score',y='race/ethnicity',data=self.df2)
		plt.title("21. 2D boxenplot")
		plt.show()
		sns.boxenplot(x='race/ethnicity',y='math score',data=self.df2)
		plt.title("22. 2D boxenplot")
		plt.show()
		plt.figure(figsize=(12,6))
		sns.boxenplot(x='race/ethnicity',y='math score',data=self.df2,hue='gender')
		plt.title("23. 3D boxenplot with hue='gender'")
		plt.show()

	def comparison_jointplot(self):
		sns.jointplot(x='math score',y='reading score',data=self.df2)
		plt.title("24. jointplot")
		plt.show()
		sns.jointplot(x='math score',y='reading score',data=self.df2,kind='hex')
		plt.title("25. jointplot with kind='hex'")
		plt.show()
		sns.jointplot(x='math score',y='reading score',data=self.df2,kind='kde')
		plt.title("26. jointplot with kind='kde")
		plt.show()

		sns.jointplot(x='math score',y='reading score',data=self.df2,hue='gender')
		plt.title("27. jointplot with hue='gender'")
		plt.show()


		sns.jointplot(x='math score',y='reading score',data=self.df2,hue='gender', kind='kde')
		plt.title("28. jointplot with hue and kind='kde")
		plt.show()


	def comparison_pairplot(self):
		# Can be CPU and RAM intensive specially for large dataframes with many columns.
		# So, better to filter down to only the columns you are interested in before calling pairplot() to full dataframe.

		sns.pairplot(self.df2)
		plt.title("29. pairplot")
		plt.show()
		sns.pairplot(self.df2,hue='gender',palette='viridis')
		plt.title("30. pairplot with hue and palette")
		plt.show()
		sns.pairplot(self.df2,hue='gender',palette='viridis',corner=True)
		plt.title("31. pairplot, corner=True remove dups")
		plt.show()
		sns.pairplot(self.df2,hue='gender',palette='viridis',diag_kind='hist')
		plt.title("32. pairplot with hue, palette, diag_kind='hist'")
		plt.show()

	def catplot(self):

		# Kind Options are: “point”, “bar”, “strip”, “swarm”, “box”, “violin”, or “boxen”
		sns.catplot(x='gender',y='math score',data=self.df2,kind='box') # like a generalized version of 'boxplot'
		plt.show()

		sns.catplot(x='gender',y='math score',data=self.df2,kind='box',row='lunch') # like hue='lunch' shown below
		plt.show()
		sns.catplot(x='gender',y='math score',data=self.df2,kind='box',hue='lunch')
		plt.title("35. with kind=box, hue='lunch'")
		plt.show()
		sns.catplot(x='gender',y='math score',data=self.df2,kind='box',col='lunch')
		plt.show()

		sns.catplot(x='gender',y='math score',data=self.df2,kind='box',row='lunch',col='test preparation course')
		plt.show()

		sns.catplot(x='gender',y='math score',data=self.df2,kind='box',row='lunch',col='race/ethnicity')
		plt.show()

	def pair_grid(self):
		g = sns.PairGrid(self.df2)
		g = g.map_upper(sns.scatterplot)
		g = g.map_diag(sns.kdeplot)
		g = g.map_lower(sns.kdeplot)
		plt.show()

		g = sns.PairGrid(self.df2, hue="gender")
		g = g.map_upper(sns.scatterplot)
		g = g.map_diag(sns.kdeplot)
		g = g.map_lower(sns.kdeplot)
		g.add_legend()
		plt.show()

		g = sns.PairGrid(self.df2)
		g = g.map_upper(sns.scatterplot)
		g = g.map_diag(sns.histplot, lw=2)
		g = g.map_lower(sns.kdeplot, colors="red")
		plt.show()

		g = sns.PairGrid(self.df2, hue="gender", palette="viridis",hue_kws={"marker": ["o", "+"]})
		g = g.map_upper(sns.scatterplot, linewidths=1, edgecolor="w", s=40)
		g = g.map_diag(sns.histplot)
		g = g.map_lower(sns.kdeplot)
		g = g.add_legend()
		plt.show()

	def facet_grid(self):
		sns.FacetGrid(data=self.df2,col='gender',row='lunch')
		plt.show()

		g = sns.FacetGrid(data=self.df2,col='gender',row='lunch')
		g = g.map(plt.scatter, "math score", "reading score", edgecolor="w")
		g.add_legend()
		plt.show()

		g = sns.FacetGrid(data=self.df2,col='gender',row='lunch')
		g = g.map(plt.scatter, "math score", "reading score", edgecolor="w")
		g.add_legend()
		plt.subplots_adjust(hspace=0.4, wspace=1)
		plt.show()

	def heat_map(self):
		"""
		NOTE: pass dataframe only containing numerical data, otherwise won't get the right visualization matrix
			  If not sure about the data types, you can pass df.corr() to the heatmap as .corr() will extract
			  only the numerical data, and pass the correlation matric to the heatmap.

			  sns.heatmap(self.df3.corr())
			  cmap='virdis' and also 'annot=True' to display the values'
			  	>> sns.heatmap(self.df3.corr(), cmap='viridis', annot=True)

		"""
		self.df3 = self.df3.set_index('Countries')
		print("rates after setting index to Countries: \n", self.df3)

		sns.heatmap(self.df3)
		plt.title("heatmap without filtering data")
		plt.show()

		rates = self.df3.drop('Life expectancy',axis=1)

		sns.heatmap(rates)
		plt.title("heatmap after dropping 'Life expectancy'")
		plt.show()

		sns.heatmap(rates,linewidth=0.5)
		plt.title("heatmap with linewidth=0.5")
		plt.show()

		sns.heatmap(rates,linewidth=0.5,annot=True)
		plt.title("heatmap with linewidth and annot=True")
		plt.show()

		# Note how its not palette here
		sns.heatmap(rates,linewidth=0.5,annot=True,cmap='viridis')
		plt.title("heatmap with linewidth, annot, cmap='viridis'")
		plt.show()

		# Set colorbar based on value from dataset
		sns.heatmap(rates,linewidth=0.5,annot=True,cmap='viridis',center=40)
		plt.title("heatmap with linewidth, annot, cmap, and center=40")
		plt.show()

		# Set colorbar based on value from dataset
		sns.heatmap(rates,linewidth=0.5,annot=True,cmap='viridis',center=1)
		plt.title("heatmap, same but center=1")
		plt.show()

		sns.clustermap(rates,col_cluster=False)
		plt.title("clustermap with col_cluster=False")
		plt.show()

		sns.clustermap(rates,col_cluster=False,figsize=(12,8),cbar_pos=(-0.1, .2, .03, .4))
		plt.title("clustermap with col_cluster=False, figsize, cbar_pos")

		plt.show()









if __name__ == "__main__":
	seabornPlots = SeabornPlots()
	# seabornPlots.rugplot_1D()
	# seabornPlots.displot_and_histplot()
	# seabornPlots.kdeplot()
	# seabornPlots.scatterplot()
	# seabornPlots.categorical_countplot()
	# seabornPlots.categorical_barplot()
	# seabornPlots.categorical_boxplot()
	# seabornPlots.categorical_violinplot()
	# seabornPlots.categorical_swarmplot()
	# seabornPlots.categorical_boxenplot()
	# seabornPlots.comparison_jointplot()
	# seabornPlots.comparison_pairplot()
	# seabornPlots.catplot()
	# seabornPlots.pair_grid()
	# seabornPlots.facet_grid()
	seabornPlots.heat_map()



