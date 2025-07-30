

import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



class PlottingResiduals:
	def __init__(self):
		pass

	

	def quartet1(self):
		quartet = pd.read_csv("/Users/tess/Desktop/MLE2025/projects/UNZIP_FOR_NOTEBOOKS_FINAL/DATA/anscombes_quartet/anscombes_quartet1.csv")
		self.visualize(quartet)

	def quartet2(self):
		quartet = pd.read_csv('/Users/tess/Desktop/MLE2025/projects/UNZIP_FOR_NOTEBOOKS_FINAL/DATA/anscombes_quartet/anscombes_quartet2.csv')
		self.visualize(quartet)


	def quartet4(self):
		quartet = pd.read_csv('/Users/tess/Desktop/MLE2025/projects/UNZIP_FOR_NOTEBOOKS_FINAL/DATA/anscombes_quartet/anscombes_quartet4.csv')
		self.visualize(quartet)

	def visualize(self, quartet):
		quartet['pred_y'] = 3 + 0.5 * quartet['x']
		quartet['residual'] = quartet['y'] - quartet['pred_y']

		sns.scatterplot(data=quartet,x='x',y='y')
		sns.lineplot(data=quartet,x='x',y='pred_y',color='red')
		plt.vlines(quartet['x'],quartet['y'],quartet['y']-quartet['residual']) # plot #1
		plt.show()

		sns.kdeplot(quartet['residual']) # plot #2
		plt.show()

		sns.scatterplot(data=quartet,x='y',y='residual')
		plt.axhline(y=0, color='r', linestyle='--') # plot #3
		plt.show()

		
		"""
		Plotting Residuals
		It's also important to plot out residuals and check for normal distribution, this helps us understand if Linear Regression was a valid model choice.

		"""
		# test_predictions = model.predict(X_test) # =============== No Model so commenting it out ===============
		# test_res = y_test - test_predictions
		# sns.scatterplot(x=y_test,y=test_res)
		# plt.axhline(y=0, color='r', linestyle='--') # plot #4
		# plt.show()

		# print("After Plot #4")
		# sns.displot(test_res,bins=25,kde=True) # plot #5
		# plt.show()

		# print("After Plot #5")
		# fig, ax = plt.subplots(figsize=(6,8),dpi=100)

		# _ = sp.stats.probplot(test_res,plot=ax)



if __name__ == "__main__":
	plottingResiduals = PlottingResiduals()
	plottingResiduals.quartet1()
	plottingResiduals.quartet2()
	plottingResiduals.quartet4()
	
