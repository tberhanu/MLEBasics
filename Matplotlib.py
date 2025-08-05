import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d.axes3d import Axes3D

class MatPlotlibBasics:
	"""
	Didn't cover statistical plots yet, like histograms or scatterplots. 
	Will use the seaborn library to create those plots instead. 
	Matplotlib is capable of creating those plots, but seaborn is easier to use (and built on top of matplotlib!).

	"""
	def __init__(self):
		pass

	def plt_function(self):
		x = np.arange(0,10)
		y = 2*x

		plt.plot(x, y) 
		plt.xlabel('X Axis Title Here')
		plt.ylabel('Y Axis Title Here')
		plt.title('String Title Here')
		plt.savefig('saved_example3.png')
		plt.show()


		

	def plt_object(self):
		x = np.arange(0,10)	
		y = 2*x	
		fig = plt.figure()# Create Figure (empty canvas)
		axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # left, bottom, width, height (range 0 to 1)
		axes.plot(x, y)# Plot on that set of axes
		plt.show()

		a = np.linspace(0,10,11)
		b = a ** 4
		fig = plt.figure()
		# axes = fig.add_axes([0, 0, 1, 1]) # tick numbers not visible here
		axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # Added padding
		axes.plot(a, b)
		axes.set_xlabel("X-axis")
		axes.set_ylabel("Y-axis")
		axes.set_title("Plot of a⁴ vs. a")

		plt.show()

	def multiple_axes(self):
		a = np.linspace(0,10,11)
		b = a ** 4
		# Creates blank canvas
		fig1 = plt.figure()

		axes1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8]) # Large figure
		axes2 = fig1.add_axes([0.2, 0.2, 0.5, 0.5]) # Smaller figure

		# Larger Figure Axes 1
		axes1.plot(a, b)
		# Use set_ to add to the axes figure
		axes1.set_xlabel('X Label')
		axes1.set_ylabel('Y Label')
		axes1.set_title('Big Figure')

		# Insert Figure Axes 2
		axes2.plot(a,b)
		axes2.set_title('Small Figure')
		plt.show()

		# You can add as many axes on the same figure as you want, even outside of the main figure.
		fig2 = plt.figure()

		axes1 = fig2.add_axes([0.1, 0.1, 1, 1]) # Full figure
		axes2 = fig2.add_axes([0.2, 0.5, 0.25, 0.25]) # Smaller figure
		# If left + width > 1 or bottom + height > 1, parts (or all) of the axes go off-screen.
		# axes3 = fig2.add_axes([1, 1, 0.25, 0.25]) # Starts at top right corner!
		axes3 = fig2.add_axes([0.74, 0.74, 0.25, 0.25])

		# Larger Figure Axes 1
		axes1.plot(a, b)
		# Use set_ to add to the axes figure
		axes1.set_xlabel('X Label')
		axes1.set_ylabel('Y Label')
		axes1.set_title('Big Figure')

		# Insert Figure Axes 2
		axes2.plot(a,b)
		axes2.set_xlim(8,10)
		axes2.set_ylim(4000,10000)
		axes2.set_xlabel('X')
		axes2.set_ylabel('Y')
		axes2.set_title('Zoomed In')

		# Insert Figure Axes 3
		axes3.plot(a,b)
		plt.show()

	def zoom_in_axes(self):
		a = np.linspace(0,10,11)
		b = a ** 4
		# Creates blank canvas
		fig = plt.figure()

		axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # Large figure
		axes2 = fig.add_axes([0.2, 0.5, 0.25, 0.25]) # Smaller figure

		# Larger Figure Axes 1
		axes1.plot(a, b)

		# Use set_ to add to the axes figure
		axes1.set_xlabel('X Label')
		axes1.set_ylabel('Y Label')
		axes1.set_title('Big Figure')

		# Insert Figure Axes 2
		axes2.plot(a,b)
		axes2.set_xlim(8,10)
		axes2.set_ylim(4000,10000)
		axes2.set_xlabel('X')
		axes2.set_ylabel('Y')
		axes2.set_title('Zoomed In')
		plt.show()

	def fig_params(self):
		a = np.linspace(0,10,11)
		b = a ** 4
		fig = plt.figure(figsize=(12,8),dpi=100)
		axes1 = fig.add_axes([0.1, 0.1, 0.9, 0.9])
		axes1.plot(a,b)
		axes1.set_xlabel('X')
		axes1.set_ylabel('Y')
		plt.show()
		fig.savefig('myfigure.png',bbox_inches='tight')


	def subplots(self):
		a = np.linspace(0,10,11)
		b = a ** 4
		x = np.arange(0,10)
		y = 2 * x

		fig,axes = plt.subplots(nrows=2,ncols=2,figsize=(12,8)) # 2 by 2, total of 4 plots

		# SET YOUR AXES PARAMETERS FIRST

		# Parameters at the axes level
		axes[0][0].plot(a,b)
		axes[0][0].set_title('(0, 0) axes level Title')


		axes[1][1].plot(x,y)
		axes[1][1].set_xlabel('(1, 1) axes Label')

		axes[0][1].plot(y,x)
		axes[1][0].plot(b,a)

		# THEN SET OVERALL FIGURE PARAMETERS

		# Parameters at the Figure level
		fig.suptitle("Figure Level Title",fontsize=16)

		# Use left,right,top, bottom to stretch subplots
		# Use wspace,hspace to add spacing between subplots
		fig.subplots_adjust(left=None,
		    bottom=None,
		    right=None,
		    top=None,
		    wspace=0.5,
		    hspace=0.1,)

		plt.show()
		fig.savefig('mysubplots.png',bbox_inches='tight')

	def legends(self):
		"""
		Legends Location:
			ax.legend(loc=1) # upper right corner
			ax.legend(loc=2) # upper left corner
			ax.legend(loc=3) # lower left corner
			ax.legend(loc=4) # lower right corner

		Most common to choose:
			ax.legend(loc=0) # let matplotlib decide the optimal location

		"""
		x = np.arange(0,10)
		y = 2 * x

		fig = plt.figure()
		ax = fig.add_axes([0.1,0.1,0.9,0.9])
		ax.plot(x, x**2, label="x**2")
		ax.plot(x, x**3, label="x**3")
		ax.legend(loc=0)
		plt.title("Legends Location")
		plt.show()

	def styling_colors_linewidths_linetypes(self):
		x = np.arange(0,10)
		y = 2 * x

		# MATLAB style line color and style 
		fig, ax = plt.subplots()
		ax.plot(x, x**2, 'b.-') # blue line with dots
		ax.plot(x, x**3, 'g--') # green dashed line
		plt.title("blue green")
		plt.show()

		fig, ax = plt.subplots()
		ax.plot(x, x+1, color="blue", alpha=0.5) # half-transparant
		ax.plot(x, x+2, color="#8B008B")        # RGB hex code
		ax.plot(x, x+3, color="#FF8C00")        # RGB hex code 
		plt.show("color RGB hex code")
		plt.show()

		fig, ax = plt.subplots(figsize=(12,6))
		# Use linewidth or lw
		ax.plot(x, x-1, color="red", linewidth=0.25)
		ax.plot(x, x-2, color="red", lw=0.50)
		ax.plot(x, x-3, color="red", lw=1)
		ax.plot(x, x-4, color="red", lw=10)
		plt.show("lw=linewidth")
		plt.show()

		# possible linestype options ‘--‘, ‘–’, ‘-.’, ‘:’, ‘steps’
		fig, ax = plt.subplots(figsize=(12,6))
		ax.plot(x, x-1, color="green", lw=3, linestyle='-') # solid
		ax.plot(x, x-2, color="green", lw=3, ls='-.') # dash and dot
		ax.plot(x, x-3, color="green", lw=3, ls=':') # dots
		ax.plot(x, x-4, color="green", lw=3, ls='--') # dashes
		plt.title("ls=linestyle")
		plt.show()

	def custom_linestyle(self):
		x = np.arange(0,10)
		y = 2 * x

		fig, ax = plt.subplots(figsize=(12,6))
		lines = ax.plot(x,x)
		print("type(lines) is LIST: ", type(lines))
		plt.title("About to add more linestyle")
		plt.show()

		fig, ax = plt.subplots(figsize=(12,6))
		# custom dash
		lines = ax.plot(x, x+8, color="black", lw=5)
		lines[0].set_dashes([10, 10]) # format: line length, space length
		plt.title("set_dashes")
		plt.show()

		fig, ax = plt.subplots(figsize=(12,6))
		# custom dash
		lines = ax.plot(x, x+8, color="black", lw=5)
		lines[0].set_dashes([1, 1,1,1,10,10]) # format: line length, space length
		plt.title("set_dashes2")
		plt.show()

		fig, ax = plt.subplots(figsize=(12,6))
		# Use marker for string code
		# Use markersize or ms for size
		ax.plot(x, x-1,marker='+',markersize=20)
		ax.plot(x, x-2,marker='o',ms=20) #ms can be used for markersize
		ax.plot(x, x-3,marker='s',ms=20,lw=0) # make linewidth zero to see only markers
		ax.plot(x, x-4,marker='1',ms=20)
		plt.title("ms=markersize")
		plt.show()

		fig, ax = plt.subplots(figsize=(12,6))
		# marker size and color
		ax.plot(x, x, color="black", lw=1, ls='-', marker='s', markersize=20, 
		        markerfacecolor="red", markeredgewidth=8, markeredgecolor="blue")
		plt.title("more styles added")
		plt.show()

	def advanced_cmds(self):
		x = np.arange(0,10)

		fig, axes = plt.subplots(1, 2, figsize=(10,4))
		axes[0].plot(x, x**2, x, np.exp(x))
		axes[0].set_title("Normal scale")
		axes[1].plot(x, x**2, x, np.exp(x))
		axes[1].set_yscale("log")
		axes[1].set_title("Logarithmic scale (y)")
		plt.show()

		fig, ax = plt.subplots(figsize=(10, 4))
		ax.plot(x, x**2, x, x**3, lw=2)
		ax.set_xticks([1, 2, 3, 4, 5])
		ax.set_xticklabels([r'$\alpha$', r'$\beta$', r'$\gamma$', r'$\delta$', r'$\epsilon$'], fontsize=18)
		yticks = [0, 50, 100, 150]
		ax.set_yticks(yticks)
		ax.set_yticklabels(["$%.1f$" % y for y in yticks], fontsize=18); # use LaTeX formatted labels
		plt.title("alpha, beta, gamma, delta, epsilon")
		plt.show()

		fig, ax = plt.subplots(1, 1)
		ax.plot(x, x**2, x, np.exp(x))
		ax.set_title("scientific notation")
		ax.set_yticks([0, 50, 100, 150])
		formatter = ticker.ScalarFormatter(useMathText=True) # ticker imported
		formatter.set_scientific(True) 
		formatter.set_powerlimits((-1,1)) 
		ax.yaxis.set_major_formatter(formatter) 
		plt.show()

		# distance between x and y axis and the numbers on the axes
		matplotlib.rcParams['xtick.major.pad'] = 5
		matplotlib.rcParams['ytick.major.pad'] = 5

		fig, ax = plt.subplots(1, 1)
		ax.plot(x, x**2, x, np.exp(x))
		ax.set_yticks([0, 50, 100, 150])
		ax.set_title("label and axis spacing")
		# padding between axis label and axis numbers
		ax.xaxis.labelpad = 5
		ax.yaxis.labelpad = 5
		ax.set_xlabel("x")
		ax.set_ylabel("y")
		plt.show()


		# restore defaults
		matplotlib.rcParams['xtick.major.pad'] = 3
		matplotlib.rcParams['ytick.major.pad'] = 3

		fig, ax = plt.subplots(1, 1)
		ax.plot(x, x**2, x, np.exp(x))
		ax.set_yticks([0, 50, 100, 150])
		ax.set_title("title")
		ax.set_xlabel("x")
		ax.set_ylabel("y")
		fig.subplots_adjust(left=0.15, right=.9, bottom=0.1, top=0.9)
		plt.show()

		fig, axes = plt.subplots(1, 2, figsize=(10,3))
		# default grid appearance
		axes[0].plot(x, x**2, x, x**3, lw=2)
		axes[0].grid(True)
		# custom grid appearance
		axes[1].plot(x, x**2, x, x**3, lw=2)
		axes[1].grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
		plt.show()

		fig, ax = plt.subplots(figsize=(6,2))
		ax.spines['bottom'].set_color('blue')
		ax.spines['top'].set_color('blue')
		ax.spines['left'].set_color('red')
		ax.spines['left'].set_linewidth(2)
		# turn off axis spine to the right
		ax.spines['right'].set_color("none")
		ax.yaxis.tick_left() # only ticks on the left side
		plt.show()

		fig, ax1 = plt.subplots()
		ax1.plot(x, x**2, lw=2, color="blue")
		ax1.set_ylabel(r"area $(m^2)$", fontsize=18, color="blue")
		for label in ax1.get_yticklabels():
		    label.set_color("blue")
		    
		ax2 = ax1.twinx()
		ax2.plot(x, x**3, lw=2, color="red")
		ax2.set_ylabel(r"volume $(m^3)$", fontsize=18, color="red")
		for label in ax2.get_yticklabels():
		    label.set_color("red")
		plt.title("TWIN AXES")
		plt.show()

		fig, ax = plt.subplots()
		ax.spines['right'].set_color('none')
		ax.spines['top'].set_color('none')
		ax.xaxis.set_ticks_position('bottom')
		ax.spines['bottom'].set_position(('data',0)) # set position of x spine to x=0
		ax.yaxis.set_ticks_position('left')
		ax.spines['left'].set_position(('data',0))   # set position of y spine to y=0
		xx = np.linspace(-0.75, 1., 100)
		ax.plot(xx, xx**3)
		plt.title("Axes where x and y is zero")
		plt.show()

		### Other 2D plot styles
		# In addition to the regular `plot` method, there are a number of other functions for generating different kind of plots. 
		# See the matplotlib plot gallery for a complete list of available plot types: http://matplotlib.org/gallery.html. Some of the more useful ones are show below:
		n = np.array([0,1,2,3,4,5])
		fig, axes = plt.subplots(1, 4, figsize=(12,3))
		axes[0].scatter(xx, xx + 0.25*np.random.randn(len(xx)))
		axes[0].set_title("scatter")
		axes[1].step(n, n**2, lw=2)
		axes[1].set_title("step")
		axes[2].bar(n, n**2, align="center", width=0.5, alpha=0.5)
		axes[2].set_title("bar")
		axes[3].fill_between(x, x**2, x**3, color="green", alpha=0.5);
		axes[3].set_title("fill_between")
		plt.show()

		fig, ax = plt.subplots()
		ax.plot(xx, xx**2, xx, xx**3)
		ax.text(0.15, 0.2, r"$y=x^2$", fontsize=20, color="blue")
		ax.text(0.65, 0.1, r"$y=x^3$", fontsize=20, color="green")
		plt.title("Text Annotation")
		plt.show()

		### Figures with multiple subplots and insets
		#### subplots
		fig, ax = plt.subplots(2, 3)
		fig.tight_layout()
		plt.show()
		#### subplot2grid
		fig = plt.figure()
		ax1 = plt.subplot2grid((3,3), (0,0), colspan=3)
		ax2 = plt.subplot2grid((3,3), (1,0), colspan=2)
		ax3 = plt.subplot2grid((3,3), (1,2), rowspan=2)
		ax4 = plt.subplot2grid((3,3), (2,0))
		ax5 = plt.subplot2grid((3,3), (2,1))
		fig.tight_layout()

		#### gridspec
		fig = plt.figure()
		gs = gridspec.GridSpec(2, 3, height_ratios=[2,1], width_ratios=[1,2,1])
		for g in gs:
		    ax = fig.add_subplot(g)		    
		fig.tight_layout()

		#### add_axes
		fig, ax = plt.subplots()
		ax.plot(xx, xx**2, xx, xx**3)
		fig.tight_layout()
		# inset
		inset_ax = fig.add_axes([0.2, 0.55, 0.35, 0.35]) # X, Y, width, height
		inset_ax.plot(xx, xx**2, xx, xx**3)
		inset_ax.set_title('zoom near origin')
		# set axis range
		inset_ax.set_xlim(-.2, .2)
		inset_ax.set_ylim(-.005, .01)
		# set axis tick locations
		inset_ax.set_yticks([0, 0.005, 0.01])
		inset_ax.set_xticks([-0.1,0,.1])
		plt.show()

		### Colormap and contour figures
		alpha = 0.7
		phi_ext = 2 * np.pi * 0.5

		def flux_qubit_potential(phi_m, phi_p):
		    return 2 + alpha - 2 * np.cos(phi_p) * np.cos(phi_m) - alpha * np.cos(phi_ext - 2*phi_p)

		phi_m = np.linspace(0, 2*np.pi, 100)
		phi_p = np.linspace(0, 2*np.pi, 100)
		X,Y = np.meshgrid(phi_p, phi_m)
		Z = flux_qubit_potential(X, Y).T

		#### pcolor
		fig, ax = plt.subplots()
		p = ax.pcolor(X/(2*np.pi), Y/(2*np.pi), Z, cmap=matplotlib.cm.RdBu, vmin=abs(Z).min(), vmax=abs(Z).max())
		cb = fig.colorbar(p, ax=ax)

		#### imshow
		fig, ax = plt.subplots()
		im = ax.imshow(Z, cmap=matplotlib.cm.RdBu, vmin=abs(Z).min(), vmax=abs(Z).max(), extent=[0, 1, 0, 1])
		im.set_interpolation('bilinear')
		cb = fig.colorbar(im, ax=ax)

		#### contour
		fig, ax = plt.subplots()
		cnt = ax.contour(Z, cmap=matplotlib.cm.RdBu, vmin=abs(Z).min(), vmax=abs(Z).max(), extent=[0, 1, 0, 1])
		plt.show()

		## 3D figures
		"""
		To use 3D graphics in matplotlib, we first need to create an instance of the `Axes3D` class. 
		3D axes can be added to a matplotlib figure canvas in exactly the same way as 2D axes; 
		or, more conveniently, by passing a `projection='3d'` keyword argument to the `add_axes` or `add_subplot` methods.

		"""

		#### Surface plots
		fig = plt.figure(figsize=(14,6))
		# `ax` is a 3D-aware axis instance because of the projection='3d' keyword argument to add_subplot
		ax = fig.add_subplot(1, 2, 1, projection='3d')
		p = ax.plot_surface(X, Y, Z, rstride=4, cstride=4, linewidth=0)
		# surface_plot with color grading and color bar
		ax = fig.add_subplot(1, 2, 2, projection='3d')
		p = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, linewidth=0, antialiased=False)
		cb = fig.colorbar(p, shrink=0.5)
		plt.show()

		#### Wire-frame plot
		fig = plt.figure(figsize=(8,6))
		ax = fig.add_subplot(1, 1, 1, projection='3d')
		p = ax.plot_wireframe(X, Y, Z, rstride=4, cstride=4)
		plt.show()

		#### Coutour plots with projections
		fig = plt.figure(figsize=(8,6))

		ax = fig.add_subplot(1,1,1, projection='3d')

		ax.plot_surface(X, Y, Z, rstride=4, cstride=4, alpha=0.25)
		cset = ax.contour(X, Y, Z, zdir='z', offset=-np.pi, cmap=matplotlib.cm.coolwarm)
		cset = ax.contour(X, Y, Z, zdir='x', offset=-np.pi, cmap=matplotlib.cm.coolwarm)
		cset = ax.contour(X, Y, Z, zdir='y', offset=3*np.pi, cmap=matplotlib.cm.coolwarm)

		ax.set_xlim3d(-np.pi, 2*np.pi);
		ax.set_ylim3d(0, 3*np.pi);
		ax.set_zlim3d(-np.pi, 2*np.pi);
		plt.show()





if __name__ == "__main__":
	mat = MatPlotlibBasics()
	# matplotlib.plt_function()
	# matplotlib.plt_object()
	# matplotlib.multiple_axes()
	# matplotlib.zoom_in_axes()
	# matplotlib.fig_params()
	# matplotlib.subplots()
	# matplotlib.legends()
	# matplotlib.custom_linestyle()
	mat.advanced_cmds()




