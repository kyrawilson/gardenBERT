import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="white")
arg0_pallete = sns.light_palette("#BB5566", as_cmap=True)
arg1_pallete = sns.light_palette("#DDAA33", as_cmap=True)
sns.set(font_scale=2)
bbox_inches='tight'

df = pd.read_csv("../hyperparameter_results.tsv", delimiter="\t", header=0)

arg0_scores = df.pivot_table('arg0_score', 'sample_size', 'weight_decay')[::-1]#.drop(index=50)
arg0_selecs = df.pivot_table('arg0_selec', 'sample_size', 'weight_decay')[::-1]#.drop(index=50)
arg1_scores = df.pivot_table('arg1_score', 'sample_size', 'weight_decay')[::-1]#.drop(index=50)
arg1_selecs = df.pivot_table('arg1_selec', 'sample_size', 'weight_decay')[::-1]#.drop(index=50)

arg0_score_plot = sns.heatmap(arg0_scores, annot=True, vmin=.5, vmax=1, cmap=arg0_pallete, square=True)
arg0_score_plot.set_title('Agent Classifier Accuracy')
arg0_score_plot.set_xlabel('Regularization Constant')
arg0_score_plot.set_ylabel('Training Size')
arg0_score_plot.figure.savefig("arg0_score.png")
plt.figure().clear()

arg0_selec_plot = sns.heatmap(arg0_selecs, annot=True, vmin=-0.1, vmax=0.5, cmap=arg0_pallete, square=True)
arg0_selec_plot.set_title('Agent Classifier Selectivity')
arg0_selec_plot.set_xlabel('Regularization Constant')
arg0_selec_plot.set_ylabel('Training Size')
arg0_selec_plot.figure.savefig("arg0_selec.png")
plt.figure().clear()

arg1_score_plot = sns.heatmap(arg1_scores, annot=True, vmin=.5, vmax=1, cmap=arg1_pallete, square=True)
arg1_score_plot.set_title('Goal Classifier Accuracy')
arg1_score_plot.set_xlabel('Regularization Constant')
arg1_score_plot.set_ylabel('Training Size')
arg1_score_plot.figure.savefig("arg1_score.png")
plt.figure().clear()

arg1_selec_plot = sns.heatmap(arg1_selecs, annot=True, vmin=-0.1, vmax=0.5, cmap=arg1_pallete,square=True)
arg1_selec_plot.set_title('Goal Classifier Selectivity')
arg1_selec_plot.set_xlabel('Regularization Constant')
arg1_selec_plot.set_ylabel('Training Size')

arg1_selec_plot.figure.savefig("arg1_selec.png")
plt.figure().clear()
