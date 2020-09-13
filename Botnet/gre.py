import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
students=[1,2,1,4]
langs=['A','B','A','B']
y_pos = np.arange(len(langs))
c=[]
for i in range(len(langs)):
    if i%2==0:
        c.append('y')
    else:
        c.append('b')
plt.bar(y_pos, students, color=c)
plt.xticks(y_pos, langs)

legend_elements = [Patch(facecolor='y', edgecolor='g',
                     label='SVM'),
               Patch(facecolor='b', edgecolor='r',
                     label='NB')]

# Create the figure
plt.legend(handles=legend_elements, loc='upper right')
plt.show()
