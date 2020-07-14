import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np


#plt.figure()    
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
line = plt.Line2D((0, 0), (0, 42.1), lw=2.5)
line1 = plt.Line2D((0, 16.35), (42.1, 53.65), lw=2.5)
line2 = plt.Line2D((16.35, 33.15), (53.65, -4.90), lw=2.5)
line3 = plt.Line2D((33.15, 12), (-4.90, -4.90), lw=2.5)
line4 = plt.Line2D((12, 12), (-4.90, 0), lw=2.5, ls='-' )
line5 = plt.Line2D((12, 0), (0, 0), lw=2.5)
line6 = plt.Line2D((13.35, 30.15), (52, -4.90), lw=2.5, ls=':' )
line7 = plt.Line2D((3,3), (0, 42.1), lw=2.5, ls=':' )
line8 = plt.Line2D((10.85,10.85), (0, 39), lw=2.5, ls=':' )
line9 = plt.Line2D((10.85,19.3), (39, 0), lw=2.5, ls=':' )
line10 = plt.Line2D((19.3,12), (0, 0), lw=2.5, ls=':' )
plt.gca().add_line(line)
plt.gca().add_line(line1)
plt.gca().add_line(line2)
plt.gca().add_line(line3)
plt.gca().add_line(line4)
plt.gca().add_line(line5)
plt.gca().add_line(line6)
plt.gca().add_line(line7)
plt.gca().add_line(line8)
plt.gca().add_line(line9)
plt.gca().add_line(line10)

#plt.axis([0, 16, 0, 16])     
plt.grid(False)                         # set the grid


ax=plt.gca()                            # get the axis
ax.xaxis.set_ticks(np.arange(0, 55, 5))
ax.set_xlim(ax.get_xlim()[::-1])        # invert the axis
#ax.yaxis.tick_left()                     # and move the X-Axis      
ax.yaxis.set_ticks(np.arange(-10, 60, 5)) # set y-ticks
ax.yaxis.tick_right()                    # remove right y-Ticks

data_set_dir = 'D:\\Project\\Dissertation\\example.txt'
def animate(i):
    graph_data = open(data_set_dir,'r').read()
    lines = graph_data.split('\n')
    xs = []
    ys = []
    for line in lines:
        if len(line) > 1:
            x, y = line.split(',')
            xs.append(float(x))
            ys.append(float(y))
    #plt.clear()
    plt.scatter(xs, ys)
ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()
