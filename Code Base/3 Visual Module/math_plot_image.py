import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

img = mpimg.imread("D:\\Project\\Dissertation\\floor_plan_samp.png")
print(img)
plt.imshow(img, extent=[33.15,0,-4.90,42.1])
ax=plt.gca()  
ax.yaxis.tick_right()        
plt.grid(False)                        

plt.show()
