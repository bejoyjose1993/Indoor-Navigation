import matplotlib.pyplot as plt
import matplotlib.animation as animation

test_raw_dir ="D:\\Project\\Dissertation\\raw_example.csv"

fig = plt.figure()

def animate(i):

    from random import random
#    data  = load_data(test_set_dir)
    data  = load_test_data(test_raw_dir)
    

    my_data = format_data(data)
    x_test = get_req_test_data(my_data)
    x_test = x_test.fillna(0)
    test_mean = x_test.mean(axis = 0).to_frame().transpose()



    if dest_reached:
        #exit()    
ani = animation.FuncAnimation(fig, animate, interval=1000)