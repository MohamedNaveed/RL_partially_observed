from __future__ import division

import numpy as np
import matplotlib.patches as fill
#import pylab
import math
import time
from random import uniform
import matplotlib.pyplot as pylab
import matplotlib.colors as mcolors

params = {'axes.labelsize':12,
            'font.size':12,
            'legend.fontsize':12,
            'xtick.labelsize':12,
            'ytick.labelsize':12,
            'text.usetex':True,
            'figure.figsize':[4.5,3]}
pylab.rcParams.update(params)


PLOT_RED = True #True #False #
PLOT_BLUE = True
PLOT_GREEN = True


Noise_level = [0,100] #file index 1.6 - 64 1.0 - 54 .4-42

epsilonList = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5]

if __name__=='__main__':

    #filenames
    if PLOT_RED:
        
        filename = "/home/naveed/Documents/RL/naveed_codes/data/sac_cartpole/sac_cartpole_monte_carlo_epsi0.csv"

        file_red = open(filename,"r")

        lines = file_red.read().splitlines()

        for i in range(len(lines)):
            data = lines[i].split(',')

            if data[0] != "epsilon":

                if i == 1:
                    epsilon_red = float(data[0])
                    cost_red = float(data[3])
                    cost_var_red = float(data[4])

                else:
                    if float(data[0]) in epsilonList:
                        epsilon_red = np.append(epsilon_red, float(data[0]))
                        cost_red = np.append(cost_red, float(data[3]))
                        cost_var_red = np.append(cost_var_red, float(data[4]))



        cost_std_red = np.sqrt(cost_var_red)

    
    if PLOT_BLUE:
        
        filename_blue = "/home/naveed/Documents/RL/naveed_codes/data/sac_cartpole/sac_cartpole_monte_carlo_epsi20.csv"
        

        file_blue = open(filename_blue,"r")

        lines = file_blue.read().splitlines()

        for i in range(len(lines)):
            data = lines[i].split(',')
            print(f"i = {i} data = {data}")
            if data[0] != "epsilon":

                if i == 1:
                    epsilon_blue = float(data[0])
                    cost_blue = float(data[3])
                    cost_var_blue = float(data[4])

                else:
                    if float(data[0]) in epsilonList:
                        epsilon_blue = np.append(epsilon_blue, float(data[0]))
                        cost_blue = np.append(cost_blue, float(data[3]))
                        cost_var_blue = np.append(cost_var_blue, float(data[4]))



        cost_std_blue = np.sqrt(cost_var_blue)

    if PLOT_GREEN:
        
        filename_green = "/home/naveed/Documents/RL/naveed_codes/data/sac_cartpole/sac_cartpole_monte_carlo_epsi10.csv"

        file_green = open(filename_green,"r")

        lines = file_green.read().splitlines()

        for i in range(len(lines)): #len(lines)
            data = lines[i].split(',')

            if data[0] != "epsilon":

                if i == 1:
                    epsilon_green = float(data[0])
                    cost_green = float(data[3])
                    cost_var_green = float(data[4])

                else:
                    if float(data[0]) in epsilonList:
                        epsilon_green = np.append(epsilon_green, float(data[0]))
                        cost_green = np.append(cost_green, float(data[3]))
                        cost_var_green = np.append(cost_var_green, float(data[4]))



        cost_std_green = np.sqrt(cost_var_green)

    #min cost
    if PLOT_BLUE and PLOT_GREEN and PLOT_RED:

        Min_cost = min(cost_blue[0], cost_red[0], cost_green[0])

    elif PLOT_GREEN and PLOT_RED:

        Min_cost = min(cost_red[0], cost_green[0])


    elif PLOT_BLUE and PLOT_GREEN:
        Min_cost = min(cost_blue[0], cost_green[0])

    elif PLOT_BLUE and PLOT_RED:
        Min_cost = min(cost_blue[0], cost_red[0])


    elif PLOT_RED:
        Min_cost = cost_red[0]

    elif PLOT_BLUE:
        Min_cost = cost_blue[0]

    Min_cost = 1 #normalization
    epsilon_scale_factor = 100

    #plotting
    if PLOT_RED:
        pylab.fill_between(epsilon_scale_factor*epsilon_red[Noise_level[0]:Noise_level[1]+1],
                        (cost_red[Noise_level[0]:Noise_level[1]+1]-cost_std_red[Noise_level[0]:Noise_level[1]+1])/Min_cost,
                        (cost_red[Noise_level[0]:Noise_level[1]+1]+cost_std_red[Noise_level[0]:Noise_level[1]+1])/Min_cost,
                        alpha=0.35,linewidth=0,color='#ff7f0e')

        pylab.plot(epsilon_scale_factor*epsilon_red[Noise_level[0]:Noise_level[1]+1],
                    cost_red[Noise_level[0]:Noise_level[1]+1]/Min_cost,
                    linewidth=3,marker='.',markersize=10,color='#ff7f0e',label=r"SAC $\epsilon = 0\%$")
    


    if PLOT_GREEN:
        pylab.fill_between(epsilon_scale_factor*epsilon_green[Noise_level[0]:Noise_level[1]+1],
                        (cost_green[Noise_level[0]:Noise_level[1]+1] - cost_std_green[Noise_level[0]:Noise_level[1]+1])/Min_cost,
                        (cost_green[Noise_level[0]:Noise_level[1]+1] + cost_std_green[Noise_level[0]:Noise_level[1]+1])/Min_cost,
                        alpha=0.25,linewidth=0,color='#2ca02c')

        pylab.plot(epsilon_scale_factor*epsilon_green[Noise_level[0]:Noise_level[1]+1],cost_green[Noise_level[0]:Noise_level[1]+1]/Min_cost,
                   linewidth=3,linestyle='--',marker='',markersize=10,color='#2ca02c',label=r"SAC $\epsilon = 10\%$")
        
    if PLOT_BLUE:
        pylab.fill_between(epsilon_scale_factor*epsilon_blue[Noise_level[0]:Noise_level[1]+1],
                        (cost_blue[Noise_level[0]:Noise_level[1]+1]-cost_std_blue[Noise_level[0]:Noise_level[1]+1])/Min_cost,
                        (cost_blue[Noise_level[0]:Noise_level[1]+1]+cost_std_blue[Noise_level[0]:Noise_level[1]+1])/Min_cost,
                        alpha=0.25,linewidth=0,color='#1f77b4')

        pylab.plot(epsilon_scale_factor*epsilon_blue[Noise_level[0]:Noise_level[1]+1],
                    cost_blue[Noise_level[0]:Noise_level[1]+1]/Min_cost,
                    linewidth=3,linestyle=':',marker='',markersize=10,color='#1f77b4', label=r"SAC $\epsilon = 20\%$")


    ##legends
    #if PLOT_BLUE and PLOT_RED and PLOT_GREEN:
    legend = pylab.legend(loc= 'upper left')
    
    pylab.ylim(-0.2,8)
    #pylab.ylim(600,1150)
    #pylab.ylim(1500,4000)
    #pylab.xlim(-0.01,1.0)
    pylab.xlim(-1,31)
    pylab.xticks([0, 5, 10, 20, 30])
    pylab.grid(alpha=0.2)
    #pylab.xlim(-.05,1.6)

    #regimes
    #pylab.plot([0.2,0.2],[0,16],'k-',linewidth=2)
    #pylab.plot([1.25,1.25],[0,16],'k-',linewidth=2)

    #plt.text(0.05,1.5,'Low noise',rotation=90,rotation_mode='anchor',fontsize=18)
    #plt.text(0.25,1.5,'Medium noise',rotation=90,rotation_mode='anchor',fontsize=18)
    #plt.text(1.35,5,'High noise',rotation=90,rotation_mode='anchor',fontsize=18)
    #frame = legend.get_frame()
    #frame.set_facecolor('0.9')
    #frame.set_edgecolor('0.9')

    pylab.xlabel(r'Std dev of process noise (\% of max. control)')
    #pylab.ylabel(r'$J/\bar{J}$')
    pylab.ylabel(r'L2-norm of terminal state error')
    #pylab.title('Cost vs error percentage for 3 agent(s) ')

    pylab.savefig('/home/naveed/Documents/RL/naveed_codes/plots/'+'cartpole_sac.pdf', format='pdf',bbox_inches='tight',pad_inches = 0.02) #1- TLQR, 2- TLQR replan, 3 - MPC, 4 - MPC_fast
    #pylab.savefig('/home/naveed/Dropbox/Research/Data/AISTATS/'+'cost_car_LQR_PFC_comp.pdf', format='pdf',bbox_inches='tight',pad_inches = 0.02)
    
    if PLOT_RED:
        file_red.close()


    if PLOT_BLUE:
        file_blue.close()

    if PLOT_GREEN:
        file_green.close()