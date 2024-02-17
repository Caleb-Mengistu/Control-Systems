import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import numpy as np
import random

trials= 3 #How many cubes drop before animation ends
trials_global = trials
incl_angle=np.pi/6.0 #Inclined angle in radians
g=10 #Acceralation from gravity in m/s^2
mass_cart=10 #Mass of cart in kg

#PID constants
K_p=300 #Porptional constant
K_d=300 #Derivative constant
K_i=10 #Integral constant

# Generate random x-positions for a falling cube
def set_x_ref(incl_angle):
    rand_x=random.uniform(0,120)
    rand_y=random.uniform(20+120*np.tan(incl_angle)+6.5,40+120*np.tan(incl_angle)+6.5)
    return rand_x,rand_y

dt=0.02 #time interval
t0=0 #start time
t_end=5 #end time
t=np.arange(t0,t_end+dt,dt) #Creates time array

#Create force of gravity w/ Fg = m * g
F_g=mass_cart*g

#Predefining arrays for simulation as zero arrays
displ_rail=np.zeros((trials,len(t))) #Displacement along the rail over time for each trial
v_rail=np.zeros((trials,len(t))) #Velocity along the rail over time for each trial
a_rail=np.zeros((trials,len(t))) #Acceleration along the rail over time for each trial
pos_x_train=np.zeros((trials,len(t))) #X-coordinate position of the train over time for each trial
pos_y_train=np.zeros((trials,len(t))) #Y-coordinate position of the train over time for each trial
e=np.zeros((trials,len(t))) #Error over time for each trial.
e_der=np.zeros((trials,len(t))) #Derivative of the error over time for each trial
e_int=np.zeros((trials,len(t))) #Integral of the error term over time for each trial
pos_x_cube=np.zeros((trials,len(t))) #X position of the cart over time for each trial
pos_y_cube=np.zeros((trials,len(t))) #Y position of the cart over time for each trial

F_ga_t=F_g*np.sin(incl_angle) # Tangential component of the gravity force

#The initial x and y positions, of the cart 
init_pos_x=120
init_pos_y=120*np.tan(incl_angle)+6.5

#Inital vleocaity and acceralation of the cart
init_vel_rail=0
init_a_rail=0

#The intital displacement of the rail
init_displ_rail=(init_pos_x**2+init_pos_y**2)**(0.5) 

init_pos_x_global=init_pos_x # Used for determining the dimensions of the animation window.

#These vairables keep track of which trial is occurring
trials_magn=trials
history=np.ones(trials)

while(trials>0):
    pos_x_cube_ref=set_x_ref(incl_angle)[0] # Cube's initial x position
    pos_y_cube_ref=set_x_ref(incl_angle)[1] # Cube's initial y position
    times=trials_magn-trials #The trial which is occurring
    pos_x_cube[times]=pos_x_cube_ref
    pos_y_cube[times]=pos_y_cube_ref-g/2*t**2
    win=False
    delta=1

    for i in range(1,len(t)):

        if i==1:
            #Enters the inital position
            displ_rail[times][0]=init_displ_rail
            pos_x_train[times][0]=init_pos_x
            pos_y_train[times][0]=init_pos_y
            v_rail[times][0]=init_vel_rail
            a_rail[times][0]=init_a_rail
        
        # Computes the horizontal error
        e[times][i-1]=pos_x_cube_ref-pos_x_train[times][i-1]

        #Updates the derivative and integral of the error arrays 
        if i>1:
            e_der[times][i-1]=(e[times][i-1]-e[times][i-2])/dt
            e_int[times][i-1]=e_int[times][i-2]+(e[times][i-2]+e[times][i-1])/2*dt
        
        #Checks for last step
        if i==len(t)-1:
            e[times][-1]=e[times][-2]
            e_der[times][-1]=e_der[times][-2]
            e_int[times][-1]=e_int[times][-2]
        
        #Updates values and arrays
        F_a=K_p*e[times][i-1]+K_d*e_der[times][i-1]+K_i*e_int[times][i-1] #Updates acceralation using PID
        F_net=F_a+F_ga_t #Updates net force of cart
        a_rail[times][i]=F_net/mass_cart #Updates acceleration of the cart 
        v_rail[times][i]=v_rail[times][i-1]+(a_rail[times][i-1]+a_rail[times][i])/2*dt #Updates the velocity of the cart using the trapezoidal rule
        displ_rail[times][i]=displ_rail[times][i-1]+(v_rail[times][i-1]+v_rail[times][i])/2*dt #Updates the displacement along the rail using the trapezoidal rule
        pos_x_train[times][i]=displ_rail[times][i]*np.cos(incl_angle) #Calculates the x-coordinate position of the cart
        pos_y_train[times][i]=displ_rail[times][i]*np.sin(incl_angle)+6.5 #Calculates the y-coordinate position of the cart

        if (pos_x_train[times][i]-5<pos_x_cube[times][i]+3 and pos_x_train[times][i]+5>pos_x_cube[times][i]-3) or win==True: # checks if the x-coordinate of the cart is within a certain range of the x-coordinate of the cube, allowing for a margin of +/-3 units
            if (pos_y_train[times][i]+3<pos_y_cube[times][i]-2 and pos_y_train[times][i]+8>pos_y_cube[times][i]+2) or win==True:
                win=True
                if delta==1:
                    change=pos_x_train[times][i]-pos_x_cube[times][i] # Gets the change in the x-coordinate position of the cube 
                    delta=0
                pos_x_cube[times][i]=pos_x_train[times][i]-change #
                pos_y_cube[times][i]=pos_y_train[times][i]+5

        #Updates initial values for the next trial after a collision
    init_displ_rail=displ_rail[times][-1]
    init_pos_x=pos_x_train[times][-1]+v_rail[times][-1]*np.cos(incl_angle)*dt
    init_pos_y=pos_y_train[times][-1]+v_rail[times][-1]*np.sin(incl_angle)*dt
    init_vel_rail=v_rail[times][-1]
    init_a_rail=a_rail[times][-1]
    history[times]=delta
    trials=trials-1

# Animation
len_t = len(t)
frame_amount = len(t) * trials_global

def update_plot(num):
    # Update the platform position
    platform.set_data([pos_x_train[int(num/len_t)][num-int(num/len_t)*len_t]-3.1,\
    pos_x_train[int(num/len_t)][num-int(num/len_t)*len_t]+3.1],\
    [pos_y_train[int(num/len_t)][num-int(num/len_t)*len_t],\
    pos_y_train[int(num/len_t)][num-int(num/len_t)*len_t]])

    # Update the cube position
    cube.set_data([pos_x_cube[int(num/len_t)][num-int(num/len_t)*len_t]-1,\
    pos_x_cube[int(num/len_t)][num-int(num/len_t)*len_t]+1],\
    [pos_y_cube[int(num/len_t)][num-int(num/len_t)*len_t],\
    pos_y_cube[int(num/len_t)][num-int(num/len_t)*len_t]])

    # Display success message if all trials are successful
    if trials_magn*len_t==num+1 and num>0: # All attempts must be successful
        if sum(history)==0:
            success.set_text('CONGRATS! YOU DID IT!')
        else:
            again.set_text('DON’T GIVE UP! YOU CAN DO IT!')

    # Update displacement on rails plot
    displ_rail_f.set_data(t[0:(num-int(num/len_t)*len_t)],
        displ_rail[int(num/len_t)][0:(num-int(num/len_t)*len_t)])

    # Update velocity on rails plot
    v_rail_f.set_data(t[0:(num-int(num/len_t)*len_t)],
        v_rail[int(num/len_t)][0:(num-int(num/len_t)*len_t)])

    # Update acceleration on rails plot
    a_rail_f.set_data(t[0:(num-int(num/len_t)*len_t)],
        a_rail[int(num/len_t)][0:(num-int(num/len_t)*len_t)])

    # Update horizontal error plot
    e_f.set_data(t[0:(num-int(num/len_t)*len_t)],
        e[int(num/len_t)][0:(num-int(num/len_t)*len_t)])

    # Update derivative of horizontal error plot
    e_der_f.set_data(t[0:(num-int(num/len_t)*len_t)],
        e_der[int(num/len_t)][0:(num-int(num/len_t)*len_t)])

    # Update integral of horizontal error plot
    e_int_f.set_data(t[0:(num-int(num/len_t)*len_t)],
        e_int[int(num/len_t)][0:(num-int(num/len_t)*len_t)])

    return displ_rail_f,v_rail_f,a_rail_f,e_f,e_der_f,e_int_f,platform,cube,success,again

# Create the figure and subplots for animation
fig=plt.figure(figsize=(16,9),dpi=120,facecolor=(0.8,0.8,0.8))
gs=gridspec.GridSpec(4,3)

# Create the main window
ax_main=fig.add_subplot(gs[0:3,0:2],facecolor=(0.9,0.9,0.9))
plt.xlim(0,init_pos_x_global)
plt.ylim(0,init_pos_x_global)
plt.xticks(np.arange(0,init_pos_x_global+1,10))
plt.yticks(np.arange(0,init_pos_x_global+1,10))
plt.grid(True)

title = ax_main.text(0,122,'Magnetic Train Simulation',size=12)
 
# Plot the rail and platform
rail=ax_main.plot([0,init_pos_x_global],[5,init_pos_x_global*np.tan(incl_angle)+5],'k',linewidth=6)
platform, = ax_main.plot([],[],'b',linewidth=18)
cube, = ax_main.plot([],[],'k',linewidth=14)

# Add success message
bbox_props_success = dict(boxstyle='square',fc=(0.9,0.9,0.9),ec='g',lw='1')
success = ax_main.text(40,60,'',size='20',color='g',bbox=bbox_props_success)

# Add failure message
bbox_props_again = dict(boxstyle='square',fc=(0.9,0.9,0.9),ec='r',lw='1')
again = ax_main.text(30,60,'',size='20',color='r',bbox=bbox_props_again)

# Plot windows for various data
ax1v = fig.add_subplot(gs[0,2],facecolor=(0.9,0.9,0.9))
ax2v = fig.add_subplot(gs[1,2],facecolor=(0.9,0.9,0.9))
ax3v = fig.add_subplot(gs[2,2],facecolor=(0.9,0.9,0.9))
ax1h = fig.add_subplot(gs[3,0],facecolor=(0.9,0.9,0.9))
ax2h = fig.add_subplot(gs[3,1],facecolor=(0.9,0.9,0.9))
ax3h = fig.add_subplot(gs[3,2],facecolor=(0.9,0.9,0.9))

# Animation function
pid_ani = animation.FuncAnimation(fig,update_plot,
    frames=frame_amount,interval=20,repeat=False,blit=True)
plt.show()
