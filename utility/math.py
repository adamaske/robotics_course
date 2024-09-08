from robot_control_class import RobotControl
import numpy as np


rc = RobotControl()

start_pos = (0, 0, 0)
speed_kmt = 6
speed_ms = speed_kmt / 3.6

speed_rds = 0.2

rot = 0
rot_step = 5

def do_maze():

    #move until we hit a wall, then decide which way to turn based on laser input

    

    while not out_of_maze():

        if not close_to_wall(): #
            rc.move_straight()
        

        rc.stop_robot()
        turn_rc()

    
        l_t = 5
        r_t = 5

        lasers = rc.get_laser_full()
        l_sum = np.sum(lasers[0:360]) 
        r_sum = np.sum(lasers[361:719]) 
        

    return 
    rc.rotate(90)

    rc.move_straight_time( 'forward', speed_ms, 1)
    rc.rotate(90)

    rc.move_straight_time( 'forward', speed_ms, 1)
    rc.rotate(90)

    rc.move_straight_time( 'forward', speed_ms, 1)
    rc.rotate(90)

    pass
def out_of_maze():
    return False

def turn_rc():
    lasers = rc.get_laser_full()

    l_w = np.sum(lasers[0:360]) / 360.0
    r_w = np.sum(lasers[361:719]) / 360.0

    if(l_w < r_w):
        # turn right
        rc.rotate(90)    
    else:
        # turn left 
        rc.rotate(-90)  

def close_to_wall():
    dist = rc.get_front_laser() 
    close =  dist < 0.75 
    print("close_to_wall : ", close)
    print("dist_to_wall: ", dist)
    return close


if __name__ == "__main__":
   
   #Using all the functions, help the robot get out of the maze
    do_maze()

    rc.stop_robot()
    exit()

    # ue all the methods
    
    dir = rc.get_laser(12) # direction between 0 and 719

    lasers = rc.get_laser_full()
    
    move_straight()

    stop_robot()

    move_straight_time( motion, speed, time) #'forward' or 'backward'
    #speed in m/s
    #time in seconds

    turn( clockwise, speed, time) # 'clockwise' or 'counter-clockwise'
    #speed in rad/s
    #time in seconds

    rotate(degrees) #degrees to rotate

    pass