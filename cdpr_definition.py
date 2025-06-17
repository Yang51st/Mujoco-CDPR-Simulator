import mujoco as mj
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

machine_frame_length=8
machine_frame_width=8
machine_frame_height=8

#Fixed frame is centre of world, at height 0, mobile frame is centre of cube.
#Order is FLD clockwise, then move to FLU clockwise.

proximal_anchor_points=np.array([
    [-machine_frame_length/2,machine_frame_width/2,0],
    [machine_frame_length/2,machine_frame_width/2,0],
    [machine_frame_length/2,-machine_frame_width/2,0],
    [-machine_frame_length/2,-machine_frame_width/2,0],
    [-machine_frame_length/2,machine_frame_width/2,machine_frame_height],
    [machine_frame_length/2,machine_frame_width/2,machine_frame_height],
    [machine_frame_length/2,-machine_frame_width/2,machine_frame_height],
    [-machine_frame_length/2,-machine_frame_width/2,machine_frame_height]
])

end_effector_length=0.5
end_effector_width=0.5
end_effector_height=0.5

distal_anchor_points=np.array([
    [-end_effector_length/2, end_effector_width/2, -end_effector_height/2],
    [ end_effector_length/2, end_effector_width/2, -end_effector_height/2],
    [ end_effector_length/2,-end_effector_width/2, -end_effector_height/2],
    [-end_effector_length/2,-end_effector_width/2, -end_effector_height/2],
    [-end_effector_length/2, end_effector_width/2, end_effector_height/2],
    [ end_effector_length/2, end_effector_width/2, end_effector_height/2],
    [ end_effector_length/2,-end_effector_width/2, end_effector_height/2],
    [-end_effector_length/2,-end_effector_width/2, end_effector_height/2],
])