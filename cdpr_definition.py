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

def inverse_kinematics(target_xyz=[0,0,0], target_orientation=[0,0,0]):
   desired_cable_vectors=np.zeros((8,3))
   target_xyz = np.array(target_xyz)
   target_orientation = np.array(target_orientation)
   rotation = R.from_rotvec(target_orientation, degrees=True)
   for cable_index in range(len(proximal_anchor_points)):
      desired_cable_vectors[cable_index,:]=proximal_anchor_points[cable_index,:]-target_xyz-rotation.apply(distal_anchor_points[cable_index,:])
      #print(distal_anchor_points[cable_index,:])
      #print(rotation.apply(distal_anchor_points[cable_index,:]))
   return [np.linalg.norm(desired_cable_vectors[cable_index,:]) for cable_index in range(len(proximal_anchor_points))]