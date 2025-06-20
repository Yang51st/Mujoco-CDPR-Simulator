import mujoco as mj
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.linalg import null_space

machine_frame_length=6
machine_frame_width=7
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

end_effector_length=0.3
end_effector_width=0.4
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

lower_tension_limit=0
upper_tension_limit=200

def inverse_kinematics(target_xyz=[0,0,0], target_orientation=[0,0,0], wrench=[0,0,-9.81*10,0,0,0]):

    desired_cable_vectors=np.zeros((8,3))
    target_xyz = np.array(target_xyz)
    target_orientation = np.array(target_orientation)
    rotation = R.from_rotvec(target_orientation, degrees=True)
    wrench = np.array(wrench)
    
    for cable_index in range(len(proximal_anchor_points)):
        desired_cable_vectors[cable_index,:]=proximal_anchor_points[cable_index,:]-target_xyz-rotation.apply(distal_anchor_points[cable_index,:])
    
    first_stage_cable_lengths = [np.linalg.norm(desired_cable_vectors[cable_index,:]) for cable_index in range(len(proximal_anchor_points))]
    second_stage_cable_lengths = force_distribution(desired_cable_vectors, rotation, wrench)

    return first_stage_cable_lengths - second_stage_cable_lengths

def force_distribution(cable_vectors, rotation, wrench, kp=50):
    """
    The control input for each CDPR slider joint is related to the force applied by the cable through the relation of
    force = kp * (control_input - actual_length), so to produce a desired length to reach the target position there
    must be additional control logic added. The cable lengths returned by this function will have to be subtracted from
    ones calculated by the inverse kinematics function to produce the desired cable lengths, since shortening cables means
    you are trying to apply more forces.
    """

    desired_cable_forces = np.zeros(8)
    
    cable_unit_vectors=cable_vectors.T/np.linalg.norm(cable_vectors.T, axis=0)
    moment_arm_vectors=np.cross(rotation.apply(distal_anchor_points).T, cable_unit_vectors, axisa=0, axisb=0, axisc=0)
    structure_matrix=np.concatenate([cable_unit_vectors,moment_arm_vectors], axis=0)
    sm_pinv=np.linalg.pinv(structure_matrix)
    sm_kernel=null_space(structure_matrix)

    if sm_kernel.shape==(8,2):
        #8 by 2 multiplied by 2 by 1 gives 8 by 1 result.
        desired_cable_forces = sm_pinv @ (-1*wrench) + sm_kernel @ np.array([100,100])
        homogeneous_sol=sm_pinv @ (-1*wrench)
        lower_limit=np.ones(8)*lower_tension_limit-homogeneous_sol
        upper_limit=np.ones(8)*upper_tension_limit-homogeneous_sol
        limits=np.concatenate([lower_limit, upper_limit], axis=0)
        kernels=np.concatenate([sm_kernel, sm_kernel], axis=0)
        for i in range(16-1): #tl=n1*x+n2*y
            for j in range(i+1,16):
                possible_solution=np.zeros(2)
                try:
                    possible_solution=np.linalg.solve(np.array([kernels[i],kernels[j]]), limits[[i,j]])
                except np.linalg.LinAlgError:
                    pass
                #print(possible_solution)
                if np.all(sm_kernel@possible_solution >= lower_limit) and np.all(sm_kernel@possible_solution <= upper_limit):
                    desired_cable_forces = homogeneous_sol + sm_kernel@possible_solution
                    print("Found solution within limits.")
                    return desired_cable_forces/kp
    else:
        desired_cable_forces = sm_pinv @ (-1*wrench)
        print("Structure matrix likely not full rank, CDPR may be at singularity.")

    #print(desired_cable_forces)
    return desired_cable_forces/kp

#print(inverse_kinematics(target_xyz=[0,0,machine_frame_height/3], target_orientation=[0.00001,0.00001,0.00001], wrench=[0,0,-9.81*10,0,0,0]))