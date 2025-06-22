from cdpr_base import *

class CableDrivenParallelRobot(CDPR_Base):

    def __init__(self, machine_frame_lwh=[6, 7, 8], end_effector_lwh=[0.3, 0.4, 0.5], end_effector_mass=10, proportionality_constant=50, control_margin=30, cable_length_limit=20, lower_tension_limit=0, upper_tension_limit=1500):
        super().__init__(machine_frame_lwh, end_effector_lwh, end_effector_mass, proportionality_constant, control_margin, cable_length_limit, lower_tension_limit, upper_tension_limit)

    def inverse_kinematics(self, wrench, target_xyz=[0, 0, 0], target_orientation=[0, 0, 0]):
        desired_cable_vectors = np.zeros((self.num_cables, 3))
        target_xyz = np.array(target_xyz)
        target_orientation = np.array(target_orientation)
        rotation = R.from_rotvec(target_orientation, degrees=True)
        wrench = np.array(wrench)

        for cable_index in range(len(self.proximal_anchor_points)):
            desired_cable_vectors[cable_index, :] = self.proximal_anchor_points[cable_index, :] - target_xyz - rotation.apply(self.distal_anchor_points[cable_index, :])

        first_stage_cable_lengths = [np.linalg.norm(desired_cable_vectors[cable_index, :]) for cable_index in range(len(self.proximal_anchor_points))]
        second_stage_cable_lengths = self.force_distribution(desired_cable_vectors, rotation, wrench, self.proportionality_constant)

        return first_stage_cable_lengths - second_stage_cable_lengths
    
    def force_distribution(self, cable_vectors, rotation, wrench, kp):
        """
        The control input for each CDPR slider joint is related to the force applied by the cable through the relation of
        force = kp * (control_input - actual_length), so to produce a desired length to reach the target position there
        must be additional control logic added. The cable lengths returned by this function will have to be subtracted from
        ones calculated by the inverse kinematics function to produce the desired cable lengths, since shortening cables means
        you are trying to apply more forces.
        """

        desired_cable_forces = np.zeros(self.num_cables)

        cable_unit_vectors=cable_vectors.T/np.linalg.norm(cable_vectors.T, axis=0)
        moment_arm_vectors=np.cross(rotation.apply(self.distal_anchor_points).T, cable_unit_vectors, axisa=0, axisb=0, axisc=0)
        structure_matrix=np.concatenate([cable_unit_vectors,moment_arm_vectors], axis=0)
        sm_pinv=np.linalg.pinv(structure_matrix)
        sm_kernel=null_space(structure_matrix)

        homogeneous_sol=sm_pinv @ (-1*wrench)

        if sm_kernel.shape==(self.num_cables,self.num_cables-self.num_dof):
            lower_limit=np.ones(self.num_cables)*self.lower_tension_limit-homogeneous_sol
            upper_limit=np.ones(self.num_cables)*self.upper_tension_limit-homogeneous_sol
            limits=np.concatenate([lower_limit, upper_limit], axis=0)
            kernels=np.concatenate([sm_kernel, sm_kernel], axis=0)
            possible_solutions=[]
            for i in range(limits.shape[0]-1):
                for j in range(i+1,limits.shape[0]):
                    possible_solution=np.zeros(self.num_cables-self.num_dof)
                    try:
                        possible_solution=np.linalg.solve(np.array([kernels[i],kernels[j]]), limits[[i,j]])
                    except np.linalg.LinAlgError:
                        pass
                    if np.all(sm_kernel@possible_solution >= lower_limit) and np.all(sm_kernel@possible_solution <= upper_limit):
                        possible_solutions.append(possible_solution.tolist())
            
            possible_solution=[list(sol) for sol in set(tuple(sol) for sol in possible_solutions)]
            #plt.scatter(*zip(*possible_solutions), color='red', label='Possible Solutions')
            possible_solutions= np.array(possible_solutions)
            if len(possible_solution) > 2:
                convex_hull = ConvexHull(possible_solutions)
                chull_area=convex_hull.volume #This gets the area for 2D points, scipy is weird like that.
                center_x=0
                center_y=0
                for vind in range(len(convex_hull.vertices)-1):
                    center_x += (possible_solutions[convex_hull.vertices[vind],0]+possible_solutions[convex_hull.vertices[vind+1],0])*(possible_solutions[convex_hull.vertices[vind],0]*possible_solutions[convex_hull.vertices[vind+1],1]-possible_solutions[convex_hull.vertices[vind+1],0]*possible_solutions[convex_hull.vertices[vind],1])
                    center_y += (possible_solutions[convex_hull.vertices[vind],1]+possible_solutions[convex_hull.vertices[vind+1],1])*(possible_solutions[convex_hull.vertices[vind],0]*possible_solutions[convex_hull.vertices[vind+1],1]-possible_solutions[convex_hull.vertices[vind+1],0]*possible_solutions[convex_hull.vertices[vind],1])
                center_x /= 6*chull_area
                center_y /= 6*chull_area
                center=np.array([center_x, center_y])
                #plt.scatter(center[0], center[1], color='green', label='Centroid of Convex Hull')
                desired_cable_forces = homogeneous_sol + sm_kernel @ center
            elif len(possible_solution) > 0:
                desired_cable_forces = homogeneous_sol + sm_kernel @ np.mean(possible_solutions, axis=0)
            else:
                desired_cable_forces = homogeneous_sol
            #plt.show()
        else:
            desired_cable_forces = homogeneous_sol
            print("Structure matrix likely not full rank, CDPR may be at singularity.")

        return desired_cable_forces/kp