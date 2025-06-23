from cdpr_base import *

class CableDrivenParallelRobot(CDPR_Base):

    def __init__(self, machine_frame_lwh=[15.40, 10.32, 17.80], end_effector_lwh=[1.24*np.sqrt(2), 1.24*np.sqrt(2), 0.85], end_effector_mass=1, proportionality_constant=50, control_margin=30, cable_length_limit=40):
        super().__init__(machine_frame_lwh, end_effector_lwh, end_effector_mass, proportionality_constant, control_margin, cable_length_limit)
        self.calculated_cable_tensions=[[] for _ in range(self.num_cables)]

    def reset_cable_tensions_list(self):
        self.calculated_cable_tensions=[[] for _ in range(self.num_cables)]

    def set_cable_tension_limits(self, lower_tension_limit=0, upper_tension_limit=0):
        self.lower_tension_limit = lower_tension_limit
        if upper_tension_limit == 0:
            self.upper_tension_limit = self.control_margin * self.proportionality_constant
        else:
            self.upper_tension_limit = upper_tension_limit

    def inverse_kinematics(self, wrench, target_xyz=[0, 0, 0], target_orientation=[0, 0, 0], plot_solution=False):
        desired_cable_vectors = np.zeros((self.num_cables, 3))
        target_xyz = np.array(target_xyz)
        target_orientation = np.array(target_orientation)
        rotation = R.from_rotvec(target_orientation, degrees=True)
        wrench = np.array(wrench)

        for cable_index in range(len(self.proximal_anchor_points)):
            desired_cable_vectors[cable_index, :] = self.proximal_anchor_points[cable_index, :] - target_xyz - rotation.apply(self.distal_anchor_points[cable_index, :])

        first_stage_cable_lengths = [np.linalg.norm(desired_cable_vectors[cable_index, :]) for cable_index in range(len(self.proximal_anchor_points))]
        second_stage_cable_lengths = self.force_distribution(desired_cable_vectors, rotation, wrench, self.proportionality_constant, plot_solution)

        return first_stage_cable_lengths - second_stage_cable_lengths
    
    def force_distribution(self, cable_vectors, rotation, wrench, kp, plot_solution=False):
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
            if plot_solution:
                for i in range(limits.shape[0]):
                    plt.axline((0,limits[i]/kernels[i][1]),(limits[i]/kernels[i][0], 0), color='blue', label=f'Limit {i+1}')
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
            
            possible_solutions=[list(sol) for sol in set(tuple(sol) for sol in possible_solutions)]
            #print(len(possible_solution), "possible solutions found.")
            if plot_solution:
                plt.scatter(*zip(*possible_solutions), color='red', label='Possible Solutions')
            possible_solutions= np.array(possible_solutions)
            if len(possible_solutions) > 2:
                try:
                    convex_hull = ConvexHull(possible_solutions)
                    chull_area=convex_hull.volume #This gets the area for 2D points, scipy is weird like that.
                    center_x=0
                    center_y=0
                    for vind in range(len(convex_hull.vertices)-1):
                        center_x += (possible_solutions[convex_hull.vertices[vind],0]+possible_solutions[convex_hull.vertices[vind+1],0])*(possible_solutions[convex_hull.vertices[vind],0]*possible_solutions[convex_hull.vertices[vind+1],1]-possible_solutions[convex_hull.vertices[vind+1],0]*possible_solutions[convex_hull.vertices[vind],1])
                        center_y += (possible_solutions[convex_hull.vertices[vind],1]+possible_solutions[convex_hull.vertices[vind+1],1])*(possible_solutions[convex_hull.vertices[vind],0]*possible_solutions[convex_hull.vertices[vind+1],1]-possible_solutions[convex_hull.vertices[vind+1],0]*possible_solutions[convex_hull.vertices[vind],1])
                    center_x /= (6*chull_area)
                    center_y /= (6*chull_area)
                    center=np.array([center_x, center_y])
                except:
                    center= np.mean(possible_solutions, axis=0)
                if plot_solution:
                    plt.scatter(center[0], center[1], color='green', label='Centroid of Convex Hull')
                desired_cable_forces = homogeneous_sol + sm_kernel @ center
            elif len(possible_solutions) > 0:
                if plot_solution:
                    plt.scatter(np.mean(possible_solutions, axis=0)[0], np.mean(possible_solutions, axis=0)[1], color='green', label='Centroid of Convex Hull')
                desired_cable_forces = homogeneous_sol + sm_kernel @ np.mean(possible_solutions, axis=0)
            else:
                desired_cable_forces = homogeneous_sol
            if plot_solution:
                plt.xlim(50, 500)
                plt.ylim(-200, 600)
                plt.show()
        else:
            desired_cable_forces = homogeneous_sol
            print("Structure matrix likely not full rank, CDPR may be at singularity.")

        for i in range(self.num_cables):
            if desired_cable_forces[i] < self.lower_tension_limit:
                desired_cable_forces[i] = self.lower_tension_limit
            if desired_cable_forces[i] > self.upper_tension_limit:
                desired_cable_forces[i] = self.upper_tension_limit
            self.calculated_cable_tensions[i].append(desired_cable_forces[i])

        return desired_cable_forces/kp

"""
cdpr= CableDrivenParallelRobot()
desired_position=[1,1,8]
desired_orientation=[5,5,2]
cdpr.set_cable_tension_limits(lower_tension_limit=10, upper_tension_limit=250)
cdpr.inverse_kinematics(wrench=[20,30,40,5,8,10], target_xyz=desired_position, target_orientation=desired_orientation, plot_solution=True)
"""