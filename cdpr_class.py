from cdpr_base import *

class CableDrivenParallelRobot(CDPR_Base):

    def __init__(self, machine_frame_lwh=[15.40, 10.32, 17.80], end_effector_lwh=[1.24*np.sqrt(2), 1.24*np.sqrt(2), 0.85], end_effector_mass=1, proportionality_constant=50, control_margin=30, cable_length_limit=40, p_constant=0.05, i_constant=0.001, d_constant=0.01):
        super().__init__(machine_frame_lwh, end_effector_lwh, end_effector_mass, proportionality_constant, control_margin, cable_length_limit)
        self.calculated_cable_tensions=[[] for _ in range(self.num_cables)]
        self.p_constant=p_constant
        self.i_constant=i_constant
        self.d_constant=d_constant
        self.previous_errors = np.zeros(self.num_cables)
        self.cumulative_errors = np.zeros(self.num_cables)
        self.first_pid_run=True

    def reset_cable_tensions_list(self):
        self.calculated_cable_tensions=[[] for _ in range(self.num_cables)]

    def set_cable_tension_limits(self, lower_tension_limit=0, upper_tension_limit=0):
        self.lower_tension_limit = lower_tension_limit
        if upper_tension_limit == 0:
            self.upper_tension_limit = self.control_margin * self.proportionality_constant
        else:
            self.upper_tension_limit = upper_tension_limit

    def sense_cable_lengths(self, cdpr_data):
        cable_lengths=[]
        for _ in range(self.num_cables):
            cable_lengths.append(np.linalg.norm(self.proximal_anchor_points[_]-cdpr_data.sensor(f'distal_pos_{_}').data))
        return np.array(cable_lengths)
    
    def pid_control(self, wrench, target_xyz, target_orientation, cdpr_data):
        rotation = R.from_rotvec(target_orientation, degrees=True)
        desired_cable_vectors=self.inverse_kinematics(target_xyz, rotation)
        desired_cable_lengths = [np.linalg.norm(desired_cable_vectors[cable_index, :]) for cable_index in range(len(self.proximal_anchor_points))]
        desired_cable_forces=self.force_distribution(desired_cable_vectors, rotation, wrench)
        sensed_cable_lengths = self.sense_cable_lengths(cdpr_data)
        cable_errors=sensed_cable_lengths-desired_cable_lengths
        p_control=self.p_constant * cable_errors
        if self.first_pid_run:
            self.previous_errors = cable_errors
            self.cumulative_errors = np.zeros(self.num_cables)
            self.first_pid_run = False
        i_control=self.i_constant * (self.cumulative_errors + cable_errors)
        d_control=self.d_constant * (cable_errors - self.previous_errors)
        self.previous_errors = cable_errors
        self.cumulative_errors += cable_errors

        return p_control+i_control+d_control
    
    def ol_joint_position_control(self, wrench, target_xyz, target_orientation):
        rotation = R.from_rotvec(target_orientation, degrees=True)
        desired_cable_vectors = self.inverse_kinematics(target_xyz, rotation)
        desired_cable_forces = self.force_distribution(desired_cable_vectors, rotation, wrench)

        return self.cable_length_limit*np.ones(self.num_cables) + desired_cable_forces/self.proportionality_constant - np.linalg.norm(desired_cable_vectors, axis=1)
    
    def cl_joint_position_control(self, target_xyz, target_orientation, cdpr_data):
        rotation = R.from_rotvec(target_orientation, degrees=True)
        desired_cable_vectors = self.inverse_kinematics(target_xyz, rotation)
        desired_cable_lengths=np.linalg.norm(desired_cable_vectors, axis=1)
        sensed_cable_lengths = self.sense_cable_lengths(cdpr_data)
        cable_errors = sensed_cable_lengths - desired_cable_lengths
        p_control = self.p_constant * cable_errors
        if self.first_pid_run:
            self.previous_errors = cable_errors
            self.cumulative_errors = np.zeros(self.num_cables)
            self.first_pid_run = False
        i_control = self.i_constant * (self.cumulative_errors + cable_errors)
        d_control = self.d_constant * (cable_errors - self.previous_errors)
        self.previous_errors = cable_errors
        self.cumulative_errors += cable_errors

        return self.cable_length_limit*np.ones(self.num_cables) + p_control + i_control + d_control - np.linalg.norm(desired_cable_vectors, axis=1)

    def inverse_kinematics(self, target_xyz, rotation):
        desired_cable_vectors = np.zeros((self.num_cables, 3))
        target_xyz = np.array(target_xyz)

        for cable_index in range(len(self.proximal_anchor_points)):
            desired_cable_vectors[cable_index, :] = self.proximal_anchor_points[cable_index, :] - target_xyz - rotation.apply(self.distal_anchor_points[cable_index, :])
        
        return desired_cable_vectors
    
    def force_distribution(self, cable_vectors, rotation, wrench, plot_solution=False):
        """
        The control input for each CDPR slider joint is related to the force applied by the cable through the relation of
        force = kp * (control_input - actual_length), so to produce a desired length to reach the target position there
        must be additional control logic added. The cable lengths returned by this function will have to be subtracted from
        ones calculated by the inverse kinematics function to produce the desired cable lengths, since shortening cables means
        you are trying to apply more forces.
        """

        desired_cable_forces = np.zeros(self.num_cables)
        wrench = np.array(wrench)

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
                #plt.xlim(50, 500)
                #plt.ylim(-200, 600)
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

        return desired_cable_forces
    
    def forward_kinematics(self, cable_lengths):
        radius_lower_bounds=self.proximal_anchor_points-(cable_lengths+np.linalg.norm(self.distal_anchor_points,axis=1)).reshape((8,1))*np.ones((8,3))
        radius_upper_bounds=self.proximal_anchor_points+(cable_lengths+np.linalg.norm(self.distal_anchor_points,axis=1)).reshape((8,1))*np.ones((8,3))
        lower_indices=np.argmax(radius_lower_bounds,axis=0)
        upper_indices=np.argmin(radius_upper_bounds,axis=0)
        lower_corner=np.array([radius_lower_bounds[lower_indices[0],0],
                               radius_lower_bounds[lower_indices[1],1],
                               radius_lower_bounds[lower_indices[2],2]])
        upper_corner=np.array([radius_upper_bounds[upper_indices[0],0],
                                 radius_upper_bounds[upper_indices[1],1],
                                 radius_upper_bounds[upper_indices[2],2]])
        if np.all(lower_corner > upper_corner):
            print("Cannot perform forward kinematics.")
            return None
        
        lower_corner = np.concatenate((lower_corner, [-180, -180, -180]), axis=0)
        upper_corner = np.concatenate((upper_corner, [180, 180, 180]), axis=0)
        initial_guess = (lower_corner + upper_corner) / 2

        def to_be_minimized(input_vector):
            total_error=0
            r=input_vector[0:3]
            rotation = R.from_rotvec(input_vector[3:], degrees=True)
            for cable_index in range(self.num_cables):
                cable_vector = self.proximal_anchor_points[cable_index] - r - rotation.apply(self.distal_anchor_points[cable_index])
                cable_length = np.linalg.norm(cable_vector)
                total_error += np.square((np.square(cable_length) - np.square(cable_lengths[cable_index])))
            return total_error
        
        result = least_squares(to_be_minimized, initial_guess, bounds=(lower_corner, upper_corner), method='trf', max_nfev=10000)
        if result.success:
            return result.x
        else:
            print("Forward kinematics failed to converge.")
            print("Error message:", result.message)
            return None

"""
cdpr= CableDrivenParallelRobot()
cdpr.set_cable_tension_limits()
print(cdpr.forward_kinematics(cable_lengths=np.array([8.05615199,
                                                8.05615199,
                                                8.05615199,
                                                8.05615199,
                                                18.76710113,
                                                18.76710113,
                                                18.76710113,
                                                18.76710113])))
"""