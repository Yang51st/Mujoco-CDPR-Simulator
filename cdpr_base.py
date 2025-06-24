import mujoco as mj
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull
from scipy.linalg import null_space
from scipy.optimize import least_squares

class CDPR_Base:

    def __init__(self, machine_frame_lwh, end_effector_lwh, end_effector_mass, proportionality_constant, control_margin, cable_length_limit):
        
        self.num_cables = 8
        self.num_dof = 6

        self.machine_frame_length = machine_frame_lwh[0]
        self.machine_frame_width = machine_frame_lwh[1]
        self.machine_frame_height = machine_frame_lwh[2]

        self.end_effector_length = end_effector_lwh[0]
        self.end_effector_width = end_effector_lwh[1]
        self.end_effector_height = end_effector_lwh[2]

        self.end_effector_mass = end_effector_mass
        self.end_effector_weight = end_effector_mass * 9.81
        self.static_wrench = [0, 0, -self.end_effector_weight, 0, 0, 0]
        self.proportionality_constant = proportionality_constant
        self.control_margin = control_margin
        self.cable_length_limit = cable_length_limit

        #Fixed frame is centre of world, at height 0, mobile frame is centre of cube.
        #Order is FLD clockwise, then move to FLU clockwise.

        self.proximal_anchor_points = np.array([
            [-self.machine_frame_length/2, self.machine_frame_width/2, 0],
            [self.machine_frame_length/2, self.machine_frame_width/2, 0],
            [self.machine_frame_length/2, -self.machine_frame_width/2, 0],
            [-self.machine_frame_length/2, -self.machine_frame_width/2, 0],
            [-self.machine_frame_length/2, self.machine_frame_width/2, self.machine_frame_height],
            [self.machine_frame_length/2, self.machine_frame_width/2, self.machine_frame_height],
            [self.machine_frame_length/2, -self.machine_frame_width/2, self.machine_frame_height],
            [-self.machine_frame_length/2, -self.machine_frame_width/2, self.machine_frame_height]
        ])

        self.distal_anchor_points = np.array([
            [-self.end_effector_length/2,  self.end_effector_width/2, -self.end_effector_height/2],
            [ self.end_effector_length/2,  self.end_effector_width/2, -self.end_effector_height/2],
            [ self.end_effector_length/2,-self.end_effector_width/2, -self.end_effector_height/2],
            [-self.end_effector_length/2,-self.end_effector_width/2, -self.end_effector_height/2],
            [-self.end_effector_length/2,  self.end_effector_width/2,  self.end_effector_height/2],
            [ self.end_effector_length/2,  self.end_effector_width/2,  self.end_effector_height/2],
            [ self.end_effector_length/2,-self.end_effector_width/2,  self.end_effector_height/2],
            [-self.end_effector_length/2,-self.end_effector_width/2,  self.end_effector_height/2],
        ])
    
    def create_mujoco_spec(self, cross_config=False):
        cdpr_spec=mj.MjSpec.from_file("base.xml") #Some settings can only be easily set in the base file.
        cdpr_spec.modelname = "cdpr_8_cables_6_dof"
        cdpr_spec.option.gravity= [0, 0, -9.81]
        cdpr_spec.worldbody.add_light(
                                        type=mj.mjtLightType.mjLIGHT_DIRECTIONAL,
                                        diffuse=[0.8,0.8,0.8],
                                        specular=[0.2,0.2,0.2],
                                    )
        
        cdpr_spec.worldbody.add_camera(name="main_camera",
                                    pos=[0, -self.machine_frame_width*1.75, self.machine_frame_height*1.75],
                                    xyaxes=[1, 0, 0, 0, 1, 1],
                                    )
        cdpr_spec.worldbody.add_camera(name="side_camera",
                                    pos=[2*self.machine_frame_length, 0, self.machine_frame_height/2],
                                    xyaxes=[0, 1, 0, 0, 0, 1],
                                    )
        cdpr_spec.worldbody.add_camera(name="top_camera",
                                    pos=[0, 0, 2*self.machine_frame_height],
                                    xyaxes=[1, 0, 0, 0, 1, 0],
                                    )

        cdpr_spec.worldbody.add_geom(
            name="floor",
            type=mj.mjtGeom.mjGEOM_PLANE,
            size=[0, 0, 1],
            rgba=[0.5, 0.5, 0.5, 1],
            material="matplane",
        )
        for anchor_ind,anchor_coord in enumerate(self.proximal_anchor_points):
            cdpr_spec.worldbody.add_site(
                name=f"proximal_anchor_{anchor_ind}",
                pos=anchor_coord,
                size=[0.1, 0.1, 0.1],
                rgba=[1, 0, 0, 1],
            )

        end_effector = cdpr_spec.worldbody.add_body(
            name="end_effector",
            pos=[0, 0, self.end_effector_height/2],
        )
        end_effector.add_geom(
            type=mj.mjtGeom.mjGEOM_BOX,
            size=[self.end_effector_length/2, self.end_effector_width/2, self.end_effector_height/2],
            rgba=[0, 0.9, 0, 1],
            mass=self.end_effector_mass, #If this value is too high, the cables will not be able to move the end effector. Also, mass and density are different!
        )
        end_effector.add_joint(
            type=mj.mjtJoint.mjJNT_FREE,
        )
        for anchor_ind,anchor_coord in enumerate(self.distal_anchor_points):
            end_effector.add_site(
                name=f"distal_anchor_{anchor_ind}",
                pos=anchor_coord,
                size=[0.1, 0.1, 0.1],
                rgba=[1, 0, 0, 1],
            )

        for i in range(len(self.proximal_anchor_points)):
            tendon=cdpr_spec.add_tendon( #The tendon contracts with negative control values, so opposite to CDPR convention.
                name=f'cable_tendon_{i}',
                limited=True,
                damping=0, #Higher this value is, the more resistant to movement the cable will be, so lower is more cablistic.
                range=[0, self.cable_length_limit], #Limits on the length of the tendon, still exerts force within this range.
                width=0.05, #Width has no effect on the amount of force the tendon can exert, for visualization only.
                rgba=[0, 0, 0.9, 1],
                stiffness=0,
                springlength=[0, self.cable_length_limit],
                frictionloss=0.1,
            )
            if cross_config:
                tendon.wrap_site(f'proximal_anchor_{i}')
                tendon.wrap_site(f'distal_anchor_{(i+4)%8}')
            else:
                tendon.wrap_site(f'distal_anchor_{i}')
                tendon.wrap_site(f'proximal_anchor_{i}')
            slider = cdpr_spec.worldbody.add_body(
                pos=self.proximal_anchor_points[i],
            )
            slider.add_geom(
                type=mj.mjtGeom.mjGEOM_SPHERE,
                size=[0.2, 1, 1],
                rgba=[0.0, 0.9, 0.0, 1],
                mass=self.end_effector_mass/2,
                contype=0,
                conaffinity=0,
            )
            slider_joint=slider.add_joint(
                name=f'slider_joint_{i}',
                type=mj.mjtJoint.mjJNT_SLIDE,
                axis=np.array([0, 1, 0])*np.sign(self.proximal_anchor_points[i][1]),
                range=[0, self.cable_length_limit],
            )
            slider_site=slider.add_site(
                name=f'slider_site_{i}',
                pos=[0, 0, 0],
                size=[0.1, 0.1, 0.1],
                rgba=[0.9, 0, 0, 1],
            )
            tendon.wrap_site(slider_site.name)
            tendon_actuator=cdpr_spec.add_actuator(
                target=slider_joint.name,
                trntype=mj.mjtTrn.mjTRN_JOINT,
                ctrllimited=True,
                ctrlrange=[0, self.cable_length_limit+self.control_margin], #If this value is above the tendon length range, then the extra controlability will allow for higher cable tensions.
                biastype=mj.mjtBias.mjBIAS_AFFINE,
            )
            tendon_actuator.set_to_position(kp=self.proportionality_constant, dampratio=1)
            cdpr_spec.add_sensor(
                name=f"distal_pos_{i}",
                type=mj.mjtSensor.mjSENS_FRAMEPOS,
                objtype=mj.mjtObj.mjOBJ_SITE,
                objname=f'distal_anchor_{(i+4)%8}' if cross_config else f'distal_anchor_{i}',
            )
            cdpr_spec.add_sensor(
                name=f"cable_tendon_force_{i}",
                type=mj.mjtSensor.mjSENS_TENDONLIMITFRC,
                objtype=mj.mjtObj.mjOBJ_TENDON,
                objname=tendon.name,
            )

        cdpr_spec.add_sensor(
            name="end_effector_position",
            type=mj.mjtSensor.mjSENS_FRAMEPOS,
            objtype=mj.mjtObj.mjOBJ_BODY,
            objname=end_effector.name,
        )
        cdpr_spec.add_sensor(
            name="end_effector_xv",
            type=mj.mjtSensor.mjSENS_FRAMEXAXIS,
            objtype=mj.mjtObj.mjOBJ_BODY,
            objname=end_effector.name,
        )
        cdpr_spec.add_sensor(
            name="end_effector_yv",
            type=mj.mjtSensor.mjSENS_FRAMEYAXIS,
            objtype=mj.mjtObj.mjOBJ_BODY,
            objname=end_effector.name,
        )
        cdpr_spec.add_sensor(
            name="end_effector_zv",
            type=mj.mjtSensor.mjSENS_FRAMEZAXIS,
            objtype=mj.mjtObj.mjOBJ_BODY,
            objname=end_effector.name,
        )

        self.cdpr_spec = cdpr_spec

    def get_mujoco_model(self):
        return self.cdpr_spec.compile()

    def create_xml(self):
        with open("cdpr_8_6.xml", "w") as file:
            file.write(self.cdpr_spec.to_xml())