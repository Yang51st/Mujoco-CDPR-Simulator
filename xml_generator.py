from cdpr_definition import *

cdpr_spec=mj.MjSpec.from_file("base.xml") #Some settings can only be easily set in the base file.
cdpr_spec.modelname = "cdpr_8_cables_6_dof"
cdpr_spec.option.gravity= [0, 0, -9.81]
cdpr_spec.worldbody.add_light(
                                type=mj.mjtLightType.mjLIGHT_DIRECTIONAL,
                                diffuse=[0.8,0.8,0.8],
                                specular=[0.2,0.2,0.2],
                              )
cdpr_spec.worldbody.add_camera(name="main_camera",
                               pos=[0, -10, 14],
                               xyaxes=[1, 0, 0, 0, 1, 1],
                               )

cdpr_spec.worldbody.add_geom(
    name="floor",
    type=mj.mjtGeom.mjGEOM_PLANE,
    size=[0, 0, 1],
    rgba=[0.5, 0.5, 0.5, 1],
    material="matplane",
)
for anchor_ind,anchor_coord in enumerate(proximal_anchor_points):
    cdpr_spec.worldbody.add_site(
        name=f"proximal_anchor_{anchor_ind}",
        pos=anchor_coord,
        size=[0.1, 0.1, 0.1],
        rgba=[1, 0, 0, 1],
    )

end_effector = cdpr_spec.worldbody.add_body(
    name="end_effector",
    pos=[0, 0, machine_frame_height / 2],
)
end_effector.add_geom(
    type=mj.mjtGeom.mjGEOM_BOX,
    size=[end_effector_length/2, end_effector_width/2, end_effector_height/2],
    rgba=[0, 0.9, 0, 1],
    mass=10, #If this value is too high, the cables will not be able to move the end effector. Also, mass and density are different!
)
joint = end_effector.add_joint(
    type=mj.mjtJoint.mjJNT_FREE,
)
for anchor_ind,anchor_coord in enumerate(distal_anchor_points):
    end_effector.add_site(
        name=f"distal_anchor_{anchor_ind}",
        pos=anchor_coord,
        size=[0.1, 0.1, 0.1],
        rgba=[1, 0, 0, 1],
    )

cross_config=False
for i in range(len(proximal_anchor_points)):
    tendon=cdpr_spec.add_tendon( #The tendon contracts with negative control values, so opposite to CDPR convention.
        name=f'cable_tendon_{i}',
        limited=True,
        damping=0, #Higher this value is, the more resistant to movement the cable will be, so lower is more cablistic.
        range=[0, 20], #Limits on the length of the tendon, still exerts force within this range.
        width=0.05, #Width has no effect on the amount of force the tendon can exert, for visualization only.
        rgba=[0, 0, 0.9, 1],
        stiffness=0,
        springlength=[0, 20],
        frictionloss=0.1,
    )
    if cross_config:
        tendon.wrap_site(f'proximal_anchor_{i}')
        tendon.wrap_site(f'distal_anchor_{(i+4)%8}')
    else:
        tendon.wrap_site(f'distal_anchor_{i}')
        tendon.wrap_site(f'proximal_anchor_{i}')
    slider = cdpr_spec.worldbody.add_body(
        pos=proximal_anchor_points[i],
    )
    slider.add_geom(
        type=mj.mjtGeom.mjGEOM_SPHERE,
        size=[0.2, 1, 1],
        rgba=[0.0, 0.9, 0.0, 1],
        mass=5,
        contype=0,
        conaffinity=0,
    )
    slider_joint=slider.add_joint(
        name=f'slider_joint_{i}',
        type=mj.mjtJoint.mjJNT_SLIDE,
        axis=np.array([0, 1, 0])*np.sign(proximal_anchor_points[i][1]),
        range=[0, 20],
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
        ctrlrange=[0, 50], #If this value is above the tendon length range, then the extra controlability will allow for higher cable tensions.
        biastype=mj.mjtBias.mjBIAS_AFFINE,
    )
    tendon_actuator.set_to_position(kp=50, dampratio=1)
    cdpr_spec.add_sensor(
        type=mj.mjtSensor.mjSENS_TENDONPOS,
        objtype=mj.mjtObj.mjOBJ_TENDON,
        objname=tendon.name,
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

cdpr_spec.compile()
with open("cdpr_8_6.xml", "w") as file:
    file.write(cdpr_spec.to_xml())