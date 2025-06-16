import mujoco

cdpr_model=mujoco.MjSpec.from_file("base.xml") #Some settings can only be easily set in the base file.
cdpr_model.modelname = "cdpr_8_cables_6_dof"
cdpr_model.option.gravity= [0, 0, -9.81]
cdpr_model.worldbody.add_light(
                                type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                                diffuse=[0.8,0.8,0.8],
                                specular=[0.2,0.2,0.2],
                              )

#Fixed frame is centre of world, at height 0, mobile frame is centre of cube.
#Order is FLD clockwise, then move to FLU clockwise.

machine_frame_length=8
machine_frame_width=8
machine_frame_height=8
proximal_anchor_points=[
    [-machine_frame_length/2,machine_frame_width/2,0],
    [machine_frame_length/2,machine_frame_width/2,0],
    [machine_frame_length/2,-machine_frame_width/2,0],
    [-machine_frame_length/2,-machine_frame_width/2,0],
    [-machine_frame_length/2,machine_frame_width/2,machine_frame_height],
    [machine_frame_length/2,machine_frame_width/2,machine_frame_height],
    [machine_frame_length/2,-machine_frame_width/2,machine_frame_height],
    [-machine_frame_length/2,-machine_frame_width/2,machine_frame_height]
]

end_effector_length=0.5
end_effector_width=0.5
end_effector_height=0.5
distal_anchor_points=[
    [-end_effector_length/2, end_effector_width/2, -end_effector_height/2],
    [ end_effector_length/2, end_effector_width/2, -end_effector_height/2],
    [ end_effector_length/2,-end_effector_width/2, -end_effector_height/2],
    [-end_effector_length/2,-end_effector_width/2, -end_effector_height/2],
    [-end_effector_length/2, end_effector_width/2, end_effector_height/2],
    [ end_effector_length/2, end_effector_width/2, end_effector_height/2],
    [ end_effector_length/2,-end_effector_width/2, end_effector_height/2],
    [-end_effector_length/2,-end_effector_width/2, end_effector_height/2],
]

cdpr_model.worldbody.add_geom(
    name="floor",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[0, 0, 1],
    rgba=[0.5, 0.5, 0.5, 1],
    material="matplane",
)
for anchor_ind,anchor_coord in enumerate(proximal_anchor_points):
    cdpr_model.worldbody.add_site(
        name=f"proximal_anchor_{anchor_ind}",
        pos=anchor_coord,
        size=[0.1, 0.1, 0.1],
        rgba=[1, 0, 0, 1],
    )

end_effector = cdpr_model.worldbody.add_body(
    name="end_effector",
    pos=[0, 0, machine_frame_height / 2],
)
end_effector.add_geom(
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=[end_effector_length/2, end_effector_width/2, end_effector_height/2],
    rgba=[0, 0.9, 0, 1],
    density=10, #If this value is too high, the cables will not be able to move the end effector.
)
joint = end_effector.add_joint(
    type=mujoco.mjtJoint.mjJNT_FREE,
)
for anchor_ind,anchor_coord in enumerate(distal_anchor_points):
    end_effector.add_site(
        name=f"distal_anchor_{anchor_ind}",
        pos=anchor_coord,
        size=[0.1, 0.1, 0.1],
        rgba=[1, 0, 0, 1],
    )

for i in range(len(proximal_anchor_points)):
    tendon=cdpr_model.add_tendon( #The tendon contracts with negative control values, so opposite to CDPR convention.
        name=f'cable_tendon_{i}',
        limited=True,
        damping=1, #Higher this value is, the more resistant to movement the cable will be, so lower is more cablistic.
        range=[0, 10], #Limits on the length of the tendon, still exerts force within this range.
        width=0.05, #Width has no effect on the amount of force the tendon can exert, for visualization only.
        rgba=[0, 0, 0.9, 1],
    )
    tendon.wrap_site(f'distal_anchor_{i}')
    tendon.wrap_site(f'proximal_anchor_{i}')
    tendon_actuator=cdpr_model.add_actuator(
        target=tendon.name,
        trntype=mujoco.mjtTrn.mjTRN_TENDON,
        ctrllimited=True,
        ctrlrange=[0, 10],
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        forcerange=[-1000, 0],
    )
    tendon_actuator.set_to_position(kp=50)
    cdpr_model.add_sensor(
        type=mujoco.mjtSensor.mjSENS_TENDONPOS,
        objtype=mujoco.mjtObj.mjOBJ_TENDON,
        objname=tendon.name,
    )
    cdpr_model.add_sensor(
        type=mujoco.mjtSensor.mjSENS_TENDONACTFRC,
        objtype=mujoco.mjtObj.mjOBJ_TENDON,
        objname=tendon.name,
    )

cdpr_model.add_sensor(
    name="end_effector_position",
    type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
    objtype=mujoco.mjtObj.mjOBJ_BODY,
    objname=end_effector.name,
)
cdpr_model.add_sensor(
    name="end_effector_quaternion",
    type=mujoco.mjtSensor.mjSENS_FRAMEQUAT,
    objtype=mujoco.mjtObj.mjOBJ_BODY,
    objname=end_effector.name,
)

cdpr_model.compile()
with open("cdpr_8_6.xml", "w") as file:
    file.write(cdpr_model.to_xml())