"""Write the default physics settings of traquad into the imported USD, so the asset works as is
in Isaac Sim / Isaac Lab (Isaac Lab actuators still override the joints they control).

- Physics variant set to 'physx' (the importer leaves it unselected: no physics otherwise)
- articulation self-collisions off
- HFE: PD drive Kp 60 Nm/rad, Kd 0.6 Nm s/rad (real robot, mulsbc_ws traquad8_dji.xacro), max --hip_max_torque
  (10 Nm, to be set from the motor K_t x 80 A x gear 9)
- ankles: passive damping 0.01 Nm s/rad. The controllers (Isaac Lab action, isaac_sim/ankle_control.py) add a
  velocity control of the ankle when the track is in the air: the frame turns with the drive sprocket
- driven wheels: velocity drive (stiffness 0), armature --wheel_armature. The motor of a track is described at the
  track: --track_max_torque and --track_damping (velocity gain), both referred to the 15 mm wheel radius, are split
  over the 4 wheels of the track
- internal friction of the tracks (belt, sprocket, gear): Coulomb --track_friction and viscous --track_viscous at the
  track joint (45 mm sprocket), as joint friction of the driven wheels (PhysX static / dynamic friction effort and
  viscous coefficient), referred to the 15 mm wheels and split over them
- HFE zero: --hfe_offset c = geometric leg angle - encoder angle of the real robot [rad] (alpha = sigma q_HFE; identified
  on the Qualisys runs of 2026-10-23, isaac_sim/real_id). The joint frame on the leg (localRot1) is turned by -sigma c
  about the joint axis, so that the simulated joint angle is the robot's encoder angle: the same references and
  stances as on the robot. Idempotent (the applied offset is stored in traquad:hfeOffset)
- contact offset --contact_offset on the colliders of wheels, rollers and track bodies (rest offset 0)
- rollers: passive, damping --roller_damping, dry (Coulomb) friction --roller_friction [Nm]

usage: ./isaac.sh finalize_usd.py --usd assets/traquad/traquad.usda [--roller_damping 1e-4] [--roller_friction 0.10]
"""
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument('--usd', required=True)
parser.add_argument('--roller_damping', type=float, default=1e-4)
# roller dry friction identified on the Qualisys runs of 2026-10-23 together with mu 0.45 (isaac_sim/real_id)
parser.add_argument('--roller_friction', type=float, default=0.10)
# hips and tracks use the same actuator (same K_t): moteus max_current_A 80 / 40 A, gear 9 / 15, so the track motor
# gives 40 x 15 / (80 x 9) = 0.83 times the hip torque at the sprocket. Velocity gain of the real robot (moteus Kd 0.9 N m s/rad at the sprocket, Kp 0): about
# 8 times lower than the 2 Nm s/rad used before, it leaves the tracks behind their target under load (the simulated
# rotation in place drops from about 95% to 60% of the command)
parser.add_argument('--hip_max_torque', type=float, default=10.0, help='torque limit of a hip motor [Nm]')
parser.add_argument('--track_max_torque', type=float, default=None,
                    help='torque limit of a track [Nm at r 15 mm] (default: 0.83 x hip limit at the 45 mm sprocket, '
                         '2.78 Nm = 185 N of belt force with 10 Nm)')
parser.add_argument('--track_damping', type=float, default=0.9 * (0.015 / 0.045) ** 2,
                    help='velocity gain of a track [Nm s/rad]: real Kd 0.9 at the 45 mm sprocket referred to the 15 mm wheels')
# measured on the straight runs of 2026-10-23 (torque of the track drives vs speed, Qualisys): 1.16 + 0.013 |w| N m
parser.add_argument('--track_friction', type=float, default=1.16, help='Coulomb friction of a track [N m at the sprocket]')
parser.add_argument('--track_viscous', type=float, default=0.013,
                    help='viscous friction of a track [N m s/rad at the sprocket]')
parser.add_argument('--wheel_armature', type=float, default=0.001, help='armature of each driven wheel [kg m^2]')
parser.add_argument('--hfe_offset', type=float, default=-0.05,
                    help='geometric leg angle - encoder angle [rad] (alpha = sigma q_HFE)')
parser.add_argument('--contact_offset', type=float, default=0.003, help='wheels, rollers and track bodies [m]')
SPROCKET_RADIUS = 0.045   # m of belt per rad of the track joint (wheel_rad of the robot, Qualisys 2026-10-22)
args, _ = parser.parse_known_args()
if args.track_max_torque is None:
    args.track_max_torque = args.hip_max_torque * (40.0 * 15.0) / (80.0 * 9.0) * 0.015 / SPROCKET_RADIUS

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({'headless': True})

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdPhysics  # noqa: E402

DEG = math.pi / 180.0   # USD angular drive gains are per degree


def drive(prim, kp, kd, fmax):
    d = UsdPhysics.DriveAPI.Apply(prim, 'angular')
    d.CreateTypeAttr().Set('force')
    d.CreateStiffnessAttr().Set(kp * DEG)
    d.CreateDampingAttr().Set(kd * DEG)
    d.CreateMaxForceAttr().Set(fmax)
    d.CreateTargetPositionAttr().Set(0.0)
    d.CreateTargetVelocityAttr().Set(0.0)


stage = Usd.Stage.Open(args.usd)
root = stage.GetDefaultPrim()
root.GetVariantSets().GetVariantSet('Physics').SetVariantSelection('physx')
stage.Load()

driven_per_track = 4
wheel_max = args.track_max_torque / driven_per_track
wheel_kd = args.track_damping / driven_per_track
wheel_friction = args.track_friction * (0.015 / SPROCKET_RADIUS) / driven_per_track
wheel_viscous = args.track_viscous * (0.015 / SPROCKET_RADIUS) ** 2 / driven_per_track


def joint_friction(prim, coulomb, viscous=None):
    prim.ApplyAPI('PhysxJointAxisAPI', 'angular')
    values = {'staticFrictionEffort': coulomb, 'dynamicFrictionEffort': coulomb}
    if viscous is not None:
        values['viscousFrictionCoefficient'] = viscous * DEG   # per degree, as the drive gains
    for attr, v in values.items():
        a = prim.GetAttribute(f'physxJointAxis:angular:{attr}')
        if not a:
            a = prim.CreateAttribute(f'physxJointAxis:angular:{attr}', Sdf.ValueTypeNames.Float)
        a.Set(v)


def drive_wheel(prim):
    drive(prim, 0.0, wheel_kd, wheel_max)
    PhysxSchema.PhysxJointAPI.Apply(prim).CreateArmatureAttr().Set(args.wheel_armature)
    joint_friction(prim, wheel_friction, wheel_viscous)

counts = {'hfe': 0, 'wheel': 0, 'ankle': 0, 'roller': 0, 'articulation': 0, 'contact_offset': 0}
for prim in stage.Traverse():
    name = prim.GetName()
    path = prim.GetPath()
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        PhysxSchema.PhysxArticulationAPI.Apply(prim).CreateEnabledSelfCollisionsAttr().Set(False)
        counts['articulation'] += 1
    if prim.HasAPI(UsdPhysics.CollisionAPI) and (
            '/wheel_' in str(path) or '/body_left_' in str(path) or '/body_right_' in str(path)):
        col = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        col.CreateContactOffsetAttr().Set(args.contact_offset)
        col.CreateRestOffsetAttr().Set(0.0)
        counts['contact_offset'] += 1
    if not prim.IsA(UsdPhysics.RevoluteJoint):
        continue
    if name.endswith('_HFE'):
        drive(prim, 60.0, 0.6, args.hip_max_torque); counts['hfe'] += 1
        # zero of the joint: alpha_geometric = alpha_encoder + c, q_geometric = q + sigma c, i.e. R(q) R(sigma c) between
        # the joint frames: localRot1 <- localRot1 R_axis(-sigma c) (only the difference to the offset already applied)
        sigma = 1.0 if name[:2] in ('LF', 'RH') else -1.0
        applied = prim.GetAttribute('traquad:hfeOffset')
        old = applied.Get() if applied and applied.Get() is not None else 0.0
        delta = args.hfe_offset - old
        if abs(delta) > 1e-12:
            j = UsdPhysics.Joint(prim)
            axis = {'X': Gf.Vec3d(1, 0, 0), 'Y': Gf.Vec3d(0, 1, 0), 'Z': Gf.Vec3d(0, 0, 1)}[
                UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get()]
            r1 = Gf.Rotation(Gf.Quatd(j.GetLocalRot1Attr().Get()))
            r1 = Gf.Rotation(axis, math.degrees(-sigma * delta)) * r1   # GfRotation: left operand applied first
            j.GetLocalRot1Attr().Set(Gf.Quatf(r1.GetQuat()))
        prim.CreateAttribute('traquad:hfeOffset', Sdf.ValueTypeNames.Double, custom=True).Set(args.hfe_offset)
    elif name.startswith('joint_wheel_'):
        drive_wheel(prim); counts['wheel'] += 1
    elif name.endswith('_ankle'):
        drive(prim, 0.0, 0.01, 1000.0); counts['ankle'] += 1
    elif '_roller_' in name:
        drive(prim, 0.0, args.roller_damping, 1000.0)
        # dry friction: the roller turns only when the torque on it exceeds this value
        joint_friction(prim, args.roller_friction)
        counts['roller'] += 1
stage.GetRootLayer().Save()

print('finalized', counts, f'wheel kd {wheel_kd:g} max {wheel_max:g} Nm friction {wheel_friction:g} Nm + '
      f'{wheel_viscous:g} Nm s/rad', flush=True)
app.close()
