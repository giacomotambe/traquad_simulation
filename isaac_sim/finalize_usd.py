"""Write the default physics settings of traquad into the imported USD, so the asset works as is
in Isaac Sim / Isaac Lab (Isaac Lab actuators still override the joints they control).

- Physics variant set to 'physx' (the importer leaves it unselected: no physics otherwise)
- articulation self-collisions off
- HFE: PD drive Kp 100 Nm/rad, Kd 10 Nm s/rad, max 5 Nm
- ankles: passive, damping 0.05 Nm s/rad
- driven wheels: velocity drive (stiffness 0), armature --wheel_armature. The motor of a track is described at the
  track: --track_max_torque and --track_damping (velocity gain), both referred to the 15 mm wheel radius, are split
  over the driven wheels of the track (4 in the roller model, 1 in the cylinder model)
- contact offset --contact_offset on the colliders of wheels, rollers and track bodies (rest offset 0)
- roller model (track_model:=rollers):
  - rollers: passive, damping --roller_damping, dry (Coulomb) friction --roller_friction [Nm]
- cylinder model (track_model:=cylinder, detected from the mimic joints):
  - wheel 2 of each track is driven; wheels 1, 3, 4 follow it through the mimic joints written by the importer
    (NewtonMimicAPI, read by PhysX too): no drive, and a finite (huge) limit on leader and followers, which PhysX
    requires for mimic joints

usage: ./isaac.sh finalize_usd.py --usd assets/traquad/traquad.usda [--roller_damping 1e-4] [--roller_friction 0.06]
"""
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument('--usd', required=True)
parser.add_argument('--roller_damping', type=float, default=1e-4)
parser.add_argument('--roller_friction', type=float, default=0.06)
# track motor: 40 Nm per track (10 Nm per wheel, as the model that matched the real robot's rotation; with 0.6 Nm per
# track the robot turned at 16% of the command). A velocity gain much lower than 2 Nm s/rad leaves the wheels behind
# their target under load
parser.add_argument('--track_max_torque', type=float, default=40.0, help='torque limit of a track [Nm at r 15 mm]')
parser.add_argument('--track_damping', type=float, default=2.0, help='velocity gain of a track [Nm s/rad]')
parser.add_argument('--wheel_armature', type=float, default=0.001, help='armature of each driven wheel [kg m^2]')
parser.add_argument('--contact_offset', type=float, default=0.003, help='wheels, rollers and track bodies [m]')
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({'headless': True})

from pxr import PhysxSchema, Sdf, Usd, UsdPhysics  # noqa: E402

DEG = math.pi / 180.0   # USD angular drive gains are per degree
MIMIC_LIMIT = 1e7       # [deg], ~28000 turns: PhysX mimic joints need finite limits on leader and followers


def drive(prim, kp, kd, fmax):
    d = UsdPhysics.DriveAPI.Apply(prim, 'angular')
    d.CreateTypeAttr().Set('force')
    d.CreateStiffnessAttr().Set(kp * DEG)
    d.CreateDampingAttr().Set(kd * DEG)
    d.CreateMaxForceAttr().Set(fmax)
    d.CreateTargetPositionAttr().Set(0.0)
    d.CreateTargetVelocityAttr().Set(0.0)


def set_limits(prim, limit):
    j = UsdPhysics.RevoluteJoint(prim)
    j.CreateLowerLimitAttr().Set(-limit)
    j.CreateUpperLimitAttr().Set(limit)


stage = Usd.Stage.Open(args.usd)
root = stage.GetDefaultPrim()
root.GetVariantSets().GetVariantSet('Physics').SetVariantSelection('physx')
stage.Load()

# mimic joints written by the importer from the URDF <mimic> tags (cylinder model)
mimic_of = {}
for prim in stage.Traverse():
    rel = prim.GetRelationship('newton:mimicJoint')
    if rel and rel.GetTargets():
        mimic_of[prim.GetPath()] = rel.GetTargets()[0]
leaders = set(mimic_of.values())
model = 'cylinder' if mimic_of else 'rollers'
driven_per_track = 1 if model == 'cylinder' else 4
wheel_max = args.track_max_torque / driven_per_track
wheel_kd = args.track_damping / driven_per_track


def drive_wheel(prim):
    drive(prim, 0.0, wheel_kd, wheel_max)
    PhysxSchema.PhysxJointAPI.Apply(prim).CreateArmatureAttr().Set(args.wheel_armature)

counts = {'hfe': 0, 'wheel': 0, 'wheel_follower': 0, 'ankle': 0, 'roller': 0, 'articulation': 0, 'contact_offset': 0}
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
        drive(prim, 100.0, 10.0, 5.0); counts['hfe'] += 1
    elif path in mimic_of:
        # follower wheel: moved by the mimic constraint only
        prim.RemoveAPI(UsdPhysics.DriveAPI, 'angular')
        set_limits(prim, MIMIC_LIMIT)
        counts['wheel_follower'] += 1
    elif path in leaders:
        drive_wheel(prim)
        set_limits(prim, MIMIC_LIMIT)
        counts['wheel'] += 1
    elif name.startswith('joint_wheel_'):
        drive_wheel(prim); counts['wheel'] += 1
    elif name.endswith('_ankle'):
        drive(prim, 0.0, 0.05, 1000.0); counts['ankle'] += 1
    elif '_roller_' in name:
        drive(prim, 0.0, args.roller_damping, 1000.0)
        # dry friction: the roller turns only when the torque on it exceeds this value
        prim.ApplyAPI('PhysxJointAxisAPI', 'angular')
        for attr in ('staticFrictionEffort', 'dynamicFrictionEffort'):
            a = prim.GetAttribute(f'physxJointAxis:angular:{attr}')
            if not a:
                a = prim.CreateAttribute(f'physxJointAxis:angular:{attr}', Sdf.ValueTypeNames.Float)
            a.Set(args.roller_friction)
        counts['roller'] += 1
stage.GetRootLayer().Save()
print('finalized', model, counts, f'wheel kd {wheel_kd:g} max {wheel_max:g} Nm', flush=True)
app.close()
