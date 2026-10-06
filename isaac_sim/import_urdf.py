"""Convert a traquad URDF to USD with the Isaac Sim 6.1 URDF importer (python API, works with the pip install).

Same options as the standalone urdf_import.py used before: merge fixed joints, floating base, force drives with
position targets (gains and targets are set later by finalize_usd.py or at runtime).
The asset is written to <out_dir>/<robot name>/<robot name>.usda.

usage: ./isaac.sh import_urdf.py --urdf robot.urdf --out_dir out/
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--urdf', required=True)
parser.add_argument('--out_dir', required=True)
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({'headless': True})

from isaacsim.core.experimental.utils.app import enable_extension  # noqa: E402

enable_extension('isaacsim.asset.importer.urdf')

from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig  # noqa: E402

config = URDFImporterConfig(
    urdf_path=args.urdf,
    usd_path=args.out_dir,
    merge_fixed_joints=True,
    fix_base=False,
    joint_drive_type='force',
    joint_target_type='position',
    robot_type='Default',
)
path = URDFImporter(config).import_urdf()
print('Import complete:', path, flush=True)
app.close()
