"""Generate the barrel-shaped roller meshes of the barrel track model (track_model:=barrel in track.xacro).

A wheel of radius R carries n rollers. Each roller turns about a chord of the wheel at distance d from the wheel axis
(roller frame: axis x along the chord, origin at the chord midpoint, +y pointing away from the wheel axis). Its surface
is the solid of revolution of the wheel's outer arc over +-pi/n: rho(s) = sqrt(R^2 - s^2) - d, s in [-R sin(pi/n),
R sin(pi/n)]. The rollers together have the wheel circle as outer envelope: the axle height does not ripple as the
wheel turns (only the tessellation is left, M chords along the profile and N sides around the axis), as in real omni
wheels. d is chosen so that the roller ends have radius END_RADIUS.

Run from this folder:  python3 make_barrel_roller.py [n ...]   (default: 4 6 8) -> barrel_roller_n<n>.stl
"""
import math
import struct
import sys

R = 0.015            # wheel radius [m]
END_RADIUS = 0.0024  # roller radius at its ends [m]
M = 5                # chords along the profile (6 x 10 = 60 hull vertices: PhysX GPU takes at most 64)
N = 10               # sides around the roller axis


def chord_distance(n):
    return R * math.cos(math.pi / n) - END_RADIUS


def barrel(n):
    d = chord_distance(n)
    half = R * math.sin(math.pi / n)
    rings = []
    for i in range(M + 1):
        s = -half + 2 * half * i / M
        rho = math.sqrt(R * R - s * s) - d
        rings.append([(s, rho * math.cos(2 * math.pi * k / N), rho * math.sin(2 * math.pi * k / N)) for k in range(N)])
    tris = []
    for i in range(M):
        for k in range(N):
            a, b = rings[i][k], rings[i][(k + 1) % N]
            c, e = rings[i + 1][k], rings[i + 1][(k + 1) % N]
            tris += [(a, c, b), (b, c, e)]
    for ring, sign in ((rings[0], -1), (rings[-1], 1)):   # end caps
        center = (ring[0][0], 0.0, 0.0)
        for k in range(N):
            p, q = ring[k], ring[(k + 1) % N]
            tris.append((center, q, p) if sign < 0 else (center, p, q))
    return tris


def write_stl(path, tris):
    with open(path, 'wb') as f:
        f.write(b'barrel roller'.ljust(80, b' '))
        f.write(struct.pack('<I', len(tris)))
        for a, b, c in tris:
            u = [b[i] - a[i] for i in range(3)]
            v = [c[i] - a[i] for i in range(3)]
            nrm = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
            ln = math.sqrt(sum(x * x for x in nrm)) or 1.0
            f.write(struct.pack('<3f', *[x / ln for x in nrm]))
            for p in (a, b, c):
                f.write(struct.pack('<3f', *p))
            f.write(b'\0\0')


if __name__ == '__main__':
    for n in [int(x) for x in sys.argv[1:]] or [4, 6, 8]:
        write_stl(f'barrel_roller_n{n}.stl', barrel(n))
        print(f'barrel_roller_n{n}.stl: d = {chord_distance(n) * 1000:.2f} mm, roller radius '
              f'{(R - chord_distance(n)) * 1000:.2f} mm (middle) / {END_RADIUS * 1000:.1f} mm (ends), '
              f'length {2 * R * math.sin(math.pi / n) * 1000:.2f} mm')
