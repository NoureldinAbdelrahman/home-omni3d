"""Minimal pure-Python stand-in for PyMesh, enough for AtlasNet templates/media."""
import numpy as np


class Mesh:
    def __init__(self, vertices, faces=None):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int64) if faces is not None else np.zeros((0, 3), dtype=np.int64)
        self._attrs = {}

    def get_attribute_names(self):
        return sorted(self._attrs.keys())

    def get_attribute(self, name):
        return self._attrs[name]

    def add_attribute(self, name):
        self._attrs.setdefault(name, np.zeros(len(self.vertices)))

    def set_attribute(self, name, value):
        self._attrs[name] = np.asarray(value)


def form_mesh(vertices, faces=None):
    return Mesh(vertices, faces)


def merge_meshes(meshes):
    verts = np.concatenate([m.vertices for m in meshes], axis=0)
    faces = []
    offset = 0
    for m in meshes:
        faces.append(m.faces + offset)
        offset += len(m.vertices)
    return Mesh(verts, np.concatenate(faces, axis=0) if faces else None)


def generate_icosphere(radius=1.0, center=None, subdivisions=3):
    """Icosahedron subdivision -> unit-ish sphere mesh."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=np.float64)
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int64)

    verts /= np.linalg.norm(verts, axis=1, keepdims=True)

    for _ in range(int(subdivisions)):
        mid_cache = {}
        new_faces = []
        vlist = verts.tolist()

        def midpoint(i, j):
            key = (min(i, j), max(i, j))
            if key in mid_cache:
                return mid_cache[key]
            v = (np.array(vlist[i]) + np.array(vlist[j])) / 2.0
            v /= np.linalg.norm(v)
            vlist.append(v.tolist())
            idx = len(vlist) - 1
            mid_cache[key] = idx
            return idx

        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]])
        verts = np.array(vlist, dtype=np.float64)
        faces = np.array(new_faces, dtype=np.int64)

    verts = verts * float(radius)
    if center is not None:
        verts = verts + np.asarray(center, dtype=np.float64)
    return Mesh(verts, faces)


def save_mesh(path, mesh, *attribute_names, ascii=True, **kwargs):
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(getattr(mesh, "faces", np.zeros((0, 3), dtype=np.int64)), dtype=np.int64)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for v in verts:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    return path


def load_mesh(path, **kwargs):
    verts, faces = [], []
    with open(path) as f:
        lines = f.readlines()
    i = 0
    if lines and lines[0].strip() == "ply":
        n_v = n_f = 0
        while i < len(lines):
            parts = lines[i].strip().split()
            if parts[:2] == ["element", "vertex"]:
                n_v = int(parts[2])
            elif parts[:2] == ["element", "face"]:
                n_f = int(parts[2])
            elif parts and parts[0] == "end_header":
                i += 1
                break
            i += 1
        for _ in range(n_v):
            verts.append([float(x) for x in lines[i].split()[:3]])
            i += 1
        for _ in range(n_f):
            toks = lines[i].split()
            faces.append([int(x) for x in toks[1:4]])
            i += 1
    else:
        for line in lines:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(x.split("/")[0]) - 1 for x in line.split()[1:4]])
    return Mesh(np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int64) if faces else None)
