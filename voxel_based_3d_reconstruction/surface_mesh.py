import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage import measure
from skimage.draw import ellipsoid



def compute_volume(all_vis_voxels, xb, yb, zb, stepsize):
    volume = np.zeros((2*xb,2*yb,2*zb)).astype(bool)
    all_vis_voxels = all_vis_voxels//stepsize
    for [x,y,z] in all_vis_voxels:
        volume[z+zb][x+xb][y] = True
        #side: volume[z+zb][x+xb][y]
        #back: volume[x+xb][z+zb][y]
        #front: volume[x+xb][-z+zb][y]
    return volume

"""
Code from
https://scikit-image.org/docs/stable/auto_examples/edges/plot_marching_cubes.html
"""
def print_mesh_of_volume(volume):
    # Use marching cubes to obtain the surface mesh of these ellipsoids
    verts, faces, normals, values = measure.marching_cubes(volume, 0)

    # Display resulting triangular mesh using Matplotlib. This can also be done
    # with mayavi (see skimage.measure.marching_cubes docstring).
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")

    ax.set_xlim(70, 120)
    ax.set_ylim(70, 120)
    ax.set_zlim(0, 50)

    plt.tight_layout()
    plt.show()
