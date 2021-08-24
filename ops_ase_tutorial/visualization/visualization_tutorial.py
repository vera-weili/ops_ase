from pathlib import Path

from ase.io.vasp import read_vasp
from ase.visualize import view
import numpy as np

from random import randint
from visualization.mview1 import Mview, ViewData
from mayavi import mlab
from time import sleep
from tvtk.tools import visual


# atoms = read_vasp(Path(__file__).parent / 'POSCAR')
# view(atoms)
figsize=(800, 800)
bgcolor=(0,0,0)
poscar_files = ['POSCAR', 'POSCAR_1', 'POSCAR_2']

from mayavi import mlab
from tvtk.tools import visual
# Create a figure
f = mlab.figure(size=(500,500))
# Tell visual to use this as the viewer.
visual.set_viewer(f)

# A silly visualization.
# data = mlab.test_plot3d()

# Even sillier animation.
# b1 = visual.box()
# b2 = visual.box(x=4.)
# b3 = visual.box(x=-4)
# b1.v = 5.0

view_data = Mview(poscar_files[0], ViewData(atoms={}))
@mlab.show
@mlab.animate(delay=1000)
def anim():
    """Animate the b1 box."""
    for i in range(100):
        Mview(poscar_files[i%len(poscar_files)], view_data=view_data)
    # while 1:
    #     data.mlab_source.x += b1.v*0.1
    #     b1.x = b1.x + b1.v*0.1
    #     if b1.x > 2.5 or b1.x < -2.5:
    #         b1.v = -b1.v
        yield

# Run the animation.
f.scene.render()
anim()


