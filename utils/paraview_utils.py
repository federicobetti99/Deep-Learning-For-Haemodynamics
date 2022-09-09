import pickle
import sys
sys.path.append('/usr/lib/paraview/site-packages')
sys.path.append('/usr/lib/paraview/')
from paraview.simple import *
from paraview import servermanager
from paraview import numpy_support as ns
import numpy as np
import scipy.stats as st


# PARAVIEW UTILS

def take_snapshots():
    """
    Take snapshots from Paraview
    :return:
    """
    count = 0
    num_views = 5
    animationScene = GetAnimationScene()
    tk = GetTimeKeeper()
    timesteps = tk.TimestepValues
    warp = WarpByVector()
    warp.Vectors = "displacement"
    Show(warp)
    Render()
    vel = ResampleToImage()
    Show(vel)
    Render()
    Hide(warp)
    Hide(reader)
    view = GetActiveView()
    view.ViewSize = (200, 200)
    view.Background = [0.7294117647058823, 0.7411764705882353, 0.7137254901960784]
    view.OrientationAxesVisibility = 0
    Render()
    readerRep = GetRepresentation()
    ColorBy(readerRep, ("POINTS", "velocity"))
    readerRep.Representation = "Volume"
    readerRep.SetScalarBarVisibility(view, False)
    Render()
    lut = GetColorTransferFunction("velocity")
    lut.ApplyPreset("X Ray", True)
    lut.RescaleTransferFunction(0.0, 15.0)
    lut.InvertTransferFunction()
    lut.UseAboveRangeColor = True
    Render()
    for i in range(len(timesteps)):
        animationScene.AnimationTime = timesteps[i]
        for j in range(num_views):
            set_view(j)
            initial_noise = np.sqrt(3) * st.norm.rvs(size=1)
            camera = GetActiveCamera()
            camera.Roll(initial_noise)
            SaveScreenshot("snapshots_" + str(branch) + "/snap_" + str(count) + ".png")
            camera.Roll(-initial_noise)
            count += 1
            noise = np.sqrt(3) * st.norm.rvs(size=1)
            if j == 0 or j == 1 or j == 3:
                camera.Azimuth(30 + noise)
            if j == 1 or j == 4:
                camera.Azimuth(-30 + noise)
            Render()
            SaveScreenshot("snapshots_" + str(branch) + "/snap_" + str(count) + ".png")
            count += 1
            if j == 0 or j == 1 or j == 3:
                camera.Azimuth(-30 - noise)
            if j == 1 or j == 4:
                camera.Azimuth(30 - noise)
            if j == 2 or j == 4:
                reset_view(j)


def set_view(num_view):
    """

    """
    camera = GetActiveCamera()

    if num_view == 0:
        camera.SetViewUp(1.0, 0.0, 0.0)
        camera.Elevation(90)
        camera.SetViewUp(0.0, 0.0, 1.0)
        camera.Roll(45)
        Render()

    if num_view == 1:
        camera.Roll(-135)
        Render()

    if num_view == 2:
        camera.Roll(-135)
        Render()

    if num_view == 3:
        camera.SetViewUp(0.0, -1.0, 0.0)
        camera.Elevation(90)
        camera.SetViewUp(0.0, 0.0, 1.0)
        camera.Roll(-45)
        camera.Azimuth(10)
        Render()

    if num_view == 4:
        camera.Azimuth(-10)
        camera.Roll(225)
        camera.Azimuth(50)
        Render()


def reset_view(num_view):

    camera = GetActiveCamera()

    if num_view == 2:
        camera.SetViewUp(0.0, 0.0, -1.0)
        camera.Elevation(-90)
        camera.SetViewUp(0.0, 1.0, 0.0)
        Render()

    if num_view == 4:
        camera.Azimuth(-50)
        camera.Roll(-180)
        camera.SetViewUp(0.0, 0.0, -1.0)
        camera.Elevation(-90)
        camera.SetViewUp(0.0, 1.0, 0.0)
        Render()


def compute_normals(indices_list):
    """
    Compute and store normal vectors to a set of points
    """
    ExtractSurface()
    GenerateSurfaceNormals()
    Render()
    source = GetActiveSource()
    d = servermanager.Fetch(source)
    Normals = ns.vtk_to_numpy(d.GetPointData().GetArray("Normals"))
    NOI = {}
    for i in range(len(indices_list)):
        current_normals = []
        for j in range(len(indices_list[i])):
            current_normals.append(Normals[indices_list[i][j]])
        NOI.update({i: current_normals})

    filename = "normals.pkl"
    with open(filename, "wb") as handle:
        pickle.dump(NOI, handle, protocol=pickle.HIGHEST_PROTOCOL)


reader = XDMFReader(FileNames="/home/betti/snapshots/snapshots_0/block0.xmf")
Show(reader)
Render()
indices_0 = [2201, 1875, 2221, 730, 1872, 1565, 1260, 1840, 2151, 2037, 2038, 1839, 1838, 363, 2150,
             2068, 1160, 295, 247, 248, 361, 362, 57, 1951, 2220, 2389, 2219, 2285, 2053, 2055, 2148]
indices_1 = [339, 337, 1512, 901, 863, 1566, 1381, 445, 444, 899, 900, 1401, 1462, 1489, 1491, 77, 153,
             533, 655, 656, 279, 395, 549, 291, 44, 45, 1377, 1181, 2343, 548]
indices_2 = [513, 262, 303, 816, 817, 2007, 1285, 1283, 304, 471, 472, 419, 1067, 553, 1802,
             2045, 743, 1025, 699, 744, 687, 688, 2064, 898, 700, 422, 864, 421, 425, 426]
indices_3 = [384, 406, 407, 416, 544, 383, 487, 515, 728, 301, 654, 367, 201, 202, 203, 127, 157,
             350, 385, 2250, 1282, 2287, 857, 941, 71, 72, 64, 1277, 1159]
list_indices = [indices_0, indices_1, indices_2, indices_3]
compute_normals(list_indices)
