# PolyZamboni - Papercraft in Blender

![banner](images/PolyZamboniBannerNoText.jpg)

Turn your low-poly creations into papercraft instructions with this Blender addon. You no longer need any additional software or good faith in some automatic unfolding algorithm. PolyZamboni provides you with all tools necessary to create high quality paper models!  

## Requirements
This addon uses some external packages not included in the Python installation that comes with Blender. All required packages are listed in `requirements.txt`.

### Here is one way of installing these packages via pip:
Open your preferred command shell and navigate to the Python binary folder of your Blender installation.  

```bash
cd 'some_path_prefix\Blender Foundation\Blender 4.3\4.3\python\bin'
```
For reference, on my Windows machine the exact path to the folder is    
`'C:\Program Files\Blender Foundation\Blender 4.3\4.3\python\bin'`.

Then install the missing packages using pip.

```bash
./python -m pip install -r "some_path_prefix\PolyZamboni\requirements.txt" --target="..\lib\site-packages" --upgrade
```
Make sure to give the correct path to the requirements file by replacing `some_path_prefix` with the location of the PolyZamboni folder on your computer.

### Some notes:
- Make sure that you have permission to write and modify files in the python folder. The easiest way would be to open your shell as admin.
- This installation guide uses Windows path formats, but the same steps should also do the trick on Linux and macOS systems (not tested yet).

## Addon installation
Now the addon can be added to blender. For this, zip the `PolyZamboni` folder containing this readme file and the `__init__.py` file. Then go into Blender and go to **Edit-><ins>P</ins>references**. On the top left of the **Add-ons** tab, you can now click on **<ins>I</ins>nstall from Disk** and select the zipped `PolyZamboni` folder. The addon should now be enabled and ready to go.

![screenshot](images/addon_install_from_disc.png)

## Usage
With the addon enabled, select any mesh object and navigate to the PolyZamboni panel in the viewport sidebar (toggle with `'N'`) and click on the `Unfold this mesh` button. This starts the unfolding process of the selected mesh with no initial cuts.  
Not all meshes can be processed by PolyZamboni. The following things prevent this addon from working for your mesh:

1. Non-manifold vertices or edges
2. Faces that touch at more than one edge
3. Faces that can not be triangulated via PolyZambonis custom triangulation angorithm
4. Faces that are highly non-planar

You will get a warning whenever one of these fail-cases occur and the option to select all faces that need to be fixed.

### Get started
<p float="middle">
    <img src ="images/SpotUnfoldingProcess.gif" width=350>
    <img src ="images/SpotUnfoldingResult.png" width=350>
</p>

### Interactive editing 

<img src ="images/SpotFeedbackAnnotated.png" width=700>