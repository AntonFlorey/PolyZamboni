# PolyZamboni - The ultimate papercraft addon for Blender

Some cool text describing this addon...

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
Now the addon can be added to blender. For this, zip the `PolyZamboni` folder containing this readme file and the `__init__.py` file. Then go into Blender and go to **Edit-><ins>P</ins>references**. On the top left of the **Add-ons** tab, you can then click on **<ins>I</ins>nstall from Disk** and select the zipped `PolyZamboni` folder. The addon should now be enabled and ready to go :D

![screenshot](images/addon_install_from_disc.png)