# PolyZamboni - The ultimate papercraft Blender addon :)

Some cool text describing this addon...

## Requirements installation
This addon uses some external packages not included in the Python installation that comes with Blender. All required packages are listed in `requirements.txt`.

Here is one way of installing these packages via pip:

### Step 1
Find the Python binary path of your Blender installation. The path should look like this:  
`'some_path_prefix\Blender Foundation\Blender 4.3\4.3\python\bin'`  
In case of my windows machine, I get the following path:  
`'C:\Program Files\Blender Foundation\Blender 4.3\4.3\python\bin'`

### Step 2
Find the path to the `requirements.txt` file of this package. The path should look like this:  
`'some_path_prefix\PolyZamboni\requirements.txt'`  
In case of my windows machine, I get the following path:  
`'E:\Dev\PolyZamboni\requirements.txt'`

### Step 3
Open your preferred command shell and navigate to the Python binary folder of your Blender installation.  

```bash
cd 'some_path_prefix\Blender Foundation\Blender 4.3\4.3\python\bin'
```

Then install the missing packages using pip.

```bash
./python -m pip install -r "some_path_prefix\PolyZamboni\requirements.txt" --target="..\lib\site-packages" --upgrade
```

### Some notes:
- Make sure that you have permission to write and modify files in the python folder. The easiest way would be to open your shell as admin.
- This installation guide uses Windows path formats and commands, but the same steps should also do the trick on Linux and macOS systems (not tested yet).

## Addon installation
Now the addon can be added to blender