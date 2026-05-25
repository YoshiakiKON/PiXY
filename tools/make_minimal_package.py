import zipfile
import os
base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
files = ['Main.py','Ui.py','Util.py','CalcCentroid.py','rendering.py','interactions.py','widgets.py','tables.py','Strings.py','pyproject.toml','requirements.txt','LICENSE','CITATION.cff','README.md','RELEASE_NOTES_v1.3.3.md','PiXY.png','PiXY_Pix.png','PiXY_XY.png','PiXY_icon.ico']
zipname = os.path.join(base, 'dist', 'PiXY_v1.3.3_minimal.zip')
os.makedirs(os.path.dirname(zipname), exist_ok=True)
with zipfile.ZipFile(zipname, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for f in files:
        p = os.path.join(base, f)
        if os.path.exists(p):
            z.write(p, arcname=f)
            print('added', f)
        else:
            print('missing', f)
print('WROTE', zipname)

