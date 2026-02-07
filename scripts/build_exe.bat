
REM Build script for Neutracker EXE
REM Requires: pip install pyinstaller

python -m PyInstaller --name neutracker ^
            --onefile ^
            --windowed ^
            --add-data "neutracker;neutracker" ^
            neutracker/gui.py

echo "Build complete. Check dist/neutracker.exe"
