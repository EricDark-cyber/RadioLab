@echo off
rem Lance RadioLab (double-clic). Python 3.10+ 64 bits doit etre installe.
cd /d "%~dp0"
python -m radiolab
if errorlevel 1 pause
