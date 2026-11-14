@echo off
rem Fabrique RadioLab.exe : un dossier autonome qui fonctionne sans installer Python.
rem A lancer une fois, sur un PC ou RadioLab fonctionne deja (Python et les bibliotheques installes).
cd /d "%~dp0"
python -m pip install --upgrade pyinstaller
if errorlevel 1 goto erreur
python -m PyInstaller --noconfirm --clean --windowed --name RadioLab --collect-all pyqtgraph --collect-all sounddevice run_radiolab.py
if errorlevel 1 goto erreur
rem Les DLL de l Airspy doivent se trouver a cote de RadioLab.exe.
for %%F in (airspy.dll libusb-1.0.dll pthreadVC2.dll pthreadVCE2.dll libwinpthread-1.dll) do if exist "%%F" copy /y "%%F" "dist\RadioLab\" >nul
echo.
echo Termine : le dossier dist\RadioLab contient RadioLab.exe.
echo Copier ce dossier entier sur l autre PC (le pilote de l Airspy doit y etre installe).
pause
exit /b 0
:erreur
echo.
echo La fabrication a echoue : lire les messages ci-dessus.
pause
exit /b 1
