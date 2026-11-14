"""Point d'entrée pour fabriquer RadioLab.exe (voir fabriquer_exe.bat). Pour un lancement normal : RadioLab.bat."""
import sys
from radiolab.gui import main

if __name__ == "__main__":
    sys.exit(main())
