import os
import sys

if "--app-mode" in sys.argv:
    option_index = sys.argv.index("--app-mode")
    mode_index = option_index + 1
    if mode_index < len(sys.argv):
        os.environ["RULA_APP_MODE"] = sys.argv[mode_index]
        del sys.argv[option_index:mode_index + 1]

from .server import main

main()
