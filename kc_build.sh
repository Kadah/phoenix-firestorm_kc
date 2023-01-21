#!/usr/bin/env bash

eval "$(conda shell.bash hook)"
conda activate sl

export AUTOBUILD_VARIABLES_FILE=$HOME/Dev/sl/fs-build-variables/variables

autobuild installables edit fmodstudio platform=linux64 hash=1e840d7e3d9a71f64031f77161e26605 url=file:////opt/sl_3p/fmodstudio-2.02.11-linux64-230210406.tar.bz2

time autobuild configure -A 64 -c ReleaseFS_open -- -DPACKAGE:BOOL=On --chan="KC" --fmodstudio -DUSE_SDL2:BOOL=On --clean

time autobuild build -A 64 -c ReleaseFS_open -- -DPACKAGE:BOOL=On --chan="KC" --fmodstudio -DUSE_SDL2:BOOL=On
