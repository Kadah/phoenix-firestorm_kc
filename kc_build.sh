#!/usr/bin/env bash

eval "$(conda shell.bash hook)"
conda activate sl

export AUTOBUILD_VARIABLES_FILE=$HOME/Dev/sl/fs-build-variables/variables

autobuild installables edit fmodstudio platform=linux64 hash=5734ad11b2c5079366e059797e09f57d url=file:////opt/sl_3p/fmodstudio-2.02.15-linux64-232270400.tar.bz2

time autobuild configure -A 64 -c ReleaseFS_open -- -DPACKAGE:BOOL=On --chan="KC" --fmodstudio -DUSE_SDL2:BOOL=On --clean

time autobuild build -A 64 -c ReleaseFS_open -- -DPACKAGE:BOOL=On --chan="KC" --fmodstudio -DUSE_SDL2:BOOL=On
