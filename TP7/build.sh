#!/bin/bash
TARGETS="exo1 exo2 exo3 exo4 exo5"
if [ "$1" != "" ]; then
    TARGETS=$1
fi
echo '\e[1;32;41mTarget is ' ${TARGETS} '\e[0m'
if [ -d linux ]; then
    echo '\e[1;32;41mFolder "linux" already exits\e[0m'
else
    mkdir linux
fi
cd linux
cmake ..
for exo in ${TARGETS}
    do
        echo '\e[1;32;104mCompilation de' ${exo} '\e[0m'
        cmake --build . --config Release --target ${exo}
    done
cd ..
