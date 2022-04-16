#!/bin/bash
if [ "$BASH" != "/bin/bash" ]; then
    /bin/bash ./build.sh $1
    exit $?
fi

TARGETS="PremierPas Communication NaiveBroadcast PipeBroadcast"
# Exit immediately if a command exits with a non-zero status.
set -e
if [ "$1" != "" ]; then
    TARGETS=$1
fi

echo -e '\e[1;32;41mTarget is ' ${TARGETS} '\e[0m'
if [ -d linux ]; then
    echo -e '\e[1;32;41mFolder "linux" already exits\e[0m'
else
    mkdir linux
fi
cd linux
cmake ..
for exo in ${TARGETS}
    do
        echo -e '\e[1;32;104mCompilation de' ${exo} '\e[0m'
        cmake --build . --config Release --target ${exo}
    done
cd ..
