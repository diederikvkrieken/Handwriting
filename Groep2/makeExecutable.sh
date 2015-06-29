#!/bin/bash
clear

function test {
    "$@"
    local status=$?
    if [ $status -ne 0 ]; then
        echo "!!!! error with $1 !!!!" >&2
        exit 0
    fi
    return $status
}

echo "1. Removing old build folder"
test rm -rf build/

echo "2. Making new empty build folder"
test mkdir build/

echo "3. Copying classifier files"
test cp *_RF.pkl build/
test cp VRF.pkl build/

# add your files here with cp (see example below)
# cp ... build/

echo "4. Freezing"
test cxfreeze recognizerMulti.py --target-dir build --include-modules=scipy.special._ufuncs_cxx,scipy.sparse.csgraph._validation,scipy.integrate.vode,scipy.integrate.lsoda,skimage._shared.geometry,email.mime.text,email.mime.image,email.mime.multipart,email.mime.audio,email.mime.message,matplotlib.backends.backend_qt5agg,skimage.filters.rank.core_cy,sklearn.utils.lgamma,sklearn.tree._utils,sklearn.utils.weight_vector,sklearn.neighbors.typedefs,sklearn.utils.sparsetools._graph_validation

clear
echo "Done!"