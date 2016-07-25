git submodule init
git submodule update
python setup.py build_ext --inplace
export PYTHONPATH="$PYTHONPATH:`pwd`"
cd other_trackers
python setup_struck.py build_ext --inplace
python setup_opentld.py build_ext --inplace
cd -
