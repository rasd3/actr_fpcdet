cd det3d/ops/dcn 
python3 setup.py build_ext --inplace

cd .. && cd  iou3d_nms
python3 setup.py build_ext --inplace

cd .. && cd furthest_point_sample
python3 setup.py build_ext --inplace

cd .. && cd gather_points
python3 setup.py build_ext --inplace

cd .. && cd group_points
python3 setup.py build_ext --inplace

cd .. && cd ball_query
python3 setup.py build_ext --inplace

cd .. && cd knn
python3 setup.py build_ext --inplace
