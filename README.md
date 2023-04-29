# Deep-Visual-Inertial-Odometry
## To preprocess the data, please first download the dataset and  run alignGT_trajectory.py script
``python3 Code/scripts/alignGT_trajectory.py``
## Then to train VO network, please run
``python3 Code/main.py --mode train --network_type vo --checkpoint_path ./checkpoint_vo``
## To test VO network, please run
``python3 Code/main.py --mode test --network_type vo --checkpoint_path ./checkpoint_vo``
## Then to train IO network, please run
``python3 Code/main.py --mode train --network_type io --checkpoint_path ./checkpoint_io``
## To test IO network, please run
``python3 Code/main.py --mode test --network_type io --checkpoint_path ./checkpoint_io``