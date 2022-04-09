python main.py --env_name predator_prey --nagents 3 --nprocesses 16 --num_epochs 2000 --hid_size 128 --detach_gap 10 --lrate 0.001 --dim 5 --max_steps 20 --commnet --vision 0 --recurrent


OMP_NUM_THREADS=1 python main.py --env_name traffic_junction --nagents 5 --nprocesses 16 --num_epochs 2000 --hid_size 128 --detach_gap 10 --lrate 0.001 --dim 6 --max_steps 20 --commnet --vision 0 --recurrent  --add_rate_min 0.1 --add_rate_max 0.3 --curr_start 250 --curr_end 1250 --difficulty easy


OMP_NUM_THREADS=1 python main.py --env_name traffic_junction --nagents 10 --nprocesses 16 --num_epochs 2000 --hid_size 128 --detach_gap 10 --lrate 0.001 --dim 14 --max_steps 40 --commnet --vision 0 --recurrent  --add_rate_min 0.2 --add_rate_max 0.05 --curr_start 250 --curr_end 1250 --difficulty medium


MP_NUM_THREADS=1 python main.py --env_name traffic_junction --nagents 20 --nprocesses 16 --num_epochs 2000 --hid_size 128 --detach_gap 10 --lrate 0.001 --dim 18 --max_steps 80 --commnet --vision 0 --recurrent  --add_rate_min 0.02 --add_rate_max 0.05 --curr_start 250 --curr_end 1250 --difficulty hard
