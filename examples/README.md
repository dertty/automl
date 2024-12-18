### This folder contains examples of launching AutoML

___
#### Ways to launch AutoML:
1. From `ipykernel` in `jupyter`. Example: `All_together.ipynb`. ***For small datasets and basic data analysis***.
2. From `.py` file in terminal. Example: `launch_automl.py`. ***For big datasets and experiments***.

#### Why NOT to launch in `jupyter`:
1. ipykernel behavior may be unpredictable. Kernel may shutdown/restart suddenly.
2. After reloading the web-page standard output will stop displaying logs.

___
#### How to launch from terminal effectively:
1. Use `tmux` to create the terminal session that will not shutdown after you disconnect from server. `tmux` is **not** installed on `Linux` distros by default.
2. Use `nohup` to create the background process that will not shutdown after you disconnect from server. `nohup` is installed on `Linux` distros by default.


#### Sample code for launching AutoML with `tmux`
```sh
# create new tmux session
tmux new -s launch_automl

# launch automl inside the created session
python launch_automl.py

# IMPORTANT!!! Detach from session. Otherwise it will be killed when you disconnect
# Ctrl-b + d

# kill the session
tmux kill-session -t launch_automl
```


#### Sample code for launching AutoML with `nohup`
```sh
# creates the background process, stores the process output to nohup.out
nohup python launch_automl.py &

# view logs
vim nohup.out

# Get the PID
ps -ef | grep -i launch_automl

# Kill the process by PID
kill XXX

# display the logs in terminal
# shows recent 15 lines of nohup.out every 10 seconds
watch -n 10 tail -n 15 nohup.out
```






