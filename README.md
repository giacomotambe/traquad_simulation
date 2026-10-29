# TraQuad simulation

Simulation of the TraQuad (Mulinex on tracks) in Isaac Sim 6.1 / Isaac Lab 3.0, plus the older Gazebo setup below.

- `isaac_sim/`: the robot asset (`assets/traquad`), the replica of the real controllers
  (`traquad_real_controller.py`), the replay of the real runs against the Qualisys ground truth (`real_replay.py`,
  `real_id/`) and the test scripts. Start from `isaac_sim/README.md`: current model, identified parameters, how to run,
  changelog.
- `isaaclab_overlay/`: the TraQuad tasks and robot config for Isaac Lab, linked into `isaaclab_traquad/` by
  `isaaclab_overlay/setup_isaaclab.sh`.
- `open_traquad.py`: the robot in Isaac Lab with joystick-like velocity commands.
- `real_robot/traquad_missions.py`: identification missions for the real robot (run on the NUC).
- `src/mulinex_description`: xacro of the robot (`traquad.xacro`, tracks in `track.xacro`).
- Real runs with the Qualisys: github.com/giacomotambe/traquad_bags; robot code: github.com/giacomotambe/mulsbc_ws
  (SBC) and github.com/giacomotambe/mulinex_app (NUC).

# Dockerized Gazebo Harmonic Simulator 
## introduction
This repository contains a docker compose structure to simulate robots in Gazebo Sim Fortress supported by ROS2 humble on ubuntu 22.04.
The Repository purpose is to provide an easy set up to allow robotic software developer to test their own controller in simulation.
## prerequisite  

Docker Engine: follow the docker engine installation [here](https://docs.docker.com/engine/install/ubuntu/) and then enable the docker usage as non-root user as specified [here](https://docs.docker.com/engine/install/linux-postinstall/)

Install Nvidia Driver: TODO

Nvidia-Container-Toolkit: follow the nvidia-container-toolkit following [this](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Build 

``` bash compose_build.bash ```

## Start Compose 

``` docker compose up -d ```

``` docker compose exec ros2_humble_sim bash```

This command launches a docker with ros inside it. The first time you open it, you have to run 

``` colcon build --symlink-install```


## NVIDIA stuff test 

To test the correct isntallation of the nvidia support you have to start the container and call

``` nvidia-smi```

## HOW TO BUILD: 
If, after deleting the folders "launch", "install" and "log", when trying to build, you will receive an error: ...

Do the following:
- ```rm -rf build install log```
- ```colcon build --packages-select pi3hat_moteus_int_msgs --symlink-install```
- ```source install/setup.bash```
- ```colcon build --symlink-install```
- ```source install/setup.bash```

Ignore the warnings and just do whatever you need to.


## LUNCH ROBOT SIMULATION

#### TERMINAL 1:
This terminal is used to run Gazebo

``` source install/setup.bash ```

``` ros2 launch mulinex_ignition gz_harmonic_sim_w_rbt.launch.py ```
#### TERMINAL 2:
This terminal is used to run the node to move the robot

``` source install/setup.bash ```

``` ros2 run mulinex_ignition-py getup ```



## Move the robot
There are three controllers: 
- "omni_control" that, given a reference velocity, solves the inverse kinematics and moves the wheels to target the reference velocity.
    - topic from which it reads: /omni_control/command
    - topic message type: pi3hat_moteus_int_msgs/msg/OmniMulinexCommand 
    - To test this controller, ```ros2 topic pub /omni_control/command pi3hat_moteus_int_msgs/msg/OmniMulinexCommand "{v_x: 0.0, v_y: 10.0, omega: 10.0}"```
    Or, alternatively, run ``` ros2 run mulinex_ignition_py move_wheels ```
- "pd_control" that moves the leg joints using a PD controller.
    - topic from which it reads: /pd_controller/command
    - topic message type:JointState (from Sensor_msgs.msg)
    - To test this controller, run ``` ros2 run mulinex_ignition_py getup ```
- "wheels_vel_cnt", that moves the wheels individually, with a velocity reference.
    - topic from which it reads: /wheels_vel_controller/wheels_velocity_cmd
    - topic message type: WheelVelocityCommand (path: docker_simulator/src/custom_interfaces/msg/WheelVelocityCommand.msg)
    - To test this controller, run ```ros2 run mulinex_ignition_py move_wheels_2```

## RUN THE INFERENCE

### SETUP
- Go to IsaacSim > logs > desired network that you trained
- Copy the files:
    - nn/omniquad_flat.pth
    - params/agent.yaml
    - params/env.yaml
- Paste the 3 files in the Gazebo Folder: src/rlg_quad_controller/models


### RUN
On every terminal you start: 

``` docker compose exec ros2_humble_sim bash```

``` source install/setup.bash ```


#### TERMINAL 1:
This terminal is used to launch Gazebo, the controllers, etc.
``` ros2 launch mulinex_ignition gz_harmonic_sim_w_rbt_PD_wheels.launch.py ```

#### TERMINAL 2:
The keyboard to control the robot
``` ros2 run teleop_twist_keyboard teleop_twist_keyboard ```

#### TERMINAL 3:
The inference node
``` ros2 launch rlg_quad_controller omniquad_inference_robot.launch.py ```

## USEFUL COMMANDS
- ``` ros2 run rqt_graph rqt_graph ``` shows a graph with all the nodes and all the topics and who writes where
- LANCIARE UN CONTROLLER:  ```ros2 run controller_manager spawner pd_controller --param-file src/mulinex_ignition/config/mulinex_mf.yaml```
- Verificare i topic emessi da Gazebo (per poi poter fare il bridge): 
    - per vedere la lista: ```ign topic -l```
    - per fare l'echo: ```ign topic -e -t /mulinex/imu```
- Avvio manuale del bridge Gazebo --> ROS: ```ros2 run ros_gz_bridge parameter_bridge /mulinex/imu@sensor_msgs/msg/Imu[ignition.msgs.IMU```
- convertire il file rosbag .db3 in .mcap: 
    - salvarsi il file convert.yaml, entrarci e modificare il nome dell'output con il nome desiderato
    - lanciare il comando ```ros2 bag convert -i <nome_cartella_rosbag> -o convert.yaml```

## REGISTRARE LA ROSBAG:
Lanciare il comando ``` ros2 launch mulinex_ignition_py rosbag_rec.launch.py ```

## CONNETTERSI AL ROBOT FISICO:
Su un nuovo terminale, ti connetti al robot: 
- apri il docker
- ```ssh mulsbc@100.100.100.3```  --> password: 123456
- ```sudo su``` --> password 123456
- ```cd mulinex_ws/```
- ```source install/setup.bash```
- ```export ROS_DOMAIN_ID=1```
- ```ros2 launch pi3hat_hw_interface start_MJBots_Pi3Hat_hw_Interface.py urdf_file:=omni_mulinex.xacro conf_file:=omniquad_joy.yaml```
- Quando poi vorrai spegnere il robot, digita ```shutdown now```, poi puoi staccare la corrente al robot

Su un nuovo terminale, lancerai i nodi dal tuo PC per controllare il robot:
- apri il docker
- ```export ROS_DOMAIN_ID=1```
- ```ros2 launch mulinex_ignition_py controller_start_launch.py```
- ora lancia tutti i nodi che vuoi:
    - Getup: ```ros2 run mulinex_ignition_py getup_omni_1```
    - run della rosbag: ```ros2 bag play rosbag/run_20250827_140849 --topics /joint_controller/command```
    - INFERENCE:
        - su un terminale, devi lanciare il nodo che fa da bridge tra i topic di ROS e quelli del robot: ```ros2 run mulinex_ignition_py bridge_node```
        - su un altro terminale, lanci il nodo di inference: ```ros2 launch rlg_quad_controller omniquad_inference.launch.py```
        - su un terzo terminale, lancia la teleop_twist_keyboard




## TROUBLESHOOT
- If ``` colcon build ``` does not work the first time you launch it during the installation, try to do: 
    - ``` docker rm -f ros2_humble_simulator``` 
    - ``` docker compose up -d``` 
    - and then, again: ``` docker compose exec ros2_humble_sim bash```
- PROBLEMI INTERCORSI NEL PASSAGGIO GAZEBO --> ROS2:
    - scrittura di tutto il codice per controllers e inference
    - nel GitHub dei PhD di Unipi era presente una inference un po' diversa rispetto a quella di Gazebo. Quindi, a parità di obs dati in input alla rete, si ottenevano actions diverse. Abbiamo sostituito l'inference copiando e incollando la stessa funzione.
    - su IsaacSim i "joints_pos" all'interno del vettore obs contiene angoli di deviazione rispetto alla posizione di default (Knee: 120°, Hip: 60°; entrambi espressi in radianti) e quindi sono valori molto vicini a 0. Invece, su ROS2 le letture sono degli angoli rispetto alla posizione orizzontale, quindi occorre sottrarre il valore di default delle gambe prima di passare il vettore obs alla rete.
    - I valori delle angular velocity e delle projected gravity erano uguali a zero. Questo era dovuto al fatto che il bridge Gazebo->ROS non era settato bene. Gazebo pubblicava su un topic "mulinex/imu" mentre ROS cercava di creare il bridge per il topic "omnicar/imu". Si è quindi sostituito questo nome del topic in:
        - mulinex.xacro
        - file di launch "gz_harmonic_sim_W_rbt_PD_wheels.launch.py"
        - file di inference "inference_controller.py"

