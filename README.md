## Commander's LoRa RX: Location Fusion & Visualisation

Replay data reception and Visualise path in a pack of 10 Odometry outputs by

```
python3 rx_lora.py LST 10
```

We investigated localisation performance in the Oxford building scenario as below. Red triangles are deployed LoRa RX nodes. We mainly traversed the square-shaped corridors and the middle room 209. LiDAR provided cm-magnitude of accuracy which is regarded as ground-truth path marked in grey. Extended Kalman Filter (EKF) and Particle Filters (PF) are utilized to conduct location fusion.

![rhb](https://github.com/zdai257/LoRaRX/blob/main/Trajectory/rhb_RXs.png)

### Extended Kalman Filter

Three-round right-hand search:

![left3](https://github.com/zdai257/LoRaRX/blob/main/Trajectory/demo0324Left3.png)

Two-round right-hand search entering Vicon room:

![leftvicon2](https://github.com/zdai257/LoRaRX/blob/main/Trajectory/demo0324LeftVicon2.png)

Two-round left-hand search entering Vicon room:

![rightvicon2](https://github.com/zdai257/LoRaRX/blob/main/Trajectory/demo0324RightVicon2.png)

### SIR Particle Filter

Two-round right-hand search using Particle Filter at 1Hz real-time refresh rate:

![pf_left2](https://github.com/zdai257/LoRaRX/blob/main/Trajectory/demo0.png)
