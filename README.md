## Commander's LoRa RX: Location Fusion & Visualisation

Replay data reception and Visualise path in a pack of 10 Odometry outputs by

```
python3 rx_lora.py LST 10
```

### Extended Kalman Filter

Three-round right-hand search:

![left2_fine](https://github.com/zdai257/LoRaRX/blob/main/Trajectory/demo0324Left3.png)

Two-round right-hand search entering Vicon room:

![left3_fine](https://github.com/zdai257/LoRaRX/blob/main/Trajectory/demo0324LeftVicon2.png)

Two-round left-hand search entering Vicon room:

![left3_fine](https://github.com/zdai257/LoRaRX/blob/main/Trajectory/demo0324RightVicon2.png)

### SIR Particle Filter

Two-round right-hand search using Particle Filter at 1Hz real-time refresh rate:

![pf_left2](https://github.com/zdai257/LoRaRX/blob/main/Trajectory/demo0.png)
