## Commander's LoRa RX: Location Fusion & Visualisation

Replay data reception and Visualise path in a pack of 10 Odometry outputs by

```
./rx_lora.py LST 10
```

We investigated localisation performance in a department building scenario as below. Red triangles are deployed LoRa RX nodes. We mainly traversed the square-shaped corridors and the middle office. Extended Kalman Filter (EKF) and Particle Filters (PF) are experimented for positioning fusion.

![rhb](https://github.com/zdai257/LoRaRX/blob/main/Trajectory/rhb_RXs.png)

### Extended Kalman Filter

Three-round right-hand search along corridor:

![left3](https://github.com/zdai257/LoRaRX/blob/main/Trajectory/demo0324Left3.png)

Two-round walk and searching the middle room:

![leftvicon2](https://github.com/zdai257/LoRaRX/blob/main/Trajectory/demo0324LeftVicon2.png)

In and out from a gound-floor apartment to an open car park with DeepTIO:

![ApartmentInOut3](https://github.com/zdai257/LoRaRX/blob/main/Trajectory/ApartmentInOut3.png)

### SIR Particle Filter

Two-round right-hand search using Particle Filter instead:

![pf_left2](https://github.com/zdai257/LoRaRX/blob/main/Trajectory/demo0.png)
