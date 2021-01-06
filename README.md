## Commander's LoRa RX: Location Log & Visualisation

Replay data reception and Visualise path in a pack of 10 Odometry outputs by

```
python3 rx_lora.py P2P 10
```

### SIR Particle Filter

Two rounds of left-hand search:

![left2](https://github.com/zdai257/LoRaRX/blob/main/demo0.png)

### Extended Kalman Filter

Two rounds of left-hand search with Constant-Angular-Velocity-and-Acceleration EKF which is decoupled from RSSI-based range:

![live_plot](https://github.com/zdai257/LoRaRX/blob/main/demo1.png)

Two rounds of left-Hand Search using the same model as above with heavy RSSI average filtering:

![sim_ekf_plot](https://github.com/zdai257/LoRaRX/blob/main/demo2.png)
