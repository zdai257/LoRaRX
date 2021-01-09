## Commander's LoRa RX: Location Log & Visualisation

Replay data reception and Visualise path in a pack of 10 Odometry outputs by

```
python3 rx_lora.py P2P 10
```

### SIR Particle Filter

Two rounds of left-hand search with Particle Filter:

![pf_left2](https://github.com/zdai257/LoRaRX/blob/main/demo0.png)

### Extended Kalman Filter

Two rounds of left-hand search with EKF which uses absolute 2D poses as measurement (fine-grained plot with 10Hz refresh rate):

![left2_fine](https://github.com/zdai257/LoRaRX/blob/main/demo10.png)

THREE rounds of left-Hand Search using the same model as above. Path constraining effect is explicit:

![left3](https://github.com/zdai257/LoRaRX/blob/main/demo2.png)

![left3_fine](https://github.com/zdai257/LoRaRX/blob/main/demo20.png)
