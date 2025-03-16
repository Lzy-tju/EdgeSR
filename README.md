# EdgeSR: Reparameterization-Driven Fast Thermal Super-Resolution for Edge Electro-Optical Device

#### Changhong Fu*, Ziyu Lu, Mengyuan Li, Zijie Zhang, Haobo Zuo

## Abstract

>Super-resolution (SR) can greatly promote the development of edge electro-optical (EO) devices.
>However, most existing SR models struggle to simultaneously achieve effective thermal reconstruction and real-time inference on EO devices with limited computing resources.
>To address these issues, this work proposes a novel fast thermal SR model (EdgeSR) for edge EO devices.
>Specifically, reparameterized scale-integrated convolutions (RepSConv) are proposed to deeply explore high-frequency features, incorporating multi-scale information and enhancing the scale-awareness of the backbone during the training phase.
>Furthermore, an interactive reparameterization module (IRM), combining historical high-frequency with low-frequency information, is introduced to guide the extraction of high-frequency features, ultimately boosting the high-quality reconstruction of thermal images.
>Edge EO deployment-oriented reparameterization (EEDR) is designed to reparameterize all modules into standard convolutions that are hardware-friendly for edge EO devices, onboard real-time inference.
>Additionally, a new benchmark for thermal SR on cityscapes (CS-TSR) is built.
>The experimental results on this benchmark show that, compared to state-of-the-art (SOTA) lightweight SR networks, EdgeSR delivers superior reconstruction quality and faster inference speed on edge EO devices.
>In real-world applications, EdgeSR exhibits robust performance on edge EO devices, making it suitable for real-world deployment.
>The code and demo is available at https://github.com/vision4robotics/EdgeSR.

![This figure shows the workflow of EdgeSR.](https://github.com/2004kiki/EdgeSR/blob/main/EdgeSR/figure/overall.png)

###Demo
