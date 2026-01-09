# Lane-Keeping-Assistant-Using-Computer-Vision
Lane Keeping Assistant System (LKAS) using computer vision for real-time lane detection and path estimation.


# Lane Keeping Assistant System (LKAS)

## Overview
This project implements a Lane Keeping Assistant System (LKAS) using computer vision techniqueswith a Raspberry Pi Camera and STM32 microcontroller.  
The system detects road lane markings in real time and estimates the vehicle’s path to assist in steering control.
The Raspberry Pi performs image acquisition and lane detection, while the STM32 handles low-level motor control and steering actuation.

## System Architecture
- **Raspberry Pi + Camera**  
  - Captures real-time road images  
  - Performs image processing and lane detection  
  - Calculates lane curvature and vehicle offset  

- **STM32 Microcontroller**  
  - Receives steering commands from Raspberry Pi  
  - Controls motors and steering mechanism  
  - Handles real-time embedded control tasks  

## Key Features
- Camera calibration and distortion correction  
- Perspective (Bird’s Eye View) transformation  
- Binary thresholding for lane extraction  
- Lane detection and road curvature estimation  
- Vehicle path and steering angle calculation  
- Raspberry Pi ↔ STM32 serial communication  
- Real-time lane keeping assistance  

## Hardware Used
- Raspberry Pi (any supported model)
- Raspberry Pi Camera Module
- STM32 Microcontroller
- DC Motors / Servo Motor
- Motor Driver Module
- Power Supply

## Software & Tools
- Python
- OpenCV
- NumPy
- STM32 HAL / Embedded C
- CAN Communication using MCP2515 CAN Module

## Working Principle
1. The Raspberry Pi camera captures the road image.
2. The image is preprocessed and converted to a binary image.
3. Perspective transformation provides a top-down road view.
4. Lane lines are detected and tracked.
5. Vehicle offset and steering angle are calculated.
6. Steering commands are sent to the STM32 via UART.
7. STM32 controls the motors to maintain lane position.

## Results
The system successfully detects lane boundaries and estimates the vehicle path in real time, enabling basic lane keeping functionality suitable for ADAS research and educational purposes.

## Applications
- Advanced Driver Assistance Systems (ADAS)
- Autonomous vehicle research
- Embedded systems and robotics projects
- Academic and educational demonstrations

## Author
**Akash Kumar**  
GitHub: https://github.com/ak1704317

