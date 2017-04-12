Simple real-time stereo implementation using CUDA. This is a baseline method
that uses NCC and 2D averaging. Your mileage may vary.

The main stereo code can be found in src/StereoMatcher.h src/StereoMatcher.cc.
It requires cudaArray (https://github.com/trueprice/cudaArray).

Executables are located in src/calibration.cc (for calibrating the stereo rig
using a checkerboard) and real_time_stereo.cc. These programs require OpenCV
(http://opencv.org/) and the FLIR FlyCapture SDK
(https://www.ptgrey.com/flycapture-sdk).
