include_directories(
    .
    ${CUDA_INCLUDE_DIRS}
    ${FlyCapture2_INCLUDE_DIR}
)

set(LIBRARIES
    ${FlyCapture2_LIBRARY}
    ${GLOG_LIBRARIES}
    ${OpenCV_LIBS}
)

#cuda_add_library(harris_corner_detection
#  HarrisCornerDetector.h
#  HarrisCornerDetector.cu
#)

cuda_add_library(stereo_matching
  StereoMatcher.h
  StereoMatcher.cu
)

#add_executable(calibration calibration.cc PGRCaptureManager.cc)
#target_link_libraries(calibration ${LIBRARIES} harris_corner_detection)
#install(TARGETS calibration DESTINATION bin/)

add_executable(calibration calibration.cc PGRCaptureManager.cc)
target_link_libraries(calibration ${LIBRARIES})
install(TARGETS calibration DESTINATION bin/)

add_executable(real_time_stereo real_time_stereo.cc PGRCaptureManager.cc)
target_link_libraries(real_time_stereo
  ${LIBRARIES}
  stereo_matching
)
install(TARGETS real_time_stereo DESTINATION bin/)
