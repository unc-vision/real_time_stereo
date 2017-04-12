// Author: True Price <jtprice at cs.unc.edu>
//
// BSD License
// Copyright (C) 2017  The University of North Carolina at Chapel Hill
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of the original author nor the names of contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
// THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
// CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
// NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <array>
#include <fstream>
#include <vector>

#include "opencv2/opencv.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/cudastereo.hpp"

#include "PGRCaptureManager.h"
#include "StereoMatcher.h"

#include "glog/logging.h"

const unsigned int WINDOW_SIZE = 11;
const unsigned int MIN_DISPARITY = 1;
const unsigned int MAX_DISPARITY = 64;
const float MIN_DEPTH = 1.f;
const float MAX_DEPTH = 20.f;

//------------------------------------------------------------------------------
//
// main()
//
//------------------------------------------------------------------------------

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " calib_file" << std::endl;
    return 0;
  }

  const std::string calib_path = argv[1];

  PGRCaptureManager capture_manager;

  if (capture_manager.NumCameras() < 2) {
    LOG(FATAL) << "This program requires at least two connected cameras.";
  }

  if (capture_manager.NumCameras() > 2) {
    LOG(INFO) << "More than 2 cameras detected; only the first two will be "
                 "used for stereo capture.";
  }

  //
  // load calibration file
  //

  std::ifstream calib_file(calib_path);

  std::vector<cv::Size> image_sizes;
  std::vector<cv::Mat> K, distortion_coeffs, R, t;

  double fx, fy, cx, cy;

  while (calib_file >> fx >> fy) {
    image_sizes.emplace_back(fx, fy);

    K.push_back(cv::Mat::eye(3, 3, CV_64F));
    distortion_coeffs.push_back(cv::Mat::zeros(4, 1, CV_64F));
    R.push_back(cv::Mat::zeros(3, 3, CV_64F));
    t.push_back(cv::Mat::zeros(3, 1, CV_64F));

    calib_file >> fx >> fy >> cx >> cy;
    K.back().at<double>(0, 0) = fx;
    K.back().at<double>(1, 1) = fy;
    K.back().at<double>(0, 2) = cx;
    K.back().at<double>(1, 2) = cy;

    for (unsigned int i = 0; i < 4; ++i) {
      calib_file >> fx; // reuse fx
      distortion_coeffs.back().at<double>(i) = fx;
    }

    for (unsigned int i = 0; i < 3; ++i) {
      for (unsigned int j = 0; j < 3; ++j) {
        calib_file >> fx; // reuse fx
        R.back().at<double>(i, j) = fx;
      }
    }

    for (unsigned int i = 0; i < 3; ++i) {
      calib_file >> fx; // reuse fx
      t.back().at<double>(i) = fx;
    }
  }

  calib_file.close();

  // compute the metric distance between the two cameras
  const float baseline = cv::norm(t[1]);

  //
  // initialize stereo rectification maps
  //

  cv::Mat ur_map1, ur_map2;
  float focal_length;

  {
    cv::Mat R1, R2, P1, P2, Q;
    cv::stereoRectify(K[0], distortion_coeffs[0], K[1], distortion_coeffs[1],
                      image_sizes[0], R[1], t[1], R1, R2, P1, P2, Q,
                      CV_CALIB_ZERO_DISPARITY, 1);

    cv::Mat tmp;
    initUndistortRectifyMap(K[0], distortion_coeffs[0], R1, P1, image_sizes[0],
                            CV_32FC2, ur_map1, tmp);
    initUndistortRectifyMap(K[1], distortion_coeffs[1], R2, P2, image_sizes[1],
                            CV_32FC2, ur_map2, tmp);

    focal_length = P1.at<double>(0, 0);
  }

  //
  // set up our matcher
  //

  const StereoMatcher::Options options = {
      (unsigned int)image_sizes[0].width, (unsigned int)image_sizes[0].height,
      WINDOW_SIZE, MIN_DISPARITY, MAX_DISPARITY, baseline, focal_length,
      MIN_DEPTH, MAX_DEPTH};

  StereoMatcher stereo_matcher(options);

  stereo_matcher.initUndistortRectifyMaps(ur_map1.data, ur_map2.data);

  //
  // main loop
  //

  capture_manager.StartCapture();

  cv::namedWindow("Stereo Stream");
  cv::namedWindow("Depth");

  char keypress; // capture user input

  // stereo-rectified intensity images
  std::array<cv::Mat, 2> images = {
      cv::Mat(image_sizes[0].height, image_sizes[0].width, CV_32FC1),
      cv::Mat(image_sizes[0].height, image_sizes[0].width, CV_32FC1)};

  cv::Mat depth_image(image_sizes[0].height, image_sizes[0].width, CV_32F);

  do {
    const std::vector<cv::Mat> &orig_images = capture_manager.Capture(true);

    stereo_matcher.init_frame(orig_images[0].data, orig_images[1].data);
    stereo_matcher.match();
    stereo_matcher.calculate_depth(); // comment out to view disparity images

    images[0] = orig_images[0];
    images[1] = orig_images[1];
    stereo_matcher.download_depth(depth_image.data);

    // concatenate for display purposes
    cv::Mat image_pair(images[0].rows, 2 * images[0].cols, CV_8UC3);
    cv::hconcat(images[0], images[1], image_pair);

    cv::imshow("Stereo Stream", image_pair);

    // use this line for visualizing disparity images
    //cv::Mat depth_display = depth_image * (255. / MAX_DISPARITY);
    
    cv::log(depth_image, depth_image);
    const float max_log_depth = cv::log(MAX_DEPTH);
    cv::Mat depth_display = 255. - depth_image * (255. / max_log_depth);

    depth_display.convertTo(depth_display, CV_8UC1);
    cv::applyColorMap(depth_display, depth_display, cv::COLORMAP_BONE);

    cv::imshow("Depth", depth_display);

    keypress = (cv::waitKey(1) & 0xff);
  } while (keypress != 'q');

  return 0;
}
