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

#include <chrono>
#include <fstream>
#include <math.h>
#include <thread>
#include <vector>

#include "opencv2/opencv.hpp"

#include "PGRCaptureManager.h"

#include "glog/logging.h"

using namespace std::chrono;

const unsigned int INITIAL_WAIT_TIME = 1000; // ms until first capture
const unsigned int DEFAULT_WAIT_TIME = 500; // ms between captures
const unsigned int DEFAULT_NUM_IMAGES = 100; // number of images to capture

//------------------------------------------------------------------------------

// combines the given set of images (assumed to be all the same resolution) into
// a grid
cv::Mat createOverviewImage(const std::vector<cv::Mat> &images,
                            float scale_x = 1.f, float scale_y = 1.f) {
  const unsigned int num_cameras = images.size();
  const unsigned int height = images[0].rows;
  const unsigned int width = images[0].cols;

  cv::Mat out_image;

  if (num_cameras == 1) {
    // 1 image: simply return that image
    out_image = images[0];
  } else if (num_cameras == 2) {
    // 2 images: concatentate side-by-side
    out_image = cv::Mat(height, 2 * width, CV_8UC3);
    cv::hconcat(images[0], images[1], out_image);
  } else {
    // 3+ images: form a grid
    const unsigned int num_cols = std::ceil(std::sqrt((float)num_cameras));
    const unsigned int num_rows = std::ceil(num_cameras / (float)num_cols);
    out_image = cv::Mat(num_rows * height, num_cols * width, CV_8UC3);

    unsigned int i = 0;
    for (unsigned int r; r < num_rows; ++r) {
      for (unsigned int c; c < num_cols; ++c, ++i) {
        images[i].copyTo(
            out_image(cv::Rect(r * height, c * width, height, width)));
      }
    }
  }

  cv::resize(out_image, out_image, cv::Size(), scale_x, scale_y);

  return out_image;
}

//------------------------------------------------------------------------------
//
// main()
//
//------------------------------------------------------------------------------

int main(int argc, char **argv) {
  if (argc < 5) {
    std::cout << "Usage: " << argv[0]
              << " out_file chessboard_rows chessboard_cols "
                 "actual_chessboard_square_size [distance of camera i > 0 to "
                 "camera 0]" << std::endl;
    return 0;
  }

  PGRCaptureManager capture_manager;

  const std::string out_file(argv[1]);
  const cv::Size chessboard_size(std::stoi(argv[2]), std::stoi(argv[3]));
  const float square_size = std::stof(argv[4]);

  std::vector<float> baselines;
  for (unsigned int i = 5; i < argc; ++i) {
    baselines.push_back(std::stof(argv[i]));
  }

  const unsigned int num_cameras = capture_manager.NumCameras();
  // TODO (True): need command-line parameters for these
  const unsigned int wait_time = DEFAULT_WAIT_TIME;
  const unsigned int max_num_images_to_capture = DEFAULT_NUM_IMAGES;

  //
  // start capture
  //

  capture_manager.StartCapture();

  // give the user a few seconds to get ready
  std::cout << std::endl
            << "Starting capture in " << INITIAL_WAIT_TIME << "ms" << std::endl;
  std::this_thread::sleep_for(milliseconds(INITIAL_WAIT_TIME));
  std::cout << "Starting capture now!" << std::endl;

  // per frame, per camera
  std::vector<std::vector<cv::Mat>> captured_images;

  //
  // main loop
  //

  cv::namedWindow("Video Streams");

  auto last_timestamp = high_resolution_clock::now();

  do {
    const std::vector<cv::Mat> &images = capture_manager.Capture(true);

    const auto now = high_resolution_clock::now();
    const auto span = duration_cast<milliseconds>(now - last_timestamp).count();

    // if enough time has passed, collect another image
    if (span > wait_time) {
      captured_images.emplace_back();
      for (auto &image : images) {
        captured_images.back().push_back(image.clone());
      }
      last_timestamp = now;
    }

    // concatenate for display purposes
    cv::imshow("Video Streams", createOverviewImage(images));

    // the user can press q to exit at any time
    if ((cv::waitKey(1) & 0xff) == 'q') {
      break;
    }
  } while (captured_images.size() < max_num_images_to_capture);

  //
  // calibrate
  //
  
  cv::namedWindow("Image pair");

  std::cout << std::endl
            << "Calibrating...hit 'n' to throw out captures, or hit any key to "
               "accept them" << std::endl;

  // detected chessboard points for camera, for each image:
  // points2D[camera[image[point_idx]]]
  std::vector<std::vector<std::vector<cv::Point2f>>> points2D(num_cameras);

  unsigned int num_detected = 0; // # images where a chessboard was found

  for (unsigned int i = 0; i < captured_images.size(); ++i) {
    cv::imshow("Captured Images", createOverviewImage(captured_images[i]));
    if ((cv::waitKey() & 0xff) == 'n') {
      continue;
    }
  
    // TODO (True): in the future, probably want to count observations in each
    // image separately
    bool all_found = true; // false if the chessboard isn't in every image

    std::vector<std::vector<cv::Point2f>> observed_points(num_cameras);

    for (unsigned int c = 0; c < num_cameras; ++c) {
      all_found &= cv::findChessboardCorners(
          captured_images[i][c], chessboard_size, observed_points[c],
          cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK |
              cv::CALIB_CB_NORMALIZE_IMAGE);

      if (!all_found) {
        break;
      }

      // refine the accuracy of the detected points
      cv::cornerSubPix(
          captured_images[i][c], observed_points[c], cv::Size(11, 11),
          cv::Size(-1, -1),
          cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30,
                           0.1));
    }

    // if points were found in all images,
    if (all_found) {
      for (unsigned int c = 0; c < num_cameras; ++c) {
        points2D[c].push_back(observed_points[c]);
      }

      ++num_detected;
    }
  }

  // quit early if nothing detected
  if (num_detected == 0) {
    std::cout << "No chessboards detected!" << std::endl;
    return 0;
  }

  std::cout << "Chessboard detected in " << num_detected << " images"
            << std::endl;

  //
  // calibrate
  //

  std::cout << "Estimating intrinsics" << std::endl;

  // create 3D points for the chessboard corners
  std::vector<std::vector<cv::Point3f>> corners3D(1);
  corners3D[1].reserve(chessboard_size.height * chessboard_size.width);

  for (int i = 0; i < chessboard_size.height; ++i) {
    for (int j = 0; j < chessboard_size.width; ++j) {
      corners3D[0].push_back(
          cv::Point3f((float)j * square_size, (float)i * square_size, 0));
    }
  }

  // tile for the number of detected images
  corners3D.resize(num_detected, corners3D[0]);

  // now, actually calibrate
  std::vector<cv::Mat> camera_matrices(num_cameras);
  std::vector<cv::Mat> distortion_coeffs(num_cameras);
//  std::vector<std::vector<cv::Point2f>> image_plane_points(num_cameras);

  for (unsigned int c = 0; c < num_cameras; ++c) {
    const unsigned int image_width = captured_images[0][c].cols;
    const unsigned int image_height = captured_images[0][c].rows;

    camera_matrices[c] = cv::Mat::eye(3, 3, CV_64F);
    distortion_coeffs[c] = cv::Mat::zeros(4, 1, CV_64F);
    std::vector<cv::Mat> rvecs, tvecs; // unused

    cv::calibrateCamera(corners3D, points2D[c],
                        cv::Size(image_width, image_height), camera_matrices[c],
                        distortion_coeffs[c], rvecs, tvecs);

/*
    // undistort all points for this camera, converting to normalized coords
    for (auto &p2D : points2D[c]) {
      std::vector<cv::Point2f> new_points;
      cv::undistortPoints(p2D, new_points, camera_matrices[c],
                          distortion_coeffs[c], cv::noArray(),
                          camera_matrices[c]);

      image_plane_points[c].insert(image_plane_points[c].end(),
                                   new_points.begin(), new_points.end());
    }
*/

    const auto &K = camera_matrices[c];
    const auto &d = distortion_coeffs[c];
    std::cout << K.at<double>(0, 0) << " " << K.at<double>(1, 1) << " "
              << K.at<double>(0, 2) << " " << K.at<double>(1, 2) << " "
              << d.at<double>(0) << " " << d.at<double>(1) << " "
              << d.at<double>(2) << " " << d.at<double>(3) << std::endl;
  }

  std::cout << "Calculating pose" << std::endl;

  // calculate poses relative to the first camera
  std::vector<cv::Mat> R, t;

  // first pose is identity
  R.push_back(cv::Mat::eye(3, 3, CV_64F));
  t.push_back(cv::Mat::zeros(3, 1, CV_64F));

  for (unsigned int c = 1; c < num_cameras; ++c) {
    cv::Mat E, F, Rc, tc;

/*
    E = cv::findEssentialMat(image_plane_points[0], image_plane_points[c]);
    cv::recoverPose(E, image_plane_points[0], image_plane_points[c], Rc, tc);
    cv::normalize(tc, tc); // just in case
*/

    const unsigned int image_width = captured_images[0][c].cols;
    const unsigned int image_height = captured_images[0][c].rows;
    cv::stereoCalibrate(corners3D, points2D[0], points2D[c], camera_matrices[0],
                        distortion_coeffs[0], camera_matrices[c],
                        distortion_coeffs[c],
                        cv::Size(image_width, image_height), Rc, tc, E, F,
                        CV_CALIB_USE_INTRINSIC_GUESS);

    cv::normalize(tc, tc);

    if (c - 1 < baselines.size()) {
      tc *= baselines[c - 1];
    }

    R.push_back(Rc);
    t.push_back(tc);
  }

  std::cout << "Saving" << std::endl;

  // save the calibration
  std::ofstream fout(out_file);

  for (unsigned int c = 0; c < num_cameras; ++c) {
    const unsigned int image_width = captured_images[0][c].cols;
    const unsigned int image_height = captured_images[0][c].rows;
    const auto &K = camera_matrices[c];
    const auto &d = distortion_coeffs[c];
    const auto &Rc = R[c];
    const auto &tc = t[c];

    fout << image_width << " " << image_height << " " << K.at<double>(0, 0)
         << " " << K.at<double>(1, 1) << " " << K.at<double>(0, 2) << " "
         << K.at<double>(1, 2) << " " << d.at<double>(0) << " "
         << d.at<double>(1) << " " << d.at<double>(2) << " " << d.at<double>(3)
         << " " << Rc.at<double>(0, 0) << " " << Rc.at<double>(0, 1) << " "
         << Rc.at<double>(0, 2) << " " << Rc.at<double>(1, 0) << " "
         << Rc.at<double>(1, 1) << " " << Rc.at<double>(1, 2) << " "
         << Rc.at<double>(2, 0) << " " << Rc.at<double>(2, 1) << " "
         << Rc.at<double>(2, 2) << " " << tc.at<double>(0) << " "
         << tc.at<double>(1) << " " << tc.at<double>(2) << std::endl;
  }

  fout.close();

  std::cout << "done" << std::endl;

  return 0;
}
