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

#include "PGRCaptureManager.h"

#include "utility.h"

//------------------------------------------------------------------------------

PGRCaptureManager::PGRCaptureManager() {
  // print build info
  FlyCapture2::FC2Version fc2Version;
  FlyCapture2::Utilities::GetLibraryVersion(&fc2Version);

  LOG(INFO) << "FlyCapture2 library version: " << fc2Version.major << "."
            << fc2Version.minor << "." << fc2Version.type << "."
            << fc2Version.build << std::endl;

  // get number of cameras
  CHECK_PGR(bus_manager_.GetNumOfCameras(&num_cameras_))

  LOG(INFO) << "Number of cameras detected: " << num_cameras_ << std::endl;

  cameras_.resize(num_cameras_);
  raw_images_.resize(num_cameras_);
  pgr_images_.resize(num_cameras_);
  images_.resize(num_cameras_);

  for (unsigned int i = 0; i < num_cameras_; ++i) {
    FlyCapture2::PGRGuid guid;
    CHECK_PGR(bus_manager_.GetCameraFromIndex(i, &guid))

    // Connect to the camera
    cameras_[i].reset(new FlyCapture2::Camera);
    CHECK_PGR(cameras_[i]->Connect(&guid))

    // Get the camera information
    FlyCapture2::CameraInfo camInfo;
    CHECK_PGR(cameras_[i]->GetCameraInfo(&camInfo))

    LOG(INFO) << std::endl
              << "*** CAMERA INFORMATION ***" << std::endl
              << "Serial number -" << camInfo.serialNumber << std::endl
              << "Camera model - " << camInfo.modelName << std::endl
              << "Camera vendor - " << camInfo.vendorName << std::endl
              << "Sensor - " << camInfo.sensorInfo << std::endl
              << "Resolution - " << camInfo.sensorResolution << std::endl
              << "Firmware version - " << camInfo.firmwareVersion << std::endl
              << "Firmware build time - " << camInfo.firmwareBuildTime
              << std::endl
              << std::endl;
  }
}

//------------------------------------------------------------------------------

PGRCaptureManager::~PGRCaptureManager() {
  for (auto &camera : cameras_) {
    camera->StopCapture();
    camera->Disconnect();
  }
}

//------------------------------------------------------------------------------

void PGRCaptureManager::StartCapture() {
  CHECK_PGR(FlyCapture2::Camera::StartSyncCapture(
      num_cameras_, (const FlyCapture2::Camera **)cameras_.data()))
}

//------------------------------------------------------------------------------

const std::vector<cv::Mat> &PGRCaptureManager::Capture(
    bool grayscale /*= false*/) {
  // TODO (True): skip opencv, output as float?
  const auto format = (grayscale) ? FlyCapture2::PIXEL_FORMAT_MONO8
                                  : FlyCapture2::PIXEL_FORMAT_BGRU;
  const auto cv_format = (grayscale) ? CV_8UC1 : CV_8UC4;

  for (unsigned int i = 0; i < num_cameras_; ++i) {
    CHECK_PGR(cameras_[i]->RetrieveBuffer(&raw_images_[i]))
    CHECK_PGR(raw_images_[i].Convert(format, &pgr_images_[i]))

    images_[i] = cv::Mat(pgr_images_[i].GetRows(), pgr_images_[i].GetCols(),
                         cv_format, pgr_images_[i].GetData());
  }

  return images_;
}

//------------------------------------------------------------------------------

// TODO (True): currently just a debug function, but it should be used to check
// for synchronization of all input cameras
//void CheckSystemSynchronization(const FlyCapture2::Image &image1,
//                                const FlyCapture2::Image &image2) {
//  const auto ts1 = image2.GetTimeStamp();
//  const auto ts2 = image2.GetTimeStamp();
//
//  LOG(INFO) << "TS1 = [" << ts1.cycleSeconds << " " << ts1.cycleCount << "] ; "
//            << "TS2 = [" << ts2.cycleSeconds << " " << ts2.cycleCount << "]"
//            << std::endl;
//}

