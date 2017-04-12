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

#ifndef CAPTURE_MANAGER_H_
#define CAPTURE_MANAGER_H_

#include <memory>
#include <vector>

#include "flycapture/FlyCapture2.h"
#include "opencv2/opencv.hpp"

class PGRCaptureManager {
 public:
  PGRCaptureManager();

  ~PGRCaptureManager();

  inline unsigned int NumCameras() const;
  inline const std::vector<cv::Mat> &Images() const;

  void StartCapture();
  const std::vector<cv::Mat> &Capture(bool grayscale = false);

 private:
  unsigned int num_cameras_;

  FlyCapture2::BusManager bus_manager_;

  std::vector<std::unique_ptr<FlyCapture2::Camera>> cameras_;

  std::vector<FlyCapture2::Image> raw_images_, pgr_images_;

  std::vector<cv::Mat> images_;
};

unsigned int PGRCaptureManager::NumCameras() const { return num_cameras_; }

const std::vector<cv::Mat> &PGRCaptureManager::Images() const {
  return images_;
}

#endif // CAPTURE_MANAGER_H_
