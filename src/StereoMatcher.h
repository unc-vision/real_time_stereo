#ifndef STEREO_MATCHER_H_
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

#define STEREO_MATCHER_H_

#include <memory>

class StereoMatcher {
 public:
  class StereoMatcherImpl;

  struct Options {
    unsigned int width, height;
    unsigned int window_size;
    unsigned int min_shift; // minimum disparity, inclusive
    unsigned int max_shift; // maximum disparity, exclusive
    float baseline;
    float focal_length;
    float min_depth, max_depth;

    inline unsigned int shift() const { return max_shift - min_shift; }
  };

  // points in the left image are matched with points in the right image,
  // searching only to the left (negative disparity)
  StereoMatcher(const Options &options);

  ~StereoMatcher();

  void initUndistortRectifyMaps(const void *ur_map1, const void *ur_map2);

  void init_frame(void *image1, void *image2);
  void match();
  void calculate_depth();

  void download_image1(void *image) const;
  void download_image2(void *image) const;
  void download_depth(void *image) const;

 private:
  Options options_;
  std::unique_ptr<StereoMatcherImpl> impl_;
};

#endif // STEREO_MATCHER_H_
