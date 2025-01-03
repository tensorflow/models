/* Copyright 2017 The TensorFlow Authors All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/lum.h>

#include <iostream>
#include <memory>
#include <vector>

using pcl::PointCloud;
using pcl::PointXYZ;

template <typename PointSource, typename PointTarget, typename Scalar = float>
class IterativeClosestPointExposed
    : public pcl::IterativeClosestPoint<PointSource, PointTarget, Scalar> {
 public:
  pcl::CorrespondencesPtr getCorrespondencesPtr() {
    for (uint32_t i = 0; i < this->correspondences_->size(); i++) {
      pcl::Correspondence currentCorrespondence = (*this->correspondences_)[i];
      std::cout << "Index of the source point: "
                << currentCorrespondence.index_query << std::endl;
      std::cout << "Index of the matching target point: "
                << currentCorrespondence.index_match << std::endl;
      std::cout << "Distance between the corresponding points: "
                << currentCorrespondence.distance << std::endl;
      std::cout << "Weight of the confidence in the correspondence: "
                << currentCorrespondence.weight << std::endl;
    }
    return this->correspondences_;
  }
};

int main(int argc, char** argv) {
  PointCloud<PointXYZ>::Ptr cloud_in(new PointCloud<PointXYZ>);
  PointCloud<PointXYZ>::Ptr cloud_out(new PointCloud<PointXYZ>);

  // Fill in the CloudIn data
  cloud_in->width = 5;
  cloud_in->height = 1;
  cloud_in->is_dense = false;
  cloud_in->points.resize(cloud_in->width * cloud_in->height);
  for (size_t i = 0; i < cloud_in->points.size(); ++i) {
    cloud_in->points[i].x = 1024 * rand() / (RAND_MAX + 1.0f);
    cloud_in->points[i].y = 1024 * rand() / (RAND_MAX + 1.0f);
    cloud_in->points[i].z = 1024 * rand() / (RAND_MAX + 1.0f);
  }
  std::cout << "Saved " << cloud_in->points.size()
            << " data points to input:" << std::endl;
  for (size_t i = 0; i < cloud_in->points.size(); ++i)
    std::cout << "    " << cloud_in->points[i].x << " " << cloud_in->points[i].y
              << " " << cloud_in->points[i].z << std::endl;
  *cloud_out = *cloud_in;
  std::cout << "size:" << cloud_out->points.size() << std::endl;
  for (size_t i = 0; i < cloud_in->points.size(); ++i)
    cloud_out->points[i].x = cloud_in->points[i].x + 0.7f;
  std::cout << "Transformed " << cloud_in->points.size()
            << " data points:" << std::endl;
  for (size_t i = 0; i < cloud_out->points.size(); ++i)
    std::cout << "    " << cloud_out->points[i].x << " "
              << cloud_out->points[i].y << " " << cloud_out->points[i].z
              << std::endl;

  IterativeClosestPointExposed<PointXYZ, PointXYZ> icp;
  icp.setInputSource(cloud_in);
  icp.setInputTarget(cloud_out);
  PointCloud<PointXYZ> Final;
  icp.align(Final);
  std::cout << "has converged:" << icp.hasConverged()
            << " score: " << icp.getFitnessScore() << std::endl;
  Eigen::Matrix4f transform = icp.getFinalTransformation();
  std::cout << transform << std::endl;
  icp.getCorrespondencesPtr();

  Eigen::Matrix3f m;

  m = Eigen::AngleAxisf(0.25 * M_PI, Eigen::Vector3f::UnitX()) *
      Eigen::AngleAxisf(0.5 * M_PI, Eigen::Vector3f::UnitY()) *
      Eigen::AngleAxisf(0.33 * M_PI, Eigen::Vector3f::UnitZ());

  std::cout << "original rotation:" << std::endl;
  std::cout << m << std::endl << std::endl;

  Eigen::Vector3f ea = m.eulerAngles(0, 1, 2);
  std::cout << "to Euler angles:" << std::endl;
  std::cout << ea << std::endl << std::endl;

  Eigen::Matrix3f n;
  n = Eigen::AngleAxisf(ea[0], Eigen::Vector3f::UnitX()) *
      Eigen::AngleAxisf(ea[1], Eigen::Vector3f::UnitY()) *
      Eigen::AngleAxisf(ea[2], Eigen::Vector3f::UnitZ());
  std::cout << "recalc original rotation:" << std::endl;
  std::cout << n << std::endl << std::endl;

  Eigen::Vector3f c0 = m.col(0);
  Eigen::Vector3f c1 = m.col(1);
  Eigen::Vector3f c2 = m.col(2);

  float sx = c0.norm();
  float sy = c1.norm();
  float sz = c2.norm();

  c0 /= sx;
  c1 /= sy;
  c2 /= sz;

  float rx = atan2(c1(2), c2(2));
  float ry = atan2(-c0(2), sqrt(c1(2) * c1(2) + c2(2) * c2(2)));
  float rz = atan2(c0(1), c0(0));

  std::cout << "Scales:" << std::endl;
  std::cout << sx << std::endl
            << sy << std::endl
            << sz << std::endl
            << std::endl;

  std::cout << "Euler angles:" << std::endl;
  std::cout << rx << std::endl
            << ry << std::endl
            << rz << std::endl
            << std::endl;

  Eigen::Vector6f v6f;
  v6f << sx, sy, sz, rx, ry, rz;
  std::cout << "Vector6f:" << std::endl;
  std::cout << v6f << std::endl << std::endl;

  return (0);
}
