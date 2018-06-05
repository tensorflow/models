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

#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/lum.h>
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/work_sharder.h"

using ::pcl::PointCloud;
using ::pcl::PointXYZ;
using ::tensorflow::DEVICE_CPU;
using ::tensorflow::OpKernel;
using ::tensorflow::OpKernelConstruction;
using ::tensorflow::OpKernelContext;
using ::tensorflow::Shard;
using ::tensorflow::Status;
using ::tensorflow::Tensor;
using ::tensorflow::TensorShape;
using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

#define kTransformSize 6               // tx, ty, tz, rx, ry, rz.
#define kMaxCorrespondenceDistance -1  // -1 = no limit.

// Extend IterativeClosestPoint to expose this->correspondences_.
template <typename PointSource, typename PointTarget, typename Scalar = float>
class IterativeClosestPointExposed
    : public pcl::IterativeClosestPoint<PointSource, PointTarget, Scalar> {
 public:
  pcl::CorrespondencesPtr getCorrespondencesPtr() {
    return this->correspondences_;
  }
};

REGISTER_OP("Icp")
    .Attr("T: realnumbertype")
    .Input("p: T")
    .Input("ego_motion: T")
    .Input("q: T")
    .Output("transform: T")
    .Output("residual: T")
    .SetShapeFn([](InferenceContext* c) {
      // TODO(rezama): Add shape checks to ensure:
      // p and q have the same rank.
      // The last dimension for p and q is always 3.
      // ego_motion has shape [B, kTransformSize].
      ShapeHandle p_shape;
      ShapeHandle q_shape;
      ShapeHandle ego_motion_shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 3, &p_shape));
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(2), 3, &q_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &ego_motion_shape));
      if (c->RankKnown(p_shape)) {
        c->set_output(0, c->MakeShape({c->Dim(p_shape, 0), kTransformSize}));
      } else {
        c->set_output(0, c->UnknownShape());
      }
      c->set_output(1, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(
Aligns two point clouds and returns the alignment transformation and residuals.

p: A [B, k, 3] or [B, m, n, 3] Tensor containing x, y, z coordinates of source
  point cloud before being transformed using ego_motion.
ego_motion: A [B, 6] Tensor containing [rx, ry, rz, tx, ty, tz].
q: A [B, k', 3] or [B, m', n', 3] Tensor containing x, y, z coordinates of
  target point cloud.

transform: A [B, 6] Tensor representing the alignment transformation containing
  [rx, ry, rz, tx, ty, tz].
residual: A Tensor with the same shape as p containing the residual
  q_c_i - transformation(p_i) for all points in p, where c_i denotes the index
  of the matched point for p_i in q.
)doc");

template <typename T>
class IcpOp : public OpKernel {
 public:
  explicit IcpOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // TODO(rezama): Add shape checks to ensure:
    // p and q have the same rank.
    // The last dimension for p and q is always 3.
    // ego_motion has shape [B, kTransformSize].

    const Tensor& P_tensor = context->input(0);
    const Tensor& Q_tensor = context->input(2);
    const Tensor& ego_motion_tensor = context->input(1);
    const TensorShape P_shape(P_tensor.shape());
    const TensorShape Q_shape(Q_tensor.shape());
    const TensorShape ego_motion_shape(ego_motion_tensor.shape());
    const int batch_size = P_shape.dim_size(0);
    const int P_rank = P_shape.dims();
    const bool is_organized = P_rank == 4;
    // P_cloud_size = k * 3 or m * n * 3.
    int P_cloud_size, Q_cloud_size;
    if (is_organized) {
      P_cloud_size =
          P_shape.dim_size(1) * P_shape.dim_size(2) * P_shape.dim_size(3);
      Q_cloud_size =
          Q_shape.dim_size(1) * Q_shape.dim_size(2) * Q_shape.dim_size(3);
    } else {
      P_cloud_size = P_shape.dim_size(1) * P_shape.dim_size(2);
      Q_cloud_size = Q_shape.dim_size(1) * Q_shape.dim_size(2);
    }
    auto P_flat = P_tensor.flat<T>();
    auto Q_flat = Q_tensor.flat<T>();
    #define PIDX(b, i, j) b * P_cloud_size + i * 3 + j
    #define QIDX(b, i, j) b * Q_cloud_size + i * 3 + j
    auto ego_motion_flat = ego_motion_tensor.flat<T>();

    // Create output tensors.
    Tensor* transform_tensor = nullptr;
    Tensor* residual_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({batch_size, kTransformSize}),
                                &transform_tensor));
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, P_shape, &residual_tensor));
    auto output_transform = transform_tensor->template flat<T>();
    auto output_residual = residual_tensor->template flat<T>();

    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());

    auto do_work = [context, &P_shape, &Q_shape, is_organized, &P_flat, &Q_flat,
                    &output_transform, &output_residual, P_cloud_size,
                    Q_cloud_size, &ego_motion_flat,
                    this](int64 start_row, int64 limit_row) {
      for (size_t b = start_row; b < limit_row; ++b) {
        PointCloud<PointXYZ>::Ptr cloud_in(new PointCloud<PointXYZ>);
        PointCloud<PointXYZ>::Ptr cloud_target(new PointCloud<PointXYZ>);

        // Fill in the source cloud data.
        if (is_organized) {
          cloud_in->width = P_shape.dim_size(2);
          cloud_in->height = P_shape.dim_size(1);
        } else {
          cloud_in->width = P_shape.dim_size(1);
          cloud_in->height = 1;
        }
        cloud_in->is_dense = false;
        cloud_in->points.resize(cloud_in->width * cloud_in->height);
        for (size_t i = 0; i < cloud_in->points.size(); ++i) {
          cloud_in->points[i].x = P_flat(PIDX(b, i, 0));
          cloud_in->points[i].y = P_flat(PIDX(b, i, 1));
          cloud_in->points[i].z = P_flat(PIDX(b, i, 2));
        }
        // Fill in the target cloud data.
        if (is_organized) {
          cloud_target->width = Q_shape.dim_size(2);
          cloud_target->height = Q_shape.dim_size(1);
        } else {
          cloud_target->width = Q_shape.dim_size(1);
          cloud_target->height = 1;
        }
        cloud_target->is_dense = false;
        cloud_target->points.resize(cloud_target->width * cloud_target->height);
        for (size_t i = 0; i < cloud_target->points.size(); ++i) {
          cloud_target->points[i].x = Q_flat(QIDX(b, i, 0));
          cloud_target->points[i].y = Q_flat(QIDX(b, i, 1));
          cloud_target->points[i].z = Q_flat(QIDX(b, i, 2));
        }
        // Apply ego-motion.
        Eigen::Vector6f ego_motion_vector;
        int s = b * kTransformSize;
        // TODO(rezama): Find out how to use slicing here.
        ego_motion_vector << ego_motion_flat(s),
                             ego_motion_flat(s + 1),
                             ego_motion_flat(s + 2),
                             ego_motion_flat(s + 3),
                             ego_motion_flat(s + 4),
                             ego_motion_flat(s + 5);
        Eigen::Matrix4f ego_motion_mat =
            ComposeTransformationMatrix(ego_motion_vector);
        PointCloud<PointXYZ>::Ptr cloud_in_moved(new PointCloud<PointXYZ>());
        pcl::transformPointCloud(*cloud_in, *cloud_in_moved, ego_motion_mat);

        // Run ICP.
        IterativeClosestPointExposed<PointXYZ, PointXYZ> icp;
        if (kMaxCorrespondenceDistance > 0) {
          icp.setMaxCorrespondenceDistance(kMaxCorrespondenceDistance);
        }
        icp.setInputSource(cloud_in_moved);
        icp.setInputTarget(cloud_target);
        PointCloud<PointXYZ> Final;
        icp.align(Final);
        Eigen::Matrix4f transform = icp.getFinalTransformation();

        // Execute the transformation.
        PointCloud<PointXYZ>::Ptr transformed_cloud(new PointCloud<PointXYZ>());
        pcl::transformPointCloud(*cloud_in_moved, *transformed_cloud,
                                 transform);

        // Compute residual.
        pcl::CorrespondencesPtr correspondences = icp.getCorrespondencesPtr();
        for (size_t i = 0; i < cloud_in->points.size(); i++) {
          output_residual(PIDX(b, i, 0)) = 0;
          output_residual(PIDX(b, i, 1)) = 0;
          output_residual(PIDX(b, i, 2)) = 0;
        }
        for (size_t i = 0; i < correspondences->size(); i++) {
          pcl::Correspondence corr = (*correspondences)[i];
          PointXYZ p_i_trans = transformed_cloud->points[corr.index_query];
          PointXYZ q_i = cloud_target->points[corr.index_match];
          PointXYZ residual_i;
          // Compute residual_i = q_i - p_i_trans
          residual_i.getArray3fMap() =
              q_i.getArray3fMap() - p_i_trans.getArray3fMap();
          output_residual(PIDX(b, corr.index_query, 0)) = residual_i.x;
          output_residual(PIDX(b, corr.index_query, 1)) = residual_i.y;
          output_residual(PIDX(b, corr.index_query, 2)) = residual_i.z;
        }

        // Decompose transformation.
        Eigen::Vector3f rotation = DecomposeRotationScale(transform);
        output_transform(b * kTransformSize + 0) = transform(0, 3);  // tx.
        output_transform(b * kTransformSize + 1) = transform(1, 3);  // ty.
        output_transform(b * kTransformSize + 2) = transform(2, 3);  // tz.
        output_transform(b * kTransformSize + 3) = rotation(0);      // rx.
        output_transform(b * kTransformSize + 4) = rotation(1);      // ry.
        output_transform(b * kTransformSize + 5) = rotation(2);      // rz.
      }
    };  // End of closure.

    // Incredibly rough estimate of clock cycles for do_work
    const int64 cost = 50 * P_cloud_size * P_cloud_size;
    Shard(worker_threads.num_threads, worker_threads.workers, batch_size, cost,
          do_work);
  }

  Eigen::Vector3f DecomposeRotationScale(const Eigen::Matrix4f& mat) {
    // Get columns.
    Eigen::Vector3f c0 = mat.col(0).head(3);
    Eigen::Vector3f c1 = mat.col(1).head(3);
    Eigen::Vector3f c2 = mat.col(2).head(3);

    // Compute scale.
    float sx = c0.norm();
    float sy = c1.norm();
    float sz = c2.norm();

    // Normalize rotations.
    c0 /= sx;
    c1 /= sy;
    c2 /= sz;

    // Compute Euler angles.
    float rx = atan2(c1(2), c2(2));
    float ry = atan2(-c0(2), sqrt(c1(2) * c1(2) + c2(2) * c2(2)));
    float rz = atan2(c0(1), c0(0));

    // Return as a vector.
    Eigen::Vector3f rotation;
    rotation << rx, ry, rz;
    return rotation;
  }

  Eigen::Matrix4f ComposeTransformationMatrix(const Eigen::VectorXf& v) {
    float tx = v(0);
    float ty = v(1);
    float tz = v(2);
    float rx = v(3);
    float ry = v(4);
    float rz = v(5);

    Eigen::Transform<float, 3, Eigen::Affine> t;
    t = Eigen::Translation<float, 3>(tx, ty, tz);
    t *= Eigen::AngleAxisf(rx, Eigen::Vector3f::UnitX());
    t *= Eigen::AngleAxisf(ry, Eigen::Vector3f::UnitY());
    t *= Eigen::AngleAxisf(rz, Eigen::Vector3f::UnitZ());

    Eigen::Matrix4f mat = t.matrix();
    return mat;
  }
};

REGISTER_KERNEL_BUILDER(Name("Icp")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T"),
                        IcpOp<float>);
REGISTER_KERNEL_BUILDER(Name("Icp")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<double>("T"),
                        IcpOp<double>);
