// @file im2row_gpu.cu
// @brief Stack image patches as matrix rows (GPU)
// @author Andrea Vedaldi

/*
Copyright (C) 2014-15 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "im2row.hpp"
#include "../datacu.hpp"
#include <iostream>

using namespace vl ;

/* ---------------------------------------------------------------- */
/*                                                           im2row */
/* ---------------------------------------------------------------- */

template <typename T> __global__ void
im2row_forward_kernel(T* stacked,
                      T const* data,
                      const int numPatchesX,
                      const int numPatchesY,
                      const int numPatchSlices,
                      const int width,
                      const int height,
                      const int windowWidth,
                      const int windowHeight,
                      const int strideX,
                      const int strideY,
                      const int padLeft,
                      const int padTop,
                      const int dilateX,
                      const int dilateY)
{
  /* each kernel copies the pixels in an image patch for one channel */
  int index = threadIdx.x + blockIdx.x * blockDim.x ;
  if (index < numPatchSlices) {
    /*
     get the patch slice (x,y,z) to copy
     */
    int x = index ;
    int y = x / numPatchesX ;
    int z = y / numPatchesY ;
    x %= numPatchesX ;
    y %= numPatchesY ;

    /*
     pick the top-left corer of the patch slice in the input image
     */
    int x_data = x * strideX - padLeft ;
    int y_data = y * strideY - padTop ;
    data += (z * height + y_data) * width + x_data ;

    /*
     pick the column of the stacked image which contains this patch,
     and move down along the column at the beginning of the patch slice
     */
    int patchSliceOffset = (windowWidth*windowHeight) * z ;
    stacked += (numPatchesY * patchSliceOffset + y) * numPatchesX + x ;

    /*
     copy the patch slice
     */
    int windowExtentX = (windowWidth - 1) * dilateX + 1;
    int windowExtentY = (windowHeight - 1) * dilateY + 1;
    for (int v = 0 ; v < windowExtentY ; v += dilateY) {
      for (int u = 0 ; u < windowExtentX ; u += dilateX) {
        if (y_data + v >= 0 &&
            y_data + v < height &&
            x_data + u >= 0 &&
            x_data + u < width) {
          *stacked = data[v * width + u] ;
        } else {
          *stacked = 0 ;
        }
        stacked += (numPatchesX*numPatchesY) ;
      }
    }
  }
}

/* ---------------------------------------------------------------- */
/*                                           im2row backward kernel */
/* ---------------------------------------------------------------- */

// The next two functions assume b > 0.
__forceinline__ __device__
int floordiv(int a, int b)
{
  int q = a/b ;
  if (a >= 0 || a == q*b) return q ;
  return q - 1 ;
}

__forceinline__ __device__
int ceildiv(int a, int b)
{
  int q = a/b ;
  if (a <= 0 || a == q*b) return q ;
  return q + 1 ;
}


int floordiv_cpu(int a, int b)
{
  int q = a/b ;
  if (a >= 0 || a == q*b) return q ;
  return q - 1 ;
}

int ceildiv_cpu(int a, int b)
{
  int q = a/b ;
  if (a <= 0 || a == q*b) return q ;
  return q + 1 ;
}

#if 0
template <typename T> void
im2row_backward_kernel_fake(
int index,
                            T* data,
                       T const* stacked,
                       const int numPatchesX,
                       const int numPatchesY,
                       const int dataVolume,
                       const int width,
                       const int height,
                       const int depth,
                       const int windowWidth,
                       const int windowHeight,
                       const int strideX,
                       const int strideY,
                       const int padLeft,
                       const int padTop,
                       const int dilateX,
                       const int dilateY,
                       const int gcdx, const int gcdy,
                       const int xbar, const int ybar,
                       const int ubar, const int vbar)
{
 // int index = 143 ;
  if (index < dataVolume)
  {
    T accumulator = 0 ;
    /*
     The goal of this kernel is to accumulate data[index]=data[x_data,y_data]
     all elements of the patch matrix that received copies of data[index] in the forward
     pass. To do this, we need to find which patches (x,y) that contain
     copies of this pixel and the relative offsets (u,v) within each such
     patch.

     First, we find which patches (x,y) contain copies of pixel (x_data,y_data)
     in the input tensor. The input tensor coordiante (x_data,y_data) of
     pixel  (u,v) in patch (x,y) are related by equations:

     x_data = x * strideX + u * dilateX - padLeft,
     y_data = y * strideY + v * dilateY - padTop.

     Hence:

     x * strideX = x_data - u * dilateX + padLeft,
     same for y.

     Now we find all values of (x,y) that can be generated by this equation.
     These gives us the patches (x,y) that must be summed. We have:

     strideX * x + dilateX * u = x_data + padLeft.

     where x and u are integers. This is a linear Diophantine equation.
     Rewrite it as:

     ax + bu = c, where

     a = strideX,
     b = dilateY,
     c = x_data + padLeft.

     This equation has a solution only if the greatest common divisor
     g = gcd(a,b) of a and b divides c as well. In this case,
     let (x0,u0) be a solution (i.e. a x0 + b u0 = c); all other solutions
     are in the form

     x_k = x0 + Dx * k,  Dx = b/g,
     u_k = u0 - Du * k,  Du = a/g.

     Next, we look for the values of k such that x_k and u_k are within
     bounds:

     1) 0 <= x_k <= Iw - 1
     2) 0 <= u_k <= Ww - 1

     Thus

     0) recall: gcd(a,b) must divide c
     1) ceil(- x0/Dx) <= k <= floor((Iw - 1 - x0)/Dx)
     2) ceil((u0 - Ww + 1)/Du) <= k <= floor(u0/Du)

     Thus we need to look for the k in the interval

     k_min = ceil(max(-x0/Dx, (u0 - Ww + 1)/Du)),
     k_max = floor(min((Iw - 1 - x0)/Dx,u0/Du).

     Toghether with (*) and the corresponding equations for y,
     this produces a list of patches (x_k,y_p) that contains
     pixel (x_data,y_data) (the list can be empty).

     Furthermore, x_data is mapped to a specific pixel in
     patch x_k whose coordiante is u_k, also given above.
     */

    int x_data = index ;
    int y_data = x_data / width ;
    int z = y_data / height ;
    x_data %= width ;
    y_data %= height ;

    int cx = x_data + padLeft ;
    int cy = y_data + padTop ;
    int qx = cx / gcdx ;
    int qy = cy / gcdy ;

    printf("x_data:%4d y_data:%4d | "
           "cx:%3d qx:%3d gcdx:%3d dx:%3d |"
           "cy:%3d qy:%3d gcdy:%3d dy:%3d\n",
           x_data, y_data,
           cx,qx,gcdx,cx - gcdx * qx,
           cy,qy,gcdy,cy - gcdy * qy) ;

    if (cx != gcdx * qx || cy != gcdy * qy) {  return ; }

    int x0 = xbar * qx ;
    int u0 = ubar * qx ;
    int y0 = ybar * qy ;
    int v0 = vbar * qy ;

//    ax + bu = c, where
//    a = strideX,
//    b = dilateY,
//    c = x_data + padLeft.

    printf("checkx:%d\n", strideX*x0+dilateY*u0-x_data-padLeft) ;
    printf("checky:%d\n", strideY*y0+dilateY*v0-y_data-padTop) ;

    int Dx = dilateX / gcdx ;
    int Du = strideX / gcdx ;
    int Dy = dilateY / gcdy ;
    int Dv = strideY / gcdy ;

    int kmin1 = ceildiv_cpu(-x0,Dx) ;
    int kmax1 = floordiv_cpu(numPatchesX - 1 - x0,Dx) ;
    int kmin2 = ceildiv_cpu(u0 - windowWidth + 1,Du) ;
    int kmax2 = floordiv_cpu(u0,Du) ;
    int kmin = max(kmin1,kmin2) ;
    int kmax = min(kmax1,kmax2) ;

    int qmin1 = ceildiv_cpu(-y0,Dy) ;
    int qmax1 = floordiv_cpu(numPatchesY - 1 - y0,Dy) ;
    int qmin2 = ceildiv_cpu(v0 - windowHeight + 1,Dv) ;
    int qmax2 = floordiv_cpu(v0,Dv) ;
    int qmin = max(qmin1,qmin2) ;
    int qmax = min(qmax1,qmax2) ;

    printf("Dy:%3d Dv:%3d\n", Dy, Dv) ;
    printf("q: %3d to %3d (qmin1:%3d qmin2:%3d qmax1:%3d qmax2:%3d)\n",
           qmin,qmax,qmin1,qmin2,qmax1,qmax2) ;


    /*
     Now we have kmin <= k <= kmax, qmin <= q <= qmax and

     x_k = x0 + Dx * k,     u_k = u0 - Du * k,
     y_q = y0 + Dy * q,     v_q = v0 - Dv * q.

     Thus for each (k,q) in the allowable range, we visit
     patch (x_k,y_q) and pixel (u_k,v_q) within it.

     (x_k,y_q) tells us which row of the patch matix to look for, and
     (u_k,v_q) tells us which column. Linearizing all this:

     pm_row(k,q) = y_q * numPatchesX + x_k,
     pm_col(k,q) = ((z * windowHeight) + v_q) * windowWidth + u_k.

     This is further linearized into an index:

     pm_index(k,q) = (numPatchesX*numPatchesY) * pm_col(k,q) + pm_row(k,q)

     Substituting everything

     pm_row(k,q)
     = (y0 + Dy * q) * numPatchesX + x0 + Dx * k
     = (numPatchesX * Dy) * q + Dx * k + (y0 * numPatchesX + x0)
     = rqc * q + rkc * k + roc

     pm_col(k,q)
     = ((z * windowHeight) + v0 - Dv * q) * windowWidth + u0 - Du * k
     = - (windowWidth * Dv) * q - (Du) * k + (windowHeight * windowWidth * z + v0 * windowWidth + u0)
     = cqc * q + ckc * k + coc ;

     pm_index(k,q)
     = (numPatchesX*numPatchesY) * (cqc * q + ckc * k + coc) + rqc * q + rkc * k + roc
     = (numPatchesX*numPatchesY * cqc + rqc) * q + (numPatchesX*numPatchesY * ckc + rkc) * k + (numPatchesX*numPatchesY * coc + roc)
     = iqc * q + ikc * k + ioc
     */

    int rqc = numPatchesX * Dy ;
    int rkc = Dx ;
    int roc = numPatchesX * y0 + x0 ;

    int cqc = - windowWidth * Dv ;
    int ckc = - Du ;
    int coc = windowWidth * (windowHeight * z + v0) + u0 ;

    int np = numPatchesX * numPatchesY ;
    int iqc = np * cqc + rqc ;
    int ikc = np * ckc + rkc ;
    int ioc = np * coc + roc ;

    stacked += ioc ;
    for (int q = qmin ; q <= qmax ; ++ q) {
      for (int k = kmin ; k <= kmax ; ++ k) {
        int index_ = iqc * q + ikc * k + ioc ;
        printf("index:%4d x:%3d y:%3d k:%3d q:%3d\n", index, x0+Dx*k, y0+Dy*q, k, q) ;

        accumulator += 1;//stacked[iqc * q + ikc * k] ;
      }
    }
   // data[index] = accumulator;
  }
}
#endif

template <typename T> __global__ void
im2row_backward_kernel(T* data,
                        T const* stacked,
                       const int numPatchesX,
                       const int numPatchesY,
                       const int dataVolume,
                       const int width,
                       const int height,
                       const int depth,
                       const int windowWidth,
                       const int windowHeight,
                       const int strideX,
                       const int strideY,
                       const int padLeft,
                       const int padTop,
                       const int dilateX,
                       const int dilateY,
                       const int gcdx, const int gcdy,
                       const int xbar, const int ybar,
                       const int ubar, const int vbar)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < dataVolume)
  {
    T accumulator = 0 ;
    /*
     The goal of this kernel is to accumulate data[index]=data[x_data,y_data]
     all elements of the patch matrix that received copies of data[index] in the forward
     pass. To do this, we need to find which patches (x,y) that contain
     copies of this pixel and the relative offsets (u,v) within each such
     patch.

     First, we find which patches (x,y) contain copies of pixel (x_data,y_data)
     in the input tensor. The input tensor coordiante (x_data,y_data) of
     pixel  (u,v) in patch (x,y) are related by equations:

       x_data = x * strideX + u * dilateX - padLeft,
       y_data = y * strideY + v * dilateY - padTop.

     Now we find all values of (x,y) that can be generated by this equation.
     These gives us the patches (x,y) that must be summed. We have:

       strideX * x + dilateX * u = x_data + padLeft.

     where x and u are integers. This is a linear Diophantine equation.
     Rewrite it as:

       ax + bu = c, where

       a = strideX,
       b = dilateY,
       c = x_data + padLeft.

     This equation has a solution only if the greatest common divisor
     g = gcd(a,b) of a and b divides c as well. In this case,
     let (x0,u0) be a solution (i.e. a x0 + b u0 = c); all other solutions
     are in the form

       x_k = x0 + Dx * k,  Dx = b/g,
       u_k = u0 - Du * k,  Du = a/g.

     Next, we look for the values of k such that x_k and u_k are within
     bounds:

       1) 0 <= x_k <= Pw - 1
       2) 0 <= u_k <= Ww - 1

     Thus

       0) recall: gcd(a,b) must divide c
       1) ceil(- x0/Dx) <= k <= floor((Iw - 1 - x0)/Dx)
       2) ceil((u0 - Ww + 1)/Du) <= k <= floor(u0/Du)

     Thus we need to look for the k in the interval

       k_min = ceil(max(-x0/Dx, (u0 - Ww + 1)/Du)),
       k_max = floor(min((Pw - 1 - x0)/Dx,u0/Du).

     Toghether with (*) and the corresponding equations for y,
     this produces a list of patches (x_k,y_p) that contains
     pixel (x_data,y_data) (the list can be empty).

     Furthermore, x_data is mapped to a specific pixel in
     patch x_k whose coordiante is u_k, also given above.
     */

    int x_data = index ;
    int y_data = x_data / width ;
    int z = y_data / height ;
    x_data %= width ;
    y_data %= height ;

    int cx = x_data + padLeft ;
    int cy = y_data + padTop ;
    int qx = cx / gcdx ;
    int qy = cy / gcdy ;

    if (cx != gcdx * qx || cy != gcdy * qy) { data[index] = 0 ; return ; }

    int x0 = xbar * qx ;
    int u0 = ubar * qx ;
    int y0 = ybar * qy ;
    int v0 = vbar * qy ;

    int Dx = dilateX / gcdx ;
    int Du = strideX / gcdx ;
    int Dy = dilateY / gcdy ;
    int Dv = strideY / gcdy ;

    int kmin1 = ceildiv(-x0,Dx) ;
    int kmax1 = floordiv(numPatchesX - 1 - x0,Dx) ;
    int kmin2 = ceildiv(u0 - windowWidth + 1,Du) ;
    int kmax2 = floordiv(u0,Du) ;
    int kmin = max(kmin1,kmin2) ;
    int kmax = min(kmax1,kmax2) ;

    int qmin1 = ceildiv(-y0,Dy) ;
    int qmax1 = floordiv(numPatchesY - 1 - y0,Dy) ;
    int qmin2 = ceildiv(v0 - windowHeight + 1,Dv) ;
    int qmax2 = floordiv(v0,Dv) ;
    int qmin = max(qmin1,qmin2) ;
    int qmax = min(qmax1,qmax2) ;

    /*
     Now we have kmin <= k <= kmax, qmin <= q <= qmax and

     x_k = x0 + Dx * k,     u_k = u0 - Du * k,
     y_q = y0 + Dy * q,     v_q = v0 - Dv * q.

     Thus for each (k,q) in the allowable range, we visit
     patch (x_k,y_q) and pixel (u_k,v_q) within it.

     (x_k,y_q) tells us which row of the patch matix to look for, and
     (u_k,v_q) tells us which column. Linearizing all this:

     pm_row(k,q) = y_q * numPatchesX + x_k,
     pm_col(k,q) = ((z * windowHeight) + v_q) * windowWidth + u_k.

     This is further linearized into an index:

     pm_index(k,q) = (numPatchesX*numPatchesY) * pm_col(k,q) + pm_row(k,q)

     Substituting everything

     pm_row(k,q)
     = (y0 + Dy * q) * numPatchesX + x0 + Dx * k
     = (numPatchesX * Dy) * q + Dx * k + (y0 * numPatchesX + x0)
     = rqc * q + rkc * k + roc

     pm_col(k,q)
     = ((z * windowHeight) + v0 - Dv * q) * windowWidth + u0 - Du * k
     = - (windowWidth * Dv) * q - (Du) * k + (windowHeight * windowWidth * z + v0 * windowWidth + u0)
     = cqc * q + ckc * k + coc ;

     pm_index(k,q)
     = (numPatchesX*numPatchesY) * (cqc * q + ckc * k + coc) + rqc * q + rkc * k + roc
     = (numPatchesX*numPatchesY * cqc + rqc) * q + (numPatchesX*numPatchesY * ckc + rkc) * k + (numPatchesX*numPatchesY * coc + roc)
     = iqc * q + ikc * k + ioc

     */

    int rqc = numPatchesX * Dy ;
    int rkc = Dx ;
    int roc = numPatchesX * y0 + x0 ;

    int cqc = - windowWidth * Dv ;
    int ckc = - Du ;
    int coc = windowWidth * (windowHeight * z + v0) + u0 ;

    int np = numPatchesX * numPatchesY ;
    int iqc = np * cqc + rqc ;
    int ikc = np * ckc + rkc ;
    int ioc = np * coc + roc ;

    stacked += ioc ;
    for (int q = qmin ; q <= qmax ; ++ q) {
      for (int k = kmin ; k <= kmax ; ++ k) {
        accumulator += stacked[iqc * q + ikc * k] ;
      }
    }
    data[index] = accumulator;
  }
}

namespace vl { namespace impl {

  template<typename type>
  struct im2row<vl::VLDT_GPU, type>
  {

    /* ------------------------------------------------------------ */
    /*                                                      forward */
    /* ------------------------------------------------------------ */

    static vl::ErrorCode
    forward(Context & context,
            type* stacked,
            type const* data,
            size_t width,
            size_t height,
            size_t depth,
            size_t windowWidth,
            size_t windowHeight,
            size_t strideX,
            size_t strideY,
            size_t padLeft,
            size_t padRight,
            size_t padTop,
            size_t padBottom,
            int dilateX,
            int dilateY)
    {
      /* Each kernel instance copies a feature dimension of a patch */

      int windowExtentX = (windowWidth - 1)*dilateX + 1 ;
      int windowExtentY = (windowHeight - 1)*dilateY + 1 ;
      int numPatchesX = (width + (padLeft + padRight) - windowExtentX)/strideX + 1 ;
      int numPatchesY = (height + (padTop + padBottom) - windowExtentY)/strideY + 1 ;
      int numPatchSlices = numPatchesX * numPatchesY * depth ;

      im2row_forward_kernel<type>
      <<< divideAndRoundUp(numPatchSlices, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (stacked,
       data,
       numPatchesX,
       numPatchesY,
       numPatchSlices,
       width, height,
       windowWidth, windowHeight,
       strideX, strideY,
       padLeft, padTop,
       dilateX, dilateY) ;

      return context.setError(context.getCudaHelper().catchCudaError(__func__)) ;
    }

    /* ------------------------------------------------------------ */
    /*                                                     backward */
    /* ------------------------------------------------------------ */

    static vl::ErrorCode
    backward(Context & context,
             type* data,
             type const* stacked,
             size_t width,
             size_t height,
             size_t depth,
             size_t windowWidth,
             size_t windowHeight,
             size_t strideX,
             size_t strideY,
             size_t padLeft,
             size_t padRight,
             size_t padTop,
             size_t padBottom,
             int dilateX,
             int dilateY)
    {
      /*
       Each kernel integrates all contributions to a particular element
       of data.
       */

      int windowExtentX = (windowWidth - 1)*dilateX + 1 ;
      int windowExtentY = (windowHeight - 1)*dilateY + 1 ;
      int numPatchesX = (width + (padLeft + padRight) - windowExtentX)/strideX + 1 ;
      int numPatchesY = (height + (padTop + padBottom) - windowExtentY)/strideY + 1 ;
      int dataVolume = width * height * depth ;

      int xbar ;
      int ubar ;
      int gcdx = vl::gcd(strideX, dilateX, xbar, ubar) ;

      int ybar ;
      int vbar ;
      int gcdy = vl::gcd(strideY, dilateY, ybar, vbar) ;

#if 0
      for (int i = 0 ; i < dataVolume ; ++i) {
      im2row_backward_kernel_fake<type>
      (i,
       data,
       stacked,
       numPatchesX,
       numPatchesY,
       dataVolume,
       width, height, depth,
       windowWidth, windowHeight,
       strideX, strideY,
       padLeft, padTop,
       dilateX, dilateY,
       gcdx, gcdy, xbar, ybar, ubar, vbar) ;
      }
#endif

      im2row_backward_kernel<type>
      <<< divideAndRoundUp(dataVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (data,
       stacked,
       numPatchesX,
       numPatchesY,
       dataVolume,
       width, height, depth,
       windowWidth, windowHeight,
       strideX, strideY,
       padLeft, padTop,
       dilateX, dilateY,
       gcdx, gcdy, xbar, ybar, ubar, vbar) ;

      return context.setError(context.getCudaHelper().catchCudaError(__func__)) ;
    }

  } ;

} }

// Instantiations
template struct vl::impl::im2row<vl::VLDT_GPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::im2row<vl::VLDT_GPU, double> ;
#endif
