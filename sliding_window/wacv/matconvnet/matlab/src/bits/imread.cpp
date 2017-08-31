// @file imread.cpp
// @brief Image reader 
// @author Andrea Vedaldi

/*
Copyright (C) 2015-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "imread.hpp"
#include <cstring>

vl::ImageShape::ImageShape()
: height(0), width(0), depth(0)
{ }

vl::ImageShape::ImageShape(size_t height, size_t width, size_t depth)
: height(height), width(width), depth(depth)
{ }

vl::ImageShape::ImageShape(ImageShape const & im)
: height(im.height), width(im.width), depth(im.depth)
{ }

vl::ImageShape & vl::ImageShape::operator =(vl::ImageShape const & im)
{
  height = im.height ;
  width = im.width ;
  depth = im.depth ;
  return *this ;
}

bool vl::ImageShape::operator == (vl::ImageShape const & im)
{
  return
  (height == im.height) &
  (width == im.width) &
  (depth == im.depth) ;
}

size_t vl::ImageShape::getNumElements() const
{
  return height*width*depth ;
}

void vl::ImageShape::clear()
{
  height = 0 ;
  width = 0 ;
  depth = 0 ;
}

vl::Image::Image()
: shape(), memory(NULL)
{ }

vl::Image::Image(Image const & im)
: shape(im.shape), memory(im.memory)
{ }

vl::Image::Image(vl::ImageShape const & shape, float * memory)
: shape(shape), memory(memory)
{ }

vl::ImageShape const & vl::Image::getShape() const { return shape ; }
float * vl::Image::getMemory() const { return memory ; }

void vl::Image::clear()
{
  shape.clear() ;
  memory = 0 ;
}
