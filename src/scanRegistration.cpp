// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// Further contributions copyright (c) 2017, Stefan Glaser
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// This is an implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

#include <cmath>
#include <vector>

#include <loam_velodyne/common.h>
#include <opencv/cv.h>
#include <nav_msgs/Odometry.h>
#include <opencv/cv.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <velodyne_pointcloud/point_types.h> 
#include <velodyne_pointcloud/rawdata.h>

using std::sin;
using std::cos;
using std::atan;
using std::atan2;
using std::sqrt;



const float SCAN_PERIOD = 0.1;

/**
 * The number of cloud messages that should be skipped after initial startup.
 *
 * Skipping the first couple of messages allows the internal IMU buffers to fill up with actual data (I guess).
 */
const int SYSTEM_INIT_DELAY = 20;

/** Counter holding the number of remaining cloud messages to skip after initial startup. */
int systemInitCount = SYSTEM_INIT_DELAY;


/** The number of (equally distributed) regions used to distribute the feature extraction within a scan ring. */
size_t N_FEATURE_REGIONS = 6;

/** The number of surrounding points (+/- region around a point) used to calculate a point curvature. */
size_t CURVATURE_REGION = 5;


/** Buffer for holding point curvatures within the currently processed region. */
std::vector<float> regionCurvatures(200);

/** Buffer for holding point labels within the currently processed region. */
std::vector<int16_t> regionLabels(200);

/** Sorted list of indices (based on point curvature) within the currently processed region. */
std::vector<size_t> regionSortIdx(200);

/** Flag vector indicating if a neighboring point within the current scan ring was already picked. */
std::vector<int16_t> ringNeighborPicked(3000);

int imuPointerFront = 0;
int imuPointerLast = -1;
const int imuQueLength = 200;

float imuRollStart = 0, imuPitchStart = 0, imuYawStart = 0;
float imuRollCur = 0, imuPitchCur = 0, imuYawCur = 0;

float imuVeloXStart = 0, imuVeloYStart = 0, imuVeloZStart = 0;
float imuShiftXStart = 0, imuShiftYStart = 0, imuShiftZStart = 0;

float imuVeloXCur = 0, imuVeloYCur = 0, imuVeloZCur = 0;
float imuShiftXCur = 0, imuShiftYCur = 0, imuShiftZCur = 0;

float imuShiftFromStartXCur = 0, imuShiftFromStartYCur = 0, imuShiftFromStartZCur = 0;
float imuVeloFromStartXCur = 0, imuVeloFromStartYCur = 0, imuVeloFromStartZCur = 0;

double imuTime[imuQueLength] = {0};
float imuRoll[imuQueLength] = {0};
float imuPitch[imuQueLength] = {0};
float imuYaw[imuQueLength] = {0};

float imuAccX[imuQueLength] = {0};
float imuAccY[imuQueLength] = {0};
float imuAccZ[imuQueLength] = {0};

float imuVeloX[imuQueLength] = {0};
float imuVeloY[imuQueLength] = {0};
float imuVeloZ[imuQueLength] = {0};

float imuShiftX[imuQueLength] = {0};
float imuShiftY[imuQueLength] = {0};
float imuShiftZ[imuQueLength] = {0};

ros::Publisher pubLaserCloud;
ros::Publisher pubCornerPointsSharp;
ros::Publisher pubCornerPointsLessSharp;
ros::Publisher pubSurfPointsFlat;
ros::Publisher pubSurfPointsLessFlat;
ros::Publisher pubImuTrans;



void ShiftToStartIMU(float pointTime)
{
  imuShiftFromStartXCur = imuShiftXCur - imuShiftXStart - imuVeloXStart * pointTime;
  imuShiftFromStartYCur = imuShiftYCur - imuShiftYStart - imuVeloYStart * pointTime;
  imuShiftFromStartZCur = imuShiftZCur - imuShiftZStart - imuVeloZStart * pointTime;

  float x1 = cos(imuYawStart) * imuShiftFromStartXCur - sin(imuYawStart) * imuShiftFromStartZCur;
  float y1 = imuShiftFromStartYCur;
  float z1 = sin(imuYawStart) * imuShiftFromStartXCur + cos(imuYawStart) * imuShiftFromStartZCur;

  float x2 = x1;
  float y2 = cos(imuPitchStart) * y1 + sin(imuPitchStart) * z1;
  float z2 = -sin(imuPitchStart) * y1 + cos(imuPitchStart) * z1;

  imuShiftFromStartXCur = cos(imuRollStart) * x2 + sin(imuRollStart) * y2;
  imuShiftFromStartYCur = -sin(imuRollStart) * x2 + cos(imuRollStart) * y2;
  imuShiftFromStartZCur = z2;
}



void VeloToStartIMU()
{
  imuVeloFromStartXCur = imuVeloXCur - imuVeloXStart;
  imuVeloFromStartYCur = imuVeloYCur - imuVeloYStart;
  imuVeloFromStartZCur = imuVeloZCur - imuVeloZStart;

  float x1 = cos(imuYawStart) * imuVeloFromStartXCur - sin(imuYawStart) * imuVeloFromStartZCur;
  float y1 = imuVeloFromStartYCur;
  float z1 = sin(imuYawStart) * imuVeloFromStartXCur + cos(imuYawStart) * imuVeloFromStartZCur;

  float x2 = x1;
  float y2 = cos(imuPitchStart) * y1 + sin(imuPitchStart) * z1;
  float z2 = -sin(imuPitchStart) * y1 + cos(imuPitchStart) * z1;

  imuVeloFromStartXCur = cos(imuRollStart) * x2 + sin(imuRollStart) * y2;
  imuVeloFromStartYCur = -sin(imuRollStart) * x2 + cos(imuRollStart) * y2;
  imuVeloFromStartZCur = z2;
}



void TransformToStartIMU(PointType *p)
{
  float x1 = cos(imuRollCur) * p->x - sin(imuRollCur) * p->y;
  float y1 = sin(imuRollCur) * p->x + cos(imuRollCur) * p->y;
  float z1 = p->z;

  float x2 = x1;
  float y2 = cos(imuPitchCur) * y1 - sin(imuPitchCur) * z1;
  float z2 = sin(imuPitchCur) * y1 + cos(imuPitchCur) * z1;

  float x3 = cos(imuYawCur) * x2 + sin(imuYawCur) * z2;
  float y3 = y2;
  float z3 = -sin(imuYawCur) * x2 + cos(imuYawCur) * z2;

  float x4 = cos(imuYawStart) * x3 - sin(imuYawStart) * z3;
  float y4 = y3;
  float z4 = sin(imuYawStart) * x3 + cos(imuYawStart) * z3;

  float x5 = x4;
  float y5 = cos(imuPitchStart) * y4 + sin(imuPitchStart) * z4;
  float z5 = -sin(imuPitchStart) * y4 + cos(imuPitchStart) * z4;

  p->x = cos(imuRollStart) * x5 + sin(imuRollStart) * y5 + imuShiftFromStartXCur;
  p->y = -sin(imuRollStart) * x5 + cos(imuRollStart) * y5 + imuShiftFromStartYCur;
  p->z = z5 + imuShiftFromStartZCur;
}



void AccumulateIMUShift()
{
  float roll = imuRoll[imuPointerLast];
  float pitch = imuPitch[imuPointerLast];
  float yaw = imuYaw[imuPointerLast];
  float accX = imuAccX[imuPointerLast];
  float accY = imuAccY[imuPointerLast];
  float accZ = imuAccZ[imuPointerLast];

  float x1 = cos(roll) * accX - sin(roll) * accY;
  float y1 = sin(roll) * accX + cos(roll) * accY;
  float z1 = accZ;

  float x2 = x1;
  float y2 = cos(pitch) * y1 - sin(pitch) * z1;
  float z2 = sin(pitch) * y1 + cos(pitch) * z1;

  accX = cos(yaw) * x2 + sin(yaw) * z2;
  accY = y2;
  accZ = -sin(yaw) * x2 + cos(yaw) * z2;

  int imuPointerBack = (imuPointerLast + imuQueLength - 1) % imuQueLength;
  double timeDiff = imuTime[imuPointerLast] - imuTime[imuPointerBack];
  if (timeDiff < SCAN_PERIOD) {

    imuShiftX[imuPointerLast] = float(imuShiftX[imuPointerBack] + imuVeloX[imuPointerBack] * timeDiff
                              + accX * timeDiff * timeDiff / 2);
    imuShiftY[imuPointerLast] = float(imuShiftY[imuPointerBack] + imuVeloY[imuPointerBack] * timeDiff
                              + accY * timeDiff * timeDiff / 2);
    imuShiftZ[imuPointerLast] = float(imuShiftZ[imuPointerBack] + imuVeloZ[imuPointerBack] * timeDiff
                              + accZ * timeDiff * timeDiff / 2);

    imuVeloX[imuPointerLast] = float(imuVeloX[imuPointerBack] + accX * timeDiff);
    imuVeloY[imuPointerLast] = float(imuVeloY[imuPointerBack] + accY * timeDiff);
    imuVeloZ[imuPointerLast] = float(imuVeloZ[imuPointerBack] + accZ * timeDiff);
  }
}



/**
 * Handler function for incoming point clouds.
 *
 * This function implements the scan-registration part of the LOAM framework.
 * It will:
 * - discard invalid points
 * - sort points based on their scan ring
 * - extract corner and flat feature points
 *
 * In case IMU information is available, it is used to register the points
 * into the local system at the end of the measurement interval.
 *
 * @param laserCloudMsg The new point cloud message.
 */
void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
{
  // check system state
  if (systemInitCount > 0) {
    // not initialized yet
    systemInitCount--;

    return;
  } else if (systemInitCount == 0) {
    // initialize system
  }

  uint16_t nScanRings = 0;
  std::vector<size_t> ringStartIndices;
  std::vector<size_t> ringEndIndices;
  std::vector<pcl::PointCloud<PointType> > laserCloudScans;
  ringStartIndices.reserve(16);
  ringEndIndices.reserve(16);
  laserCloudScans.reserve(16);

  double timeScanCur = laserCloudMsg->header.stamp.toSec();
  pcl::PointCloud<velodyne_pointcloud::PointXYZIR> laserCloudIn;
  pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);
  size_t cloudSize = laserCloudIn.points.size();

  float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
  float endOri = float(-atan2(laserCloudIn.points[cloudSize - 1].y,
                        laserCloudIn.points[cloudSize - 1].x) + 2 * M_PI);
  if (endOri - startOri > 3 * M_PI) {
    endOri -= 2 * M_PI;
  } else if (endOri - startOri < M_PI) {
    endOri += 2 * M_PI;
  }

  bool halfPassed = false;
  PointType point;


  // --------------------------------------------------------------------------
  // Step 1.1: Extract, sort and register new points in local system
  // -- Extract:  Discard NaN and zero valued points.
  // -- Sort:     Sort new points based on their ring number.
  // -- Register: Use IMU information (if available) to register points in the local system.
  for (size_t i = 0; i < cloudSize; i++) {
    point.x = laserCloudIn.points[i].y;
    point.y = laserCloudIn.points[i].z;
    point.z = laserCloudIn.points[i].x;

    // discard NaN valued points
    if (!pcl_isfinite (point.x) ||
        !pcl_isfinite (point.y) ||
        !pcl_isfinite (point.z)) {
      ROS_DEBUG("Found NaN point!");
      continue;
    }

    // discard zero valued points
    if (point.x * point.x * point.y * point.y + point.z * point.z < 0.0001) {
      continue;
    }

    // fetch scan ring index and resize scan ring buffers if necessary
    uint16_t ringID = laserCloudIn.points[i].ring;
    if (ringID >= nScanRings){
      nScanRings = ringID + 1;
      ringStartIndices.resize(nScanRings, 0);
      ringEndIndices.resize(nScanRings, 0);
      laserCloudScans.resize(nScanRings);
    }

    // extract horizontal point orientation
    float ori = -atan2(point.x, point.z);
    if (!halfPassed) {
      if (ori < startOri - M_PI / 2) {
        ori += 2 * M_PI;
      } else if (ori > startOri + M_PI * 3 / 2) {
        ori -= 2 * M_PI;
      }

      if (ori - startOri > M_PI) {
        halfPassed = true;
      }
    } else {
      ori += 2 * M_PI;

      if (ori < endOri - M_PI * 3 / 2) {
        ori += 2 * M_PI;
      } else if (ori > endOri + M_PI / 2) {
        ori -= 2 * M_PI;
      } 
    }

    float relTime = (ori - startOri) / (endOri - startOri);
    point.intensity = ringID + SCAN_PERIOD * relTime;

    // register new points into local system using the IMU
    if (imuPointerLast >= 0) {
      float pointTime = relTime * SCAN_PERIOD;
      while (imuPointerFront != imuPointerLast) {
        if (timeScanCur + pointTime < imuTime[imuPointerFront]) {
          break;
        }
        imuPointerFront = (imuPointerFront + 1) % imuQueLength;
      }

      if (timeScanCur + pointTime > imuTime[imuPointerFront]) {
        imuRollCur = imuRoll[imuPointerFront];
        imuPitchCur = imuPitch[imuPointerFront];
        imuYawCur = imuYaw[imuPointerFront];

        imuVeloXCur = imuVeloX[imuPointerFront];
        imuVeloYCur = imuVeloY[imuPointerFront];
        imuVeloZCur = imuVeloZ[imuPointerFront];

        imuShiftXCur = imuShiftX[imuPointerFront];
        imuShiftYCur = imuShiftY[imuPointerFront];
        imuShiftZCur = imuShiftZ[imuPointerFront];
      } else {
        int imuPointerBack = (imuPointerFront + imuQueLength - 1) % imuQueLength;
        float ratioFront = float((timeScanCur + pointTime - imuTime[imuPointerBack])
                         / (imuTime[imuPointerFront] - imuTime[imuPointerBack]));
        float ratioBack = float((imuTime[imuPointerFront] - timeScanCur - pointTime)
                        / (imuTime[imuPointerFront] - imuTime[imuPointerBack]));

        imuRollCur = imuRoll[imuPointerFront] * ratioFront + imuRoll[imuPointerBack] * ratioBack;
        imuPitchCur = imuPitch[imuPointerFront] * ratioFront + imuPitch[imuPointerBack] * ratioBack;
        if (imuYaw[imuPointerFront] - imuYaw[imuPointerBack] > M_PI) {
          imuYawCur = float(imuYaw[imuPointerFront] * ratioFront + (imuYaw[imuPointerBack] + 2 * M_PI) * ratioBack);
        } else if (imuYaw[imuPointerFront] - imuYaw[imuPointerBack] < -M_PI) {
          imuYawCur = float(imuYaw[imuPointerFront] * ratioFront + (imuYaw[imuPointerBack] - 2 * M_PI) * ratioBack);
        } else {
          imuYawCur = imuYaw[imuPointerFront] * ratioFront + imuYaw[imuPointerBack] * ratioBack;
        }

        imuVeloXCur = imuVeloX[imuPointerFront] * ratioFront + imuVeloX[imuPointerBack] * ratioBack;
        imuVeloYCur = imuVeloY[imuPointerFront] * ratioFront + imuVeloY[imuPointerBack] * ratioBack;
        imuVeloZCur = imuVeloZ[imuPointerFront] * ratioFront + imuVeloZ[imuPointerBack] * ratioBack;

        imuShiftXCur = imuShiftX[imuPointerFront] * ratioFront + imuShiftX[imuPointerBack] * ratioBack;
        imuShiftYCur = imuShiftY[imuPointerFront] * ratioFront + imuShiftY[imuPointerBack] * ratioBack;
        imuShiftZCur = imuShiftZ[imuPointerFront] * ratioFront + imuShiftZ[imuPointerBack] * ratioBack;
      }

      if (i == 0) {
        imuRollStart = imuRollCur;
        imuPitchStart = imuPitchCur;
        imuYawStart = imuYawCur;

        imuVeloXStart = imuVeloXCur;
        imuVeloYStart = imuVeloYCur;
        imuVeloZStart = imuVeloZCur;

        imuShiftXStart = imuShiftXCur;
        imuShiftYStart = imuShiftYCur;
        imuShiftZStart = imuShiftZCur;
      } else {
        ShiftToStartIMU(pointTime);
        VeloToStartIMU();
        TransformToStartIMU(&point);
      }
    }

    // store point in corresponding ring cloud
    laserCloudScans[ringID].push_back(point);
  }

  if (nScanRings == 0) {
    // Nothing to compute... input cloud is invalid or empty.
    return;
  }


  // --------------------------------------------------------------------------
  // Step 1.2: Construct sorted, full resolution point cloud and store ring indices
  pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());

  cloudSize = 0;
  for (uint16_t ringID = 0; ringID < nScanRings; ringID++) {
    // append scan ring cloud to sorted cloud
    *laserCloud += laserCloudScans[ringID];

    ringStartIndices[ringID] = cloudSize;
    cloudSize += laserCloudScans[ringID].points.size();
    ringEndIndices[ringID] = cloudSize - 1;
  }


  // --------------------------------------------------------------------------
  // Step 1.3: Extract feature points
  pcl::PointCloud<PointType> cornerPointsSharp;
  pcl::PointCloud<PointType> cornerPointsLessSharp;
  pcl::PointCloud<PointType> surfPointsFlat;
  pcl::PointCloud<PointType> surfPointsLessFlat;

  // iterate over each scan ring and extract feature points
  for (uint16_t ringID = 0; ringID < nScanRings; ringID++) {
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);
    size_t ringStartIdx = ringStartIndices[ringID];
    size_t ringEndIdx = ringEndIndices[ringID];
    size_t ringSize = ringEndIdx - ringStartIdx;

    // skip empty scan rings
    if (ringEndIdx <= ringStartIdx) {
      continue;
    }

    // resize and reset ringNeighborPicked vector
    ringNeighborPicked.assign(ringSize, 0);


    // mark unreliable points in ring as picked
    for (size_t ringIdx = CURVATURE_REGION; ringIdx < ringSize - CURVATURE_REGION - 1; ringIdx++) {
      size_t pi = ringStartIdx + ringIdx;

      float fwdDiff = calcSquaredDiff(laserCloud->points[pi + 1], laserCloud->points[pi]);

      if (fwdDiff > 0.1) {
        float dist1 = calcPointDistance(laserCloud->points[pi]);
        float dist2 = calcPointDistance(laserCloud->points[pi + 1]);

        if (dist1 > dist2) {
          if (sqrt(calcSquaredDiff(laserCloud->points[pi + 1], laserCloud->points[pi], dist2 / dist1)) / dist2 < 0.1) {
            for (size_t i = 0; i <= CURVATURE_REGION; i++) {
              ringNeighborPicked[ringIdx - i] = 1;
            }

            continue;
          }
        } else {
          if (sqrt(calcSquaredDiff(laserCloud->points[pi + 1], dist1 / dist2, laserCloud->points[pi])) / dist1 < 0.1) {
            for (size_t i = CURVATURE_REGION + 1; i > 0; i--) {
              ringNeighborPicked[ringIdx + i] = 1;
            }
          }
        }
      }

      float backDiff = calcSquaredDiff(laserCloud->points[pi], laserCloud->points[pi - 1]);
      float distSqare = calcSquaredPointDistance(laserCloud->points[pi]);

      if (fwdDiff > 0.0002 * distSqare && backDiff > 0.0002 * distSqare) {
        ringNeighborPicked[ringIdx] = 1;
      }
    }


    // Distribute feature extraction across multiple regions of equal size.
    // Each region can provide at most 2 sharp corners and 20 less sharp corners,
    // as well as at most 4 flat surface points.
    for (size_t regionID = 0; regionID < N_FEATURE_REGIONS; regionID++) {
      size_t regionStartIdx = ((ringStartIdx + CURVATURE_REGION) * (N_FEATURE_REGIONS - regionID)
                + (ringEndIdx - CURVATURE_REGION) * regionID)
                / N_FEATURE_REGIONS;
      size_t regionEndIdx = ((ringStartIdx + CURVATURE_REGION) * (N_FEATURE_REGIONS - 1 - regionID)
                + (ringEndIdx - CURVATURE_REGION) * (regionID + 1))
                / N_FEATURE_REGIONS;
      size_t regionSize = regionEndIdx - regionStartIdx;

      // skip empty regions
      if (regionEndIdx <= regionStartIdx) {
        continue;
      }

      // resize region buffers
      regionCurvatures.resize(regionSize);
      regionSortIdx.resize(regionSize);
      regionLabels.assign(regionSize, 0);


      // calculate point curvatures and reset sorting indices
      for (size_t i = regionStartIdx, regionIdx = 0; i < regionEndIdx; i++, regionIdx++) {
        float diffX = -2 * CURVATURE_REGION * laserCloud->points[i].x;
        float diffY = -2 * CURVATURE_REGION * laserCloud->points[i].y;
        float diffZ = -2 * CURVATURE_REGION * laserCloud->points[i].z;

        for (size_t j = 1; j <= CURVATURE_REGION; j++) {
          diffX += laserCloud->points[i - j].x + laserCloud->points[i + j].x;
          diffY += laserCloud->points[i - j].y + laserCloud->points[i + j].y;
          diffZ += laserCloud->points[i - j].z + laserCloud->points[i + j].z;
        }

        regionCurvatures[regionIdx] = diffX * diffX + diffY * diffY + diffZ * diffZ;
        regionSortIdx[regionIdx] = i;
      }

      // sort features from low to high curvature
      for (size_t i = 1; i < regionSize; i++) {
        for (size_t j = i; j >= 1; j--) {
          if (regionCurvatures[regionSortIdx[j] - regionStartIdx] < regionCurvatures[regionSortIdx[j - 1] - regionStartIdx]) {
            size_t temp = regionSortIdx[j - 1];
            regionSortIdx[j - 1] = regionSortIdx[j];
            regionSortIdx[j] = temp;
          }
        }
      }


      // extract corner points
      int largestPickedNum = 0;
      for (size_t i = regionSize; i > 0;) {
        size_t idx = regionSortIdx[--i];
        size_t ringIdx = idx - ringStartIdx;
        size_t regionIdx = idx - regionStartIdx;

        if (ringNeighborPicked[ringIdx] == 0 &&
            regionCurvatures[regionIdx] > 0.1) {
        
          largestPickedNum++;
          if (largestPickedNum <= 2) {
            regionLabels[regionIdx] = 2;
            cornerPointsSharp.push_back(laserCloud->points[idx]);
            cornerPointsLessSharp.push_back(laserCloud->points[idx]);
          } else if (largestPickedNum <= 20) {
            regionLabels[regionIdx] = 1;
            cornerPointsLessSharp.push_back(laserCloud->points[idx]);
          } else {
            break;
          }

          ringNeighborPicked[ringIdx] = 1;
          for (size_t j = 1; j <= CURVATURE_REGION; j++) {
            if (calcSquaredDiff(laserCloud->points[idx + j], laserCloud->points[idx + j - 1]) > 0.05) {
              break;
            }

            ringNeighborPicked[ringIdx + j] = 1;
          }
          for (size_t j = 1; j <= CURVATURE_REGION; j++) {
            if (calcSquaredDiff(laserCloud->points[idx - j], laserCloud->points[idx - j + 1]) > 0.05) {
              break;
            }

            ringNeighborPicked[ringIdx - j] = 1;
          }
        }
      }


      // extract flat surface points
      int smallestPickedNum = 0;
      for (size_t i = 0; i < regionSize; i++) {
        size_t idx = regionSortIdx[i];
        size_t ringIdx = idx - ringStartIdx;
        size_t regionIdx = idx - regionStartIdx;

        if (ringNeighborPicked[ringIdx] == 0 &&
            regionCurvatures[regionIdx] < 0.1) {

          regionLabels[regionIdx] = -1;
          surfPointsFlat.push_back(laserCloud->points[idx]);

          // TODO: Think about why we break here after the point has been added, but before its neighbors are marked
          smallestPickedNum++;
          if (smallestPickedNum >= 4) {
            break;
          }

          ringNeighborPicked[ringIdx] = 1;
          for (size_t j = 1; j <= CURVATURE_REGION; j++) {
            if (calcSquaredDiff(laserCloud->points[idx + j], laserCloud->points[idx + j - 1]) > 0.05) {
              break;
            }

            ringNeighborPicked[ringIdx + j] = 1;
          }
          for (size_t j = 1; j <= CURVATURE_REGION; j++) {
            if (calcSquaredDiff(laserCloud->points[idx - j], laserCloud->points[idx - j + 1]) > 0.05) {
              break;
            }

            ringNeighborPicked[ringIdx - j] = 1;
          }
        }
      }


      // extract less flat surface points
      // TODO: This loop could be integrated into the above one to reduce some index lookups, etc.
      for (size_t i = 0; i < regionSize; i++) {
        if (regionLabels[i] <= 0) {
          surfPointsLessFlatScan->push_back(laserCloud->points[regionStartIdx + i]);
        }
      }
    }

    // downsize and store less flat surface points
    pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
    pcl::VoxelGrid<PointType> downSizeFilter;
    downSizeFilter.setInputCloud(surfPointsLessFlatScan);
    downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
    downSizeFilter.filter(surfPointsLessFlatScanDS);

    surfPointsLessFlat += surfPointsLessFlatScanDS;
  }


  // --------------------------------------------------------------------------
  // publish results
  sensor_msgs::PointCloud2 laserCloudOutMsg;
  pcl::toROSMsg(*laserCloud, laserCloudOutMsg);
  laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
  laserCloudOutMsg.header.frame_id = "/camera";
  pubLaserCloud.publish(laserCloudOutMsg);

  sensor_msgs::PointCloud2 cornerPointsSharpMsg;
  pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
  cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
  cornerPointsSharpMsg.header.frame_id = "/camera";
  pubCornerPointsSharp.publish(cornerPointsSharpMsg);

  sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
  pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
  cornerPointsLessSharpMsg.header.stamp = laserCloudMsg->header.stamp;
  cornerPointsLessSharpMsg.header.frame_id = "/camera";
  pubCornerPointsLessSharp.publish(cornerPointsLessSharpMsg);

  sensor_msgs::PointCloud2 surfPointsFlat2;
  pcl::toROSMsg(surfPointsFlat, surfPointsFlat2);
  surfPointsFlat2.header.stamp = laserCloudMsg->header.stamp;
  surfPointsFlat2.header.frame_id = "/camera";
  pubSurfPointsFlat.publish(surfPointsFlat2);

  sensor_msgs::PointCloud2 surfPointsLessFlat2;
  pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlat2);
  surfPointsLessFlat2.header.stamp = laserCloudMsg->header.stamp;
  surfPointsLessFlat2.header.frame_id = "/camera";
  pubSurfPointsLessFlat.publish(surfPointsLessFlat2);

  pcl::PointCloud<pcl::PointXYZ> imuTrans(4, 1);
  imuTrans.points[0].x = imuPitchStart;
  imuTrans.points[0].y = imuYawStart;
  imuTrans.points[0].z = imuRollStart;

  imuTrans.points[1].x = imuPitchCur;
  imuTrans.points[1].y = imuYawCur;
  imuTrans.points[1].z = imuRollCur;

  imuTrans.points[2].x = imuShiftFromStartXCur;
  imuTrans.points[2].y = imuShiftFromStartYCur;
  imuTrans.points[2].z = imuShiftFromStartZCur;

  imuTrans.points[3].x = imuVeloFromStartXCur;
  imuTrans.points[3].y = imuVeloFromStartYCur;
  imuTrans.points[3].z = imuVeloFromStartZCur;

  sensor_msgs::PointCloud2 imuTransMsg;
  pcl::toROSMsg(imuTrans, imuTransMsg);
  imuTransMsg.header.stamp = laserCloudMsg->header.stamp;
  imuTransMsg.header.frame_id = "/camera";
  pubImuTrans.publish(imuTransMsg);
}



/**
 * IMU message handler.
 *
 * @param imuIn The new IMU message.
 */
void imuHandler(const sensor_msgs::Imu::ConstPtr& imuIn)
{
  double roll, pitch, yaw;
  tf::Quaternion orientation;
  tf::quaternionMsgToTF(imuIn->orientation, orientation);
  tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

  imuPointerLast = (imuPointerLast + 1) % imuQueLength;

  imuTime[imuPointerLast] = imuIn->header.stamp.toSec();
  imuRoll[imuPointerLast] = float(roll);
  imuPitch[imuPointerLast] = float(pitch);
  imuYaw[imuPointerLast] = float(yaw);
  imuAccX[imuPointerLast] = float(imuIn->linear_acceleration.y - sin(roll) * cos(pitch) * 9.81);
  imuAccY[imuPointerLast] = float(imuIn->linear_acceleration.z - cos(roll) * cos(pitch) * 9.81);
  imuAccZ[imuPointerLast] = float(imuIn->linear_acceleration.x + sin(pitch) * 9.81);;

  AccumulateIMUShift();
}



/**
 * The main function.
 *
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char** argv)
{
  ros::init(argc, argv, "scanRegistration");
  ros::NodeHandle nh;


  // fetch parameters
  int intParam = nh.param("/scanRegistration/featureRegions", int(N_FEATURE_REGIONS));
  N_FEATURE_REGIONS = intParam < 1 ? 1 : size_t(intParam);
  ROS_INFO("Using  %d  feature regions per scan ring.", int(N_FEATURE_REGIONS));

  intParam = nh.param("/scanRegistration/curvatureRegion", int(CURVATURE_REGION));
  CURVATURE_REGION = intParam < 1 ? 1 : size_t(intParam);
  ROS_INFO("Using  +/- %d  points for curvature calculation.", int(CURVATURE_REGION));


  // register subscribers
  ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2> 
                                  ("/velodyne_points", 2, laserCloudHandler);

  ros::Subscriber subImu = nh.subscribe<sensor_msgs::Imu> ("/imu/data", 50, imuHandler);


  // register publishers
  pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>
                                 ("/velodyne_cloud_2", 2);

  pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>
                                        ("/laser_cloud_sharp", 2);

  pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>
                                            ("/laser_cloud_less_sharp", 2);

  pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>
                                       ("/laser_cloud_flat", 2);

  pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>
                                           ("/laser_cloud_less_flat", 2);

  pubImuTrans = nh.advertise<sensor_msgs::PointCloud2> ("/imu_trans", 5);


  ros::spin();

  return 0;
}

