/* 
* @Author: BlahGeek
* @Date:   2016-03-10
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-03-10
*/

#include <iostream>

#include "./ocam_fisheye.hpp"

using namespace vr;

OCamFisheyeCamera::OCamFisheyeCamera(const rapidjson::Value & options): Camera(options) {
    auto opt_pols = options["pol"].GetArray();
    this->model.length_pol = opt_pols.Size();
    CV_Assert(this->model.length_pol > 0);
    for(int i = 0 ; i < this->model.length_pol ; i += 1)
        this->model.pol[i] = opt_pols[i].GetDouble();

    auto opt_invpols = options["invpol"].GetArray();
    this->model.length_invpol = opt_invpols.Size();
    CV_Assert(this->model.length_invpol > 0);
    for(int i = 0 ; i < this->model.length_invpol ; i += 1)
        this->model.invpol[i] = opt_invpols[i].GetDouble();

    this->model.xc = options["xc"].GetDouble();
    this->model.yc = options["yc"].GetDouble();
    this->model.c = options["c"].GetDouble();
    this->model.d = options["d"].GetDouble();
    this->model.e = options["e"].GetDouble();

    this->model.width = options["width"].GetInt();
    this->model.height = options["height"].GetInt();
}

double OCamFisheyeCamera::get_aspect_ratio() {
    return double(this->model.width) / this->model.height;
}

// copy from ocam_functions.cpp

/*------------------------------------------------------------------------------
 CAM2WORLD projects a 2D point onto the unit sphere
    CAM2WORLD(POINT3D, POINT2D, OCAM_MODEL) 
    back-projects a 2D point (point2D), in pixels coordinates, 
    onto the unit sphere returns the normalized coordinates point3D = [x;y;z]
    where (x^2 + y^2 + z^2) = 1.
    
    POINT3D = [X;Y;Z] are the coordinates of the 3D points, such that (x^2 + y^2 + z^2) = 1.
    OCAM_MODEL is the model of the calibrated camera.
    POINT2D = [rows;cols] are the pixel coordinates of the point in pixels
    
    Copyright (C) 2009 DAVIDE SCARAMUZZA   
    Author: Davide Scaramuzza - email: davide.scaramuzza@ieee.org
  
    NOTE: the coordinates of "point2D" and "center" are already according to the C
    convention, that is, start from 0 instead than from 1.
------------------------------------------------------------------------------*/
static void cam2world(double point3D[3], double point2D[2], struct OCamFisheyeCamera::ocam_model *myocam_model)
{
 double *pol    = myocam_model->pol;
 double xc      = (myocam_model->xc);
 double yc      = (myocam_model->yc); 
 double c       = (myocam_model->c);
 double d       = (myocam_model->d);
 double e       = (myocam_model->e);
 int length_pol = (myocam_model->length_pol); 
 double invdet  = 1/(c-d*e); // 1/det(A), where A = [c,d;e,1] as in the Matlab file

 double xp = invdet*(    (point2D[0] - xc) - d*(point2D[1] - yc) );
 double yp = invdet*( -e*(point2D[0] - xc) + c*(point2D[1] - yc) );
  
 double r   = sqrt(  xp*xp + yp*yp ); //distance [pixels] of  the point from the image center
 double zp  = pol[0];
 double r_i = 1;
 int i;
 
 for (i = 1; i < length_pol; i++)
 {
   r_i *= r;
   zp  += r_i*pol[i];
 }
 
 //normalize to unit norm
 double invnorm = 1/sqrt( xp*xp + yp*yp + zp*zp );
 
 point3D[0] = invnorm*xp;
 point3D[1] = invnorm*yp; 
 point3D[2] = invnorm*zp;
}

/*------------------------------------------------------------------------------
 WORLD2CAM projects a 3D point on to the image
    WORLD2CAM(POINT2D, POINT3D, OCAM_MODEL) 
    projects a 3D point (point3D) on to the image and returns the pixel coordinates (point2D).
    
    POINT3D = [X;Y;Z] are the coordinates of the 3D point.
    OCAM_MODEL is the model of the calibrated camera.
    POINT2D = [rows;cols] are the pixel coordinates of the reprojected point
    
    Copyright (C) 2009 DAVIDE SCARAMUZZA
    Author: Davide Scaramuzza - email: davide.scaramuzza@ieee.org
  
    NOTE: the coordinates of "point2D" and "center" are already according to the C
    convention, that is, start from 0 instead than from 1.
------------------------------------------------------------------------------*/
static void world2cam(double point2D[2], double point3D[3], struct OCamFisheyeCamera::ocam_model *myocam_model)
{
 double *invpol     = myocam_model->invpol; 
 double xc          = (myocam_model->xc);
 double yc          = (myocam_model->yc); 
 double c           = (myocam_model->c);
 double d           = (myocam_model->d);
 double e           = (myocam_model->e);
 int    width       = (myocam_model->width);
 int    height      = (myocam_model->height);
 int length_invpol  = (myocam_model->length_invpol);
 double norm        = sqrt(point3D[0]*point3D[0] + point3D[1]*point3D[1]);
 double theta       = atan(point3D[2]/norm);
 double t, t_i;
 double rho, x, y;
 double invnorm;
 int i;
  
  if (norm != 0) 
  {
    invnorm = 1/norm;
    t  = theta;
    rho = invpol[0];
    t_i = 1;

    for (i = 1; i < length_invpol; i++)
    {
      t_i *= t;
      rho += t_i*invpol[i];
    }

    x = point3D[0]*invnorm*rho;
    y = point3D[1]*invnorm*rho;
  
    point2D[0] = x*c + y*d + xc;
    point2D[1] = x*e + y   + yc;
  }
  else
  {
    point2D[0] = xc;
    point2D[1] = yc;
  }
}

cv::Point2d OCamFisheyeCamera::obj_to_image_single(const cv::Point2d & lonlat) {
    auto xyz = this->sphere_lonlat_to_xyz(lonlat);
    double p_xyz[] = {xyz.x, xyz.y, xyz.z};
    double p_xy[2];

    world2cam(p_xy, p_xyz, &(this->model));

    return cv::Point2d(p_xy[1] / this->model.width, p_xy[0] / this->model.height);
}

cv::Point2d OCamFisheyeCamera::image_to_obj_single(const cv::Point2d & xy) {
    double p_xy[] = {xy.y * this->model.height, xy.x * this->model.width};
    double p_xyz[3];

    cam2world(p_xyz, p_xy, &(this->model));

    return this->sphere_xyz_to_lonlat(cv::Point3d(p_xyz[0], p_xyz[1], p_xyz[2]));
}
