/* 
* @Author: BlahGeek
* @Date:   2015-11-03
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-04-06
*/

#include "./fullframe_fisheye_cam.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace vr;

static void cubeZero_copy( double *a, int *n, double *root );
static void squareZero_copy( double *a, int *n, double *root );
static double cubeRoot_copy( double x );

#define EXCLUDE_MASK_PADDING 0

static void cubeZero_copy( double *a, int *n, double *root ){
    if( a[3] == 0.0 ){ // second order polynomial
        squareZero_copy( a, n, root );
    }else{
        double p = ((-1.0/3.0) * (a[2]/a[3]) * (a[2]/a[3]) + a[1]/a[3]) / 3.0;
        double q = ((2.0/27.0) * (a[2]/a[3]) * (a[2]/a[3]) * (a[2]/a[3]) - (1.0/3.0) * (a[2]/a[3]) * (a[1]/a[3]) + a[0]/a[3]) / 2.0;
        
        if( q*q + p*p*p >= 0.0 ){
            *n = 1;
            root[0] = cubeRoot_copy(-q + sqrt(q*q + p*p*p)) + cubeRoot_copy(-q - sqrt(q*q + p*p*p)) - a[2] / (3.0 * a[3]);
        }else{
            double phi = acos( -q / sqrt(-p*p*p) );
            *n = 3;
            root[0] =  2.0 * sqrt(-p) * cos(phi/3.0) - a[2] / (3.0 * a[3]);
            root[1] = -2.0 * sqrt(-p) * cos(phi/3.0 + M_PI/3.0) - a[2] / (3.0 * a[3]);
            root[2] = -2.0 * sqrt(-p) * cos(phi/3.0 - M_PI/3.0) - a[2] / (3.0 * a[3]);
        }
    }
    // PrintError("%lg, %lg, %lg, %lg root = %lg", a[3], a[2], a[1], a[0], root[0]);
}

static void squareZero_copy( double *a, int *n, double *root ){
    if( a[2] == 0.0 ){ // linear equation
        if( a[1] == 0.0 ){ // constant
            if( a[0] == 0.0 ){
                *n = 1; root[0] = 0.0;
            }else{
                *n = 0;
            }
        }else{
            *n = 1; root[0] = - a[0] / a[1];
        }
    }else{
        if( 4.0 * a[2] * a[0] > a[1] * a[1] ){
            *n = 0;
        }else{
            *n = 2;
            root[0] = (- a[1] + sqrt( a[1] * a[1] - 4.0 * a[2] * a[0] )) / (2.0 * a[2]);
            root[1] = (- a[1] - sqrt( a[1] * a[1] - 4.0 * a[2] * a[0] )) / (2.0 * a[2]);
        }
    }

}

static double cubeRoot_copy( double x ){
    if( x == 0.0 )
        return 0.0;
    else if( x > 0.0 )
        return pow(x, 1.0/3.0);
    else
        return - pow(-x, 1.0/3.0);
}

static double smallestRoot_copy( double *p ){
    int n,i;
    double root[3], sroot = 1000.0;
    
    cubeZero_copy( p, &n, root );
    
    for( i=0; i<n; i++){
        // PrintError("Root %d = %lg", i,root[i]);
        if(root[i] > 0.0 && root[i] < sroot)
            sroot = root[i];
    }
    
    // PrintError("Smallest Root  = %lg", sroot);
    return sroot;
}

static double CalcCorrectionRadius_copy(double *coeff )
{
    double a[4];
    int k;
    
    for( k=0; k<4; k++ )
    {
        a[k] = 0.0;//1.0e-10;
        if( coeff[k] != 0.0 )
        {
            a[k] = (k+1) * coeff[k];
        }
    }
    return smallestRoot_copy( a );
}

FullFrameFisheyeCamera::FullFrameFisheyeCamera(const rapidjson::Value & options): 
Camera(options) {

    this->size.width = options["width"].GetInt();
    this->size.height = options["height"].GetInt();

    if(options.HasMember("crop")) {
        std::vector<int> crop_args;
        for(auto x = options["crop"]["rect"].Begin() ; x != options["crop"]["rect"].End() ; x ++)
            crop_args.push_back(x->GetInt());
        crop.x = crop_args[0];
        crop.y = crop_args[2];
        crop.width = crop_args[1] - crop_args[0];
        crop.height = crop_args[3] - crop_args[2];
        crop_is_circular = options["crop"]["is_circular"].GetBool();
    }

    if(crop.area() == 0) {
        crop = cv::Rect(cv::Point(0, 0), this->size);
        crop_is_circular = false;
    }
    std::cerr << "crop: " << crop << ", " << crop_is_circular << std::endl;

    this->hfov = options["hfov"].GetDouble();
    this->center_shift.x = options["center_dx"].GetDouble();
    this->center_shift.y = options["center_dy"].GetDouble();

    auto _r = options["radial"].GetArray();
    this->radial_distortion[3] = _r[0].GetDouble();
    this->radial_distortion[2] = _r[1].GetDouble();
    this->radial_distortion[1] = _r[2].GetDouble();
    this->radial_distortion[0] = 1.0 - _r[0].GetDouble() - _r[1].GetDouble() - _r[2].GetDouble();
    this->radial_distortion[4] = (crop.width < crop.height ? crop.width : crop.height) / 2.0;
    this->radial_distortion[5] = CalcCorrectionRadius_copy(this->radial_distortion);

}

double FullFrameFisheyeCamera::get_aspect_ratio() {
    return double(this->size.width) / this->size.height;
}

#define VAR(n) (this->radial_distortion[n])

cv::Point2d FullFrameFisheyeCamera::do_radial_distort(cv::Point2d orig) {
    double r, scale;

    r = (sqrt( orig.x*orig.x + orig.y*orig.y )) / VAR(4);
    if( r < VAR(5) )
        scale = ((VAR(3) * r + VAR(2)) * r + VAR(1)) * r + VAR(0);
    else
        scale = 1000.0;  // WTF?

    return orig * scale;
}

cv::Point2d FullFrameFisheyeCamera::do_reverse_radial_distort(cv::Point2d orig) {
    double _sqrt = sqrt(orig.x * orig.x + orig.y * orig.y);

    std::vector<double> coeffs({ - _sqrt / VAR(4), VAR(0), VAR(1), VAR(2), VAR(3)});
    cv::Mat roots;
    double r = -1;

    cv::solvePoly(coeffs, roots);
    for(int i = 0 ; i < roots.rows ; i += 1) {
        double real = roots.at<double>(i, 0);
        double virt = roots.at<double>(i, 1);
        if(fabs(virt) < 1e-5 && real > 0)
            r = real;
    }
    //CV_Assert(r > 0);

    double scale;
    if(r < VAR(5) && r > 0)
        scale = _sqrt / VAR(4) / r;
    else
        scale = 1000.0;

    return orig / scale;

}

#undef VAR

cv::Point2d FullFrameFisheyeCamera::obj_to_image_single(const cv::Point2d & lonlat) {
    double lon = lonlat.x, lat = lonlat.y;

    double s = cos(lat) * cos(lon);
    double v1 = sin(lat);
    double v0 = - cos(lat) * sin(lon);
    double r = sqrt(v0 * v0 + v1 * v1);
    double theta = atan2(r, s);
    double distance = double(this->crop.width) / (this->hfov);

    double x = - (theta * v0 / r) * distance;
    double y = - (theta * v1 / r) * distance;
    // double y = - (theta * v1 / r) / (this->hfov / double(crop.width) * double(crop.height) * 0.5);

    auto ret = this->do_radial_distort(cv::Point2d(x, y));
    ret += this->center_shift;

    ret.x /= double(this->crop.width);
    ret.y /= double(this->crop.height);

    ret.x += 0.5;
    ret.y += 0.5;

    if(crop_is_circular && (ret.x - 0.5) * (ret.x - 0.5) + (ret.y - 0.5) * (ret.y - 0.5) > 0.25)
        return cv::Point2d(NAN, NAN);
    ret.x = (ret.x * this->crop.width) + crop.x;
    ret.y = (ret.y * this->crop.height) + crop.y;
    ret.x /= double(size.width);
    ret.y /= double(size.height);

    return ret;
}

cv::Point2d FullFrameFisheyeCamera::image_to_obj_single(const cv::Point2d & _xy) {
    CV_Assert(this->crop.size() == this->size && this->crop.tl() == cv::Point(0, 0));

    auto xy = _xy;

    xy.x -= 0.5;
    xy.y -= 0.5;

    xy.x *= double(this->crop.width);
    xy.y *= double(this->crop.height);

    xy -= this->center_shift;
    xy = this->do_reverse_radial_distort(xy);

    double distance = double(this->crop.width) / (this->hfov);

    double alpha = atan2(-xy.y, xy.x);

    double theta = - xy.y / distance / sin(alpha);
    if(fabs(sin(alpha)) < 1e-1)
        theta = xy.x / distance / cos(alpha);

    double lon = atan2(sin(theta) * cos(alpha), cos(theta));
    double lat = atan(tan(alpha) * sin(lon));

    return cv::Point2d(lon, lat);

}
