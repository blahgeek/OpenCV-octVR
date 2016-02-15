/* 
* @Author: BlahGeek
* @Date:   2015-11-03
* @Last Modified by:   BlahGeek
* @Last Modified time: 2016-01-21
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

FullFrameFisheyeCamera::FullFrameFisheyeCamera(const json & options): 
Camera(options) {

    this->size.width = options["width"].get<int>();
    this->size.height = options["height"].get<int>();
    if(options.find("circular_crop") != options.end()) {
        auto crop_args = options["circular_crop"].get<std::vector<double>>();
        circular_crop.x = - this->size.width * 0.5 
                          - crop_args[0] * this->size.width
                          + crop_args[2] * this->size.width;
        circular_crop.y = - this->size.height * 0.5 
                          - crop_args[1] * this->size.width
                          + crop_args[2] * this->size.width;
        circular_crop.width = this->size.width;
        circular_crop.height = this->size.height;

        this->size.height = crop_args[2] * 2 * this->size.width;
        this->size.width = crop_args[2] * 2 * this->size.width;
    }

    this->hfov = options["hfov"].get<double>();
    this->center_shift.x = options["center_dx"].get<double>();
    this->center_shift.y = options["center_dy"].get<double>();

    std::vector<double> _r = options["radial"];
    this->radial_distortion[3] = _r[0];
    this->radial_distortion[2] = _r[1];
    this->radial_distortion[1] = _r[2];
    this->radial_distortion[0] = 1.0 - _r[0] - _r[1] - _r[2];
    this->radial_distortion[4] = (size.width < size.height ? size.width : size.height) / 2.0;
    this->radial_distortion[5] = CalcCorrectionRadius_copy(this->radial_distortion);

    #define VAR(n) (this->radial_distortion[n])

    // print debug info
    json debug = {
        {"radial_distortion", {VAR(0), VAR(1), VAR(2), VAR(3), VAR(4), VAR(5)}},
        {"hfov", this->hfov},
        {"center_shift", {this->center_shift.x, this->center_shift.y}},
        {"size", {this->size.width, this->size.height}},
        {"rotate", {this->rotate_matrix.at<double>(0, 0), this->rotate_matrix.at<double>(0, 1), this->rotate_matrix.at<double>(0, 2),
                    this->rotate_matrix.at<double>(1, 0), this->rotate_matrix.at<double>(1, 1), this->rotate_matrix.at<double>(1, 2),
                    this->rotate_matrix.at<double>(2, 0), this->rotate_matrix.at<double>(2, 1), this->rotate_matrix.at<double>(2, 2),
                   }}
    };
    std::cout << debug.dump(4) << ", " << std::endl;
}

double FullFrameFisheyeCamera::get_aspect_ratio() {
    return double(this->size.width) / this->size.height;
}

cv::Point2d FullFrameFisheyeCamera::do_radial_distort(cv::Point2d orig) {
    double r, scale;

    r = (sqrt( orig.x*orig.x + orig.y*orig.y )) / VAR(4);
    if( r < VAR(5) )
        scale = ((VAR(3) * r + VAR(2)) * r + VAR(1)) * r + VAR(0);
    else
        scale = 1000.0;  // WTF?

    return orig * scale;

    #undef VAR
}

cv::Point2d FullFrameFisheyeCamera::obj_to_image_single(const cv::Point2d & lonlat) {
    double lon = lonlat.x, lat = lonlat.y;

    double s = cos(lat) * cos(lon);
    double v1 = sin(lat);
    double v0 = cos(lat) * sin(lon);
    double r = sqrt(v0 * v0 + v1 * v1);
    double theta = atan2(r, s);
    double distance = double(this->size.width) / (this->hfov);

    double x = (theta * v0 / r) * distance;
    double y = - (theta * v1 / r) * distance;
    // double y = - (theta * v1 / r) / (this->hfov / double(size.width) * double(size.height) * 0.5);

    auto ret = this->do_radial_distort(cv::Point2d(x, y));
    ret += this->center_shift;

    ret.x /= double(this->size.width);
    ret.y /= double(this->size.height);

    ret.x += 0.5;
    ret.y += 0.5;

    if(this->circular_crop.area() > 0) {
        if((ret.x - 0.5) * (ret.x - 0.5) + (ret.y - 0.5) * (ret.y - 0.5) > 0.25)
            return cv::Point2d(NAN, NAN);
        ret.x = (ret.x * this->size.width) - circular_crop.x;
        ret.y = (ret.y * this->size.height) - circular_crop.y;
        ret.x /= double(circular_crop.width);
        ret.y /= double(circular_crop.height);
    }

    return ret;
}
