/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: examples/fit-model.cpp
 *
 * Copyright 2016 Patrik Huber
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "eos/core/Image.hpp"
#include "eos/core/Image_opencv_interop.hpp"
#include "eos/core/Landmark.hpp"
#include "eos/core/LandmarkMapper.hpp"
#include "eos/core/read_pts_landmarks.hpp"
#include "eos/fitting/fitting.hpp"
#include "eos/morphablemodel/Blendshape.hpp"
#include "eos/morphablemodel/MorphableModel.hpp"
#include "eos/render/draw_utils.hpp"
#include "eos/render/texture_extraction.hpp"
#include "eos/fitting/affine_camera_estimation.hpp"

#include "glm/gtc/matrix_transform.hpp"
#include "glm/mat4x4.hpp"
#include "glm/vec4.hpp"
#include "glm/ext.hpp"

#include "Eigen/Core"

#include "boost/filesystem.hpp"
#include "boost/program_options.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "highgui.h"
#include "cv.h"

#include <iostream>
#include <experimental/optional>
#include <string>
#include <vector>
#include <cmath>

#include "eos/core/Mesh.hpp"

using namespace eos;
using namespace core;
using namespace eos::fitting;
using namespace morphablemodel;
using namespace render;
namespace po = boost::program_options;
namespace fs = boost::filesystem;
using eos::core::Landmark;
using eos::core::LandmarkCollection;
using cv::Mat;
using std::cout;
using std::endl;
using std::string;
using std::vector;
using cv::Vec3b;
using cv::Point;

using Eigen::MatrixXf;
using Eigen::Vector2f;
using Eigen::Vector4f;
using Eigen::VectorXf;
using std::vector;


/**
 * This app demonstrates estimation of the camera and fitting of the shape
 * model of a 3D Morphable Model from an ibug LFPW image with its landmarks.
 * In addition to fit-model-simple, this example uses blendshapes, contour-
 * fitting, and can iterate the fitting.
 *
 * 68 ibug landmarks are loaded from the .pts file and converted
 * to vertex indices using the LandmarkMapper.
 */

using namespace std;


Vector2f tempVec2f;
Vector3f tempVec3f;
Vector4f tempVec4f;
Vector2d tempVec2d;
Vector3d tempVec3d;
Vector4d tempVec4d;


void getCameraMatrix(cv::Mat image, const core::LandmarkCollection<Eigen::Vector2f>& landmarks, const core::LandmarkMapper& landmark_mapper, Mesh current_mesh, cv::Mat& cameraMatrix3D2D,cv::Mat& cameraMatrix2D3D  ) {

    vector<Vector4f> model_points; // the points in the 3D shape model
    vector<int> vertex_indices; // their vertex indices
    vector<Vector2f> image_points; // the corresponding 2D landmark points
    int image_height = image.rows;

for (int i = 0; i < landmarks.size(); ++i)
    {
        auto converted_name = landmark_mapper.convert(landmarks[i].name);
        if (!converted_name)
        { // no mapping defined for the current landmark
            continue;
        }
        int vertex_idx = std::stoi(converted_name.value());
        Vector4f vertex(current_mesh.vertices[vertex_idx][0], current_mesh.vertices[vertex_idx][1],
                        current_mesh.vertices[vertex_idx][2], 1.0f);
        model_points.emplace_back(vertex);
        vertex_indices.emplace_back(vertex_idx);
        image_points.emplace_back(landmarks[i].coordinates);
    }

     fitting::ScaledOrthoProjectionParameters current_pose =
        fitting::estimate_orthographic_projection_linear(image_points, model_points, true, image_height);   

    Matrix3f Roration;

    double tx = current_pose.tx, ty = current_pose.ty; double tz = 1;;
    double s = current_pose.s;

    tx*=s;ty*=s;tz*=s;

    for (int i =0; i <3; i++)
        for (int j =0; j <3; j++)
            Roration(i,j) = current_pose.R[i][j]*s;

     MatrixXf RMatrix = MatrixXf::Zero(3, 4);

    model_points.clear ();

    for (int i = 0; i < landmarks.size(); ++i)
    {
        auto converted_name = landmark_mapper.convert(landmarks[i].name);
        if (!converted_name)
        { // no mapping defined for the current landmark
            continue;
        }
        int vertex_idx = std::stoi(converted_name.value());
        Vector4f vertex(current_mesh.vertices[vertex_idx][0], current_mesh.vertices[vertex_idx][1],
                        current_mesh.vertices[vertex_idx][2], 1.0f);
            
        Vector4f vertexX = RMatrix*vertex;

        model_points.emplace_back(vertexX);
    }

     cameraMatrix3D2D = estimate_affine_camera(image_points,model_points);
     cout <<"XXX CAMERA\n";
     cout << cameraMatrix3D2D << endl<< endl;
    
}

void getCameraMatrix( Mesh current_mesh, const core::LandmarkCollection<Eigen::Vector2f>& landmarks, const core::LandmarkMapper& landmark_mapper,cv::Mat& cameraMatrix  ) {

    vector<Vector3f> model_points; // the points in the 3D shape model
    vector<int> vertex_indices; // their vertex indices
    vector<Vector2f> image_points; // the corresponding 2D landmark points
   
for (int i = 0; i < landmarks.size(); ++i)
    {
        auto converted_name = landmark_mapper.convert(landmarks[i].name);
        if (!converted_name)
        { // no mapping defined for the current landmark
            continue;
        }
        int vertex_idx = std::stoi(converted_name.value());
        Vector3f vertex(current_mesh.vertices[vertex_idx][0], current_mesh.vertices[vertex_idx][1],
                        current_mesh.vertices[vertex_idx][2]);
        model_points.emplace_back(vertex);
        vertex_indices.emplace_back(vertex_idx);
        image_points.emplace_back(landmarks[i].coordinates);
    }


    cout << model_points.size () << endl << endl;
    MatrixXf A  = MatrixXf::Zero(3*model_points.size(), 6);
    VectorXf X  (6);
    VectorXf b   (3*model_points.size ());

    /*
    |X Y Z 0 0||f1|  = |u|
    |0 0 0 Y Z||s |  = |v|
               |c1|
               |f2|
               |c2|
    */

    for (int i = 0 ; i < model_points.size (); i++) {
        //set A 
        A(i*3,0) = model_points.at(i)(0);
        A(i*3,1) = model_points.at(i)(1);
        A(i*3,2) = model_points.at(i)(2);
        A(i*3+1,3) = model_points.at(i)(1);
        A(i*3+1,4) = model_points.at(i)(2);
        A(i*3+2,5) = model_points.at(i)(2);

        // set b
        b(i*3) = image_points.at(i)(0);
        b(i*3+1) = image_points.at(i)(1);
        b(i*3+2) = 1.0f;
    }
    
    X = A.colPivHouseholderQr().solve(b);
    
    cout << "Result X " << endl;
    cout << X.transpose() << endl << endl;
    cameraMatrix = Mat::zeros(3, 3, CV_32FC1);
    cameraMatrix.at<float>(0,0) = X(0);
    cameraMatrix.at<float>(0,1) = X(1);
    cameraMatrix.at<float>(0,2) = X(2);
    cameraMatrix.at<float>(1,1) = X(3);
    cameraMatrix.at<float>(1,2) = X(4);
    cameraMatrix.at<float>(2,2) = X(5);

    cout << "manual solve 3x3 camera matrix" << endl;
    cout << cameraMatrix << endl << endl;
    
}



void project3Dto2D(Mesh current_mesh,  cv::Mat cameraMatrix ,  vector <Vector2f>& textCoor){

    cv::Vec4f model_points; // the points in the 3D shape model
 
    for (int i = 0; i < current_mesh.vertices.size(); ++i)
    {
     
        model_points[0]= current_mesh.vertices[i][0];
        model_points[1]= current_mesh.vertices[i][1];
        model_points[2]= current_mesh.vertices[i][2];
        model_points[3]= 1.0f;
    
        cv::Mat temp  =   cameraMatrix * cv::Mat(model_points);
        Vector2f temp2V; temp2V(0) = temp.at<float>(0,0);temp2V(1) = temp.at<float>(1,0);

        textCoor.push_back(temp2V);
    }  

}

void project2Dto3D(cv::Mat image,  cv::Mat cameraMatrix, vector<VectorXf>& _2DimageZ ){
  //  freopen ("2DImageZ.txt","w",stdout);
  //  cout <<"project 2D 3D" << endl;
    int image_height = image.rows;
    int image_width = image.cols;

    cv::Vec3f vec3f;
    
    
  //  cout << "inverse Matrix " << endl;
   // cout << cameraMatrix.inv() << endl <<endl;
   // cout << cameraMatrix.inv()*cameraMatrix << endl << endl;;

    for (int i =0 ; i < image_width; i++)
         for (int j =0 ; j < image_height; j++) {
     
        vec3f[0] =  i*1.0f;
        vec3f[1] =   j*1.0f ;
        vec3f[2] =   1.0f;
    
        cv::Mat temp  =   cameraMatrix.inv() * cv::Mat(vec3f);
        VectorXf temp2V(5); 
        temp2V(0)  = i*1.0f;    temp2V(1) = j*1.0f;
        temp2V(2) = temp.at<float>(0,0);    temp2V(3) = temp.at<float>(1,0);    temp2V(4) = temp.at<float>(2,0);
                
            
    //     cout << temp2V.transpose() << endl;
        _2DimageZ.push_back(temp2V);
    }  

}

void get2DimageZ(Mesh mesh,vector <VectorXf>  _2DimageZ, vector<vector <int > > mappingTriangle, vector <Vector3f>&  _2DimageRealZ ) {

    Vector3f X,Y,A,B,C, BA ,CA;
    //freopen ("CheckCross.txt","w",stdout);
    for (int i =0; i < _2DimageZ.size (); i++) {
        float u = _2DimageZ.at(i)(0);
        float v = _2DimageZ.at(i)(1);
        // z(u, v) = 1
        float x = _2DimageZ.at(i)(2);
        float y = _2DimageZ.at(i)(3);
        float z = _2DimageZ.at(i)(4);
        int idTriangle = mappingTriangle.at((int)u).at((int)v);

        if (idTriangle==-1) continue;
     //   cout <<u <<" " << v <<" " << 1.0f << " " << x << " " << y << " "<< z  << " "  << idTriangle << endl;

        // line XY
        X << u,v,1.0f; Y << x,y,z;
        Vector3f n ;n << (x - u)  , (y -v) , (z-1)  ; 
        float al = n(0), bl = n(1), cl = n(2);
    
        // Plan P (ABC)        
        A << mesh.vertices[mesh.tvi[idTriangle][0]][0] , mesh.vertices[mesh.tvi[idTriangle][0]][1] ,mesh.vertices[mesh.tvi[idTriangle][0]][2] ;
        B << mesh.vertices[mesh.tvi[idTriangle][1]][0] , mesh.vertices[mesh.tvi[idTriangle][1]][1] ,mesh.vertices[mesh.tvi[idTriangle][1]][2] ;
        C << mesh.vertices[mesh.tvi[idTriangle][2]][0] , mesh.vertices[mesh.tvi[idTriangle][2]][1] ,mesh.vertices[mesh.tvi[idTriangle][2]][2] ;

     //   cout << A.transpose() << " " << B.transpose () << " " << C.transpose () << endl;

        BA(0) =  (B(0)-A(0)); BA(1) = (B(1)-A(1)); BA(2) = (B(2)-A(2)); 
        CA(0) =  (C(0)-A(0)); CA(1) = (C(1)-A(1)); CA(2) = (C(2)-A(2)); 

        Vector3f p ; 
        p(0) = BA(1)*CA(2) - CA(1)*BA(2);
        p(1) = BA(2)*CA(0) - CA(2)*BA(0);
        p(2) = BA(0)*CA(1) - CA(0)*BA(1);

        float a= p(0), b = p(1), c = p(2); 

        float d= - ( a* A(0) + b*A(1) + c* A(2) );

        float t  = -( a*x+ b*y + c*z + d  )  / ( a*al + b*bl + c*cl );

        float crossX  = x + al*t;
        float crossY  = y + bl*t;
        float crossZ  = z + cl*t;
        Vector3f crossPoint ; 
        crossPoint << crossX,crossY,crossZ;

      //  cout << a << " " << b << " " <<  c << " " << d << endl;
       // cout << crossPoint.transpose() << endl ;
        //cout <<  ( a*crossPoint(0) + b*crossPoint(1) + c*crossPoint(2) + d ) << endl << endl;

        _2DimageRealZ.push_back(crossPoint);

    }


}

void get2DimageRealZ(Mesh mesh,vector <VectorXf>  _2DimageZ, vector<vector <int > > mappingTriangle, vector <Vector3f>&  _2DimageRealZ ) {

    Vector3f X,Y,A,B,C, BA ,CA;
    freopen ("CheckCross.txt","w",stdout);
    for (int i =0; i < _2DimageZ.size (); i++) {
        float u = _2DimageZ.at(i)(0);
        float v = _2DimageZ.at(i)(1);
        // z(u, v) = 1
        float x = _2DimageZ.at(i)(2);
        float y = _2DimageZ.at(i)(3);
        float z = _2DimageZ.at(i)(4);
        int idTriangle = mappingTriangle.at((int)u).at((int)v);

        if (idTriangle==-1) continue;
     //   cout <<u <<" " << v <<" " << 1.0f << " " << x << " " << y << " "<< z  << " "  << idTriangle << endl;

        // line XY
        X << u,v,1.0f; Y << x,y,z;
        Vector3f n ;n << (x - u)  , (y -v) , (z-1)  ; 
        float al = n(0), bl = n(1), cl = n(2);
    
        // Plan P (ABC)        
        A << mesh.vertices[mesh.tvi[idTriangle][0]][0] , mesh.vertices[mesh.tvi[idTriangle][0]][1] ,mesh.vertices[mesh.tvi[idTriangle][0]][2] ;
        B << mesh.vertices[mesh.tvi[idTriangle][1]][0] , mesh.vertices[mesh.tvi[idTriangle][1]][1] ,mesh.vertices[mesh.tvi[idTriangle][1]][2] ;
        C << mesh.vertices[mesh.tvi[idTriangle][2]][0] , mesh.vertices[mesh.tvi[idTriangle][2]][1] ,mesh.vertices[mesh.tvi[idTriangle][2]][2] ;

     //   cout << A.transpose() << " " << B.transpose () << " " << C.transpose () << endl;

        BA(0) =  (B(0)-A(0)); BA(1) = (B(1)-A(1)); BA(2) = (B(2)-A(2)); 
        CA(0) =  (C(0)-A(0)); CA(1) = (C(1)-A(1)); CA(2) = (C(2)-A(2)); 

        Vector3f p ; 
        p(0) = BA(1)*CA(2) - CA(1)*BA(2);
        p(1) = BA(2)*CA(0) - CA(2)*BA(0);
        p(2) = BA(0)*CA(1) - CA(0)*BA(1);

        float a= p(0), b = p(1), c = p(2); 

        float d= - ( a* A(0) + b*A(1) + c* A(2) );

        float t  = -( a*x+ b*y + c*z + d  )  / ( a*al + b*bl + c*cl );

        float crossX  = x + al*t;
        float crossY  = y + bl*t;
        float crossZ  = z + cl*t;
        Vector3f crossPoint ; 
        crossPoint << crossX,crossY,crossZ;

        cout << a << " " << b << " " <<  c << " " << d << endl;
        cout << crossPoint.transpose() << endl ;
        cout <<  ( a*crossPoint(0) + b*crossPoint(1) + c*crossPoint(2) + d ) << endl << endl;

        _2DimageRealZ.push_back(crossPoint);

    }


}

float getIntensity(int r, int g, int b) {
    return (0.2126*r + 0.7152*g + 0.0722*b )/256;
}

void getRGB(vector <Vector2f> textCoor ,cv::Mat image, vector <Vector3d>& textColor, vector <int>& intent  ){
    const int imgw = image.cols;
    const int imgh = image.rows;
 
    uint8_t r,g,b,grey;
   Vector3d tempVec3d;

     
    for (int i =0; i < textCoor.size (); i++) {
        b=image.at<cv::Vec3b>(textCoor.at(i)(1),textCoor.at(i)(0))[0];//b
        g=image.at<cv::Vec3b>(textCoor.at(i)(1),textCoor.at(i)(0))[1];//g
        r=image.at<cv::Vec3b>(textCoor.at(i)(1),textCoor.at(i)(0))[2];//r

        
        grey =  (int) r* 0.21 + (int)g * 0.72 + int(b)*0.07;
        grey =(int) image.at<uchar>(textCoor.at(i)(1),textCoor.at(i)(0));//b
        intent.push_back((int) grey);

        tempVec3d << (int) r, (int)g, (int) b;
        
        textColor.push_back(tempVec3d);

    }
    

}

void getRGB (Mesh current_mesh,cv::Mat image, const core::LandmarkCollection<Eigen::Vector2f>& landmarks, const core::LandmarkMapper& landmark_mapper, vector <Vector3d>& textColor, vector <int>& intent,   vector <Vector2f>& textCoor,vector<VectorXf>& _2DimageZ  ){
    
    cv::Mat cameraMatrix3D2D = Mat::zeros(3, 4, CV_32FC1);
    cv::Mat cameraMatrix2D3D = Mat::zeros(3,4, CV_32FC1);;
    getCameraMatrix(image, landmarks, landmark_mapper, current_mesh, cameraMatrix3D2D,cameraMatrix2D3D );
 
    project3Dto2D (current_mesh,cameraMatrix3D2D,textCoor);

    project2Dto3D (image,cameraMatrix3D2D,_2DimageZ);


    
    getRGB(textCoor,image,textColor,intent );
   
}

void write2DImangeZIntensity(vector <vector <float> > depthMap,  cv::Mat image,vector<vector< int > > mapping2D3D, Mesh mesh) {
    const int imgw = image.cols;
    const int imgh = image.rows;
 
    uint8_t r,g,b,grey;
     freopen("2DImageZIntensity.txt","w",stdout);
     int count = 0;
    for (int i =0; i < imgw; i++) 
        for (int  j =0 ; j < imgh; j++){
        b=image.at<cv::Vec3b>(j,i)[0];//R
        g=image.at<cv::Vec3b>(j,i)[1];//B
        r=image.at<cv::Vec3b>(j,i)[2];//G


        if ( depthMap[i][j]!=-9999  ) {
                cout << (i) <<" " << (j) << " " << (depthMap[i][j]) <<" " << getIntensity((int)r,(int)g,(int)b)  <<   endl ;
                count++;
            }
          
    }  

     //verify Result mapiing
     freopen("verifyMapping2D3D.off","w",stdout);
     cout << "COFF" << endl;
     cout << count  << " 0 0" << endl;
    for (int i =0; i < imgw; i++) 
        for (int  j =0 ; j < imgh; j++){
        b=image.at<cv::Vec3b>(j,i)[0];//R
        g=image.at<cv::Vec3b>(j,i)[1];//B
        r=image.at<cv::Vec3b>(j,i)[2];//G
        int index = mapping2D3D[i][j];


        if (index==-1) continue;
        cout << mesh.vertices[index](0) <<" " <<  mesh.vertices[index](1) << " " <<  mesh.vertices[index](2) <<" " << (int) r << " " << (int) g << " " << (int) b << " 1" <<   endl ;
          
    }  


}

void write3DTo2DMapping(vector <Vector2f>& textCoor, Mesh mesh, cv::Mat image) {
   
     freopen("3DTo2DMapping.txt","w",stdout);
    
    for (int i=0; i < textCoor.size (); i++) {
        cout << ((int)textCoor.at(i)(0) ) << " " << ((int)textCoor.at(i)(1) ) << " " << mesh.eigeinValue.at(i)<< endl;

    }

    //verify Result
    uint8_t r,g,b,grey;
    freopen ("verifyMapping3D2D.off","w",stdout);
    cout << "COFF" << endl;
    cout << textCoor.size () << " 0 0 \n" ;
    for (int k=0; k < textCoor.size (); k++) {

        int i =  ((int)textCoor.at(k)(0) );
        int j =  ((int)textCoor.at(k)(1) );

        b=image.at<cv::Vec3b>(j,i)[0];//R
        g=image.at<cv::Vec3b>(j,i)[1];//B
        r=image.at<cv::Vec3b>(j,i)[2];//G

        cout << mesh.vertices.at(k)(0) << " " << mesh.vertices.at(k)(1) << " " << mesh.vertices.at(k)(2) << " " <<(int) r <<" " <<(int) g <<" " << (int) b << " 1" << endl;

    }

}

void getEdgeFromMesh ( Mesh& mesh  ) {
tempVec2d << -1, -1;
 
for (int i =0; i < mesh.vertices.size (); i++) {    
    vector <int> rowInt;
    mesh.edge.push_back(rowInt);
      
}
cout << "done init edge\n";

int count = 0;
    for (auto& triangle: mesh.tvi) {
        int i1 = triangle[0];
        int i2 = triangle[1];
        int i3 = triangle[2];

        
        bool checka =  false,checkb = false;
        for (int i =0; i < mesh.edge.at(i1).size ();i++) {
            if (mesh.edge.at(i1)[i] ==i2 ) checka = true;
            if (mesh.edge.at(i1)[i] ==i3 ) checkb = true;
        }
        if (!checka) mesh.edge.at(i1).push_back(i2);
        if (!checkb) mesh.edge.at(i1).push_back(i3);
 

      
 
        checka =  false;checkb = false;
        for (int i =0; i < mesh.edge.at(i2).size ();i++) {
            if (mesh.edge.at(i2)[i] ==i1 ) checka = true;
            if (mesh.edge.at(i2)[i] ==i3 ) checkb = true;
        }

        if (!checka) mesh.edge.at(i2).push_back(i1);
        if (!checkb) mesh.edge.at(i2).push_back(i3);
         

        checka =  false;checkb = false;
        for (int i =0; i < mesh.edge.at(i3).size ();i++) {
            if (mesh.edge.at(i3)[i] ==i1 ) checka = true;
            if (mesh.edge.at(i3)[i] ==i2 ) checkb = true;
        }

        if (!checka) mesh.edge.at(i3).push_back(i1);
        if (!checkb) mesh.edge.at(i3).push_back(i2);
       
    }
   

}

void canculateNormalVector (Mesh& mesh ) {

 getEdgeFromMesh(mesh );
 cout << "done get edge\n";
 MatrixXf A;
 MatrixXf ATA;
 VectorXf singularValues;
 MatrixXf U;
int numOfPoint = mesh.vertices.size ();
// calculate normalvector of each vertex
    for (int i =0; i < numOfPoint; i ++) {

      A = MatrixXf::Zero( mesh.edge.at(i).size() , 3);
        
        for (int j = 0; j < mesh.edge.at(i).size (); j++) {
            RowVector3f rowVec ;
            rowVec = RowVectorXf( mesh.vertices[mesh.edge.at(i).at(j)] - mesh.vertices[i]  );
            A.row(j) = rowVec;
        }
        
        ATA = A.transpose()*A;
        // ATA = (U*S)*V
        JacobiSVD<MatrixXf> svd(ATA, ComputeThinU | ComputeThinV);
        
        singularValues =  svd.singularValues() ;
        U = svd.matrixU();
        mesh.normalVector.push_back(VectorXf(U.col(2)));
        mesh.eigeinValue.push_back(fabs(singularValues(1)));
        
    }
cout << "done calculate noralVec\n";
   
}


void getConstanceL( double& l0, double& l1, double& l2, double& l3) {

    l0 = sqrt ( 1.0f /(4*M_PI) );
    l1 = sqrt( ( 3.0f) /(4*M_PI) );
    l2 = sqrt(( 3.0f) /(4*M_PI) );
    l3 = sqrt( (3.0f) /(4*M_PI) );
}
 

void writeParameterOptimization ( Mesh mesh) {
    
    double l0,l1,l2,l3,A,B,_b, N;
    int r,g,b,grey;
    freopen ("parameters.txt","w",stdout);
    getConstanceL(l0,l1,l2,l3);
 
    for (int i =0; i < mesh.vertices.size (); i++) {

        r = mesh.colors.at(i)(0);
        g = mesh.colors.at(i)(1);
        b = mesh.colors.at(i)(2);

        N = mesh.eigeinValue.at(i);
 

        if (N==0) continue;
        if (mesh.edge.at(i).size()<2) continue;

        cout << i << " " << l0 << " " << l1 << " " << l2 << " " <<l3 <<" " << " " << N << "  " << mesh.edge.at(i).size()  << endl; 
        cout << i << " " << mesh.normalVector.at(i)(0) << " " << mesh.normalVector.at(i)(1) << " " << mesh.normalVector.at(i)(2)  << endl;
        int U1V = mesh.edge.at(i).at(0);
        int UV1 = mesh.edge.at(i).at(1);

        if (U1V==-1 || UV1==-1) continue;

        float zU1V = mesh.vertices[U1V](2);
        float zUV1 = mesh.vertices[UV1](2);

        double I = getIntensity((int) r, (int )g, (int) b);

        A = - (l1+l2)/N;
        B = - ( I + l0 - ( l3-(l1*zU1V + l2*zUV1))/N  );
        _b = mesh.vertices.at(i)(2);

        cout << i << " " << A << " " << B << " " << _b  << endl;
    }

}

int main(int argc, char* argv[])
{
    string modelfile, isomapfile, imagefile, landmarksfile, mappingsfile, contourfile, edgetopologyfile,
        blendshapesfile, outputbasename;
    try
    {
        po::options_description desc("Allowed options");
        // clang-format off
        desc.add_options()
            ("help,h", "display the help message")
            ("model,m", po::value<string>(&modelfile)->required()->default_value("share/sfm_shape_3448.bin"),
                "a Morphable Model stored as cereal BinaryArchive")
            ("image,i", po::value<string>(&imagefile)->required()->default_value("data/image_0010.png"),
                "an input image")
            ("landmarks,l", po::value<string>(&landmarksfile)->required()->default_value("data/image_0010.pts"),
                "2D landmarks for the image, in ibug .pts format")
            ("mapping,p", po::value<string>(&mappingsfile)->required()->default_value("share/ibug_to_sfm.txt"),
                "landmark identifier to model vertex number mapping")
            ("model-contour,c", po::value<string>(&contourfile)->required()->default_value("share/sfm_model_contours.json"),
                "file with model contour indices")
            ("edge-topology,e", po::value<string>(&edgetopologyfile)->required()->default_value("share/sfm_3448_edge_topology.json"),
                "file with model's precomputed edge topology")
            ("blendshapes,b", po::value<string>(&blendshapesfile)->required()->default_value("share/expression_blendshapes_3448.bin"),
                "file with blendshapes")
            ("output,o", po::value<string>(&outputbasename)->required()->default_value("out"),
                "basename for the output rendering and obj files");
        // clang-format on
        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
        if (vm.count("help"))
        {
            cout << "Usage: fit-model [options]" << endl;
            cout << desc;
            return EXIT_SUCCESS;
        }
        po::notify(vm);
    } catch (const po::error& e)
    {
        cout << "Error while parsing command-line arguments: " << e.what() << endl;
        cout << "Use --help to display a list of options." << endl;
        return EXIT_FAILURE;
    }

    // Load the image, landmarks, LandmarkMapper and the Morphable Model:
    Mat image = cv::imread(imagefile);
  
  //  cv::resize(image, image, cv::Size(), 0.5, 0.5);

    LandmarkCollection<Eigen::Vector2f> landmarks;
    try
    {
        landmarks = core::read_pts_landmarks(landmarksfile);
    } catch (const std::runtime_error& e)
    {
        cout << "Error reading the landmarks: " << e.what() << endl;
        return EXIT_FAILURE;
    }
    morphablemodel::MorphableModel morphable_model;
    try
    {
        morphable_model = morphablemodel::load_model(modelfile);
    } catch (const std::runtime_error& e)
    {
        cout << "Error loading the Morphable Model: " << e.what() << endl;
        return EXIT_FAILURE;
    }
    // The landmark mapper is used to map 2D landmark points (e.g. from the ibug scheme) to vertex ids:
    core::LandmarkMapper landmark_mapper;
    try
    {
        landmark_mapper = core::LandmarkMapper(mappingsfile);
    } catch (const std::exception& e)
    {
        cout << "Error loading the landmark mappings: " << e.what() << endl;
        return EXIT_FAILURE;
    }

    // The expression blendshapes:
    const vector<morphablemodel::Blendshape> blendshapes = morphablemodel::load_blendshapes(blendshapesfile);

    // These two are used to fit the front-facing contour to the ibug contour landmarks:
    const fitting::ModelContour model_contour =
        contourfile.empty() ? fitting::ModelContour() : fitting::ModelContour::load(contourfile);
    const fitting::ContourLandmarks ibug_contour = fitting::ContourLandmarks::load(mappingsfile);

    // The edge topology is used to speed up computation of the occluding face contour fitting:
    const morphablemodel::EdgeTopology edge_topology = morphablemodel::load_edge_topology(edgetopologyfile);

    // Draw the loaded landmarks:
    Mat outimg = image.clone();
    for (auto&& lm : landmarks)
    {
        cv::rectangle(outimg, cv::Point2f(lm.coordinates[0] - 2.0f, lm.coordinates[1] - 2.0f),
                      cv::Point2f(lm.coordinates[0] + 2.0f, lm.coordinates[1] + 2.0f), {255, 0, 0});
    }

    // Fit the model, get back a mesh and the pose:
    core::Mesh mesh;
    fitting::RenderingParameters rendering_params;
     vector<Eigen::Vector4f> model_points; // the points in the 3D shape model
    vector<int> vertex_indices; // their vertex indices
    vector<Eigen::Vector2f> image_points; // the corresponding 2D landmark points

    std::tie(mesh, rendering_params) = fitting::fit_shape_and_pose(
        morphable_model, blendshapes, landmarks, landmark_mapper, image.cols, image.rows, edge_topology,
        ibug_contour, model_contour,model_points,vertex_indices,image_points, 5, std::nullopt, 30.0f);

    // The 3D head pose can be recovered as follows:
    float yaw_angle = glm::degrees(glm::yaw(rendering_params.get_rotation()));
    // and similarly for pitch and roll.

    // Extract the texture from the image using given mesh and camera parameters:
    const Eigen::Matrix<float, 3, 4> affine_from_ortho =
        fitting::get_3x4_affine_camera_matrix(rendering_params, image.cols, image.rows);
    const core::Image4u isomap =
        render::extract_texture(mesh, affine_from_ortho, core::from_mat(image), true);

    // Draw the fitted mesh as wireframe, and save the image:
    render::draw_wireframe(outimg, mesh, rendering_params.get_modelview(), rendering_params.get_projection(),
                         fitting::get_opencv_viewport(image.cols, image.rows));
    
      cout <<"Matrix 3x4" << endl;
    cout << affine_from_ortho << endl;
   
    /*
    cout << "XXX" << endl;
      int delta = 5;

    cout << image_points.size () << endl;

    for (int i =0; i < image_points.size() ; i ++) {

        int cx = image_points.at(i)(0);
        int cy = image_points.at(i)(1);

         for (int  x = cx-delta; x>=0 && x < cx+ delta && x < image.cols ; x++) 
            for (int y = cy-delta; y>=0 &&  y < cy+delta && y < image.rows  ; y++)
                {
                 
                image.at<cv::Vec3b>(y,x)[0] = 0;//b
                image.at<cv::Vec3b>(y,x)[1] = 0;//g
                image.at<cv::Vec3b>(y,x)[2] = 255;//r

                }
   
    }

    mesh =  morphable_model.get_mean();

    int currentLandMarkId =0 ;
    vector<Vector3d>  clor ; 
    tempVec3d << 255,255,255;
    for (int i=0; i < mesh.vertices.size(); i++ )
        clor.push_back(tempVec3d);

    for (int i=0; i < vertex_indices.size(); i++ )
        clor.at(vertex_indices.at(i)) << 255,0,0;

    cout << "XXX" << endl;

    freopen ("meanfaceLandmark.off", "w",stdout);
    cout <<"COFF" << endl;
    cout << mesh.vertices.size() << " " << mesh.tvi.size () << " 0" << endl; 
    for (int i =0 ; i < mesh.vertices.size(); i ++) {
         
        cout<< mesh.vertices.at(i).transpose() << " " << clor.at(i).transpose() << " 1" << endl;
    }

     for (int i =0 ; i < mesh.tvi.size(); i ++)
        cout << 3 << " "<< mesh.tvi.at(i)[0]<< " "<< mesh.tvi.at(i)[1]<< " "<< mesh.tvi.at(i)[2] << endl;

    cv::imwrite("imageMarked.jpg", image);
    */
    


    const int imgw = image.cols;
    const int imgh = image.rows;
    uint8_t r,g,b;  
   

    cout <<"image size: " << image.cols << " x " << image.rows <<endl; 

    vector <vector <int> > mappingTriangle;
   
     for (int i =0; i < imgw; i++){
        vector <int> row;
        for (int  j =0 ; j < imgh; j++){
                 row.push_back(-1);
         //           mapping[i][j] = -1;
        }
        mappingTriangle.push_back(row);
    }


    getMapping2DTriangle(image, mesh,rendering_params.get_modelview(), rendering_params.get_projection(),
                        fitting::get_opencv_viewport(image.cols, image.rows),
                        mappingTriangle);

    cv::Mat cameraMatrix;
    vector<VectorXf> _2DimageZ ;
    getCameraMatrix(  mesh,  landmarks,   landmark_mapper, cameraMatrix  ) ;
    project2Dto3D(image,  cameraMatrix,  _2DimageZ );
    vector<Vector3f> _2DimageRealZ ;
    get2DimageZ(mesh,_2DimageZ,mappingTriangle,_2DimageRealZ );

    Vector3f A,B,C,D;
    A << _2DimageZ.at(0)(0),_2DimageZ.at(0)(1), (1.0f);
    C << _2DimageZ.at(0)(2),_2DimageZ.at(0)(3),_2DimageZ.at(0)(4);
    B << _2DimageZ.at(imgw-1)(0),_2DimageZ.at(imgw-1)(1), (1.0f);
    D << _2DimageZ.at(imgw-1)(2),_2DimageZ.at(imgw-1)(3),_2DimageZ.at(imgw-1)(4);


    float scale = getLen(A,B) / getLen(C,D);
    cout << "scale: " << scale << endl;
    
    /*
    // WRITE 2D IMAGE & IMAGE PLAN
    freopen ("_2DimageRealZ.off","w",stdout);
    cout << "COFF\n";
    cout << (_2DimageZ.size () + mesh.vertices.size () ) << " 0 0" << endl;
    
    for (int i =0; i < _2DimageZ.size (); i ++ ){
        int u = _2DimageZ.at(i)(0);
        int v = _2DimageZ.at(i)(1);
        b=imageOriginal.at<cv::Vec3b>(v,u)[0];//R
        g=imageOriginal.at<cv::Vec3b>(v,u)[1];//B
        r=imageOriginal.at<cv::Vec3b>(v,u)[2];//G
   //     cout << _2DimageZ.at(i)(0) << " "  << _2DimageZ.at(i)(1) << " "  << " 1 " << (int)r << " " << (int)g   << " " << (int) b << " 1"<< endl;
       cout << _2DimageZ.at(i)(2) << " "  << _2DimageZ.at(i)(3) << " "  << _2DimageZ.at(i)(4)  <<" "<< (int)r << " " << (int)g   << " " << (int) b << " 1"<< endl;
    } 

    for (int i=0 ; i < mesh.vertices.size () ; i ++)
        cout << mesh.vertices.at (i)( 0) << " "<< mesh.vertices.at (i)( 1) << " "<< mesh.vertices.at (i)(2) << " " << " 200 200 200 1" << endl;
    */

    vector <vector <float> > depthMap;
    vector <vector <int> > indexMap;
    vector <Vector2f> textCoor ;
    vector <vector <int> > mapping2D3D;
    vector <vector <double> > currentLen;
    tempVec2d << -1,-1;
    core::Mesh reconstructedMesh;
   
     for (int i =0; i < imgw; i++){
        vector <float> row;
        vector <double> rowDouble ;
        vector <int> rowInt ;
        for (int  j =0 ; j < imgh; j++){
                 row.push_back(-9999.0f);
                 rowDouble.push_back(9999);
                 rowInt.push_back(-1);
        }
        depthMap.push_back(row);
        currentLen.push_back(rowDouble);
        mapping2D3D.push_back(rowInt);
        indexMap.push_back(rowInt);

    }

    // init textCoor
   tempVec2f << (-1.0f),(-1.0f);
   for (int i =0; i < mesh.vertices.size (); i++ ) {
        textCoor.push_back (tempVec2f);
        mesh.neibour.push_back(tempVec2d);
    }

    int count = 0;
    
     
    // get depth & get mapping 3D to 2D
    render::add_depth_information( mesh, rendering_params.get_modelview(), rendering_params.get_projection(),
                           fitting::get_opencv_viewport(image.cols, image.rows),depthMap,textCoor,  8.0f );//2* (int) scale);
    // get mapping 2D to 3D index
    render::getMapping2D3DBy2D(outimg,mesh, rendering_params.get_modelview(), rendering_params.get_projection(),
                           fitting::get_opencv_viewport(image.cols, image.rows), mapping2D3D,currentLen );
    
     
    
   
    int _index = 0 ;
     int _scale  = 5;

    // CREATE INDEX
    for (int i =_scale; i < imgw; i+=_scale) {
        for (int  j =_scale ; j < imgh; j+=_scale){
        

        if ( depthMap[i][j]!=-9999  )
        {      
               tempVec3f << i,j,depthMap[i][j];
               reconstructedMesh.vertices.push_back(tempVec3f);
               indexMap[i][j] = _index++;
               count++;
               reconstructedMesh.neibour.push_back(tempVec2d);

            if ( indexMap[i][j-_scale] !=-1 && indexMap [i-_scale][j-_scale]!=-1 ) {
            std::array<int, 3> tempArray = {  indexMap[i][j], indexMap[i][j-_scale],indexMap [i-_scale][j-_scale]  };
            reconstructedMesh.tvi.push_back(tempArray);
            reconstructedMesh.neibour.at(indexMap [i-_scale][j-_scale] )(0) = indexMap[i][j-_scale];
            }

            if ( indexMap[i-_scale][j-_scale] !=-1 && indexMap [i-_scale][j]!=-1 ) {
             std::array<int, 3> tempArray = {   indexMap [i-_scale][j],indexMap[i][j], indexMap[i-_scale][j-_scale]};
            reconstructedMesh.tvi.push_back(tempArray);
            reconstructedMesh.neibour.at(indexMap [i-_scale][j-_scale] )(1) = indexMap [i-_scale][j];
           
            }

        } // if 99999
        } // 2nd for    
    } //1st for

  
    freopen ("depthmap.off","w",stdout);
    cout << "COFF\n";
    cout << (reconstructedMesh.vertices.size()) << " " << reconstructedMesh.tvi.size() << " 0" << endl;

    for (int k =0; k < reconstructedMesh.vertices.size(); k++) {
                int i = reconstructedMesh.vertices.at(k)(0);
                int j = reconstructedMesh.vertices.at(k)(1); 
                float z = reconstructedMesh.vertices.at(k)(2);      
                b=image.at<cv::Vec3b>(j,i)[0];//R
                g=image.at<cv::Vec3b>(j,i)[1];//B
                r=image.at<cv::Vec3b>(j,i)[2];//G
                tempVec3f << i ,j, depthMap[i][j];
                tempVec3f << (int)r , (int)g, (int)b;
                cout << (i) <<" " << (j) << " " << (z) <<  " "  << (int) r << " "  << (int) g << " " << (int) b << " 1"   <<endl ;
             
    }

    for (int i =0; i< reconstructedMesh.tvi.size(); i++) {
        cout << 3 << " " << reconstructedMesh.tvi.at(i)[0] << " " << reconstructedMesh.tvi.at(i)[1]    << " " << reconstructedMesh.tvi.at(i)[2] << endl;
    }
    count = 0;
    for (int i =0; i< reconstructedMesh.neibour.size(); i++) {
        if (reconstructedMesh.neibour.at(i)(0) !=-1 && reconstructedMesh.neibour.at(i)(1) !=-1  ) {
            int id1 =i; int id2 =reconstructedMesh.neibour.at(i)(0); int id3= reconstructedMesh.neibour.at(i)(1);
            Vector3f u = reconstructedMesh.vertices.at(id2) -  reconstructedMesh.vertices.at(id1);
            Vector3f v = reconstructedMesh.vertices.at(id3) -  reconstructedMesh.vertices.at(id1);
            cout << ((u.cross(v).transpose())/u.cross(v).norm()) << " " << ((u.cross(v).transpose())/u.cross(v).norm()).norm ()<< endl;
            count ++;
        }
    }

    cout << count << endl;
    cout << reconstructedMesh.tvi.size() << endl;
    cout <<   reconstructedMesh.neibour.size() << endl;
        
    // write parameter out
    // I, l0,l1,l2,l3,z1,z2,z3,id1,id2,id3,D
    freopen ("parameter.txt","w",stdout);
    double l0,l1,l2,l3;
    getConstanceL(l0,l1,l2,l3);
    cout << reconstructedMesh.vertices.size()  << endl;
    cout << _scale << endl;
    for (int k =0; k <reconstructedMesh.vertices.size(); k++ ) {
        int id1 =k; int id2 =reconstructedMesh.neibour.at(k)(0); int id3= reconstructedMesh.neibour.at(k)(1);
        if (id2!=-1 && id3!=-1) {
        int i = reconstructedMesh.vertices.at(k)(0);
        int j = reconstructedMesh.vertices.at(k)(1); 
           
        b=image.at<cv::Vec3b>(j,i)[0];//R
        g=image.at<cv::Vec3b>(j,i)[1];//B
        r=image.at<cv::Vec3b>(j,i)[2];//G

        float I  = getIntensity(r,g,b);
        float z1 = reconstructedMesh.vertices.at(id1)(2);   
        float z2 =  reconstructedMesh.vertices.at(id2)(2);   
        float z3 =  reconstructedMesh.vertices.at(id3)(2); 
        Vector3f u = reconstructedMesh.vertices.at(id2) -  reconstructedMesh.vertices.at(id1);
        Vector3f v = reconstructedMesh.vertices.at(id3) -  reconstructedMesh.vertices.at(id1);
        float D = (u.cross(v)).norm();

        cout << I <<" " << z1 <<" " << z2 <<" " << z3 <<" " << id1  <<" "  << id2  <<" " << id3 <<" "  << D << endl;  
        }
        else
        cout << -99999 << endl;
    }


   // return 0;
   
    fs::path outputfile = outputbasename + ".png";
    cv::imwrite(outputfile.string(), outimg);
    cv::imwrite("imageMarked.jpg", image);

    // Save the mesh as textured obj:
    outputfile.replace_extension(".obj");
    core::write_textured_obj(mesh, outputfile.string());

    // And save the isomap:
    outputfile.replace_extension(".isomap.png");
    cv::imwrite(outputfile.string(), core::to_mat(isomap));
   

 //   cout << "Finished fitting and wrote result mesh and isomap to files with basename "
  //  cout<< outputfile.stem().stem() << "." << endl;
  
    return EXIT_SUCCESS;
}
