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

void project2Dto3D(cv::Mat image,  cv::Mat cameraMatrix, vector<Vector3f>& _2DimageZ ){
    freopen ("2DImageZ.txt","w",stdout);
    cout <<"project 2D 3D" << endl;
    int image_height = image.rows;
    int image_width = image.cols;

    cv::Vec3f vec3f;

     cv::Mat RMatrix =  Mat::zeros(3, 3, CV_32FC1);
     Vector2f translation;

    for (int i=0; i <2; i++)
        translation(i) = cameraMatrix.at<float>(i,3);

    cout << "center" << endl;
   

    for (int i =0; i <3; i++)
        for (int j=0; j<3; j++)
            RMatrix.at<float>(i,j) = cameraMatrix.at<float>(i,j);
    
    
    cout << "inverse Matrix " << endl;
    cout << RMatrix.inv() << endl;

    for (int i =0 ; i < image_width; i++)
         for (int j =0 ; j < image_height; j++) {
     
        vec3f[0] =  i;
        vec3f[1] =   j ;
        vec3f[2] =   1.0;
    
        cv::Mat temp  =   RMatrix.inv() * cv::Mat(vec3f);
        Vector3f temp2V; temp2V(0) = temp.at<float>(0,0);temp2V(1) = temp.at<float>(1,0);temp2V(2) = temp.at<float>(2,0);
            
        if (fabs(temp2V(0)) < 0.05 || fabs(temp2V(1)) < 0.05 || fabs(temp2V(2)) < 0.05)
            continue;

        cout << temp2V.transpose() << endl;
        _2DimageZ.push_back(temp2V);
    }  

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

void getRGB (Mesh current_mesh,cv::Mat image, const core::LandmarkCollection<Eigen::Vector2f>& landmarks, const core::LandmarkMapper& landmark_mapper, vector <Vector3d>& textColor, vector <int>& intent,   vector <Vector2f>& textCoor,vector<Vector3f>& _2DimageZ  ){
    
    cv::Mat cameraMatrix3D2D = Mat::zeros(3, 4, CV_32FC1);
    cv::Mat cameraMatrix2D3D = Mat::zeros(3,4, CV_32FC1);;
    getCameraMatrix(image, landmarks, landmark_mapper, current_mesh, cameraMatrix3D2D,cameraMatrix2D3D );
 
    project3Dto2D (current_mesh,cameraMatrix3D2D,textCoor);

    project2Dto3D (image,cameraMatrix3D2D,_2DimageZ);


    
    getRGB(textCoor,image,textColor,intent );
   
}
 
void getEdgeFromMesh ( Mesh mesh, vector <vector <int>>& edge ) {
    
vector <vector <bool>> check;

for (int i =0; i < mesh.vertices.size (); i++) {
    vector <bool > row;
    vector <int> rowInt;
    edge.push_back(rowInt);
    for (int j =0; j < mesh.vertices.size (); j++)
        row.push_back(false);

    check.push_back(row);
}

    for (auto& triangle: mesh.tvi) {
        int i1 = triangle[0];
        int i2 = triangle[1];
        int i3 = triangle[2];

        if (!check[i1][i2]) check[i1][i2] = true;
        if (!check[i1][i3]) check[i1][i3] = true; 
        if (!check[i2][i1]) check[i2][i1] = true; 
        if (!check[i2][i3]) check[i2][i3] = true; 
        if (!check[i3][i1]) check[i3][i1] = true; 
        if (!check[i3][i2]) check[i3][i2] = true;
    }

    for (int i =0; i < mesh.vertices.size (); i++)
        for (int j =0; j < mesh.vertices.size (); j++)
         if ( check[i][j]  ) 
            edge.at(i).push_back(j);

}

void canculateNormalVector (Mesh& mesh,vector <float>& normalValue) {

    vector <vector <int> > edge;
    getEdgeFromMesh(mesh, edge);

int numOfPoint = mesh.vertices.size ();
// calculate normalvector of each vertex
    for (int i =0; i < numOfPoint; i ++) {
        MatrixXf A( edge.at(i).size() , 3);
        
        for (int j = 0; j < edge.at(i).size (); j++) {
            RowVectorXf rowVec(3);
            rowVec = RowVectorXf( mesh.vertices[edge.at(i).at(j)] - mesh.vertices[i]  );
            A.row(j) = rowVec;
        }
        
        MatrixXf ATA = A.transpose()*A;
        // ATA = (U*S)*V
        JacobiSVD<MatrixXf> svd(ATA, ComputeThinU | ComputeThinV);
        
        VectorXf singularValues =  svd.singularValues() ;
        MatrixXf U = svd.matrixU();
        mesh.normalVector.push_back(VectorXf(U.col(2)));
        normalValue.push_back(singularValues(2));
        
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
    std::tie(mesh, rendering_params) = fitting::fit_shape_and_pose(
        morphable_model, blendshapes, landmarks, landmark_mapper, image.cols, image.rows, edge_topology,
        ibug_contour, model_contour, 5, std::nullopt, 30.0f);

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
    

    vector <Vector3f > normalVector;
    vector <float> normalValue;
     ;
    canculateNormalVector(mesh,normalValue);
    vector <Vector3f> _2DimageZ;


    cv::Mat RMatrix =  Mat::zeros(3, 3, CV_32FC1);

    getCameraIntinsion(mesh,landmarks, landmark_mapper,RMatrix);
   

    project2Dto3D(image,RMatrix,_2DimageZ);
    uint8_t r,g,b,grey;
    
    freopen ("_2DimageZ.off","w",stdout);
    for (int k=0; k < _2DimageZ.size (); k++) {
        int i= (int) _2DimageZ.at(k)(0);
        int j= (int) _2DimageZ.at(k)(1);
            b=image.at<cv::Vec3b>(j,i)[0];//b
            g=image.at<cv::Vec3b>(j,i)[1];//g
            r=image.at<cv::Vec3b>(j,i)[2];//r
            cout << i << " " << j << " " << _2DimageZ.at(k)(2) << " " << (int)r<< " " << (int)g<< " " << (int)b<< " 1" << endl;
    }

    return 0;
    /*    
    const int imgw = image.cols;
    const int imgh = image.rows;
    cout <<"image size: " << image.cols << " x " << image.rows <<endl; 
    vector <vector <int> > depthMap;
   
     for (int i =0; i < imgw; i++){
        vector <int> row;
        for (int  j =0 ; j < imgh; j++){
                 row.push_back(-9999);
         //           mapping[i][j] = -1;
        }
        depthMap.push_back(row);
    }

    int count = 0;
    int bounder [6]; // maxX minX maxY minY maxZ minZ;

    bounder[0] = -19999;
    bounder[1] = 9999;
    bounder[2] = -19999;
    bounder[3] =-19999;
    bounder[4] = -19999;
    bounder[5] = 9999;

    render::add_depth_information(outimg, mesh, rendering_params.get_modelview(), rendering_params.get_projection(),
                           fitting::get_opencv_viewport(image.cols, image.rows),depthMap,bounder);
      uint8_t r,g,b;
     
  
    Mat imageOriginal = cv::imread(imagefile); 
   
    freopen ("depthmap.off","w",stdout);
    cout << "COFF\n";
     cout << (imgw*imgh) << " 0 0" << endl;
    for (int i =0; i < imgw; i++) {
        for (int  j =0 ; j < imgh; j++){
        b=imageOriginal.at<cv::Vec3b>(j,i)[0];//R
        g=imageOriginal.at<cv::Vec3b>(j,i)[1];//B
        r=imageOriginal.at<cv::Vec3b>(j,i)[2];//G


        if ( depthMap[i][j]!=-9999 &&  depthMap[i][j] <= bounder[4] &&   depthMap[i][j] >= bounder[5] )
        {         
                cout << (i) <<" " << (imgh-j) << " " << -(depthMap[i][j]) <<  " "  << (int) r << " "  << (int) g << " " << (int) b << " 1"   <<endl ;
                count ++;
        }
        } // 2nd for    
    } //1st for

    cout << count << endl;
    

    int image_height = image.rows;
    int image_width = image.cols;

    vector <vector <int>> mapping;
    vector <vector <double >> currentLen;



    for (int i =0; i < imgw; i++) {
        vector <int > row; 
        vector <double > row1; 
        for (int  j =0 ; j < imgh; j++){
               row.push_back(-1);
               row1.push_back(9999.0f);
        }
        mapping.push_back (row);
        currentLen.push_back(row1);

    }

    Mesh avgMesh = mesh;
    vector <int> intent ;
    vector <Vector3d> textColor ;
    vector <Vector2f> textCoor ;
    vector <Vector3f> _2DimageZ;
    getRGB(avgMesh,image,landmarks, landmark_mapper,textColor,intent,textCoor,_2DimageZ);
    
    render::getMapping2D3D(outimg, mesh, rendering_params.get_modelview(), rendering_params.get_projection(),
                           fitting::get_opencv_viewport(image.cols, image.rows),mapping,currentLen);
   
    freopen ("grey_image.off","w",stdout);
    cout << "COFF\n";
    cout << intent.size () << " 0 0" << endl;
  

    for (int i = 0; i < mesh.vertices.size(); ++i)
    {
     cout << avgMesh.vertices[i][0] <<" " <<avgMesh.vertices[i][1] <<" " <<avgMesh.vertices[i][2] <<" " << textColor.at(i)(0) << " " << textColor.at(i)(1) << " " << textColor.at(i)(2) << " 1"<< endl;

    }


    freopen ("Mapping2D3D_070.txt","w",stdout);
    count = 0;
    cout << imgw << " " << imgh << endl;
     for (int i =0; i < imgw; i++) 
        for (int  j =0 ; j < imgh; j++)
            if (mapping[i][j]!= -1)
            if ( depthMap[i][j]!=-9999 &&  depthMap[i][j] <= bounder[4] &&   depthMap[i][j] >= bounder[5] ){
            cout << i << " " << j << " " << mapping[i][j] << endl;
            count++;
        }
  


    freopen ("Mapping3D2D_070.txt","w",stdout);
    cout <<textCoor.size()<< endl;
    for (int i = 0; i < textCoor.size(); ++i)
    {
            cout << i << " " <<  (int)textCoor.at(i)(0)<< " " <<  (int)textCoor.at(i)(1)<< " " <<  (int)textColor.at(i)(0)<< " " <<  (int)textColor.at(i)(1)<< " " <<  (int)textColor.at(i)(2) << " " << (int)intent.at(i) << endl;
    }

    freopen ("check2D2.off","w",stdout);
    cout << "COFF\n";
    cout << count << " 0 0 " << endl;
   
    for (int k=0; k < _2DimageZ.size (); k++) {
        int i= (int) _2DimageZ.at(k)(0);
        int j= (int) _2DimageZ.at(k)(1);
        if (mapping[i][j]!= -1) 
            if ( depthMap[i][j]!=-9999 &&  depthMap[i][j] <= bounder[4] &&   depthMap[i][j] >= bounder[5] ){
            b=image.at<cv::Vec3b>(j,i)[0];//b
            g=image.at<cv::Vec3b>(j,i)[1];//g
            r=image.at<cv::Vec3b>(j,i)[2];//r
            cout << i << " " << j << " " << _2DimageZ.at(k)(2) << " " << (int)r<< " " << (int)g<< " " << (int)b<< " 1" << endl;
            }
    }
    


    // END OF GET INTENSITY


    fs::path outputfile = outputbasename + ".png";
    cv::imwrite(outputfile.string(), outimg);

    // Save the mesh as textured obj:
    outputfile.replace_extension(".obj");
    core::write_textured_obj(mesh, outputfile.string());

    // And save the isomap:
    outputfile.replace_extension(".isomap.png");
    cv::imwrite(outputfile.string(), core::to_mat(isomap));
    */

 //   cout << "Finished fitting and wrote result mesh and isomap to files with basename "
  //       << outputfile.stem().stem() << "." << endl;

    return EXIT_SUCCESS;
}
