/*
 * eos - A 3D Morphable Model fitting library written in modern C++11/14.
 *
 * File: include/eos/render/draw_utils.hpp
 *
 * Copyright 2017 Patrik Huber
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
#pragma once

#ifndef RENDER_DRAW_UTILS_HPP_
#define RENDER_DRAW_UTILS_HPP_

#include "eos/core/Mesh.hpp"
#include "eos/render/detail/render_detail.hpp"

#include "glm/gtc/matrix_transform.hpp"
#include "glm/mat4x4.hpp"
#include "glm/vec4.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <algorithm>
using namespace std;
using namespace Eigen;

namespace eos {
namespace render {

/**
 * Draws the given mesh as wireframe into the image.
 *
 * It does backface culling, i.e. draws only vertices in CCW order.
 *
 * @param[in] image An image to draw into.
 * @param[in] mesh The mesh to draw.
 * @param[in] modelview Model-view matrix to draw the mesh.
 * @param[in] projection Projection matrix to draw the mesh.
 * @param[in] viewport Viewport to draw the mesh.
 * @param[in] color Colour of the mesh to be drawn.
 */
float area(Vector2f A, Vector2f B, Vector2f M) {

        float a = sqrt((A(0) - B(0))*(A(0) - B(0)) + (A(1) - B(1))*(A(1) - B(1)));
        float b = sqrt((A(0) - M(0))*(A(0) - M(0)) + (A(1) - M(1))*(A(1) - M(1)));
        float c = sqrt((B(0) - M(0))*(B(0) - M(0)) + (B(1) - M(1))*(B(1) - M(1)));

        float p = (a + b + c) / 2;

        return sqrt(p*(p - a)*(p - b)*(p - c));
}

bool insideABC(Vector2f A, Vector2f B, Vector2f C, Vector2f M) {
    return (!(ceilf(area(A, B, M) + area(A, C, M) + area(B, C, M)) != ceilf(area(A, B, C))));
}


inline void draw_wireframe(cv::Mat image, const core::Mesh mesh, glm::mat4x4 modelview,
                           glm::mat4x4 projection, glm::vec4 viewport,
                           cv::Scalar color = cv::Scalar(0, 255, 0, 255))
{


    for (const auto& triangle : mesh.tvi)
    {
        const auto p1 = glm::project(
            {mesh.vertices[triangle[0]][0], mesh.vertices[triangle[0]][1], mesh.vertices[triangle[0]][2]},
            modelview, projection, viewport);
        const auto p2 = glm::project(
            {mesh.vertices[triangle[1]][0], mesh.vertices[triangle[1]][1], mesh.vertices[triangle[1]][2]},
            modelview, projection, viewport);
        const auto p3 = glm::project(
            {mesh.vertices[triangle[2]][0], mesh.vertices[triangle[2]][1], mesh.vertices[triangle[2]][2]},
            modelview, projection, viewport);
        if (render::detail::are_vertices_ccw_in_screen_space(glm::vec2(p1), glm::vec2(p2), glm::vec2(p3)))
        {
            cv::line(image, cv::Point(p1.x, p1.y), cv::Point(p2.x, p2.y), color);
            cv::line(image, cv::Point(p2.x, p2.y), cv::Point(p3.x, p3.y), color);
            cv::line(image, cv::Point(p3.x, p3.y), cv::Point(p1.x, p1.y), color);
        }
    }
};

inline void mapping(cv::Mat image, const core::Mesh& mesh, glm::mat4x4 modelview,
                           glm::mat4x4 projection, glm::vec4 viewport,
                            vector <Vector2f>& textCoor,
                           cv::Scalar color = cv::Scalar(0, 255, 0, 255))
{
    Vector2f temp2f;
    
    for (const auto& v : mesh.vertices)
    {
        const auto p1 = glm::project({v[0], v[1], v[2]},  modelview, projection, viewport);
       
        temp2f(0) = p1.x ;  temp2f(1) = p1.y ; 
        textCoor.push_back (temp2f);


    }
 
};

inline void add_depth_information(cv::Mat image, const core::Mesh mesh, glm::mat4x4 modelview,
                           glm::mat4x4 projection, glm::vec4 viewport,
                           vector <vector <float > >& depthMap, 
                           vector <Vector2f>  & textCoor,
                             float _scale, 
                           cv::Scalar color = cv::Scalar(0, 255, 0, 255))
{


    float scale = 10.0f;// _scale;
 
    for (const auto& triangle : mesh.tvi)
    {
        const auto p1 = glm::project(
            {mesh.vertices[triangle[0]][0], mesh.vertices[triangle[0]][1], mesh.vertices[triangle[0]][2]},
            modelview, projection, viewport);
        const auto p2 = glm::project(
            {mesh.vertices[triangle[1]][0], mesh.vertices[triangle[1]][1], mesh.vertices[triangle[1]][2]},
            modelview, projection, viewport);
        const auto p3 = glm::project(
            {mesh.vertices[triangle[2]][0], mesh.vertices[triangle[2]][1], mesh.vertices[triangle[2]][2]},
            modelview, projection, viewport);

       

        if (render::detail::are_vertices_ccw_in_screen_space(glm::vec2(p1), glm::vec2(p2), glm::vec2(p3)))
        {
            cv::line(image, cv::Point(p1.x, p1.y), cv::Point(p2.x, p2.y), color);
            cv::line(image, cv::Point(p2.x, p2.y), cv::Point(p3.x, p3.y), color);
            cv::line(image, cv::Point(p3.x, p3.y), cv::Point(p1.x, p1.y), color);

            float p1z = p1.z * scale; float p2z = p2.z * scale;float p3z = p3.z* scale;
         
            depthMap[(int)p1.x][(int)p1.y] =  p1z ;
            depthMap[(int)p2.x][(int)p2.y] =  p2z ;
            depthMap[(int)p3.x][(int)p3.y] =  p3z ;

            textCoor.at(triangle[0])(0) = (int)p1.x;textCoor.at(triangle[0])(1) = (int)p1.y;
            textCoor.at(triangle[1])(0) = (int)p2.x;textCoor.at(triangle[1])(1) = (int)p2.y;
            textCoor.at(triangle[2])(0) = (int)p3.x;textCoor.at(triangle[2])(1) = (int)p3.y;
            
              
            float A[3]  ; A[0]  =  (p2.x-p1.x); A[1]  = (p2.y-p1.y); A[2]  = (p2z-p1z); 
            float B[3]  ; B[0]  =  (p3.x-p1.x); B[1]  = (p3.y-p1.y); B[2]  = (p3z-p1z); 
            float p[3] ;
             p[0] = A[1]*B[2] - B[1]*A[2];
             p[1] = A[2]*B[0] - B[2]*A[0];
             p[2] = A[0]*B[1] - B[0]*A[1];
            float a= p[0], b = p[1], c = p[2]; 
            float d= - ( a* p1.x + b*p1.y + c* p1z );

            Vector2f A1 ,B1,C1, M;
            A1 << p1.x , p1.y;
            B1 << p2.x , p2.y;
            C1 << p3.x , p3.y;

            
            int maxX = (int) max( p1.x, max (p2.x,p3.x)), minX =  (int)min( p1.x, min (p2.x,p3.x));
            int maxY = (int) max( p1.y, max (p2.y,p3.y)), minY = (int)min( p1.y, min (p2.y,p3.y));
           // cout << "done here" << endl;
           // cout << c << endl;
            if (c!=0)
            for (int i = minX; i <=  maxX; i++)
                for (int j =minY; j <= maxY; j++) {
                    M << i , j;
                    if (insideABC(A1,B1,C1,M))
                    depthMap[i ][j ] =   -(d+a*(i ) + b*(j ))/c ;
            }

        }
    }
    cout << "DONE MAPPINGs" << endl;
};

inline double getLen (Vector3f a, Vector3f b) {
    return  (a-b).norm();
}

inline double getLen (Vector2f a, Vector2f b) {
    return  (a-b).norm();
}

inline void getMapping2D3D(cv::Mat image, const core::Mesh& mesh, glm::mat4x4 modelview,
                           glm::mat4x4 projection, glm::vec4 viewport,
                           vector <vector< int > >&  mapping,
                           vector <vector< double > >&  currentLen, float _scale,
                           cv::Scalar color = cv::Scalar(0, 255, 0, 255))
{
    float scale = _scale;

    for (const auto& triangle : mesh.tvi)
    {
        const auto p1 = glm::project(
            {mesh.vertices[triangle[0]][0], mesh.vertices[triangle[0]][1], mesh.vertices[triangle[0]][2]},
            modelview, projection, viewport);
        const auto p2 = glm::project(
            {mesh.vertices[triangle[1]][0], mesh.vertices[triangle[1]][1], mesh.vertices[triangle[1]][2]},
            modelview, projection, viewport);
        const auto p3 = glm::project(
            {mesh.vertices[triangle[2]][0], mesh.vertices[triangle[2]][1], mesh.vertices[triangle[2]][2]},
            modelview, projection, viewport);
        if (render::detail::are_vertices_ccw_in_screen_space(glm::vec2(p1), glm::vec2(p2), glm::vec2(p3)))
        {
         
            float p1z = p1.z * scale; float p2z = p2.z * scale;float p3z = p3.z* scale;
            mapping[(int)p1.x][(int)p1.y] = triangle[0] ;
            mapping[(int)p2.x][(int)p2.y] = triangle[1] ;
            mapping[(int)p3.x][(int)p3.y] =triangle[2];
            currentLen[(int)p1.x][(int)p1.y] =  0 ;
            currentLen[(int)p2.x][(int)p2.y] =  0;
            currentLen[(int)p3.x][(int)p3.y] = 0;

            Vector3f x1,x2,x3;
            x1 << p1.x , p1.y , p1z;
            x2 << p2.x , p2.y , p2z;
            x3 << p3.x , p3.y , p3z;
             
            float A[3]  ; A[0]  =  (p2.x-p1.x); A[1]  = (p2.y-p1.y); A[2]  = (p2z-p1z); 
            float B[3]  ; B[0]  =  (p3.x-p1.x); B[1]  = (p3.y-p1.y); B[2]  = (p3z-p1z); 
            float p[3] ; p[0] = A[1]*B[2] - B[1]*A[2];p[1] = A[2]*B[0] - B[2]*A[0];p[2] = A[0]*B[1] - B[0]*A[1];
            float a= p[0], b = p[1], c = p[2]; 
            float d= - ( a* p1.x + b*p1.y + c* p1z );

            
            int maxX = (int) max( p1.x, max (p2.x,p3.x)), minX =  (int)min( p1.x, min (p2.x,p3.x));
            int maxY = (int) max( p1.y, max (p2.y,p3.y)), minY = (int)min( p1.y, min (p2.y,p3.y));
           // cout << "done here" << endl;
           // cout << c << endl;
            if (c!=0)
            for (int i = minX; i <=  maxX; i++)
                for (int j =minY; j <= maxY; j++) {
                    Vector3f currentPoint;
                    currentPoint << i,j, ( -(d+a*(i ) + b*(j ))/c );
                   double L1=  getLen ( x1,currentPoint   );
                   double L2=  getLen ( x2,currentPoint   );
                   double L3=  getLen ( x3,currentPoint   );

                    
                   if (L1 < currentLen[i][j]) { currentLen[i][j] = L1; mapping[i][j] = triangle[0] ; }
                   if (L2 < currentLen[i][j]) { currentLen[i][j] = L2; mapping[i][j] = triangle[1] ; }    
                   if (L3 < currentLen[i][j]) { currentLen[i][j] = L3; mapping[i][j] = triangle[2] ; }   
                  
                }

        }
    }
    cout << "DONE MAPPINGs" << endl;
};


inline void getMapping2D3DBy2D(cv::Mat image, const core::Mesh& mesh, glm::mat4x4 modelview,
                           glm::mat4x4 projection, glm::vec4 viewport,
                           vector <vector< int > >&  mapping,
                           vector <vector< double > >&  currentLen,
                           cv::Scalar color = cv::Scalar(0, 255, 0, 255))
{
    
    for (const auto& triangle : mesh.tvi)
    {
        const auto p1 = glm::project(
            {mesh.vertices[triangle[0]][0], mesh.vertices[triangle[0]][1], mesh.vertices[triangle[0]][2]},
            modelview, projection, viewport);
        const auto p2 = glm::project(
            {mesh.vertices[triangle[1]][0], mesh.vertices[triangle[1]][1], mesh.vertices[triangle[1]][2]},
            modelview, projection, viewport);
        const auto p3 = glm::project(
            {mesh.vertices[triangle[2]][0], mesh.vertices[triangle[2]][1], mesh.vertices[triangle[2]][2]},
            modelview, projection, viewport);
        if (render::detail::are_vertices_ccw_in_screen_space(glm::vec2(p1), glm::vec2(p2), glm::vec2(p3)))
        {
         
            mapping[(int)p1.x][(int)p1.y] = triangle[0] ;
            mapping[(int)p2.x][(int)p2.y] = triangle[1] ;
            mapping[(int)p3.x][(int)p3.y] =triangle[2];
            currentLen[(int)p1.x][(int)p1.y] =  0 ;
            currentLen[(int)p2.x][(int)p2.y] =  0;
            currentLen[(int)p3.x][(int)p3.y] = 0;

            Vector2f x1,x2,x3;
            x1 << p1.x , p1.y;
            x2 << p2.x , p2.y ;
            x3 << p3.x , p3.y;
             
              
            int maxX = (int) max( p1.x, max (p2.x,p3.x)), minX =  (int)min( p1.x, min (p2.x,p3.x));
            int maxY = (int) max( p1.y, max (p2.y,p3.y)), minY = (int)min( p1.y, min (p2.y,p3.y));
           // cout << "done here" << endl;
           // cout << c << endl;
            
            for (int i = minX; i <=  maxX; i++)
                for (int j =minY; j <= maxY; j++) {

                    Vector2f currentPoint;
                    currentPoint << i,j ;

                     if (!(insideABC(x1,x2,x3,currentPoint))) continue;

                    double L1=  getLen ( x1,currentPoint   );
                    double L2=  getLen ( x2,currentPoint   );
                    double L3=  getLen ( x3,currentPoint   );
                    
                   if (L1 < currentLen[i][j]) { currentLen[i][j] = L1; mapping[i][j] = triangle[0] ; }
                   if (L2 < currentLen[i][j]) { currentLen[i][j] = L2; mapping[i][j] = triangle[1] ; }    
                   if (L3 < currentLen[i][j]) { currentLen[i][j] = L3; mapping[i][j] = triangle[2] ; }   
                  
                }

        }
    }
    cout << "DONE MAPPINGs" << endl;
};



inline void getMapping2DTriangle(cv::Mat image, const core::Mesh& mesh, glm::mat4x4 modelview,
                           glm::mat4x4 projection, glm::vec4 viewport,
                           vector <vector< int > >&  mappingTriangle,
                           cv::Scalar color = cv::Scalar(0, 255, 0, 255))
{
    int currentTriangle  =0;
    for (const auto& triangle : mesh.tvi)
    {
        const auto p1 = glm::project(
            {mesh.vertices[triangle[0]][0], mesh.vertices[triangle[0]][1], mesh.vertices[triangle[0]][2]},
            modelview, projection, viewport);
        const auto p2 = glm::project(
            {mesh.vertices[triangle[1]][0], mesh.vertices[triangle[1]][1], mesh.vertices[triangle[1]][2]},
            modelview, projection, viewport);
        const auto p3 = glm::project(
            {mesh.vertices[triangle[2]][0], mesh.vertices[triangle[2]][1], mesh.vertices[triangle[2]][2]},
            modelview, projection, viewport);
        if (render::detail::are_vertices_ccw_in_screen_space(glm::vec2(p1), glm::vec2(p2), glm::vec2(p3)))
        {
         
            mappingTriangle[(int)p1.x][(int)p1.y] = currentTriangle ;
            mappingTriangle[(int)p2.x][(int)p2.y] = currentTriangle;
            mappingTriangle[(int)p3.x][(int)p3.y] = currentTriangle;
           
            Vector2f A ,B,C, M;
            A << p1.x , p1.y;
            B << p2.x , p2.y;
            C << p3.x , p3.y;
              
            int maxX = (int) max( p1.x, max (p2.x,p3.x)), minX =  (int)min( p1.x, min (p2.x,p3.x));
            int maxY = (int) max( p1.y, max (p2.y,p3.y)), minY = (int)min( p1.y, min (p2.y,p3.y));
           // cout << "done here" << endl;
           // cout << c << endl;
            
            for (int i = minX; i <=  maxX; i++)
                for (int j =minY; j <= maxY; j++) {
                     M << i , j ;
                     if (insideABC(A,B,C,M)) 
                     mappingTriangle[i][j] = currentTriangle;
                }

        }
        currentTriangle++;
    }
    cout << "DONE MAPPINGs Triangle" << endl;
};


/**
 * Draws the texture coordinates (uv-coords) of the given mesh
 * into an image by looping over the triangles and drawing each
 * triangle's texcoords.
 *
 * Note/Todo: This function has a slight problems, the lines do not actually get
 * drawn blue, if the image is 8UC4. Well if I save a PNG, it is blue. Not sure.
 *
 * @param[in] mesh A mesh with texture coordinates.
 * @param[in] image An optional image to draw onto.
 * @return An image with the texture coordinate triangles drawn in it, 512x512 if no image is given.
 */
inline cv::Mat draw_texcoords(core::Mesh mesh, cv::Mat image = cv::Mat())
{
    using cv::Point2f;
    using cv::Scalar;
    if (image.empty())
    {
        image = cv::Mat(512, 512, CV_8UC4, Scalar(0.0f, 0.0f, 0.0f, 255.0f));
    }

    for (const auto& triIdx : mesh.tvi)
    {
        cv::line(
            image,
            Point2f(mesh.texcoords[triIdx[0]][0] * image.cols, mesh.texcoords[triIdx[0]][1] * image.rows),
            Point2f(mesh.texcoords[triIdx[1]][0] * image.cols, mesh.texcoords[triIdx[1]][1] * image.rows),
            Scalar(255.0f, 0.0f, 0.0f));
        cv::line(
            image,
            Point2f(mesh.texcoords[triIdx[1]][0] * image.cols, mesh.texcoords[triIdx[1]][1] * image.rows),
            Point2f(mesh.texcoords[triIdx[2]][0] * image.cols, mesh.texcoords[triIdx[2]][1] * image.rows),
            Scalar(255.0f, 0.0f, 0.0f));
        cv::line(
            image,
            Point2f(mesh.texcoords[triIdx[2]][0] * image.cols, mesh.texcoords[triIdx[2]][1] * image.rows),
            Point2f(mesh.texcoords[triIdx[0]][0] * image.cols, mesh.texcoords[triIdx[0]][1] * image.rows),
            Scalar(255.0f, 0.0f, 0.0f));
    }
    return image;
};

} /* namespace render */
} /* namespace eos */

#endif /* RENDER_DRAW_UTILS_HPP_ */
