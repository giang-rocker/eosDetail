function [vertex,face,coor_vertex] = read_coff(filename)

% read_off - read data from OFF file.
%
%   [vertex,face] = read_off(filename);
%
%   'vertex' is a 'nb.vert x 3' array specifying the position of the vertices.
%   'face' is a 'nb.face x 3' array specifying the connectivity of the mesh.
%
%   Copyright (c) 2003 Gabriel Peyr�


fid = fopen(filename,'r');
if( fid==-1 )
    error('Can''t open the file.');
    return;
end

str = fgets(fid);   % -1 if eof
if ~strcmp(str(1:4), 'COFF')
    disp(str(1:4));
    error('The file is not a valid COFF one.');    
end

str = fgets(fid);
[a,str] = strtok(str); nvert = str2num(a);
[a,str] = strtok(str); nface = str2num(a);




[A,cnt] = fscanf(fid,'%f %f %f', 7*nvert);
if cnt~=7*nvert
    warning('Problem in reading vertices.');
end
A = reshape(A, 7, cnt/7);
vertex = A(1:3,:);
coor_vertex = A(4:6,:);

% read Face 1  1088 480 1022
[A,cnt] = fscanf(fid,'%d %d %d %d\n', 4*nface);
if cnt~=4*nface
    warning('Problem in reading faces.');
end
A = reshape(A, 4, cnt/4);
face = A(2:4,:)+1;

fclose(fid);

