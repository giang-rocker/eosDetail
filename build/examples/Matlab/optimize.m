
[vertex,face] = read_coff('../019MeshSmall.off');
disp(size(vertex));

mesh.vertices = vertex';
mesh.faces = face';
mesh.normals = vertex';

mesh.normalVector = calculateNormal(mesh);



drawnow;
