function normalVector = calculateNormal (mesh)
%get edge table
[w.h]=size(mesh.vertices);
normalVector = -1;
[numOfTriangle,x]=size(mesh.faces);

[numOfVertex,X] = size(mesh.vertices);

edgeTable = Zeros(numOfVertex,1);

for index = 1:numOfTriangle
        i1= mesh.faces(index,1);
        i2= mesh.faces(index,2);
        i3= mesh.faces(index,3);
        
        checka= 0; checkb=0;
        [x,numOfNeibour] = size(edgeTable[i1]);
        %x should be 1       
        for iNeibour = 1: numOfNeibour
            if (edgeTable[i1,iNeubour] == i2)
                checka = 1;
            end %end of if
            if (edgeTable[i1,iNeubour] == i3)
                checkb = 1;
            end %end of if
        end %end of for
        
        if (checka==0)
           edgeTable[i1,numOfNeibour+1] =i2;
        end
        if (checkb==0)
           edgeTable[i1,numOfNeibour+1] =i3;
        end
        
        checka= 0; checkb=0;
        [x,numOfNeibour] = size(edgeTable[i2]);
        %x should be 2      
        for iNeibour = 1: numOfNeibour
            if (edgeTable[i2,iNeubour] == i1)
                checka = 1;
            end %end of if
            if (edgeTable[i2,iNeubour] == i3)
                checkb = 1;
            end %end of if
        end %end of for
        
        if (checka==0)
           edgeTable[i2,numOfNeibour+1] =i1;
        end
        if (checkb==0)
           edgeTable[i2,numOfNeibour+1] =i3;
        end
        
        checka= 0; checkb=0;
        [x,numOfNeibour] = size(edgeTable[i3]);
        %x should be 3      
        for iNeibour = 1: numOfNeibour
            if (edgeTable[i3,iNeubour] == i1)
                checka = 1;
            end %end of if
            if (edgeTable[i3,iNeubour] == i2)
                checkb = 1;
            end %end of if
        end %end of for
        
        if (checka==0)
           edgeTable[i3,numOfNeibour+1] =i1;
        end
        if (checkb==0)
           edgeTable[i3,numOfNeibour+1] =i2;
        end
       
        end % end of for


end