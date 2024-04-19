function matrix = R2d(a)
%% Description:  matrix=R2d(a)
%
%two-dimensional rotation matrix (i.e. rotation about z axis [2x2])
%
% matrix= [ cos(a) -sin(a);
%           sin(a)  cos(a); ];
%
%

    matrix= [ cos(a) -sin(a);
              sin(a)  cos(a); ];
          
end
          