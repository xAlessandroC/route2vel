function [en, R] = findOscCircle(r1,r2,r3)

X1 = r1(1);
Y1 = r1(2);
X2 = r2(1);
Y2 = r2(2);
X3 = r3(1);
Y3 = r3(2);

Coeff = [X1 Y1 1; X2 Y2 1; X3 Y3 1];
bKnown = [-X1^2 - Y1^2; -X2^2 - Y2^2; -X3^2 - Y3^2;];

Sol = Coeff\bKnown;

A_ = Sol(1);
B_ = Sol(2);
C_ = Sol(3);

Xc = -A_/2;
Yc = -B_/2;

rPC = [Xc; Yc] - [X2; Y2];
en = rPC/norm(rPC);
R = sqrt(Xc^2 + Yc^2 - C_);
        
end