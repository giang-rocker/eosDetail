%GET Lsss
function l = calculateL(normal)
l(1) = sqrt(4*pi)^(-1);
l(2) = normal(1)*sqrt(4*pi)^(-1);
l(3) = normal(2)*sqrt(4*pi)^(-1);
l(4) = normal(3)*sqrt(4*pi)^(-1);
end