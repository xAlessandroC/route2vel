function V = alias(V)
    n= length(V);  n2= n/2;
    V= [V(1,1:n2+1) conj(fliplr(V(1,2:n2)))];
end

