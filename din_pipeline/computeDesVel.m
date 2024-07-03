function [Vdes, ratio, alfa, beta] = computeDesVel(attributo,A_attributo,B_attributo,CCR,l,R,K,an_lim,kt,kc,kr,ks,kd,Vlim)

if strcmp(attributo,'tangente')
    ratio = Vlim/3.6;
    if Vlim > 50
%         beta = -1/20*log(1-(60/3.6)/ratio);
        beta = -1/20*log(1-((Vlim-10)/3.6)/ratio);
    else
        beta = -1/15*log(1-(25/3.6)/ratio);
    end
    alfa = ratio*beta;
    if Vlim > 50
        if CCR <= 700
            Vdes = kt*(Vlim - 2.79*CCR^(0.47));
        elseif CCR > 700 && CCR <= 1600
            Vdes = kt*(123.54 - 2.79*CCR^(0.47)); 
        else
            Vdes = 30;
        end
    else
        Vdes = 30;
    end
elseif strcmp(attributo,'intersezione')
    ratio = Vlim/3.6;
    beta = -1/15*log(1-(25/3.6)/ratio);
    alfa = ratio*beta;
    if strcmp(A_attributo,'ing_rotonda')
        fprintf('ing_rotonda\n');
        if l > 80
            Vdes = kr*50;
        else
            Vdes = kr*(0.58*l + 30.61);
        end
        fprintf('value is %d obtained at time %d \n',Vdes, now);
        fprintf('value after division is %d \n',(Vdes/3.6));
        if (Vdes/3.6)^2/R > an_lim
            Vdes = 3.6*sqrt(an_lim*R);
        end
    elseif strcmp(A_attributo,'rotonda')
        if l > 80
            Vdes = kr*50;
        else
            Vdes = kr*(0.33*l + 42.71);
        end
        if (Vdes/3.6)^2/R > an_lim
            Vdes = 3.6*sqrt(an_lim*R);
        end
    elseif strcmp(A_attributo,'ing_stop')
        if l > 67
            Vdes = ks*50;
        else
            Vdes = ks*(0.41*l + 31.15);
        end
    elseif strcmp(A_attributo,'stop')
        if l > 67
            Vdes = ks*50;
        else
            Vdes = ks*(0.37*l + 32.02);
        end
    elseif strcmp(A_attributo,'ing_dosso')
        Vdes = kd*0.5*(101.289 - 87.46/K);
    elseif strcmp(A_attributo,'dosso')
        Vdes = kd*0.5*(101.289 - 87.46/K);
    end 
elseif strcmp(attributo,'strada')
    ratio = Vlim/3.6;
    beta = -1/15*log(1-(25/3.6)/ratio);
    alfa = ratio*beta;
    Vdes = 50; 
elseif strcmp(attributo,'curva')
    ratio = Vlim/3.6;
    beta = -1/15*log(1-(25/3.6)/ratio);
    alfa = ratio*beta;
    if CCR < 30 && R > 200 && R < 2500
        Vdes = kc*(80 - 563/sqrt(R));
    elseif CCR > 30 && CCR < 80 && R > 100 && R < 635
        Vdes = kc*(75 - 510.56/sqrt(R));
    elseif CCR > 80 && CCR < 160 && R > 77 && R < 480
        Vdes = kc*(70 - 437.44/sqrt(R));
    elseif CCR > 160 && R > 36 && R < 300
        Vdes = kc*(65 - 346.62/sqrt(R));
    else
        Vdes = kc*40;
    end   
    if (Vdes/3.6)^2/R > an_lim
        Vdes = 3.6*sqrt(an_lim*R);
    end
 
end

if Vdes > Vlim
    
    Vdes = Vlim;
    
end

end
