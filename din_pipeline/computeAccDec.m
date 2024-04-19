function [a_des,d_des,Lta,Ltd,Vdes,a_star,d_star] = computeAccDec(vectVdes,alfa,beta,attributo,A_attributo,B_attributo,R,l,g,I1,I2,r,W_P,W,G)

a_des = zeros(1,length(vectVdes));
d_des = zeros(1,length(vectVdes));
Lta = zeros(1,length(vectVdes));
Ltd = zeros(1,length(vectVdes));
Vdes = vectVdes;
a_star = zeros(1,length(vectVdes));
d_star = zeros(1,length(vectVdes));

for i = 1:length(vectVdes)
    
    if strcmp(attributo(i),'tangente') || strcmp(attributo(i),'curva')
        if strcmp(A_attributo(i),'usc_rotonda')
            a_des(i) = 0.70;
            d_des(i) = -1.5;
        elseif strcmp(A_attributo(i),'usc_curva')
            a_des(i) = 1.328 - 0.159*log(R(i-1));
            d_des(i) = -1.5;
        elseif strcmp(A_attributo(i),'usc_stop')
            a_des(i) = 0.71;
            d_des(i) = -1.5;
        elseif strcmp(A_attributo(i),'usc_dosso')
            if vectVdes(i)/3.6 < 1.8
                ratioMe = 0.2;
            else
                ratioMe = 1.02 - 1.45/vectVdes(i)/3.6;
            end
            a_des(i) = alfa(i) - beta(i)*vectVdes(i-1)/3.6 - g*I2(i)/100*ratioMe;
            d_des(i) = -1.5;
        elseif strcmp(B_attributo(i),'ing_rotonda')
            d_des(i) = -1.14;
            a_des(i) = alfa(i) - beta(i)*vectVdes(i-1)/3.6;
        elseif strcmp(B_attributo(i),'ing_curva')
            d_des(i) = -(1.757 - 0.222*log(R(i+1)));
            a_des(i) = alfa(i) - beta(i)*vectVdes(i-1)/3.6;
        elseif strcmp(B_attributo(i),'ing_dosso')
            a_des(i) = alfa(i) - beta(i)*vectVdes(i-1)/3.6;
            if vectVdes(i)/3.6 < 1.8
                ratioMe = 0.2;
            else
                ratioMe = 1.02 - 1.45/vectVdes(i)/3.6;
            end
            d_des(i) = - a_des(i) - g*I1(i)/100*ratioMe;
        elseif strcmp(B_attributo(i),'ing_stop')
            d_des(i) = -1.17;
            a_des(i) = alfa(i) - beta(i)*vectVdes(i-1)/3.6;
        elseif strcmp(A_attributo(i),'partenza') || strcmp(A_attributo(i),'stop')
            a_des(i) = alfa(i);
            d_des(i) = -1.5;
        else
                if i == 1
                    a_des(i) = alfa(i);
                    d_des(i) = -1.5;
                else
                    a_des(i) = alfa(i) - beta(i)*vectVdes(i-1)/3.6;
                    d_des(i) = -1.5;
                end
        end
    elseif strcmp(attributo(i),'intersezione')
        if strcmp(A_attributo(i),'stop') || strcmp(A_attributo(i),'ing_rotonda')
            a_des(i) = alfa(i);
            d_des(i) = -1.5;
        elseif strcmp(B_attributo(i),'stop')
            a_des(i) = 1.5;
            d_des(i) = - alfa(i) + beta(i)*vectVdes(i)/3.6; 
        else
            a_des(i) = alfa(i) - beta(i)*vectVdes(i-1)/3.6;
            d_des(i) = -1.5;
        end     
    elseif strcmp(attributo(i),'strada')
        if strcmp(A_attributo(i),'partenza') || strcmp(A_attributo(i),'stop')
            a_des(i) = alfa(i);
            d_des(i) = -1.5;
        else 
            a_des(i) = alfa(i) - beta(i)*vectVdes(i-1)/3.6;
            d_des(i) = -1.5;
        end
    end

end

for i = 1:length(vectVdes)
    
    if vectVdes(i)/3.6 < 1.8
        ratioMe = 0.2;
    else
        ratioMe = 1.02 - 1.45/vectVdes(i)/3.6;
    end
    a_star(i) = a_des(i) - G(i)*g*ratioMe;
    d_star(i) = d_des(i) - G(i)*g*ratioMe;
    
    if a_star(i) < 0 
        if G(i) > 0.01
            a_star(i) = 0.2;
        else
            a_star(i) = 0.5;
        end
    end
    
end

for i = 1:length(vectVdes)
    
        if strcmp(A_attributo(i),'partenza') || strcmp(A_attributo(i),'stop') || strcmp(A_attributo(i),'ing_rotonda') || i == 1
            v_up = vectVdes(i)/3.6;
            Lta(i) = (v_up^2)/2/a_star(i);
            if strcmp(B_attributo(i),'stop')
                Ltd(i) = (v_up^2)/2/abs(d_star(i));
            else
                if vectVdes(i) <= vectVdes(i+1)
                    v_up = vectVdes(i+1)/3.6;
                    v_down = vectVdes(i)/3.6;
                    Lta(i+1) = (v_up^2 - v_down^2)/2/a_star(i+1);
                else
                    v_up = vectVdes(i)/3.6;
                    v_down = vectVdes(i+1)/3.6;
                    Ltd(i) = (v_up^2 - v_down^2)/2/abs(d_star(i));
                end
            end
        elseif strcmp(B_attributo(i),'stop') || strcmp(B_attributo(i),'ing_rotonda')
            v_up = vectVdes(i)/3.6;
            Ltd(i) = (v_up^2)/2/abs(d_star(i));
            if vectVdes(i) <= vectVdes(i-1)
                v_up = vectVdes(i-1)/3.6;
                v_down = vectVdes(i)/3.6;
                Ltd(i-1) = (v_up^2 - v_down^2)/2/abs(d_star(i-1));
            elseif vectVdes(i) > vectVdes(i-1)
                v_up = vectVdes(i)/3.6;
                v_down = vectVdes(i-1)/3.6;
                Lta(i) = (v_up^2 - v_down^2)/2/a_star(i);
            end
        else  
            if vectVdes(i) <= vectVdes(i-1)
                v_up = vectVdes(i-1)/3.6;
                v_down = vectVdes(i)/3.6;
                Ltd(i-1) = (v_up^2 - v_down^2)/2/abs(d_star(i-1));
            elseif vectVdes(i) > vectVdes(i-1)
                v_up = vectVdes(i)/3.6;
                v_down = vectVdes(i-1)/3.6;
                Lta(i) = (v_up^2 - v_down^2)/2/a_star(i);
            elseif vectVdes(i) > vectVdes(i+1)
                v_up = vectVdes(i)/3.6;
                v_down = vectVdes(i+1)/3.6;
                Ltd(i) = (v_up^2 - v_down^2)/2/abs(d_star(i));
            elseif vectVdes(i) <= vectVdes(i+1)
                v_up = vectVdes(i+1)/3.6;
                v_down = vectVdes(i)/3.6;
                Lta(i+1) = (v_up^2 - v_down^2)/2/a_star(i+1);
            end 
        end
end

end