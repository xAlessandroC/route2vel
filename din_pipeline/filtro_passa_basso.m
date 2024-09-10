function [freq,X,xf,Xf] = filtro_passa_basso(x,fc,f0)
    % x: signal
    % fc: sampling frequence
    % f0: cut frequence   


% SIGNAL FFT
    N= length(x);  
    if mod(N,2) == 0
       x_ = x;
       N_ = length(x_);
    else
       x_ = [x, x(end)];
       N_ = length(x_);
    end
    
    N2 = 0.5*N_;
    freq= linspace(0,(N_-1)*fc/N_,N_);
    for j=1:length(x_(:,1))
        X(j,:)= fft(x_(j,:));  
        X(j,:)= alias(X(j,:));
    end

% FILTRO PASSA-BASSO
%     N= length(x);  N2= 0.5*N;
%     freq= linspace(0,(N-1)*fc/N,N);
    [~,indx]=min((freq-f0).^2);
    for j=1:length(x_(:,1))
        Xf(j,:)= X(j,:);  
        Xf(j,indx:N2+1)= zeros(1,N2-indx+2);
        Xf(j,:)= alias(Xf(j,:));
        xf_(j,:)= ifft(Xf(j,:));
    end
    
    if mod(N,2) == 0
       xf = xf_; 
    else
       xf = xf_(1:end-1); 
    end
end

