%--------------------------------------------------------------------------
% Model Speed Simulation for ROBOPAC Acc2Twin
%
%    Author:     Roberto Di Leva
%    Email:      roberto.dileva@unibo.it 
%    Date:       October 2024
%--------------------------------------------------------------------------
%%Begin
%clear all
%close all
%clc


function aa = acc2Twin(kt,kc,kr,ks,kd,csvFile)
	fprintf('M: il file input e: %s\n', csvFile);
	%[csvFileName,path] = uigetfile('*.csv');
	[base_folder,name,ext] = fileparts(csvFile)
	%outputFileName = [csvFileName(1:end-4),'_output'];   
	outputFileName = 'test';

	[rx,ry,rz,r_mat_order,l,CCR,th,G,R,Vlim,attributo,A_attributo,B_attributo] = parserFuncDisi2Din(csvFile);

	n = length(l);
	step = 10;
	fig = 0;
	breaking_type = 'nominal';
	breaking_pattern = {'A','B','C','D'};
	K = zeros(1,n);
	I1 = zeros(1,n);
	I2 = zeros(1,n);
	an_lim = 5;

	W_P = 121.7; %[kg/kW]
	P   = 298.3; %[kW]
	W   = 36288; %[kg]
	tau = 0.92;
	g   = 9.81;
           
	%% CALC&STORE
	one = figure('visible','off')
	set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
	subplot(2,2,1)
	title('Dato filtrato DIN','interpreter','latex','FontSize',14)
	hold on
	box on
	grid on
	axis equal
	xlim([min(rx)-500 max(rx)+500])
	ylim([min(ry)-500 max(ry)+500])
	plot(rx,ry,'Color','black','LineStyle','--','LineWidth',0.5)
	xlabel('x [m]','interpreter','latex','FontSize',14)
	ylabel('y [m]','interpreter','latex','FontSize',14)
	ax = gca;
	ax.TickLabelInterpreter = 'latex';
	for i = 1:size(r_mat_order,2)
    
	    Ax = [r_mat_order(2,i), r_mat_order(5,i)];
	    Ay = [r_mat_order(3,i), r_mat_order(6,i)];
	    Az = [r_mat_order(4,i), r_mat_order(7,i)];
    
	    if strcmp(attributo{i},'tangente')
	        line(Ax,Ay,'Color','blue','LineWidth',3)
	    elseif strcmp(attributo{i},'curva')
	        line(Ax,Ay,'Color','red','LineWidth',3)
	    elseif strcmp(A_attributo{i},'ing_rotonda')
        	line(Ax,Ay,'Color','yellow','LineWidth',3)
    	end
    	text(sum(Ax)/2,sum(Ay)/2,num2str(i),'FontSize',6)
    	plot(Ax,Ay,'ok','MarkerFaceColor','white','MarkerSize',3)
    
	end

	save_name = fullfile(base_folder, 'main_dato_filtrato_DIN.svg');
	set(gcf,'PaperPositionMode','auto')
	%print(save_name,'-dsvg')
	print(gcf, save_name, '-dsvg')

	l_tmp = 0;
	two = figure('visible','off')
	hold on
	grid on
	box on
	ylim([0,100])
	xlabel('\textit{l} [m]','interpreter','latex','FontSize',14)
	ylabel('\textit{v} [km/h]','interpreter','latex','FontSize',14)
	ax = gca;
	ax.TickLabelInterpreter = 'latex';
	vectVdes = zeros(1,n);
	ratio = zeros(1,n);
	alfa = zeros(1,n);
	beta = zeros(1,n);
	for i = 1:n   
    
	    [vectVdes(i), ratio(i), alfa(i), beta(i)] = computeDesVel(attributo(i),A_attributo(i),B_attributo(i),CCR(i),l(i),R(i),K(i),an_lim,kt,kc,kr,ks,kd,Vlim(i));    
    
	%     plot(linspace(l_tmp,l_tmp+l(i),round(l(i)/10)),vectVdes(i)*ones(1,round(l(i)/10)),'--k','LineWidth',0.5)
	    plot(linspace(l_tmp,l_tmp+l(i),round(l(i)/10)),vectVdes(i)*ones(1,round(l(i)/10)),'r','LineWidth',2)
	    line([l_tmp l_tmp],[0 vectVdes(i)],'Color','black','LineStyle','--','LineWidth',0.5)
	    line([l_tmp+l(i) l_tmp+l(i)],[0 vectVdes(i)],'Color','black','LineStyle','--','LineWidth',0.5)
	    text(l_tmp+l(i)/2,vectVdes(i)+1,num2str(i),'HorizontalAlignment','center',...
	        'interpreter','latex','FontSize',12)
	    l_tmp = l_tmp + l(i);
           
	end

	[a_des,d_des,Lta,Ltd,Vdes,a,d] = computeAccDec(vectVdes,alfa,beta,attributo,A_attributo,B_attributo,R,l,g,I1,I2,tau,W_P,W,G);
	l_tmp = 0;

	Vstar = zeros(1,n);
	Lta_star = zeros(1,n);
	Ltd_star = zeros(1,n);
	Vend  = zeros(1,n);
	tauA  = zeros(1,n);
	tauB  = zeros(1,n);
	tauC  = zeros(1,n);
	t_tot  = zeros(1,n);
	for i = 1:n
      
	    if Lta(i)+Ltd(i) <= l(i)
	        deltaL = l(i) - (Lta(i)+Ltd(i));
	        l_ = linspace(0,l(i),round(l(i)/2));
	        v_ = zeros(1,round(l(i)/2));
	        for j = 1:round(l(i)/2)
	            if l_(j) <= Lta(i)
	                if Lta(i) == 0
	                    v_(j) = Vdes(i);
	                else
	                    if strcmp(A_attributo(i),'partenza') || strcmp(A_attributo(i),'stop') || strcmp(A_attributo(i),'ing_rotonda')
	                        v_(j) = (Vdes(i))*l_(j)/Lta(i);
	                    else
	                        v_(j) = Vdes(i-1) + (Vdes(i) - Vdes(i-1))*l_(j)/Lta(i);  
	                    end
	                end
	            elseif l_(j) > Lta(i) + deltaL && Ltd(i) ~= 0
	                if strcmp(B_attributo(i),'stop') || strcmp(B_attributo(i),'ing_rotonda')
	                    v_(j) = Vdes(i) - (Vdes(i))*(l_(j)-(Lta(i) + deltaL))/Ltd(i);
	                else
	                    v_(j) = Vdes(i) - (Vdes(i) - Vdes(i+1))*(l_(j)-(Lta(i) + deltaL))/Ltd(i); 
	                end
        	    else
        	        v_(j) = Vdes(i); 
        	    end
        	end
    	elseif Lta(i)+Ltd(i) > l(i)
        	deltaL = 0;
        	if Ltd(i) == 0
        	    l_ = linspace(0,l(i),round(l(i)/2));
        	    v_ = zeros(1,round(l(i)/2));
        	    for j = 1:round(l(i)/2)
        	        if strcmp(A_attributo(i),'partenza') || strcmp(A_attributo(i),'stop') || strcmp(A_attributo(i),'ing_rotonda')
        	            Vstar(i) = 3.6*sqrt(2*a(i)*l(i));
        	            v_(j) = (Vstar(i))*l_(j)/l(i);
        	        else
        	            Vstar(i) = 3.6*sqrt((Vdes(i-1)/3.6)^2 + 2*a(i)*l(i));
        	            v_(j) = Vdes(i-1) + (Vstar(i) - Vdes(i-1))*l_(j)/l(i); 
        	        end
        	    end
        	    Vdes(i) = Vstar(i);
        	    v_up = Vdes(i+1)/3.6;
        	    v_down = Vstar(i)/3.6;
        	    Lta(i+1) = (v_up^2 - v_down^2)/2/a(i+1);
        	elseif Lta(i) == 0
        	    l_ = linspace(0,l(i),round(l(i)/2));
        	    v_ = zeros(1,round(l(i)/2));
        	    for j = 1:round(l(i)/2)
        	        if strcmp(B_attributo(i),'stop') || strcmp(B_attributo(i),'ing_rotonda')
        	            Vstar(i) = 3.6*sqrt(2*abs(d(i))*l(i));
        	            v_(j) = Vstar(i) - (Vstar(i))*l_(j)/l(i);
        	        else
        	            Vstar(i) = 3.6*sqrt((Vdes(i+1)/3.6)^2 + 2*abs(d(i))*l(i));
        	            v_(j) = Vstar(i) + (Vstar(i) - Vdes(i+1))*l_(j)/l(i); 
        	        end
        	    end
        	    Vdes(i) = Vstar(i);
        	    v_up = Vdes(i-1)/3.6;
        	    v_down = Vstar(i)/3.6;
        	    Ltd(i-1) = (v_up^2 - v_down^2)/2/abs(d(i-1));            
        	else          
        	    l_ = linspace(0,l(i),round(l(i)/2));
        	    v_ = zeros(1,round(l(i)/2));
        	    if strcmp(A_attributo(i),'partenza') || strcmp(A_attributo(i),'stop') || strcmp(A_attributo(i),'ing_rotonda') 
        	        vdec = 0;
        	    else 
        	        vdec = Vdes(i-1)/3.6;
        	    end
        	    if strcmp(B_attributo(i),'stop') || strcmp(B_attributo(i),'ing_rotonda')
        	        vacc = 0;
        	    else
        	        vacc = Vdes(i+1)/3.6;
        	    end
        	    Vstar(i) = 3.6*sqrt((2*a(i)*abs(d(i))*l(i)+abs(d(i))*vdec^2+a(i)*vacc^2)/(a(i)+abs(d(i))));
        	    Vdes(i) = Vstar(i);
        	    deltaL = 0;
        	    v_up = Vstar(i)/3.6;
        	    Lta_star(i) = (v_up^2 - vdec^2)/2/a(i);
        	    Ltd_star(i) = (v_up^2 - vacc^2)/2/abs(d(i));
        	    for j = 1:round(l(i)/2)
        	        if l_(j) <= Lta_star(i)
        	            if strcmp(A_attributo(i),'partenza') || strcmp(A_attributo(i),'stop') || strcmp(A_attributo(i),'ing_rotonda')
        	                v_(j) = (Vstar(i))*l_(j)/Lta_star(i);
        	            else
        	                v_(j) = Vdes(i-1) + (Vstar(i) - Vdes(i-1))*l_(j)/Lta_star(i); 
        	            end
        	        else
        	            if strcmp(B_attributo(i),'stop') || strcmp(B_attributo(i),'ing_rotonda')
        	                v_(j) = Vstar(i) - (Vstar(i))*(l_(j)-(Lta_star(i) + deltaL))/Ltd_star(i);
        	            else
        	            v_(j) = Vstar(i) - (Vstar(i) - Vdes(i+1))*(l_(j)-(Lta_star(i) + deltaL))/Ltd_star(i);
        	            end
        	        end
        	    end
        	end
    	end
        
	plot(l_tmp+l_,v_,'r','LineWidth',1.5)
    	l_tmp = l_tmp+l(i);
    
	    Vend(i) = v_(end)/3.6;
	    if i == 1
        	tauA(i) = Vdes(i)/3.6/a(i);
	    else
	        if a(i) < 0.001
	            tauA(i) = 0;
	        else
	            tauA(i) = (Vdes(i)/3.6 - Vend(i-1))/a(i);
	        end
	    end
	    tauB(i) = 3.6*deltaL/Vdes(i);
	    tauC(i) = (Vend(i) - Vdes(i)/3.6)/d(i);
	    t_tot(i) = tauA(i) + tauB(i) + tauC(i);
    
	end

	tauA(tauA<10^-3) = 0;
	tauB(tauB<10^-3) = 0;
	tauC(tauC<10^-3) = 0;
	
	time_tot = [];
	sd_tot = [];
	sdd_tot = [];
	l_tmp_array = [];
	tauA_star  = zeros(1,n);
	tauB_star  = zeros(1,n);
	tauC_star  = zeros(1,n);

	subplot(2,2,3)
	hold on
	grid on
	box on
	ylim([-5,40])
	xlabel('t [s]','interpreter','latex','FontSize',14)
	ax = gca;
	ax.TickLabelInterpreter = 'latex';
	for i = 1:n  
       
	    if i == 1
	        time = linspace(0,t_tot(i),10*round(t_tot(i)));
	        sdd = zeros(1,length(time));
	        sd = Vdes(i)/3.6*ones(1,length(time));
	        sdd(time<tauA(i)) = a(i);
	        sd(time<tauA(i))  = 0 + a(i)*time(time<tauA(i));
	        sdd(time>tauA(i)+tauB(i)) = d(i);
	        sd(time>tauA(i)+tauB(i))  = Vdes(i)/3.6 + d(i)*(time(time>tauA(i)+tauB(i)) - tauA(i) - tauB(i));
	        time_tmp = 0;
	        l_tmp = 0;
	    elseif strcmp(B_attributo(i),'stop')
	        if strcmp(breaking_type,'random')
	            V0 = Vend(i-1);
	            pattern = 'D';

	            fun = @(T)breakingPattern(T,V0,pattern);
        	    t_end = fsolve(fun,10);        

	            if strcmp(pattern,'A')
	                d_pattern = 9.81*[0, 0, -0.02, -0.05, -0.1, -0.2, -0.40, -0.35, -0.1, -0.05, 0.01, 0.05, 0.01];
	            elseif strcmp(pattern,'B')
	                d_pattern = 9.81*[0.05, 0.03, 0, -0.02, -0.05, -0.1, -0.38, -0.38, -0.32, -0.28, -0.22, -0.1, -0.08, -0.05];
	            elseif strcmp(pattern,'C')
	                d_pattern = 9.81*[-0.10, -0.15, -0.18, -0.2, -0.22, -0.25, -0.30, -0.30, -0.20, -0.10, 0, 0.02, 0.04, 0.06];
	            elseif strcmp(pattern,'D')
	                d_pattern = 9.81*[-0.05, -0.09, -0.11, -0.2, -0.23, -0.29, -0.35, -0.35, -0.29, -0.22, -0.18, -0.1, -0.05, 0];
	            end

	            t = t_end*linspace(0,1,length(d_pattern));

        	    t_spline = linspace(0,t_end,101);
        	    d_spline = spline(t,d_pattern,t_spline);
        	    v_spline = V0 + cumtrapz(t_spline,d_spline);

        	    l_spline = cumtrapz(t_spline,v_spline);
        	    l_spline(end)

	            tauA_star(i) = tauA(i);
        	    tauB_star(i) = (l(i)-l_spline(end))/Vdes(i);
        	    tauC_star(i) = t_end;

	            t_tot(i) = tauA_star(i) + tauB_star(i) + tauC_star(i);

	            time = linspace(0,t_tot(i),10*round(t_tot(i)));
        	    sdd = zeros(1,length(time));
        	    sd = Vdes(i)/3.6*ones(1,length(time));
        	    sdd(time<tauA_star(i)) = a(i);
	%             sd(time<tauA_star(i))  = Vend(i-1) + a(i)*time(time<=tauA(i));
        	    sd(time<tauA_star(i))  = Vend(i-1) + a(i)*time(time<tauA(i));
        	    sdd(time>tauA_star(i)+tauB_star(i)) = spline(t_spline+tauA_star(i)+tauB_star(i),d_spline,time(time>tauA_star(i)+tauB_star(i)));
        	    sd(time>tauA_star(i)+tauB_star(i))  = spline(t_spline+tauA_star(i)+tauB_star(i),v_spline,time(time>tauA_star(i)+tauB_star(i)));
        	    time_tmp = time_tmp + t_tot(i-1);
        	    time_ptn = time + time_tmp;
        	    l_tmp = l_tmp + l(i-1);
        	elseif strcmp(breaking_type,'nominal')
        	    time = linspace(0,t_tot(i),10*round(t_tot(i)));
        	    sdd = zeros(1,length(time));
        	    sd = Vdes(i)/3.6*ones(1,length(time));
        	    sdd(time<tauA(i)) = a(i);
	%             sd(time<tauA(i))  = Vend(i-1) + a(i)*time(time<=tauA(i));
        	    sd(time<tauA(i))  = Vend(i-1) + a(i)*time(time<tauA(i));
        	    sdd(time>tauA(i)+tauB(i)) = d(i);
        	    sd(time>tauA(i)+tauB(i))  = Vdes(i)/3.6 + d(i)*(time(time>tauA(i)+tauB(i)) - tauA(i) - tauB(i));
        	    time_tmp = time_tmp + t_tot(i-1);
        	    time_ptn = time + time_tmp;
        	    l_tmp = l_tmp + l(i-1);
        	end
	    else
	        time = linspace(0,t_tot(i),10*round(t_tot(i)));
	        sdd = zeros(1,length(time));
	        sd = Vdes(i)/3.6*ones(1,length(time));
	        sdd(time<tauA(i)) = a(i);
	        sd(time<tauA(i))  = Vend(i-1) + a(i)*time(time<tauA(i));
	        sdd(time>tauA(i)+tauB(i)) = d(i);
	        sd(time>tauA(i)+tauB(i))  = Vdes(i)/3.6 + d(i)*(time(time>tauA(i)+tauB(i)) - tauA(i) - tauB(i));
	        time_tmp = time_tmp + t_tot(i-1);
	        l_tmp = l_tmp + l(i-1);
	    end
    
 	   l_tot(i) = l_tmp;
 	   if i == 1
	        time_tot = [time_tot, time(1:end)+time_tmp];
        	sdd_tot = [sdd_tot, sdd(1:end)];
        	sd_tot = [sd_tot, sd(1:end)];
        	l_tmp_array = [l_tmp_array, l(i)];
	    else
	        time_tot = [time_tot, time(2:end)+time_tmp];
	        sdd_tot = [sdd_tot, sdd(2:end)];
	        sd_tot = [sd_tot, sd(2:end)];
	        l_tmp_array = [l_tmp_array, l(i)+l_tmp_array(end)];
	    end
	    time_tmp_array(i) = time_tot(end);

	    if strcmp(B_attributo(i),'stop') && i ~= 1
        	plot(time+time_tmp,sd,'m','LineWidth',2)
        	plot(time+time_tmp,sdd,'.--g','LineWidth',1)
	    else
	        plot(time+time_tmp,sd,'r','LineWidth',2)
	        plot(time+time_tmp,sdd,'.--b','LineWidth',1)
 	   end
	    line([time_tmp time_tmp],[0 Vdes(i)/3.6],'Color','black','LineStyle','--','LineWidth',0.5)
	    line([time_tmp+t_tot(i) time_tmp+t_tot(i)],[0 Vdes(i)/3.6],'Color','black','LineStyle','--','LineWidth',0.5)
	    line([time_tmp time_tmp+t_tot(i)],[Vdes(i)/3.6 Vdes(i)/3.6],'Color','black','LineStyle','--','LineWidth',0.5) 
	    legend('$\dot s$ [m/s]','$\ddot s$ [m/s$^2$]','interpreter','latex','FontSize',14)
            
	end

	l_road = zeros(1,length(rx));
	l_tmp = 0;
	for i = 2:length(rx)
   
	    l_tmp = l_tmp + norm([rx(i)-rx(i-1); ry(i)-ry(i-1)]);  
	    l_road(i) = l_tmp;
    
	end

	save_name =fullfile(base_folder, 'main_two_v2.svg');
	set(gcf,'PaperPositionMode','auto')
	%print(save_name,'-dsvg')
	print(gcf, save_name, '-dsvg')

	l_tot = [l_tot, l_tot(end)+l(end)];
	% l_tot = [l_tot(2:end), l_tot(end)+l(end)];
	deltaLength = 0.005*l_tot(end);
	s_tot = cumtrapz(time_tot,sd_tot);
	r_s = spline(l_road,[rx; ry],s_tot);
	r_sz = spline(l_road,rz,s_tot);

	rp_s  = [diff(r_s(1,:))./diff(s_tot); diff(r_s(2,:))./diff(s_tot)];
	rpp_s = [diff(rp_s(1,:))./diff(s_tot(1:end-1)); diff(rp_s(2,:))./diff(s_tot(1:end-1))];

	rp_s_spline  = spline(s_tot(1:end-1),rp_s,s_tot);
	rpp_s_spline = spline(s_tot(1:end-2),rpp_s,s_tot);

	ind_R = find(R);
	R_min = min(R(ind_R));
	en_new = zeros(2,length(time_tot));
	et = zeros(2,length(time_tot));
	en = zeros(2,length(time_tot));
	rd_tot = zeros(2,length(time_tot));
	rdd_tot = zeros(2,length(time_tot));
	at = zeros(2,length(time_tot));
	an = zeros(2,length(time_tot));
	rho = zeros(1,length(time_tot));
	rd_norm = zeros(1,length(time_tot));
	at_int = zeros(1,length(time_tot));
	an_int = zeros(1,length(time_tot));
	az_int = zeros(1,length(time_tot));
	for i = 1:length(time_tot)
    
	    A_ind1 = find(l_tot <= s_tot(i));
	    A_ind = A_ind1(end);
	    if A_ind > length(attributo)
	        A_ind = length(attributo);
	    end
	    if i ~= 1 && i ~= length(time_tot)
	        r1 = [r_s(1,i-1); r_s(2,i-1)];
	        r2 = [r_s(1,i); r_s(2,i)];
	        r3 = [r_s(1,i+1); r_s(2,i+1)];
	        [en_new(:,i), rho(i)] = findOscCircle(r1,r2,r3); 
	        rho(i) = R(A_ind);
	    else
	        rho(i) = 0;
	    end
	    if rho(i) < 250 && strcmp(attributo(A_ind),'tangente')
	        rdd_tot(:,i) = rp_s_spline(:,i)*sdd_tot(i);
	    elseif rho(i) < 1.5*R_min
	        rdd_tot(:,i) = rp_s_spline(:,i)*sdd_tot(i);
	    else
	        rdd_tot(:,i) = en_new(:,i)*sd_tot(i)^2/rho(i) + rp_s_spline(:,i)*sdd_tot(i);
	    end
	    rd_tot(:,i)  = rp_s_spline(:,i)*sd_tot(i);
    
	    rd_norm(i) = norm(rd_tot(:,i));
	    et(:,i) = rp_s_spline(:,i);
	    at_int(i) = rdd_tot(:,i)'*et(:,i);
	    at(:,i) = at_int(i)*et(:,i);
	    an(:,i) = rdd_tot(:,i) - at(:,i);
    
	    en(:,i) = an(:,i)/norm(an(:,i));
	    an_int(i) = rdd_tot(:,i)'*R2d(pi/2)*et(:,i); 
    
	    az_int(i) = -g*sqrt(1 - G(A_ind)^2);
	    
	end

	subplot(2,2,4)
	hold on
	grid on
	box on
	axis equal
	xlim([min(rx)-deltaLength  max(rx)+deltaLength ])
	ylim([min(ry)-deltaLength  max(ry)+deltaLength ])
	plot(r_s(1,:),r_s(2,:),'b','LineWidth',3)
	plot(r_mat_order(2,1),r_mat_order(3,1),'-og','LineWidth',2)
	for i = 1:n
	     if strcmp(B_attributo(i),'stop')
	        plot(r_mat_order(2,i),r_mat_order(3,i),'-dr','MarkerFaceColor','red','LineWidth',2)
	    elseif strcmp(A_attributo(i),'ing_rotonda')
	        plot(r_mat_order(2,i),r_mat_order(3,i),'-oy','MarkerFaceColor','yellow','LineWidth',2)
	        plot(r_mat_order(5,i),r_mat_order(6,i),'-oy','MarkerFaceColor','yellow','LineWidth',2)
	     elseif strcmp(B_attributo(i),'dosso')
	        plot(r_mat_order(2,i),r_mat_order(3,i),'-^r','MarkerFaceColor','white','LineWidth',2)
	    end
	end
	xlabel('x [m]','interpreter','latex','FontSize',14)
	ylabel('y [m]','interpreter','latex','FontSize',14)
	legend('$\mathbf{r}(s)$','interpreter','latex','FontSize',14)
	ax = gca;
	ax.TickLabelInterpreter = 'latex';

	save_name =fullfile(base_folder, 'path.svg');
	set(gcf,'PaperPositionMode','auto')
	print(gcf, save_name, '-dsvg')


	%% Total Motion-Law Profile Graphics 

	three = figure('visible','off')
	set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
	subplot(2,3,1)
	hold on
	grid on
	box on
	axis equal
	xlim([min(rx)-deltaLength  max(rx)+deltaLength ])
	ylim([min(ry)-deltaLength  max(ry)+deltaLength ])
	plot(rx,ry,'Color','black','LineStyle','--','LineWidth',0.5)
	xlabel('x [m]','interpreter','latex','FontSize',14)
	ylabel('y [m]','interpreter','latex','FontSize',14)
	ax = gca;
	ax.TickLabelInterpreter = 'latex';
	for i = 1:size(r_mat_order,2)
    
	    Ax = [r_mat_order(2,i), r_mat_order(5,i)];
	    Ay = [r_mat_order(3,i), r_mat_order(6,i)];
	    Az = [r_mat_order(4,i), r_mat_order(7,i)];
    
	    if strcmp(attributo{i},'tangente')
	        line(Ax,Ay,'Color','blue','LineWidth',3)
	    elseif strcmp(attributo{i},'curva')
	        line(Ax,Ay,'Color','red','LineWidth',3)
	    elseif strcmp(A_attributo{i},'ing_rotonda')
	        line(Ax,Ay,'Color','yellow','LineWidth',3)
	    end
	    text(sum(Ax)/2,sum(Ay)/2,[num2str(i)],'FontSize',6)
	    plot(Ax,Ay,'ok','MarkerFaceColor','white','MarkerSize',3)
    
	end
	subplot(2,3,4)
	hold on
	grid on
	box on
	axis equal
	xlim([min(rx)-deltaLength  max(rx)+deltaLength ])
	ylim([min(ry)-deltaLength  max(ry)+deltaLength ])
	plot(r_s(1,:),r_s(2,:),'b','LineWidth',3)
	plot(r_mat_order(2,1),r_mat_order(3,1),'-og','LineWidth',2)
	% for i = 1:100:1500
	%     
	%     plot(r_s(1,i),r_s(2,i),'*k','LineWidth',2)
	%     text(r_s(1,i)+5, r_s(2,i)+0, strcat('$s = $',num2str(s_tot(i)),'[m]'),'interpreter','latex','FontSize',18);
	%     
	% end
	for i = 1:n
	     if strcmp(B_attributo(i),'stop')
	        plot(r_mat_order(2,i),r_mat_order(3,i),'-dr','MarkerFaceColor','red','LineWidth',2)
	    elseif strcmp(A_attributo(i),'ing_rotonda')
	        plot(r_mat_order(2,i),r_mat_order(3,i),'-oy','MarkerFaceColor','yellow','LineWidth',2)
	        plot(r_mat_order(5,i),r_mat_order(6,i),'-oy','MarkerFaceColor','yellow','LineWidth',2)
	     elseif strcmp(B_attributo(i),'dosso')
	        plot(r_mat_order(2,i),r_mat_order(3,i),'-^r','MarkerFaceColor','white','LineWidth',2)
	    end
	end
	xlabel('x [m]','interpreter','latex','FontSize',14)
	ylabel('y [m]','interpreter','latex','FontSize',14)
	legend('$\mathbf{r}(s)$','interpreter','latex','FontSize',14)
	ax = gca;
	ax.TickLabelInterpreter = 'latex';
	subplot(2,3,2)
	hold on
	grid on
	box on
	plot(time_tot,3.6*rd_norm,'LineWidth',2)
	plot(time_tot,3.6*sd_tot,'--k','LineWidth',1)
	xlabel('t [s]','interpreter','latex','FontSize',14)
	ylabel('$||\mathbf{v}||$ [km/h]','interpreter','latex','FontSize',14)
	ax = gca;
	ax.TickLabelInterpreter = 'latex';
	subplot(2,3,3)
	hold on
	grid on
	box on
	plot(time_tot,at_int,'LineWidth',2)
	plot(time_tot,an_int,'LineWidth',2)
	plot(time_tot,az_int,'LineWidth',2)
	text(150,-6,strcat('$|{a}_t|_{max} = $',num2str(max(abs(at_int))),'[m/s$^2$]'),...
	           'interpreter','latex','FontSize',12)
	text(150,-8,strcat('$|{a}_n|_{max} = $',num2str(max(abs(an_int))),'[m/s$^2$]'),...
	            'interpreter','latex','FontSize',12)
	xlabel('t [s]','interpreter','latex','FontSize',14)
	ylabel('[m/s$^2$]','interpreter','latex','FontSize',14)
	legend('${a}_t$','${a}_n$','${a}_z$','interpreter','latex','FontSize',14)
	ax = gca;
	ax.TickLabelInterpreter = 'latex';
	subplot(2,3,5)
	hold on
	grid on
	box on
	plot(s_tot,3.6*rd_norm,'LineWidth',2)
	plot(s_tot,3.6*sd_tot,'--k','LineWidth',1)
	xlabel('s [m]','interpreter','latex','FontSize',14)
	ylabel('$||\mathbf{v}||$ [km/h]','interpreter','latex','FontSize',14)
	ax = gca;
	ax.TickLabelInterpreter = 'latex';
	subplot(2,3,6)
	hold on
	grid on
	box on
	plot(s_tot,at_int,'LineWidth',2)
	plot(s_tot,an_int,'LineWidth',2)
	plot(s_tot,az_int,'LineWidth',2)
	text(150,-6,strcat('$|{a}_t|_{max} = $',num2str(max(abs(at_int))),'[m/s$^2$]'),...
	            'interpreter','latex','FontSize',12)
	text(150,-8,strcat('$|{a}_n|_{max} = $',num2str(max(abs(an_int))),'[m/s$^2$]'),...
	            'interpreter','latex','FontSize',12)
	xlabel('s [m]','interpreter','latex','FontSize',14)
	ylabel('[m/s$^2$]','interpreter','latex','FontSize',14)
	legend('${a}_t$','${a}_n$','${a}_z$','interpreter','latex','FontSize',14)
	ax = gca;
	ax.TickLabelInterpreter = 'latex';

	save_name =fullfile(base_folder, 'kinematics.svg');
	set(gcf,'PaperPositionMode','auto')
	%print(save_name,'-dsvg')
	print(gcf, save_name, '-dsvg')
	
	
	%% Colormap Graphics
	
	four = figure('visible','off')
	hold on
	grid on
	box on
	axis equal
	xlim([min(rx)-deltaLength  max(rx)+deltaLength ])
	ylim([min(ry)-deltaLength  max(ry)+deltaLength ])
	xlabel('x [m]','interpreter','latex','FontSize',14)
	ylabel('y [m]','interpreter','latex','FontSize',14)
	ax = gca;
	ax.TickLabelInterpreter = 'latex';
	hs=surf([r_s(1,:);r_s(1,:)],[r_s(2,:);r_s(2,:)],zeros(2,length(time_tot)),[3.6*sd_tot;3.6*sd_tot],'EdgeColor','interp','LineWidth',3) 
	colormap(flip(autumn))
	view(2) 
	colorbarOpts = colorbar('Direction','reverse','TickLabelInterpreter','latex','FontSize',14);
	colorbarOpts.Label.String = '$v$ [km/h]';
	colorbarOpts.Label.Interpreter = 'latex';
	colorbarOpts.Label.FontSize = 14;

	save_name =fullfile(base_folder, 'colormap_v.svg');	
	set(gcf,'PaperPositionMode','auto')
	print(gcf, save_name, '-dsvg')
	
	five = figure('visible','off')
	hold on
	grid on
	box on
	axis equal
	xlim([min(rx)-deltaLength  max(rx)+deltaLength ])
	ylim([min(ry)-deltaLength  max(ry)+deltaLength ])
	xlabel('x [m]','interpreter','latex','FontSize',14)
	ylabel('y [m]','interpreter','latex','FontSize',14)
	ax = gca;
	ax.TickLabelInterpreter = 'latex';
	hs=surf([r_s(1,:);r_s(1,:)],[r_s(2,:);r_s(2,:)],zeros(2,length(time_tot)),[at_int;at_int],'EdgeColor','interp','LineWidth',3) 
	colormap(winter)
	view(2) 
	colorbarOpts = colorbar('Direction','reverse','TickLabelInterpreter','latex','FontSize',14);
	colorbarOpts.Label.String = '$a_t$ [m/s$^2$]';
	colorbarOpts.Label.Interpreter = 'latex';
	colorbarOpts.Label.FontSize = 14;

	save_name =fullfile(base_folder, 'colormap_at.svg');
	set(gcf,'PaperPositionMode','auto')
	%print(save_name,'-dsvg')
	print(gcf, save_name, '-dsvg')

	
	six = figure('visible','off')
	hold on
	grid on
	box on
	axis equal
	xlim([min(rx)-deltaLength  max(rx)+deltaLength ])
	ylim([min(ry)-deltaLength  max(ry)+deltaLength ])
	xlabel('x [m]','interpreter','latex','FontSize',14)
	ylabel('y [m]','interpreter','latex','FontSize',14)
	ax = gca;
	ax.TickLabelInterpreter = 'latex';
	hs=surf([r_s(1,:);r_s(1,:)],[r_s(2,:);r_s(2,:)],zeros(2,length(time_tot)),[abs(an_int);abs(an_int)],'EdgeColor','interp','LineWidth',3) 
	colormap(spring)
	view(2) 
	colorbarOpts = colorbar('Direction','reverse','TickLabelInterpreter','latex','FontSize',14);
	colorbarOpts.Label.String = '$a_n$ [m/s$^2$]';
	colorbarOpts.Label.Interpreter = 'latex';
	colorbarOpts.Label.FontSize = 14;

	save_name = fullfile(base_folder, 'colormap_an.svg');
	set(gcf,'PaperPositionMode','auto')
	%print(save_name,'-dsvg')
	print(gcf, save_name, '-dsvg')

	
	seven = figure('visible','off')
	hold on	
	grid on
	box on
	axis equal
	xlim([min(rx)-deltaLength max(rx)+deltaLength])
	ylim([min(ry)-deltaLength max(ry)+deltaLength])
	xlabel('x [m]','interpreter','latex','FontSize',14)
	ylabel('y [m]','interpreter','latex','FontSize',14)
	ax = gca;
	ax.TickLabelInterpreter = 'latex';
	hs=surf([r_s(1,:);r_s(1,:)],[r_s(2,:);r_s(2,:)],zeros(2,length(time_tot)),[sqrt(at_int.^2 + an_int.^2);sqrt(at_int.^2 + an_int.^2)],'EdgeColor','interp','LineWidth',3) 
	colormap(summer)
	view(2) 
	colorbarOpts = colorbar('Direction','reverse','TickLabelInterpreter','latex','FontSize',14);
	colorbarOpts.Label.String = '$\sqrt{a_t^2 + a_n^2}$ [m/s$^2$]';
	colorbarOpts.Label.Interpreter = 'latex';
	colorbarOpts.Label.FontSize = 14;


	%% Worst cases detection

	an_detected = sqrt(at_int.^2+an_int.^2) > 2;
	detected_index = find(an_detected);
	ind = [0 find(diff(detected_index) > 1) length(detected_index)];
	r_C_ind = [];
	k = 0;

	%XXXX 
	if isempty(detected_index)
	    fprintf('No worst cases detected')
	else
	%XXXX 

		for i = 1:length(ind)-1
		    str(i).detected_index = detected_index(ind(i)+1:ind(i+1));
		    str(i).rpp_s_detected = rpp_s_spline(:,str(i).detected_index);
		    str(i).s_detected = s_tot(str(i).detected_index);
		    str(i).rho_vec = 1./sqrt(diag(str(i).rpp_s_detected'*str(i).rpp_s_detected));
		    str(i).rho = mean(str(i).rho_vec);
		    str(i).rhomin = min(str(i).rho_vec);
		    str(i).rhomax = max(str(i).rho_vec);
		    
		%     A_ind1 = find(l_tot < str(i).s_detected(1));
		    A_ind1 = find(l_tot <= str(i).s_detected(1));
		    A_ind2 = find(l_tot > str(i).s_detected(1));
		    A_ind = [A_ind1(end) A_ind2(1)];
		
		    B_ind1 = find(l_tot < str(i).s_detected(end));
		    B_ind2 = find(l_tot > str(i).s_detected(end));
		    B_ind = [B_ind1(end) B_ind2(1) B_ind2(1)+1];
		
		    C_ind = [A_ind B_ind];
		    str(i).C_ind = unique(C_ind, 'first');
		    str(i).r_C_ind = [r_s(1,str(i).C_ind); r_s(2,str(i).C_ind)];
		    r_C_ind = [r_C_ind str(i).r_C_ind];   
		    
		    A_ind1 = find(s_tot < l_tot(str(i).C_ind(1)));
		    if isempty(A_ind1)
			A_ind = 1;
		    else
			A_ind = A_ind1(end);
		    end
	    
		    if str(i).C_ind(end) > length(l_tot)
			B_ind1 = find(s_tot > l_tot(end));
		    else
			B_ind1 = find(s_tot > l_tot(str(i).C_ind(end)));
		    end
		    if isempty(B_ind1)
			B_ind = length(time_tot);
		    else
			B_ind = B_ind1(1);
	    	end
		%     B_ind = B_ind1(1);
	    
		    str(i).t   = time_tot(A_ind:B_ind);
		    str(i).r   = r_s(:,A_ind:B_ind);
		    str(i).rz  = r_sz(:,A_ind:B_ind);
		    str(i).rp  = rd_tot(:,A_ind:B_ind);
		    str(i).rpp = [at_int(:,A_ind:B_ind);an_int(:,A_ind:B_ind);az_int(:,A_ind:B_ind)];
		    str(i).rho = rho(A_ind:B_ind);
	    
		    [str(i).an_max str(i).an_max_ind] = max(str(i).rpp(2,:));
		    an_max_ind(i) = str(i).an_max_ind;
		    k = k + 1;
		end

		ind_an_max = find(diff([0, an_max_ind]));

		for i = ind_an_max

			ind_t1 = find(str(i).t > (str(i).t(an_max_ind(i))-0.75));
	   	 	ind_t2 = find(str(i).t < (str(i).t(an_max_ind(i))+0.75));
	    		str(i).t_detected = str(i).t(ind_t1(1):ind_t2(end));
	    		str(i).rpp_detected = str(i).rpp(:,ind_t1(1):ind_t2(end));
	    
		end

		%% Worst-case Graphics

		eight = figure('visible','off')
		hold on	
		grid on
		box on
		axis equal
		xlim([min(rx)-deltaLength max(rx)+deltaLength])
		ylim([min(ry)-deltaLength max(ry)+deltaLength])
		xlabel('x [m]','interpreter','latex','FontSize',14)
		ylabel('y [m]','interpreter','latex','FontSize',14)
		ax = gca;
		ax.TickLabelInterpreter = 'latex';
		hs=surf([r_s(1,:);r_s(1,:)],[r_s(2,:);r_s(2,:)],zeros(2,length(time_tot)),[sqrt(at_int.^2 + an_int.^2);sqrt(at_int.^2 + an_int.^2)],'EdgeColor','interp','LineWidth',3) 
		colormap(summer)
		view(2) 
		colorbarOpts = colorbar('Direction','reverse','TickLabelInterpreter','latex','FontSize',14);
		colorbarOpts.Label.String = '$\sqrt{a_t^2 + a_n^2}$ [m/s$^2$]';
		colorbarOpts.Label.Interpreter = 'latex';
		colorbarOpts.Label.FontSize = 14;
		wc_i = 0;
		for i = 1:k
		    if ~isempty(str(i).t_detected)
			wc_i = wc_i + 1;
			hold on
			Ox_rectangle = min(str(i).r(1,:))-deltaLength;
			Oy_rectangle =  min(str(i).r(2,:))-deltaLength;
			b_rectanle = abs((min(str(i).r(1,:))-deltaLength) - (max(str(i).r(1,:))+deltaLength));
			h_rectanle = abs((min(str(i).r(2,:))-deltaLength) - (max(str(i).r(2,:))+deltaLength));
			rectangle('Position',[Ox_rectangle Oy_rectangle b_rectanle h_rectanle])  
			text(Ox_rectangle,Oy_rectangle+h_rectanle,num2str(wc_i),'Interpreter','latex','FontSize',14)
		    end
	    
		end

		save_name =fullfile(base_folder, 'colormap_wc.svg');
		set(gcf,'PaperPositionMode','auto')
		print(gcf, save_name, '-dsvg')
		
		wc_i = 0;
		for i =1:k
		    if ~isempty(str(i).t_detected)
			wc_i = wc_i + 1;
			t_wc = str(i).t;
			rpp_wc = str(i).rpp;
			clear rpp_wc_filt
			if t_wc(end) - t_wc(1) < 40
			    for j = 1:length(t_wc)  
			        for i_tnz = 1:3
			            if abs(rpp_wc(i_tnz,j)) < 2
			                rpp_wc_filt(i_tnz,j) = 0;
			            else
			                rpp_wc_filt(i_tnz,j) = rpp_wc(i_tnz,j);
			            end
			        end
			    end
			    T = t_wc(end)-t_wc(1);
			    freq = 100;
			    n_wc = round(freq*T)+1;
			    t_wc_spline = linspace(0,T,n_wc);
			    rpp_wc_spline = spline(t_wc-t_wc(1),rpp_wc,t_wc_spline);
			    rpp_wc_filt_spline = spline(t_wc-t_wc(1),rpp_wc_filt,t_wc_spline);
			    [freq,X,an_filt,Xf] = filtro_passa_basso(rpp_wc_spline(2,:),100,5);
			    [freq,X,at_filt,Xf] = filtro_passa_basso(rpp_wc_spline(1,:),100,5);
			    A = [t_wc_spline;rpp_wc_spline;at_filt;an_filt;sqrt(rpp_wc_spline(1,:).^2 + rpp_wc_spline(2,:).^2)];
		%             A_filt = [t_wc_spline;rpp_wc_filt_spline];

			    save_name =fullfile(base_folder, strcat('worstCase_',num2str(wc_i),'.csv'));
		%	    writematrix(A',save_name,'Delimiter', 'comma');
			    writematrix(A',save_name,'Delimiter', 'comma');
		%             writematrix(A_filt',[outputFileName, '_worstCase_filt_',num2str(wc_i),'.csv'],'Delimiter','comma'); 
			end
			nine = figure('visible','off')
			sgtitle(['Worst case ', num2str(wc_i)],'Interpreter','latex','FontSize',14)
			subplot(1,2,1)
			hold on
			grid on
			box on
			axis equal
			xlim([min(str(i).r(1,:))-deltaLength, max(str(i).r(1,:))+deltaLength])
			ylim([min(str(i).r(2,:))-deltaLength, max(str(i).r(2,:))+deltaLength])
			xlabel('x [m]','interpreter','latex','FontSize',14)
			ylabel('y [m]','interpreter','latex','FontSize',14)
			ax = gca;
			ax.TickLabelInterpreter = 'latex';
			surf([r_s(1,:);r_s(1,:)],[r_s(2,:);r_s(2,:)],zeros(2,length(time_tot)),[sqrt(at_int.^2 + an_int.^2);sqrt(at_int.^2 + an_int.^2)],'EdgeColor','interp','LineWidth',3) 
			colormap(summer)
			view(2)
			colorbarOpts = colorbar('Direction','reverse','TickLabelInterpreter','latex','FontSize',14);
			colorbarOpts.Label.String = '$\sqrt{a_t^2 + a_n^2}$ [m/s$^2$]';
			colorbarOpts.Label.Interpreter = 'latex';
			colorbarOpts.Label.FontSize = 14;
			title(['Worst case ', num2str(wc_i)],'Interpreter','latex','FontSize',14)
			subplot(1,2,2)
			hold on
			grid on
			box on
			plot(str(i).t,str(i).rpp(1:2,:),'LineWidth',2)

			xlabel('t [s]','interpreter','latex','FontSize',14)
			ylabel('[m/s$^2$]','interpreter','latex','FontSize',14)
			legend('${a}_t$','${a}_n$','interpreter','latex','FontSize',14)
			ax = gca;
			ax.TickLabelInterpreter = 'latex';

			filename=strcat('wc_',num2str(wc_i),'.svg');
			save_name =fullfile(base_folder, filename);
			set(gcf,'PaperPositionMode','auto')
			print(gcf, save_name, '-dsvg')
			end
	    	end
	    end
	aa = 1
end
