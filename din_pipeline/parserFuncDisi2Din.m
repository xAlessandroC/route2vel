function [rx,ry,rz,r_mat_order,l,CCR,th,G,R,Vlim,attributo,A_attributo,B_attributo] = parserFuncDisi2Din(csvFileName)

[base_folder,name,ext] = fileparts(csvFileName)
A = readmatrix(csvFileName,'OutputType','char');
n = size(A,1);
i_line = 1;
i_curve = 1;
i_roundabout = 1;
rx = zeros(1,n);
ry = zeros(1,n);
rz = zeros(1,n);
index_type = zeros(1,n);

%%Calculate&Store: Point coordinates (rx, ry, rz) and section types (line,
%%curve, roundabout)

disi=figure('visible','off')
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
title('Dato grezzo DISI','interpreter','latex','FontSize',14)
hold on
box on
grid on
axis equal
view([0, 90])
xlabel('x [m]','interpreter','latex','FontSize',14)
ylabel('y [m]','interpreter','latex','FontSize',14)
zlabel('z [m]','interpreter','latex','FontSize',14)
ax = gca;
ax.TickLabelInterpreter = 'latex';
for i = 1:n
    
    rx(i) = str2num(A{i,1});
    ry(i) = str2num(A{i,2});
    rz(i) = str2num(A{i,3});
    
    index_type(i) = str2num(A{i,5});
    
    if strcmp(A{i,4},'line')
        index_line(i_line) = i;
        i_line = i_line + 1;
        plot3(rx(i),ry(i),rz(i),'+b')
    elseif strcmp(A{i,4},'curve')
        plot3(rx(i),ry(i),rz(i),'*r')
        index_curve(i_curve) = i;
        i_curve = i_curve + 1;
    elseif strcmp(A{i,4},'roundabout')
        plot3(rx(i),ry(i),rz(i),'dy')
        index_roundabout(i_roundabout) = i;
        i_roundabout = i_roundabout + 1;
    end
end
plot3(rx,ry,rz,'k')
saveas(disi, fullfile(base_folder, 'disi_dato_filtrato.png'))
save_name = fullfile(base_folder, 'disi_dato_filtrato_v2.svg');
set(gcf,'PaperPositionMode','auto')
%print(save_name,'-dsvg')
print(gcf, save_name, '-dsvg')

%%Calculate&Store: Organize linear road sections according to their
%%horizontal grade

B_index = find(diff(index_type));
for i = 1:length(B_index)
    
    if i == 1
        ind1 = 1;
        ind2 = B_index(i);
    else
        ind1 = B_index(i-1) + 1;
        ind2 = B_index(i);
    end
    
%     if strcmp(A{ind1,4},'curve') && (ind2-ind1) < 17
    if strcmp(A{ind1,4},'curve') && (ind2-ind1) < 5
        for j = ind1:ind2
            A{j,4} = 'line';
        end
    end
    
    if i ~= 1 && strcmp(A{ind1,4},'line') && strcmp(A{B_index(i-1),4},'line')
        for j = ind1:ind2
            A{j,5} = A{B_index(i-1),5};
        end
        
    end
    
end

i_line = 1;
i_curve = 1;
i_roundabout = 1;
index_type = zeros(1,n);
clear {'index_curve', 'index_line', 'index_roundabout'}

din=figure('visible','off')
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
title('Dato semi-filtrato DIN','interpreter','latex','FontSize',14)
hold on
box on
grid on
axis equal
view([0, 90])
xlim([min(rx)-500 max(rx)+500])
ylim([min(ry)-500 max(ry)+500])
%xlim([0 5000])
%ylim([8000 10000])
xlabel('x [m]','interpreter','latex','FontSize',14)
ylabel('y [m]','interpreter','latex','FontSize',14)
zlabel('z [m]','interpreter','latex','FontSize',14)
ax = gca;
ax.TickLabelInterpreter = 'latex';
for i = 1:n
    
    index_type(i) = str2num(A{i,5});
    
    if strcmp(A{i,4},'line')
        index_line(i_line) = i;
        i_line = i_line + 1;
        plot3(rx(i),ry(i),rz(i),'+b')
    elseif strcmp(A{i,4},'curve')
        plot3(rx(i),ry(i),rz(i),'*r')
        index_curve(i_curve) = i;
        i_curve = i_curve + 1;
    elseif strcmp(A{i,4},'roundabout')
        plot3(rx(i),ry(i),rz(i),'dy')
        index_roundabout(i_roundabout) = i;
        i_roundabout = i_roundabout + 1;
    end

end
plot3(rx,ry,rz,'k')
saveas(din, fullfile(base_folder, 'din_dato_semi_filtrato.png'))
save_name = fullfile(base_folder, 'din_dato_semi_filtrato_v2.svg');
set(gcf,'PaperPositionMode','auto')
%print(save_name,'-dsvg')
print(gcf, save_name, '-dsvg')

% B_index = find(diff(index_type));
B_index = [find(diff(index_type)) size(A,1)];
k = 0;
k_sum = 0;
for i = 1:length(B_index)
    
    tmp_index = index_type(B_index(i));
    if i == 1
        r_mat_order_old(:,i) = [tmp_index; rx(tmp_index); ry(tmp_index); rz(tmp_index);...
                                       rx(B_index(i)); ry(B_index(i)); rz(B_index(i));];
        r_mat_order(:,i+k_sum) = [tmp_index; rx(tmp_index); ry(tmp_index); rz(tmp_index);...
                                       rx(B_index(i)); ry(B_index(i)); rz(B_index(i));];
        
    else
        r_mat_order_old(:,i) = [tmp_index; rx(tmp_index-1); ry(tmp_index-1); rz(tmp_index-1);...
                                       rx(B_index(i)); ry(B_index(i)); rz(B_index(i));];
        r_mat_order(:,i+k_sum) = [tmp_index; rx(tmp_index-1); ry(tmp_index-1); rz(tmp_index-1);...
                                       rx(B_index(i)); ry(B_index(i)); rz(B_index(i));];
    end
    if strcmp(A{r_mat_order(1,i+k_sum),4},'line')
        m = (ry(tmp_index)-ry(tmp_index+1))/(rx(tmp_index)-rx(tmp_index+1));
        q = ry(tmp_index+1) - m*rx(tmp_index+1);
        for j = tmp_index+2:B_index(i)
             y_star = m*rx(j) + q;
             e_star(j) = abs(y_star - ry(j)); 
             if e_star(j) < 20
                 k = k;
             else
                 r_mat_order(5:7,i+k_sum+k) = [rx(j-1); ry(j-1); rz(j-1)];
                 k = k+1;
                 m = (ry(j)-ry(j+1))/(rx(j)-rx(j+1));
                 q = ry(j+1) - m*rx(j+1);
                 r_mat_order(1:4,i+k_sum+k) = [j; rx(j-1); ry(j-1); rz(j-1)];                     
             end
        end  
        r_mat_order(5:7,i+k_sum+k) = [rx(B_index(i)); ry(B_index(i)); rz(B_index(i))];
        k_sum = k_sum + k;
        k = 0;
    end
    
end

%%Calculate&Store: Refine the attribute of the roads and Compute, for each
%%road section its length l, its horizontale grade CCR, its vertical grade
%%G and Export the limit velocity Vlim

attributo = cell(1,size(r_mat_order,2));
attributo(:) = {'tangente'};
A_attributo = attributo;
B_attributo = attributo;
l = zeros(1,size(r_mat_order,2));
CCR = zeros(1,size(r_mat_order,2));
th = zeros(1,size(r_mat_order,2));
G = zeros(1,size(r_mat_order,2));
Vlim = zeros(1,size(r_mat_order,2));

din=figure('visible','off')
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
title('Dato filtrato DIN','interpreter','latex','FontSize',14)
hold on
box on
grid on
axis equal
xlim([min(rx)-500 max(rx)+500])
ylim([min(ry)-500 max(ry)+500])
%xlim([0 5000])
%ylim([8000 10000])
plot(rx,ry,'Color','black','LineStyle','--','LineWidth',0.5)
xlabel('x [m]','interpreter','latex','FontSize',14)
ylabel('y [m]','interpreter','latex','FontSize',14)
ax = gca;
ax.TickLabelInterpreter = 'latex';
for i = 1:size(r_mat_order,2)
    Ax = [r_mat_order(2,i), r_mat_order(5,i)];
    Ay = [r_mat_order(3,i), r_mat_order(6,i)];
    Az = [r_mat_order(4,i), r_mat_order(7,i)];
    
    l_tmp = 0;
    if i ~= size(r_mat_order,2)
        for j = r_mat_order(1,i):(r_mat_order(1,i+1)-1)     
            Ax_ = [rx(j), rx(j+1)];
            Ay_ = [ry(j), ry(j+1)];
            l_tmp = l_tmp + sqrt(diff(Ax_)^2 + diff(Ay_)^2);
        end
    else
        for j = r_mat_order(1,i):size(A,1)-1     
            Ax_ = [rx(j), rx(j+1)];
            Ay_ = [ry(j), ry(j+1)];
            l_tmp = l_tmp + sqrt(diff(Ax_)^2 + diff(Ay_)^2);
        end
    end
    l(i) = l_tmp;
    
%     l(i) = sqrt(diff(Ax)^2 + diff(Ay)^2 + diff(Az)^2);
    G(i) = (diff(Az))/l(i);
    th(i) = atan2(diff(Ay)/l(i),diff(Ax)/l(i));
    if i ~=size(r_mat_order,2) && strcmp(A{r_mat_order(1,i+1),4},'roundabout')
        A{r_mat_order(1,i),4} = 'line';
        A{r_mat_order(1,i+2),4} = 'line';
    elseif strcmp(A{r_mat_order(1,i),4},'curve') && l(i) < 50
        A{r_mat_order(1,i),4} = 'line';        
    end
    if strcmp(A{r_mat_order(1,i),4},'line')
        line(Ax,Ay,'Color','blue','LineWidth',4)
    elseif strcmp(A{r_mat_order(1,i),4},'curve')
        line(Ax,Ay,'Color','red','LineWidth',4)
    elseif strcmp(A{r_mat_order(1,i),4},'roundabout')
        line(Ax,Ay,'Color','yellow','LineWidth',4)
    end
%     text(sum(Ax)/2,sum(Ay)/2,[num2str(i),'-',num2str(l(i))])
    plot(Ax,Ay,'ok','MarkerFaceColor','white','MarkerSize',5)
    text(sum(Ax)/2,sum(Ay)/2,[num2str(i)],'FontSize',6)

    Vlim(i) = str2num(A{r_mat_order(1,i),6});
    if Vlim(i) > 80 && Vlim(i) <= 110
        Vlim(i) = 80;
    elseif Vlim(i) > 110
        Vlim(i) = 90;
    end
    
end
CCR(2:end) = abs(180/pi*diff(th)./(l(2:end)/1000));

for i = 1:size(r_mat_order,2)
    
    if i ~=size(r_mat_order,2) && strcmp(A{r_mat_order(1,i+1),4},'roundabout') && strcmp(A{r_mat_order(1,i),4},'line')
        attributo{i} = 'tangente';
        A_attributo{i} = 'tangente';
        B_attributo{i} = 'ing_rotonda';
        attributo{i+2} = 'tangente';
        A_attributo{i+2} = 'usc_rotonda';
        B_attributo{i+2} = 'tangente';
    elseif i ~=size(r_mat_order,2) && strcmp(A{r_mat_order(1,i+1),4},'curve') && strcmp(A{r_mat_order(1,i),4},'line')
        attributo{i} = 'tangente';
        A_attributo{i} = 'tangente';
        B_attributo{i} = 'ing_curva';
        attributo{i+2} = 'tangente';
        A_attributo{i+2} = 'usc_curva';
        B_attributo{i+2} = 'tangente';
    elseif strcmp(A{r_mat_order(1,i),4},'roundabout')
        attributo{i} = 'intersezione';
        A_attributo{i} = 'ing_rotonda';
        B_attributo{i} = 'usc_rotonda';
    elseif strcmp(A{r_mat_order(1,i),4},'curve')
        attributo{i} = 'curva';
        A_attributo{i} = 'ing_curva';
        B_attributo{i} = 'usc_curva';       
    end
    
end
A_attributo{1} = 'partenza';
B_attributo{end} = 'stop';

%%Calculate&Store: Compute the curvature radius R for each identified
%%roundabout and curve section

n_circle = 361;
th_circle = linspace(0,2*pi,n_circle);
R = zeros(1,size(r_mat_order,2));
Xc = zeros(1,size(r_mat_order,2));
Yc = zeros(1,size(r_mat_order,2));
for i = 1:size(r_mat_order,2)
    
    if strcmp(attributo{i},'curva') || strcmp(A_attributo{i},'ing_rotonda')...
            || strcmp(A_attributo{i},'rotonda')
        j = round((r_mat_order(1,i)+r_mat_order(1,i+1))/2);
        X1 = rx(j);
        Y1 = ry(j);
        X2 = rx(j+1);
        Y2 = ry(j+1);
        X3 = rx(j+2);
        Y3 = ry(j+2);

        Coeff = [X1 Y1 1; X2 Y2 1; X3 Y3 1];
        bKnown = [-X1^2 - Y1^2; -X2^2 - Y2^2; -X3^2 - Y3^2;];

        Sol = Coeff\bKnown;

        A_ = Sol(1);
        B_ = Sol(2);
        C_ = Sol(3);

        Xc(i) = -A_/2;
        Yc(i) = -B_/2;
        R(i) = sqrt(Xc(i)^2 + Yc(i)^2 - C_);
        
        circle_centre = plot3(Xc(i),Yc(i),rz(j),'*k');
        circle = plot3(Xc(i)+R(i)*sin(th_circle),Yc(i)+R(i)*cos(th_circle),rz(j)*ones(1,n_circle),'--r');
    end
    
end  
saveas(din, fullfile(base_folder, 'din_dato_filtrato.png'))       
save_name = fullfile(base_folder, 'din_dato_filtrato_v2.svg');
set(gcf,'PaperPositionMode','auto')
%print(save_name,'-dsvg')
print(gcf, save_name, '-dsvg')
end
