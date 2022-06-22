% =========================================================================
% 
% This code contains the example with the Urban image in the following paper:
% 
%    Model-based deep autoencoder networks for nonlinear hyperspectral unmixing
%    H Li, RA Borsoi, T Imbiriba, P Closas, JCM Bermudez, D Erdoğmuş
%    IEEE Geoscience and Remote Sensing Letters, 2021
% =========================================================================




clear
close all
clc
warning off;


addpath(genpath('other_methods'))
addpath(genpath('DATA'))
addpath(genpath('utils'))



% load data

load('DATA/subimage_Urban_R162.mat')

[nr,nc,L] = size(im);
N = nr*nc;

r = reshape(im,nr*nc,L)';
Yim = reshape(r', nr, nc, L);




% uncomment to plot the image

% figure;
% clear rgb
% v = [50 25 10 ];
% rgb(:,:,1) = imadjust(rescale(im(:,:,v(1)),0,1));
% rgb(:,:,2) = imadjust(rescale(im(:,:,v(2)),0,1));
% rgb(:,:,3) = imadjust(rescale(im(:,:,v(3)),0,1));
% imshow(1.5*rgb) % display RGB image
% set(gca,'ytick',[],'xtick',[])
% figure, imagesc(10*im(:,:,v))
% figure, imagesc(1.7*im(:,:,[100 60 40]))
% set(gca,'xtick',[]), set(gca,'xticklabel',[]), set(gca,'ytick',[]), set(gca,'yticklabel',[])




% Extract EMs from each class
P = 3;

% [M0, V, U, Y_bar, endm_proj, Y_proj] = find_endm(r,P,'vca');
% [M0, V, U, Y_bar, endm_proj, Y_proj] = find_endm(r,P,'nfindr');
load('DATA/M0_urban.mat')
M = M0;



%%
% =========================================================================
% run the proposed method

L1=1;
L2=1e-5;
learnRate = 1e-4;
numEpochs= 100;

lambda1=L1;    % OFF DIAG
lambda2=L1;    % OFF DIAG
lambda5=1e-2;  % inv(M)*M = I
lambda3=1e-2;  % W
lambda6=1e-6;  % SAD(M,M0)
rngflag=1;

disp('run MAC-U...') 

tic
[a_NNHU1,W,MLP1,rmse_r_NNHU1] = NNHU_autoencoder_customize_1(r,M,lambda1,lambda2,lambda3,lambda5,lambda6,learnRate,numEpochs,rngflag);
time_NNHU1 = toc;

M_est_MAC_U = MLP1.Layers(28).Weights;


%%
% plot endmembers

figure;
plot(M0)
hold on
plot(M_est_MAC_U)




% =========================================================================
%  FCLS

tic
A_FCLS = FCLSU(r,M);
time_fcls = toc;

rmse_r_FCLS = sqrt(norm(r-M*A_FCLS','fro')^2);
A_FCLS_im   = reshape(A_FCLS,nr,nc,[]);




%%
% =========================================================================
%  K-Hype

% Regualrization parameter 
C = 500; % \mu in the paper = 1/C

tic
[a_est_khype,beta_khype,rmse_r_KHYPE] = KHype(r,M,C);
time_khype = toc;





%% ========================================================================
%% Halimi's TIP16 method

flag_sto_halimi = -1;
tic
[a_NL,~,gam_NLt_halimi,~,~,c_Ev_halimi,~,~] = Unmix_CDA_NL_TIP_v1(M,Yim,0,-1,flag_sto_halimi);
a_NL = a_NL(:,:,end);
time_NL_halimi = toc;
a_NL_im = reshape(a_NL',nr,nc,P);  

M_NL_halimi = [];
for i=1:P, for j=i+1:P, M_NL_halimi = [M_NL_halimi  M(:,i).*M(:,j)]; end, end
M_NL_halimi = [sqrt(2)*M_NL_halimi M.^2]; % LxD 
Y_rec_halimi  = (M*a_NL) .*(c_Ev_halimi(:,end)*ones(1,L))' ...
    + (M_NL_halimi*gam_NLt_halimi(:,:,end)) .*(c_Ev_halimi(:,end).^2*ones(1,L))';

rmse_r_NL_halimi = sqrt(sum(sum((r - Y_rec_halimi).^2)));





%%
% display results

fprintf('\n\n TIME \n')
fprintf('FCLS.............: %f \n',time_fcls)
fprintf('KHype............: %f \n',time_khype)
fprintf('Halimi NL TIP16..: %f \n',time_NL_halimi)
fprintf('MAC-U............: %f \n',time_NNHU1)

fprintf('\n\n Reconstruction errors: \n') 
fprintf('FCLS............: RMSE_R = %f \n', rmse_r_FCLS        /sqrt(L*N) )
fprintf('K-Hype..........: RMSE_R = %f \n', rmse_r_KHYPE       /sqrt(L*N) )
fprintf('Halimi TIP16....: RMSE_R = %f \n', rmse_r_NL_halimi   /sqrt(L*N) )
fprintf('MAC-U...........: RMSE_R = %f \n', rmse_r_NNHU1       /sqrt(L*N) )





% reorder as cubes
A_FCLS_im      = reshape(A_FCLS,nr,nc,[]);
a_est_khype_im = reshape(a_est_khype',nr,nc,[]);
a_NNHU1_im     = reshape(a_NNHU1',nr,nc,[]);


idx_ems2 = 1:P;
P_reduced = length(idx_ems2);
fh = figure; 
[ha, pos] = tight_subplot(4, P_reduced, 0.01, 0.1, 0.1);

maxval = 1;
for i=1:P_reduced
    kk = 0;
    axes(ha(i+kk*P_reduced)); kk = kk+1;
    imagesc(A_FCLS_im(:,:,idx_ems2(i)),[0 maxval])
    set(gca,'ytick',[],'xtick',[])
    
    axes(ha(i+kk*P_reduced)); kk = kk+1;
    imagesc(a_est_khype_im(:,:,idx_ems2(i)),[0 maxval])
    set(gca,'ytick',[],'xtick',[])
    
    axes(ha(i+kk*P_reduced)); kk = kk+1;
    imagesc(a_NL_im(:,:,idx_ems2(i)),[0 maxval])
    set(gca,'ytick',[],'xtick',[])
    
    axes(ha(i+kk*P_reduced)); kk = kk+1;
    imagesc(a_NNHU1_im(:,:,idx_ems2(i)),[0 maxval])
    set(gca,'ytick',[],'xtick',[])
end

set(fh, 'Position', [0 0 550 700])
fontsize=12;
axes(ha(1));
title('Asphalt','interpreter','latex', 'fontsize',fontsize)
axes(ha(2));
title('Tree','interpreter','latex', 'fontsize',fontsize)
axes(ha(3));
title('Ground','interpreter','latex', 'fontsize',fontsize)

kk = 0;
axes(ha(kk*P_reduced+1)); kk = kk+1;
ylabel('FCLS','interpreter','latex', 'fontsize',fontsize)
axes(ha(kk*P_reduced+1)); kk = kk+1;
ylabel('K-Hype','interpreter','latex', 'fontsize',fontsize)
axes(ha(kk*P_reduced+1)); kk = kk+1;
ylabel('CDA-NL','interpreter','latex', 'fontsize',fontsize)
axes(ha(kk*P_reduced+1)); kk = kk+1;
ylabel('MAC-U','interpreter','latex', 'fontsize',fontsize)
colormap(jet)





