clear
% close all
% clc
warning off;


disp('Using polynomial kernel...')


addpath(genpath('other_methods'))
addpath(genpath('DATA'))
addpath(genpath('utils'))


rng(10, 'twister') % reproducible



% --------------------- OPTIONS ---------------------
% Noise SNR (20 or 30 dB) 
SNR = 20; 30;


% abundance maps ----
ab_maps = 'im1_squares';
% ab_maps = 'im2_spatial'; 

% nonlinearity model (2 or 3) ----
% nlmix_fcn = 2; nlmm_str = 'blmm'; % bilinear mixing model
nlmix_fcn = 3; nlmm_str = 'pnmm'; % post-nonlinear mixing model


% estimate EMs ----
% flag_estimate_EM = false; eeaStr = 'trueEMs'; % use known EMs
flag_estimate_EM = true; eeaStr = 'VCA';  % use estimated EMs (VCA)


% Gaussian kernel bandwidth (not used) ----
% par = 2;

% =========================================================================
% generate synthetic image

% load abundance maps
if strcmp(ab_maps,'im1_squares')
    load('DATA/abundance_squares_3em.mat')
elseif strcmp(ab_maps,'im2_spatial')
    load('DATA/abundance_spatial_3em.mat')
else
    error('Unknown abundance maps selected!')
end

nr = size(A_cube,1);
nc = size(A_cube,2);
P  = size(A_cube,3);
N  = nr*nc;

% THIS ORDERING IS OK ---------------
% convert matrix to image
conv2im  = @(A)  reshape(A',nr,nc,P);
% convert image to matrix
conv2mat = @(A)  reshape(A,nr*nc,P)';
% -----------------------------------

% convert to matrix
Ath = conv2mat(A_cube);


% load endmembers ---------------------------------------------------------
[M,namesM] = load_endmembers(P);
L = size(M,1);

% generate noiseless image according to the nonlinearity model
% [Y, ~] = hypermix(M, nr*nc, 2, Ath); % Bilinear mixing model
% [Y, ~] = hypermix(M, nr*nc, 3, Ath); % Post nonlinear mixing model
[Y, ~] = hypermix(M, nr*nc, nlmix_fcn, Ath);

% add noise
pw_signal = norm(Y,'fro')^2/(nr*nc*L);
pw_noise  = pw_signal/(10^(SNR/10));
std_noise = sqrt(pw_noise);
noise     = std_noise*randn(L,nr*nc);
r = Y + noise;

% write the image over other variables too
Y = r;
Yim = reshape(Y', nr, nc, L);

% --------------------------------------------------------------
% Endmember initialization 

% save true endmember matrix
Mth = M;

% load pre-saved estimated endmembers if available, or extract it using VCA
EM_str = ['M0_' ab_maps '_SNR' num2str(SNR) '_' nlmm_str];
% if exist([EM_str '.mat'],'file') == 2
%     load(EM_str)
% else
%     [M0, V, U, Y_bar, endm_proj, Y_proj] = find_endm(r,P,'vca');
%     save(EM_str,'M0')
% end
load(EM_str)
% [M0, V, U, Y_bar, endm_proj, Y_proj] = find_endm(r,P,'vca');
% Sort M0 with respect to real/desired EM signatures to ease the comparison 
% of estimated abundance maps
id = zeros(P,1);
for k = 1:P
    for l = 1:P
        s(l) = 180*acos( (Mth(:,k).')*M0(:,l) /(norm(Mth(:,k))*norm(M0(:,l))) )/pi; 
    end
    [~, id(k)] = min(s);
end
M0 = M0(:,id);


% -------------------------------
if flag_estimate_EM
    M = M0;
    disp('Using VCA...')
else
    M = Mth;
    disp('Using TRUE EMs...')
end
       
LearnrateDecayfactor=0;                  
Results=zeros(120,6);
iteration=1;
for option=1:3
    for numEpochs=[10,30,50] 
        for learnRate=[1e-6,1e-4]
            for L1=[1,10,100] 
                lambda1=L1;
                lambda2=L1;
                lambda5=L1;
                switch option
                    case 1
                         for L2=[1e-1,1e-4,1e-6]
                             lambda3=L2;
                             lambda6=L2;
                             [a_NNHU,W,MLP] = NNHU_autoencoder_customize_1(Y,M,lambda1,lambda2,lambda3,lambda5,lambda6,learnRate,numEpochs,LearnrateDecayfactor);
                             RMSE8 = ErrComput(Ath, a_NNHU);
                             Results(iteration,:)=[RMSE8,option,numEpochs,learnRate,L1,L2];
                             iteration=iteration+1;
                         end
                    case 2
                        L2=nan;
                        [a_NNHU,W,MLP] = NNHU_autoencoder_customize_2(Y,M,lambda1,learnRate,numEpochs);
                        RMSE8 = ErrComput(Ath, a_NNHU);
                        Results(iteration,:)=[RMSE8,option,numEpochs,learnRate,L1,L2];
                        iteration=iteration+1;
                    case 3
                         for L2=[1e-1,1e-4,1e-6]
                             lambda3=L2;
                             lambda6=L2;
                             [a_NNHU,W,MLP] = NNHU_autoencoder_customize_3(Y,M,lambda1,lambda2,lambda3,lambda6,learnRate,numEpochs);
                             RMSE8 = ErrComput(Ath, a_NNHU);
                             Results(iteration,:)=[RMSE8,option,numEpochs,learnRate,L1,L2];
                             iteration=iteration+1;
                         end  
                end 
            end
        end
    end          
end
save('RMSE_results','Results');