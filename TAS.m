clear all; close all; 

rng(1)

A = 4;
T = 100000 ;                   % total number
M = 1 ;                    % batch size
N = 4;                     % offline number
mu = [0.25, 0.25,0.25,0.25];            % data coverage
w = 1;                    % weight
alpha = 1;

r = [0.7, 0.2, 0.2, 0.2];                       % reward
r_opt = max(r);
Deltamin = max(r) - max(r(r<max(r)));
iternum = 1000;
logbns = log(T) + 0.2*sqrt(log(T));
Monitor = 10000;


R_off = zeros(A,iternum,N);                 % offline data
R_off(1,:,:) = double(rand(iternum,N) < r(1));
R_off(2,:,:) = double(rand(iternum,N) < r(2));
R_off(3,:,:) = double(rand(iternum,N) < r(3));
R_off(4,:,:) = double(rand(iternum,N) < r(4));
%% BAI

if 1

rng(1)

Nk_BAI = N * mu' .* ones(A,iternum);                          % number of pulls
    
r_bar_BAI = mean(R_off,3);   % empirical mean
bk_BAI = sqrt(2 * alpha * log(T) ./ Nk_BAI);                             % bonus
Decision_k_BAI = r_bar_BAI + bk_BAI;
Regret_record_BAI = zeros(1,T);
Regret_BAI = zeros(1,iternum);
Stop_BAI = zeros(A,iternum);

for t=1:T
    
    R = zeros(A, iternum, 1);                   % online sample
    R(1,:,:) = double(rand(iternum,1) < r(1));
    R(2,:,:) = double(rand(iternum,1) < r(2));
    R(3,:,:) = double(rand(iternum,1) < r(3));
    R(4,:,:) = double(rand(iternum,1) < r(4));
    
%     mu1_hat = repmat(Nk_BAI(1,:),3,1) ./ (repmat(Nk_BAI(1,:),3,1) + Nk_BAI(2:4,:)) .* repmat(r_bar_BAI(1,:),3,1)...
%         + Nk_BAI(2:4,:) ./ (repmat(Nk_BAI(1,:),3,1) + Nk_BAI(2:4,:)) .* r_bar_BAI(2:4,:);
%     mu2_hat = repmat(Nk_BAI(2,:),3,1) ./ (repmat(Nk_BAI(2,:),3,1) + Nk_BAI([1,3,4],:)) .* repmat(r_bar_BAI(2,:),3,1)...
%         + Nk_BAI([1,3,4],:) ./ (repmat(Nk_BAI(2,:),3,1) + Nk_BAI([1,3,4],:)) .* r_bar_BAI([1,3,4],:);
%     mu3_hat = repmat(Nk_BAI(3,:),3,1) ./ (repmat(Nk_BAI(3,:),3,1) + Nk_BAI([1,2,4],:)) .* repmat(r_bar_BAI(3,:),3,1)...
%         + Nk_BAI([1,2,4],:) ./ (repmat(Nk_BAI(3,:),3,1) + Nk_BAI([1,2,4],:)) .* r_bar_BAI([1,2,4],:);
%     mu4_hat = repmat(Nk_BAI(4,:),3,1) ./ (repmat(Nk_BAI(4,:),3,1) + Nk_BAI([1,2,3],:)) .* repmat(r_bar_BAI(4,:),3,1)...
%         + Nk_BAI([1,2,3],:) ./ (repmat(Nk_BAI(4,:),3,1) + Nk_BAI([1,2,3],:)) .* r_bar_BAI([1,2,3],:);
% 
% 
%     Z1_all = repmat(Nk_BAI(1,:),3,1) .* d(repmat(r_bar_BAI(1,:),3,1), mu1_hat) + Nk_BAI(2:4,:) .* d(r_bar_BAI(2:4,:), mu1_hat);
%     Z2_all = repmat(Nk_BAI(2,:),3,1) .* d(repmat(r_bar_BAI(2,:),3,1), mu2_hat) + Nk_BAI([1,3,4],:) .* d(r_bar_BAI([1,3,4],:), mu2_hat);
%     Z3_all = repmat(Nk_BAI(3,:),3,1) .* d(repmat(r_bar_BAI(3,:),3,1), mu3_hat) + Nk_BAI([1,2,4],:) .* d(r_bar_BAI([1,2,4],:), mu3_hat);
%     Z4_all = repmat(Nk_BAI(4,:),3,1) .* d(repmat(r_bar_BAI(4,:),3,1), mu4_hat) + Nk_BAI([1,2,3],:) .* d(r_bar_BAI([1,2,3],:), mu4_hat);
%     
    
    Z1_all = (repmat(r_bar_BAI(1,:),3,1) - r_bar_BAI(2:4,:)).^2 .* (repmat(Nk_BAI(1,:),3,1) + Nk_BAI(2:4,:));
    Z2_all = (repmat(r_bar_BAI(2,:),3,1) - r_bar_BAI([1,3,4],:)).^2 .* (repmat(Nk_BAI(2,:),3,1) + Nk_BAI([1,3,4],:));
    Z3_all = (repmat(r_bar_BAI(3,:),3,1) - r_bar_BAI([1,2,4],:)).^2 .* (repmat(Nk_BAI(3,:),3,1) + Nk_BAI([1,2,4],:));
    Z4_all = (repmat(r_bar_BAI(4,:),3,1) - r_bar_BAI([1,2,3],:)).^2 .* (repmat(Nk_BAI(4,:),3,1) + Nk_BAI([1,2,3],:));
    
    Z(1,:) = min(Z1_all);
    Z(2,:) = min(Z2_all);
    Z(3,:) = min(Z3_all);
    Z(4,:) = min(Z4_all);
    Z_end = max(Z);
    
    
    Stopping_BAI = repmat( double( Z_end > 12*log(T/t) ), A,1) .* (1 - Stop_BAI);
    Stop_BAI = Stop_BAI + Stopping_BAI;
    
    Prob = [0.3625, 0.2125 ,0.2125 ,0.2125];
    ArmPull_k_BAI = randsrc(1,iternum,[1,2,3,4; Prob ]);
    IndexPull_k_BAI = [0:A:A*(iternum-1)] + ArmPull_k_BAI;
    
    ArmPull_k_BAI_mask = zeros(A,iternum);
    ArmPull_k_BAI_mask(IndexPull_k_BAI) = 1;
    
    Decision_k_BAI = Stop_BAI .* Decision_k_BAI + (1 - Stop_BAI).* ArmPull_k_BAI_mask;
    Decision_k_BAI = Stopping_BAI .* r_bar_BAI + (1 - Stopping_BAI).* Decision_k_BAI;
    

    [~,a_choose_BAI] = max(Decision_k_BAI); 
    a_choose_mask_BAI = (Decision_k_BAI ==(max(Decision_k_BAI)));
    reward_BAI = R(:,:,1) .* a_choose_mask_BAI;

    Regret_BAI = Regret_BAI + (r_opt - r(a_choose_BAI));

    r_bar_BAI = (r_bar_BAI .* Nk_BAI + reward_BAI) ./ (Nk_BAI + a_choose_mask_BAI);          % empirical mean
    Nk_BAI = Nk_BAI + a_choose_mask_BAI;
    bk_BAI = sqrt(2 * alpha * log(T) ./ Nk_BAI); 
    Regret_record_BAI(t) = mean(Regret_BAI);
    if mod(t, Monitor) == 0     
        diary on
        disp(['t = ', num2str(t), ', Regret: ', num2str(mean(Regret_BAI))])
        diary off
    end
    
end

Final_Sub_Pull_BAI = Nk_BAI(2,:)';
Final_Reg_BAI = Regret_BAI;

diary on
disp(['BAI-ETC:   Regret:', num2str(mean(Final_Reg_BAI)), ',STD:', num2str(std(Final_Sub_Pull_BAI))])
diary off
end
%%
save('TAS_ETC.mat','Regret_record_BAI');

%%
function y = d(mu1, mu2)
    y = mu1 .* log(mu1 ./ mu2) + (1-mu1) .* log((1-mu1) ./ (1-mu2));
end

