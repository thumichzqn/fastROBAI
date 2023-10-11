clear all; close all; 

rng(1)

A = 2;
T = 1000 ;                   % total number
M = 1 ;                    % batch size
N = 2;                     % offline number
mu = [0.5, 0.5];            % data coverage
w = 1;                    % weight
alpha = 1;

r = [0.7, 0.2];                       % reward
r_opt = max(r);                     
iternum = 1000000;
logbns = log(T) + 0*sqrt(log(T));
Monitor = 100000;


R_off = zeros(2,iternum,N);                 % offline data
R_off(1,:,:) = randn(iternum,N) + r(1);
R_off(2,:,:) = randn(iternum,N) + r(2);

%% eocp Parameter
diary 'mylog3.txt'
rng(1)

Nk = N * mu' .* ones(2,iternum);                          % number of pulls
    
r_bar = mean(R_off,3);   % empirical mean
bk = sqrt(2 * alpha * logbns ./ Nk);                             % bonus
Decision_k = r_bar + bk;
Regret_record = zeros(1,T);
Regret = zeros(1,iternum);

for t=1:T
    
    R = zeros(2, iternum, 1);                   % online sample
    R(1,:,:) = randn(iternum,1) + r(1);
    R(2,:,:) = randn(iternum,1) + r(2);


    if t < 16/(r(1)-r(2))^2 * alpha  * logbns
        Decision_k = r_bar + bk;                            % bonus
                                 
    elseif t-1 < 16/(r(1)-r(2))^2 * alpha  * logbns
        Decision_k = r_bar +(1)* bk;
        [~,a_choose] = max(Decision_k);
        Mistakes = sum(a_choose == 2);
    end

    [~,a_choose] = max(Decision_k); 
    a_choose_mask = double((Decision_k) > flip(Decision_k,1));
    reward = R(:,:,1) .* a_choose_mask;

    Regret = Regret + (r_opt - r(a_choose));

    r_bar = (r_bar .* Nk + reward) ./ (Nk + a_choose_mask);          % empirical mean
    Nk = Nk + a_choose_mask;
    bk = sqrt(2 * alpha * logbns ./ Nk);
    Regret_record(t) = mean(Regret);
    if mod(t, Monitor) == 0     
        diary on
        disp(['t = ', num2str(t), ', Regret: ', num2str(mean(Regret))])
        diary off
    end
    
end

%%
Final_Sub_Pull = Nk(2,:)';
Final_Reg = Regret;

diary on
disp(['Prop:   Regret:', num2str(mean(Final_Reg)), ',STD:', num2str(std(Final_Sub_Pull))])
diary off
