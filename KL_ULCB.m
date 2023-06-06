function [UCB,LCB] = KL_ULCB(r_bar, Nk, logbns, maxiter)
if (nargin<4)
    maxiter = 20;
end
epsilon = 1e-7;
indicator = (r_bar == 1);
r_bar = (1-indicator) .* r_bar + (1-epsilon) .* indicator;

indicator = (r_bar ==0);
r_bar = (1-indicator) .* r_bar + epsilon .* indicator;

[A,iternum] = size(r_bar);

% Find UCB
up = 1*ones(A,iternum);
low = r_bar;

for t_iter = 1:maxiter
    mid = (up + low)/2;
    KL_mid = r_bar .* log(r_bar./mid) + (1-r_bar).* log((1-r_bar)./(1-mid));
    up = (KL_mid > logbns ./ Nk).* mid + (KL_mid <= logbns ./ Nk) .* up;
    low = (KL_mid >= logbns ./ Nk).* low + (KL_mid < logbns ./ Nk) .* mid;
end
UCB = (up + low)/2;

% Find LCB
up = r_bar;
low = zeros(A,iternum);

for t_iter = 1:maxiter
    mid = (up + low)/2;
    KL_mid = r_bar .* log(r_bar./mid) + (1-r_bar).* log((1-r_bar)./(1-mid));
    low = (KL_mid > logbns ./ Nk).* mid + (KL_mid <= logbns ./ Nk) .* low;
    up = (KL_mid >= logbns ./ Nk).* up + (KL_mid < logbns ./ Nk) .* mid;
end
LCB = (up + low)/2;

end