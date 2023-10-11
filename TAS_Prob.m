r = [0.7, 0.2];
left = 0.0001;
right = d(r(1), r(2)) - 0.0001;
while abs(right - left) > 0.0001
    mid = (left + right) / 2;
    f = F(mid);
    if f >1
        right = mid;
    else
        left = mid;
    end
end
y_opt = (left + right)/2;
w = [1, invg(y_opt), invg(y_opt), invg(y_opt)] / (1 + 3 * invg(y_opt));
Opt_Prob = w;


function f = F(y)
    r = [0.7, 0.2];
    r_mid = (r(1) + invg(y) * r(2)) / (1 + invg(y));
    f = 3 * d(r(1), r_mid) / d(r(2), r_mid);
end

function y = d(mu1, mu2)
    y = mu1 .* log(mu1 ./ mu2) + (1-mu1) .* log((1-mu1) ./ (1-mu2));
end

function y = g(x)
    r = [0.7, 0.2];
    alpha = 1 / (1 + x);
    r_mid = alpha * r(1) + (1 - alpha) * r(2);
    d1 = r(1) * log(r(1) / r_mid) + (1 - r(1)) * log((1 - r(1)) / (1 - r_mid));
    d2 = r(2) * log(r(2) / r_mid) + (1 - r(2)) * log((1 - r(2)) / (1 - r_mid));
    I = alpha *d1 + (1 - alpha) * d2;
    y = (1 + x) * I;
end

function x = invg(y)
    right = 1;
    left = 0.001;
    while g(right) < y
        right = 2 * right;
    end
    while abs(right - left) > 10^(-4)
        mid = (right + left) / 2;
        y_mid = g(mid);
        if y_mid < y
            left = mid;
        else
            right = mid;
        end
    end
    x = (right + left) / 2;
end

