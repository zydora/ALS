function z = lassoals(A,x,b)
%https://www.mathworks.com/help/stats/lasso.html#d120e453959
%use the method of ADMM
%A = A', x = W{3}', b = y'
%A = A(:), x = W{3}(:), b = y(:)
threshold = 10e-3;lambda = 0.01;
theta = 0.01;
e = 1;z = x;u = 0;
while e>threshold
x = (A'*A + theta*eye(size(A,2)))\(A'*b + theta*(z-u));
z = S_para(x+u,lambda/theta);
u = u + x - z;
e = norm(x-z);
end
end