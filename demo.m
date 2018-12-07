% soft SVD

%% 2.1
[a,b] = size(W{3});
u = randn(a,b);
U = orth(u')';
D = eye(b);
W = U*D;
A = pinv(D*D+numda*eye(size(D)))*D*U'*Z{3};
[V,D,~] = svd(A'*D);
D = D.^(1/2);
A = V*D;
W = Z{3}*V*D*pinv(D*D+numda*eye(size(D)));
[U,D,~] = svd(W*D);
D = D.^(1/2);
W = U*D;

%% 3.1
u = randn(a,b);%1
U = orth(u')';
D = eye(b);
V = zeros(N,b);
W{i} = U*D;
A{i} = (V*D)';
Xstar =  (Z{3} - W{i}*A{i})+W{i}*A{i};%2
A{i} = (pinv(D'*D+numda*eye(size(D,1)))*D*U'*Xstar);
[V,D,~] = svd(A{3}'*D,0);
D = D.^(1/2);
A{i} = (V*D)';
Xstar =  (Z{3}' - A{i}'*W{i}')+A{i}'*W{i}';%3
W{i} = (pinv(D'*D+numda*eye(size(D,1)))*D*V'*Xstar)';
[U,D,~] = svd((W{i}*D));
D = D.^(1/2);
U = U;
W{i} = (U*D);
U = V;
[~,V,D] = svd(Xstar'*V);
V = V';
D = max(D-numda*eye(size(D,1)),0);


%% 4.1
Xstar = (Z{3}-W{3}*A{3}) + W{3}*A{3};
W{3} = Xstar*A{3}*pinv(X{3}'*X{3} + numda*eye(size(X{3},2)));
Xstar = ((Z{3}-W{3}*A{3})+W{3}*A{3})';
A{3} = Xstar*W{3}*pinv(W{3}'*W{3} + numda*eye(size(W{3},2)));
