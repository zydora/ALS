function [A, B, C] = DBTF(X,R,T,L,N)
% X three way tensor,
% R rank
% T maximum number of iterations
% L sets of initial factor matrices
% N number of partitions
[X1] = Unfold(X,1);[ X2] = Unfold(X,2); [X3] = Unfold(X,3);
pX1 = partition(X1, N)
pX2 = partition(X2, N)
pX3 = partition(X3, N)
for t = 1:T
    if t == 1
        FM = initialize_FM(L);
        for i = 1:L
            FM{i} = UpdateFactors(FM{i});
        end
        [A, B, C] = smin(FM)
    else
        [A, B, C] = UpdateFactors(A,B,C);
    end
    if error < threshold
        break
    end
end
end

function [A, B, C] = UpdateFactors(A, B, C)
% Update A to minimize "X1-A*C*B"
A = UpdateFactor(pX1,A,C,B);
B = UpdateFactor(pX2,B,C,A);
C = UpdateFactor(pX3,C,B,A);
end

function [pX] = partition(X,N)
% split
Q = size(X,2);
min = floor(Q/N);
for i = 1:N
    if i ~= N
    p(i) = X(:,i*min-min+1:i*min);
    else
        p(N) = X(:,(i-1)*min+1:end);
    end
end
pX = p(1);
for i = 2:N
    pX = [pX p(i)];
end
% Figure 5
% cache pX
end

function A = UpdateFactor(pX,A,C,B)
[pX, Mx, V] = CacheRowSummations(pX, Ms, V);
for c = 1:R %column iter
    for r = 1:P %row iter
        for a = 1:2
            for i = 1:N
                for b = 1:block
                    k = ar;% modify
                    % Cached Boolean summation of the rows
                    % Compute the error between the fetched row
                end
            end
        end
    end
    % errors
    for r = 1:P
        % update are
    end
end
end

function [pX ,Mx, V] = CacheRowSummations(pX, Ms, V)
for i = 1:N
    % distributed
end
for b = 1:block
    % figure 5
    % cache
end
end