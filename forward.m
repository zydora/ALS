function [A,Z] = forward(A,W,act)
N = size(A{1},2);
if act == 1
    for i = 1:3
        %W{i} = [b{i} W{i}];
        if i == 1 
            Z{i} = W{i}*A{i};
            A{i+1} = sigmoid(Z{i});
        elseif i == 2
            A{i} = ([ones(N,1) A{i}'])';
            Z{i} = W{i}*A{i};
            A{i+1} = sigmoid(Z{i});
        else
            A{i} = ([ones(N,1) A{i}'])';
            Z{i} = W{i}*A{i};
            A{i+1} = Z{i};
            
        end
    end
elseif act == 2
    for i = 1:3
        %W{i} = [b{i} W{i}];
        if i == 1 
            
            Z{i} = W{i}*A{i};
            A{i+1} = relu(Z{i});
        elseif i == 2
            A{i} = ([ones(N,1) A{i}'])';
            Z{i} = W{i}*A{i};
            A{i+1} = relu(Z{i});
        else
            A{i} = ([ones(N,1) A{i}'])';
            Z{i} = W{i}*A{i};
            A{i+1} = Z{i};
            
        end
    end
end
end