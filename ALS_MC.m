function [W,e] = ALS_MC(W,A,Z,y,index,mc,fm,TrainImages,TrainLabels,TestImages,TestLabels,N,MAXITER)
%% ALS Iteration
normt = 0;
itr = 1;
while (1 )
    %sweep from ltr
    if fm == 1% if update W,Z during forward procession
        for i = 1:3
            W{i} = Z{i}/A{i};
            Z{i} = W{i}*A{i};
        end
    elseif fm == 2% if not update W,Z
        [A,Z] = forward(A,W,index);
        Z{3} = y;
    end
    %sweep from rtl
    for i = 3:-1:1
        W{i} = Z{i}/A{i};
        if i ~=3
            if index == 1
                temp = max(0,min(A{i+1}(2:end,:),1));
                %tic;Z{i} = MC(temp,mc);toc;
                Z{i} = log(Z{i}./(1-Z{i}));
            elseif index == 2
                temp = relu(A{i+1}(2:end,:));
                tic;Z{i} = MC(temp,mc);toc;
                error(i) = norm(temp - Z{i});
            end
        end
    end
    
    e = norm(y-W{3}*A{3})/N;%accuracy error
    error %error between M-Z and Z
    accuracy = test(TestImages,W,index,TestLabels);
    fprintf('')
    fprintf(sprintf(' %itr iterations \n',itr));
    fprintf(sprintf(' accuracy : %f\n',accuracy));
    
    itr = itr + 1;
    if itr > MAXITER
        break;
    end
end
end