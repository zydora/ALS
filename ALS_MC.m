function [W,e] = ALS_MC(W,A,Z,y,index,mc,fm,TrainImages,TrainLabels,TestImages,TestLabels,N,MAXITER,threshold)
%% ALS Iteration
normt = 0;
itr = 1;
L = size(W,2);
while (1 )
    %sweep from ltr
    if fm == 1% if update W,Z during forward procession
        for i = 1:L
            W{i} = Z{i}/A{i};
            Z{i} = W{i}*A{i};
        end
    elseif fm == 2% if not update W,Z
        [A,Z] = forward(A,W,index);
        Z{L} = y;
    end
    %sweep from rtl
    for i = L:-1:1
        W{i} = Z{i}/A{i};
        if i ~=L
            if index == 1
                temp = max(0,min(A{i+1}(2:end,:),1));
                %tic;Z{i} = MC(temp,mc);toc;
                Z{i} = log(Z{i}./(1-Z{i}));
            elseif index == 2
                temp = relu(A{i+1}(2:end,:));
                for j = 1:size(temp,2)
                    tic;Z{i}(:,j) = reshape(MC(reshape(temp(:,j),[sqrt(size(temp,1)),sqrt(size(temp,1))]),mc),[size(temp,1),1]);toc;
                end
            end
        end
    end
    
    e = norm(y-W{L}*A{L})/N;%accuracy error
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