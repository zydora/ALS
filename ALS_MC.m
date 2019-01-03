function [W,e] = ALS_MC(W,A,Z,y,index,mc,TrainImages,TrainLabels,TestImages,TestLabels,N,MAXITER,threshold)
%% ALS Iteration
itr = 1;
L = size(W,2);lambda = 0;
while (1 )tic;
    %sweep from ltr
    for i = 1:L-1
        W{i} = Z{i}/A{i};
        %             W{i} = Z{i}*A{i}'*pinv(A{i}*A{i}'+lambda*eye(size(A{i},1)));
        if index == 1
            A{i+1} = [ones(N,1) sigmoid(W{i}*A{i})']';
        elseif index == 2
            A{i+1} = [ones(N,1) relu(W{i}*A{i})']';
        elseif index == 0
            A{i+1} = [ones(N,1) (W{i}*A{i})']';
        end
    end
    W{L} = Z{L}/A{L};
    %         W{L} = Z{L}*A{L}'*pinv(A{L}*A{L}'+lambda*eye(size(A{L},1)));
    A{L+1} = W{L}*A{L};
    %sweep from rtl
    for i = L:-1:1
        W{i} = Z{i}/A{i};
        %         W{i} = Z{i}*A{i}'*pinv(A{i}*A{i}'+lambda*eye(size(A{i},1)));
        if i ~=L
            if index == 1
                temp = max(0,min(A{i+1}(2:end,:),1));
                %tic;Z{i} = MC(temp,mc);toc;
                Z{i} = log(temp./(1-temp));
            elseif index == 2
                %temp = A{i+1}(2:end,:);
                temp = relu(A{i+1}(2:end,:));
                %                 for j = 1:size(temp,2)
                %                     Z{i}(:,j) = reshape(MC(reshape(temp(:,j),[sqrt(size(temp,1)),sqrt(size(temp,1))]),mc),[size(temp,1),1]);
                %                 end
                Z{i} = temp;
                %A{i} = pinv(W{i})*tZ{i};
            elseif index == 0
                temp = A{i+1}(2:end,:);
                %A{i} = pinv(W{i})*temp;
                Z{i} = temp;
            end
        end
    end
    toc;
    e = norm(y-W{L}*A{L})/N;%accuracy error
    accuracy = test(TestImages,W,index,TestLabels);
    fprintf('')
    fprintf(sprintf(' %itr iterations \n',itr));
    fprintf(sprintf(' test accuracy : %f\n',accuracy));
    disp('debug')
    accuracy = test(TrainImages,W,index,TrainLabels);
    fprintf(sprintf('train accuracy: %f\n',accuracy));
    
    itr = itr + 1;
    if itr > MAXITER
        break;
    end
end
end