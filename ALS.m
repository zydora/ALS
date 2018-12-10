function [W,e] = ALS(W,A,Z,y,index,mc,fm,TrainImages,TrainLabels,TestImages,TestLabels,N,MAXITER,threshold)
normt = 0;
itr = 1;
while (1 )
    %sweep from ltr
    for i = 1:3
            %E = eye(size(A{i},1));
            %E(1,1) = 0;
            W{i} = Z{i}/A{i};
            %Z{i} = W{i}*A{i};
            if i~= 3
            A{i+1} = ([ones(N,1) sigmoid(W{i}*A{i})'])';
            end
    end
    %sweep from rtl
    for i = 3:-1:1
        E = eye(size(A{i},1));
        W{i} = Z{i}/A{i};
        if i ~=3
            if index == 1
                temp = max(0.01,min(0.99,A{i+1}(2:end,:)));
                %temp = A{i+1}(2:end,:);
                temp = MC(temp,mc);
                Z{i} = log(temp./(1-temp));   
            elseif index == 2
                temp = relu(A{i+1}(2:end,:));
                Z{i} = MC(temp,mc);
            end
        end
    end
    e = norm(y-W{3}*A{3})/N;
    accuracy = test(TestImages,W,index,TestLabels);
    fprintf('')
    fprintf(sprintf(' %itr iterations ',itr));
    fprintf(sprintf(' accuracy : %f\n',accuracy));
    itr = itr + 1;
    if itr > MAXITER || e< threshold
        break;
    end
end
end
