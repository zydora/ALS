function [W,e] = bp(W,TrainImages,y,Lrate,index,TestLabels,TestImages,threshold,MAXITER,batchsize)
itr = 0;
while (1)
    %% Main Iteration
    N = size(TrainImages,2);
    L = size(W,2);
    A = cell(1,L+1);
    tempS = cell(1,L);
    if index == 1% sigmoid af
        for i = 1:N/batchsize
            for m = 1:L
                tempS{m} = 0;
            end
            for j = 1:batchsize
                A{1} = TrainImages(:,i*batchsize-batchsize+j);
                A{1} = [ones(batchsize,1) A{1}']';
                [A,Z] = forward(A,W,index);
                assistS{L} = diag(ones(size(Z{L})));
                for m = L-1:-1:1
                assistS{m} = diag((ones(size(A{m+1},1),1)-A{m+1}).*A{m+1});
                end
                e = (y(:,i)-Z{L});
                S{L} = -2*assistS{L}*e;
                S{L-1} = assistS{L-1}*(W{L}'*S{L});
                for m = L-2:-1:1
                S{m} = assistS{m}*(W{m+1}'*S{m+1}(2:end));
                end
                tempS{L} = tempS{L} + Lrate*S{L}*A{L}';
                for m = L-1:-1:1
                tempS{m} = tempS{m} + Lrate*S{m}(2:end)*A{m}';
                end
            end
            for m = L:-1:1
            W{m} = W{m}-tempS{m}/batchsize;
            end
            accuracy = test(TestImages,W,index,TestLabels);
            fprintf('')
            fprintf(sprintf('%itr iterations\n',itr));
            fprintf(sprintf('error : %f\n %f',norm(e),accuracy));
            itr = itr + 1;
        end
    elseif index == 2% relu af
        for i = 1:N/batchsize
            for m = 1:L
                tempS{m} = 0;
            end
            for j = 1:batchsize
                A{1} = TrainImages(:,i*batchsize-batchsize+j);
                A{1} = [ones(batchsize,1) A{1}']';
                [A,Z] = forward(A,W,index);
                assistS{L} = diag(ones(size(Z{L})));
                for m = L-1:-1:1
                assistS{m} = diag((ones(size(A{m+1},1),1)));
                end
                e = (y(:,i)-Z{L});
                S{L} = -2*assistS{L}*e;
                S{L-1} = assistS{L}*(W{L}'*S{L});
                for m = L-2:-1:1
                S{m} = assistS{m}*(W{m+1}'*S{m+1}(2:end));
                end
                tempS{L} = tempS{L} + Lrate*S{L}*A{L}';
                for m = L-1:-1:1
                tempS{m} = tempS{m} + Lrate*S{m}(2:end)*A{m}';
                end
            end
            for m = L:-1:1
            W{m} = W{m}-tempS{m}/batchsize;
            end
            accuracy = test(TestImages,W,index,TestLabels);
            fprintf('')
            fprintf(sprintf('%itr iterations\n',itr));
            fprintf(sprintf('error : %f\n %f',norm(e),accuracy));
            itr = itr + 1;
        end
    end
    normt = [norm(e) normt];
    accuracyt = [accuracy accuracyt];
    if norm(e)<threshold || itr > MAXITER
        break;
    end
    if normt(1)>normt(2)
        Lrate = Lrate/2;% if error begins increasing, break
    end
    if accuracyt(1)<accuracyt(2)
        break;% if accuracy begins decreasing, break
    end
end