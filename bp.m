function [W,e] = bp(W,TrainImages,y,Lrate,index,TestLabels,TestImages,threshold,MAXITER,batchsize)
itr = 0;
while (1)
    %% Main Iteration
    N = size(TrainImages,2);
    A = cell(1,4);
    if index == 1% sigmoid af
        for i = 1:N/batchsize
            tempS3 = 0;
            tempS2 = 0;
            tempS1 = 0;
            for j = 1:batchsize
                A{1} = TrainImages(:,i*batchsize-batchsize+j);
                A{1} = [ones(batchsize,1) A{1}']';
                [A,Z] = forward(A,W,index);
                assistS{3} = diag(ones(size(Z{3})));%10 10
                assistS{2} = diag((ones(257,1)-A{3}).*A{3});% 257 257
                assistS{1} = diag((ones(257,1)-A{2}).*A{2});% 257 257
                e = (y(:,i)-Z{3});% 10 1
                S{3} = -2*assistS{3}*e;% 10 1
                S{2} = assistS{2}*(W{3}'*S{3});% 257 1
                S{1} = assistS{1}*(W{2}'*S{2}(2:end));% 256 1
                tempS3 = tempS3 + Lrate*S{3}*A{3}';
                tempS2 = tempS2 + Lrate*S{2}(2:end)*A{2}';
                tempS1 = tempS1 + Lrate*S{1}(2:end)*A{1}';
            end
            W{3} = W{3}-tempS3/batchsize;% 10 257
            W{2} = W{2}-tempS2/batchsize;
            W{1} = W{1}-tempS1/batchsize;
            accuracy = test(TestImages,W,index,TestLabels);
            fprintf('')
            fprintf(sprintf('%itr iterations\n',itr));
            fprintf(sprintf('error : %f\n %f',norm(e),accuracy));
            itr = itr + 1;
        end
    elseif index == 2% relu af
        for i = 1:N/batchsize
            tempS3 = 0;
            tempS2 = 0;
            tempS1 = 0;
            for j = 1:batchsize
                A{1} = TrainImages(:,i*batchsize-batchsize+j);
                A{1} = [ones(batchsize,1) A{1}']';
                [A,Z] = forward(A,W,index);
                assistS{3} = diag(ones(size(Z{3})));%10 10
                assistS{2} = diag((ones(257,1)));% 257 257
                assistS{1} = diag((ones(257,1)));% 257 257
                e = (y(:,i)-Z{3});% 10 1
                S{3} = -2*assistS{3}*e;% 10 1
                S{2} = assistS{2}*(W{3}'*S{3});% 257 1
                S{1} = assistS{1}*(W{2}'*S{2}(2:end));% 256 1
                tempS3 = tempS3 + Lrate*S{3}*A{3}';
                tempS2 = tempS2 + Lrate*S{2}(2:end)*A{2}';
                tempS1 = tempS1 + Lrate*S{1}(2:end)*A{1}';
            end
            W{3} = W{3}-tempS3/batchsize;% 10 257
            W{2} = W{2}-tempS2/batchsize;
            W{1} = W{1}-tempS1/batchsize;
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