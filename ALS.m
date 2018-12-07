%ALS
while (1 )
    for i = 3:-1:1
    %% sweep from rtl
    
    sumW = zeros(size(A{i},1));
    for m = 1:N
        sumW = sumW + A{i}(:,m)*A{i}(:,m)';
    end
    sumA = zeros(size(A{i},1));
    for u = 1:size(W{i},1)
        sumA = sumA + W{i}(u,:)'*W{i}(u,:);
    end
    tempW = pinv(sumW + numda*ones(size(A{i},1)));
    tempA = pinv(sumA + numda*ones(size(A{i},1)));
%     
%     
%     for u = 1:size(W{i},1)
%         sumWW = zeros(size(A{i},1),1);
%         for m = 1:N
%             sumWW = sumWW + Z{i}(u,m)*A{i}(:,m);
%         end
%         W{i}(u,:) = (tempW*sumWW)';
%     end
%     for m = 1:N
%         sumAA = zeros(size(A{i},1),1);
%         for u = 1:size(W{i},1)
%             sumAA = sumAA + Z{i}(u,m)*W{i}(u,:)';
%         end
%         A{i}(:,m) = tempA*sumAA;
%     end
%     e = norm(Z{i} - W{i}*A{i})
%     error = [e error];
%     itr = itr+1;
tempWW = zeros(size(A{i},1),1);
tempAA = zeros(size(W{i},2),1);

        %% W [u,~] [~,m]
        for u = 1:size(W{i},1)
            sumWW = zeros(size(A{i},1),1);
            for m = 1:N
                sumWW = sumWW + Z{i}(u,m)*A{i}(:,m);
            end
            tempWW = [tempWW sumWW];
        end
        W{i} = (tempW*tempWW(:,1:end-1))';
        %% A
        for m = 1:N
            sumAA = zeros(size(A{i},1),1);
            for u = 1:size(W{i},1)
                sumAA = sumAA + Z{i}(u,m)*W{i}(u,:)';
            end
            tempAA = [tempAA sumAA];
        end
        A{i} = (tempA*tempAA(:,1:end-1));
        if i ~=1
            temp = A{i}(2:end,:);
            %Z{i} = log(temp./(1-temp));
            Z{i-1} = temp;
            [Z{i-1},ier] = MatrixCompletion(Z{i-1},abs(sign(Z{i-1})),10,'nuclear',10,0.1,0);
        end
        e = norm(Z{i} - W{i}*A{i})
        error = [e error];
        itr = itr + 1
    end
    A{1} = A{1}(2:end,:);
    [A,Z] = forward(A,W,2);
Z{3} = y(:,1:N);
        
end
