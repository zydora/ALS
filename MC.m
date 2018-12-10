function fx = MC(x,mc)
n1 = size(x,1); n2 = size(x,2); r = size(x,1);
sizex = size(x);
X = x(:);
IDX = find(X ~= 0);
M = opRestriction(n1*n2,IDX);
y = M(X,1);
if mc == 1
    XRec = NonCVX_MC(y,M,[n1,n2],0.1);
elseif mc == 2
    XRec = IHT_MC(y,M,[n1,n2]);
elseif mc == 3
    XRec = IST_MC(y,M,[n1,n2]);
elseif mc == 0
    XRec = x;
end
fx = XRec;
end
