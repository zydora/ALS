function r = S_para(a1,a2)
if a1>a2
    r = a1-a2;
elseif a1<a2
    if a1>-a2
    r = 0;
    else
    r = a1+a2;
    end
end
end