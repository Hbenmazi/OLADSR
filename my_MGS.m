function U = my_MGS(U, K)
[m,n] = size(U);
U = [U zeros(m,K-n)];
for i = n+1:K
    v = rand(m,1);
    v = v-mean(v);
    for j = 1: i-1
        v = v-(U(:,j)'*v)*U(:,j);
    end
%    v = l2_normalize(v');
    norm_value = norm(v);
    v = v / norm_value;
    U(:,i) = v';
end