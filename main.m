DIR = dir('train_set');
X = [];
C = [];
for i = 3:length(DIR)
    dirname = strcat('train_set/',DIR(i).name);
    disp(dirname);
    AUX_DIR = dir(dirname);
    for j = 3:length(AUX_DIR)
        auxfilename = strcat(dirname,'/',AUX_DIR(j).name);
        aux = double(imread(auxfilename))/255.0;
        d = size(aux);
        X = [X ; reshape(aux,[1,numel(aux)])];
        C = [C ; i-3];
    end
end

mX = mean(X,1);
Y = X - repmat(mX,[size(X,1),1]);
sX = mean(Y.^2,1) + 1e-6;
Y = Y ./ repmat(sX,[size(X,1),1]);
[W,sig,E] = LCA(Y,C,3,2000,25);
Z = Y*W;

figure;
plot(log(E+1e-8));

figure;
a = mX;
a = (a-min(a))/(max(a)-min(a));
subplot(1,5,1);
imshow(kron(reshape(a,d),ones(8,8)));
a = sX;
a = (a-min(a))/(max(a)-min(a));
subplot(1,5,2);
imshow(kron(reshape(a,d),ones(8,8)));
for i = 1:3
    a = W(:,i)';
    a = (a-min(a))/(max(a)-min(a));
    subplot(1,5,i+2);
    imshow(kron(reshape(a,d),ones(8,8)));
end

figure;
for i = 3:length(DIR)
    disp(DIR(i).name);
    idx = (C == i-3);
    scatter3(Z(idx,1),Z(idx,2),Z(idx,3),7,C(idx),'filled','MarkerEdgeColor','k','DisplayName',DIR(i).name);
    if i == 3
        hold on;
    end
end
colormap(colorcube);
legend(gca,'show');
hold off;
