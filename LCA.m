function [W,sig,E] = LCA(X,C,d,T,B)
    P = @(w,x,y,s) (sqrt(1.0/(1.0+2*s^2))^d)*exp(-0.5*(norm((x-y)*w)^2)/(1.0+2.0*s^2));
    W = 1e-2*randn(size(X,2),d);
    nW = 1e-4*ones(size(W));
    nu = 0.99;
    E = zeros(T,1);
    sig = 10.0;

    for t = 1:T
        disp(t);
        idxs = (1:size(X,1))';
        rper = randperm(size(idxs,1));
        idxs_x = idxs(rper(  1:  B));
        idxs_y = idxs(rper(B+1:2*B));

        x = X(idxs_x,:)+1e-4*randn(B,size(X,2));
        cx= C(idxs_x); 
        y = X(idxs_y,:)+1e-4*randn(B,size(X,2));
        cy= C(idxs_y);
        
        r = y-x;
        
        zx = zeros(B,d);
        zy = zeros(B,d);
        eps = zeros(B,1);
        for b = 1:B
            l = (sig^2)/(1.0+2.0*sig^2);
            if cx(b) == cy(b)
                zx(b,:) = (x(b,:)+l*r(b,:))*W;
                zy(b,:) = (y(b,:)-l*r(b,:))*W;
                eps(b) = d*sig^2*(1.0-l);
            else
                e = P(W,x(b,:),y(b,:),sig);
                v = e/(1.0-e);
                zx(b,:) = (x(b,:)-v*l*r(b,:))*W;
                zy(b,:) = (y(b,:)+v*l*r(b,:))*W;
                eps(b) = d*sig^2*(1.0+v*l);
            end
        end
        dW = (x'*(x*W-zx)+y'*(y*W-zy))/(d*B);
        nW = nu*nW + (1.0-nu)*dW.^2;
        W = W - 1e-3*dW./(1e-8+sqrt(nW));
        E(t) = 0.5*(norm(zx-x*W,'fro')^2+norm(zy-y*W,'fro'))/(d*B);
        sig = sqrt(mean(eps)/d+E(t));
        disp(E(t));
    end
end