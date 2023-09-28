clear;clc;
resp=struct2array(load('resp.mat'));
% use drug as row indix and cell line as column index

resp = resp';
scale1 = resp(~isnan(resp));
num = resp./max(max(scale1),abs(min(scale1)));
Drugsim_fig_mt=struct2array(load('Drugsim_fig_mt.mat'));
Cellsim_probe=struct2array(load('Cellsim_probe.mat'));

drugwisecorr = NaN(size(num,1),1);
drugwise_qt = NaN(size(num,1),1);
drugwiseerr = NaN(size(num,1),1);
drugwiseerr_qt = NaN(size(num,1),1);
drugwiserepn = NaN(size(num,1),1);

i1 = -2;i3 = -2;
%K = 45; lambda_l = 2^i1; lambda_d = 0; lambda_c = 2^i3; max_iter=50; seed=50;
%K = 15; lambda_l = 2^i1; lambda_d = 0; lambda_c = 2^i3; max_iter=500; seed=50;
K = 6; lambda_l =  0.0001; lambda_d = 0.0001; lambda_c = 0.0001; max_iter=1000; seed=50;
curnum = num;
W = ~isnan(curnum);
curnum(isnan(curnum)) = 0;
[U,V] = CMF(W,curnum,Drugsim_fig_mt,Cellsim_probe,lambda_l,lambda_d,lambda_c,K,max_iter,seed);
num = num *max(max(scale1),abs(min(scale1)));
numpred = U*V'*max(max(scale1),abs(min(scale1)));
for d = 1:size(num,1)
    curtemp1 = num(d,:);
    y1 = prctile(curtemp1,75);
    xia1 = find(curtemp1 >= y1);
    y2 = prctile(curtemp1,25);
    xia2 = find(curtemp1 <= y2);
    xia = [xia1,xia2];
    drugwise_qt(d) = corr(curtemp1(xia)',numpred(d,xia)');
    drugwiseerr_qt(d) = sqrt(sum((curtemp1(xia)-numpred(d,xia)).^2)/sum(~isnan(curtemp1(xia))));
    curtemp2 = numpred(d,:);
    curtemp2(isnan(curtemp1)) = [];
    curtemp1(isnan(curtemp1)) = [];
    drugwiserepn(d) = length(curtemp1);  
    drugwisecorr(d) = corr(curtemp1',curtemp2');
    drugwiseerr(d) = sqrt(sum((curtemp1-curtemp2).^2)/sum(~isnan(curtemp1)));
end
save('drugwise_predict1.mat');
