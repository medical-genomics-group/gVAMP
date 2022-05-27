
figure(3)
pd = fitdist(12000^2*oo','kernel','Width',1);
y = pdf(pd,12000^2*oo);
plot(12000^2*oo,y,'Color','r','LineStyle','-')
xlim([0 10000])
hold on;

figure(6)
pd_sim = fitdist(12000^2*oo_sim','kernel','Width',1);
y_sim = pdf(pd_sim,12000^2*oo_sim);
plot(12000^2*oo_sim,y_sim,'Color','b','LineStyle','-')
histogram(12000^2*oo_sim / 382390,300,'Normalization','probability')
( 1 - sqrt(6543 / 382390) )^2
( 1 + sqrt(6543 / 382390) )^2
ylabel('pdf', 'FontSize', 24)
%xlabel('', 'FontSize', 24)
title('histogram of squared singular values distribution - simulated','FontSize', 24);


figure(4)
histogram(12000^2*oo,300,'Normalization','probability', 'BinLimits',[0,20000])
histogram(12000^2*oo / 382390,300,'Normalization','probability', 'BinLimits', [0,20])
ylabel('pdf', 'FontSize', 24)
%xlabel('', 'FontSize', 24)
title('pdf of squared singualar values distribution - real','FontSize', 24);

figure(5)
plot(12000^2*(0:0.01:1-0.01),oo_est')
