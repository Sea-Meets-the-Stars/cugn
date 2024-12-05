function sf=randomplots(ctd,dtimemax,numbins)
% sf=structurefunction(ctd,iz) calculates structure function given ctd
% structure and the depth index iz and maxdtime in hours
%

% get only my data
x=ctd.x;
y=ctd.y;
time=ctd.time;

% calculate deltas
dx=x-x';
dy=y-y';
dtime=time-time';
dr=sqrt(dx.^2+dy.^2);

%plot locaations of profiles
subplot(1,3,1);
cla;
hold on;
for nn=1:ctd.missid(end)
   plot(ctd.x(ctd.missid == nn),ctd.y(ctd.missid == nn),'-x');
end
set(gca,'dataaspectratio',[1 1 1]);
hold off;
xlabel('x (km)');
ylabel('y (km)');

% plot histogram of separations
subplot(1,3,2);
jj=dtime > 0 & dtime < dtimemax;
bin=linspace(log10(min(dr(jj))),log10(max(dr(jj))),numbins);
binr=10.^bin;
histogram(dr(jj),binr)
set(gca,'xscale','log')
xlabel('\Deltar (km)');
ylabel('Count');

% plot histogram of separations
subplot(1,3,3);
bin=linspace(min(dr(jj)),max(dr(jj)),numbins);
histogram(dr(jj),bin)
xlabel('\Deltar (km)');
ylabel('Count');

sf.binr=binr;
