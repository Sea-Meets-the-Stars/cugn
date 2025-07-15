function ctd=randomsurvey(side,duration,numgliders,speed,divetime)
% 

side=side*1000;

n0=0;
for n=1:numgliders
   xwp0=side./2;
   ywp0=side./2;
   twp0=0;
   ctd.x(n0+1)=xwp0;
   ctd.y(n0+1)=ywp0;
   ctd.time(n0+1)=twp0;
   ctd.missid(n0+1)=n;
   while twp0 < duration
      xwp=rand*side;
      ywp=rand*side;
      lengthsection=sqrt((xwp-xwp0).^2+(ywp-ywp0).^2);
      timesection=lengthsection./speed;
      ndives=round(timesection./divetime);
      heading=atan2(ywp-ywp0,xwp-xwp0);
      xdive=speed*divetime*cos(heading);
      ydive=speed*divetime*sin(heading);
      xx=linspace(xwp0+xdive,xwp0+ndives*xdive,ndives)';
      yy=linspace(ywp0+ydive,ywp0+ndives*ydive,ndives)';
      tt=linspace(twp0+divetime,twp0+ndives*divetime,ndives)';
      ctd.x=[ctd.x; xx];
      ctd.y=[ctd.y; yy];
      ctd.time=[ctd.time; tt];
      xwp0=ctd.x(end);
      ywp0=ctd.y(end);
      twp0=ctd.time(end);
      ctd.missid=[ctd.missid; n*ones(size(xx))];
   end
   n0=length(ctd.missid);
end

%trim time values longer than duration
jj=ctd.time <= duration;
ctd.time=ctd.time(jj);
ctd.x=ctd.x(jj)/1000; %output km
ctd.y=ctd.y(jj)/1000; %output km
ctd.missid=ctd.missid(jj);

