function ctd=irregularsurvey(side,duration,numgliders,speed,divetime)
% 

side=side*1000;

if mod(numgliders,2) ~=0
   error('numgliders must be even')
end

lineloc=10.^linspace(log10(divetime/3600*1000),log10(side-divetime/3600*1000),numgliders/2);

n0=0;
for n=1:numgliders
   if n <= numgliders/2
      xwp0=lineloc(n);
      xwp=xwp0;
      ywp0=rand*side;
      twp0=0;
      ywp=side/2*(sign(rand-0.5)+1);
   else
      ywp0=lineloc(n-numgliders/2);
      ywp=ywp0;
      xwp0=rand*side;
      twp0=0;
      xwp=side/2*(sign(rand-0.5)+1);
   end
   ctd.x(n0+1)=xwp0;
   ctd.y(n0+1)=ywp0;
   ctd.time(n0+1)=twp0;
   ctd.missid(n0+1)=n;
   while twp0 < duration
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
      if xwp == side
         xwp=0;
      elseif xwp == 0
         xwp=side;
      end
      if ywp == side
         ywp=0;
      elseif ywp == 0
         ywp=side;
      end
   end
   n0=length(ctd.missid);
end

%trim time values longer than duration
jj=ctd.time <= duration;
ctd.time=ctd.time(jj);
ctd.x=ctd.x(jj)/1000; %output km
ctd.y=ctd.y(jj)/1000; %output km
ctd.missid=ctd.missid(jj);

