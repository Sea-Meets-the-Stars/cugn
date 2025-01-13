  % function [winddata windcoeff windfitdata] = ecmwfWind
% function [winddata windcoeff windfitdata] = ecmwfWind
% ERA Monthly averaged reanalysis
% https://data.marine.copernicus.eu/product/WIND_GLO_PHY_CLIMATE_L4_MY_012_003/download
% https://data.marine.copernicus.eu/product/WIND_GLO_PHY_CLIMATE_L4_MY_012_003/services
% Global Ocean Monthly Mean Sea Surface Wind and Stress from Scatterometer and Model. E.U. Copernicus Marine Service Information (CMEMS). Marine Data Store (MDS). DOI: 10.48670/moi-00181 (Accessed on 18 Nov 2024)
% DOI 10.48670/moi-00181
% 2017-2024
% 126 to 116 W
% 30 to 36 N


ncload2('../ecmwf/cmems_obs-wind_glo_phy_my_l4_P1M_1731978800651.nc');

ecmwf.dn = ut2dn(time);
ecmwf.lon = longitude;
ecmwf.lat = latitude;
ecmwf.taux = eastward_stress;
ecmwf.tauy = northward_stress;

f = find(ecmwf.dn >= datenum(2019, 1, 1) & ecmwf.dn <= datenum(2023, 12, 31));
ecmwf.dn = ecmwf.dn(f);
ecmwf.taux = ecmwf.taux(f, :, :);
ecmwf.tauy = ecmwf.tauy(f, :, :);

load ekmanInfo
theta = info.theta - 90;
phi = info.phi - pi/2;

% [Lon Lat] = meshgrid(ecmwf.lon, ecmwf.lat);
% [x y] = lonlat2xyRot(Lon, Lat, info.lono, info.lato, 0);
% ecmwf.xkm = x/1000;
% ecmwf.ykm = y/1000;

% as in rotateuv.m
[ecmwf.tauacross ecmwf.taualong] = rotateComplexUV(ecmwf.taux, ecmwf.tauy, -phi);

%%

maxharm = 2;

dn = [datenum(2023, 1, 1):datenum(2023, 12, 31)]';
taucoeff = harmonicFitVar(ecmwf, {'taux', 'tauy', 'tauacross', 'taualong'}, maxharm);
taufit = harmonicReconVar(taucoeff, maxharm, {'taux', 'tauy', 'tauacross', 'taualong'}, dn);

%%

disp('saving ecmwfStress: ecmwf taucoeff taufit ')
matfile = ['../ecmwf/ecmwfStress' datestr(ecmwf.dn(1), 'yyyy') 'to' datestr(ecmwf.dn(end), 'yyyy')]
save(matfile, 'ecmwf', 'taucoeff', 'taufit')
