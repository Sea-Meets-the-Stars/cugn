

# First-order structure function
from scipy.stats import norm
from xhistogram.xarray import histogram

def first_order():

    # First order structure function
    sf1_mn = mSF_15.du1.mean(dim='time')[indx:indf]
    sf1_std = mSF_15.du1.std(dim='time')[indx:indf]
    rr1 = mSF_15.dr.mean(dim='time')[indx:indf].values
    du1 = mSF_15.du1.isel(mid_rbins=np.arange(indx, indf)).chunk({'mid_rbins':len(mSF_15.mid_rbins), 'time': 100})


    in1 = 2
    in2 = 5
    in3 = 10
    in4 = 18

    du1r = 0.4

    dull_mn = mSF_15.ulls.mean(dim='time')[indx:indf]
    dutt_mn = mSF_15.utts.mean(dim='time')[indx:indf]
    dull_std = mSF_15.ulls.std(dim='time')[indx:indf]
    dutt_std = mSF_15.utts.std(dim='time')[indx:indf]


    # Bins
    d1_bins = np.arange(-3, 3.5, du1r)#np.arange(-1e-2, 1e-2, 6e-5)/sf1_std[in1].values
    d2_bins = np.arange(-3, 3.5, du1r)#np.arange(-1e-2, 1e-2, 2e-4)/sf1_std[in2].values
    d3_bins = np.arange(-3, 3.5, du1r)#np.arange(-1e-2, 1e-2, 6e-4)/sf1_std[in3].values
    d4_bins = np.arange(-3, 3.5, du1r)#np.arange(-1e-2, 1e-2, 1e-3)/sf1_std[in4].values


    # Histograms
    sf1h0 =  histogram(du1.isel(mid_rbins=in1)/sf1_std[in1].values, bins=d1_bins, dim=['time'], density=True)
    sf1h2 = histogram(du1.isel(mid_rbins=in2)/sf1_std[in2].values, bins=d2_bins, dim=['time'], density=True)
    sf1h10 = histogram(du1.isel(mid_rbins=in3)/sf1_std[in3].values, bins=d3_bins, dim=['time'], density=True)
    sf1h15 = histogram(du1.isel(mid_rbins=in4)/sf1_std[in4].values, bins=d4_bins, dim=['time'], density=True)

    # Constructs Gaussian
    sf1p0 = norm.pdf(d1_bins, sf1_mn[in1]/sf1_std[in1].values, 1)
    sf1p2 = norm.pdf(d2_bins, sf1_mn[in2]/sf1_std[in2].values, 1)
    sf1p10 = norm.pdf(d3_bins, sf1_mn[in3]/sf1_std[in3].values, 1)
    sf1p15 = norm.pdf(d4_bins, sf1_mn[in4]/sf1_std[in4].values, 1)


    # Calculates kurtosis
    from scipy.stats import skew, kurtosis
    sf1_skew = np.zeros((len(rr1),))
    sf1_kurt = sf1_skew*0.

    for ii in range(len(rr1)):
        
        sf1_skew[ii] = skew(du1.isel(mid_rbins=ii).values, axis=0, bias=True)
        sf1_kurt[ii] = kurtosis(du1.isel(mid_rbins=ii).values, axis=0, fisher=True, bias=True)