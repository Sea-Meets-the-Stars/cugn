""" Code relqted to plotting. """


Sn_lbls = dict(
    S1=r'$<\delta u_L> \;\; \rm [m/s]$',
    S1_duL=r'$<\delta u_L> \;\; \rm [m/s]$',
    S1_duT=r'$<\delta u_T> \;\; \rm [m/s]$',
    S1_duLT=r'$<\delta u_L + \delta u_T> \;\; \rm [m/s]$',
    S1_dT=r'$<\delta T> \;\; \rm [K]$',
    S2=r'$<\delta u_L^2> \;\; \rm [m/s]^2$',
    S2_duT=r'$<\delta u_T^2> \;\; \rm [m/s]^2$',
    S2_duLT=r'$<\delta u_L^2 + \delta u_T^2> \;\; \rm [m/s]^2$',
    S2_dT=r'$<\delta T^2> \;\; \rm [K]^2$',
    S3=r'$<\delta u_L^3> \;\; \rm [m/s]^3$',
    S3_duT=r'$<\delta u_T^3> \;\; \rm [m/s]^3$',
    S3_duLT=r'$<\delta u_L (\delta u_L^2 + \delta u_T^2)> \;\; \rm [m/s]^3$',
    S3_dT=r'$<\delta T^3> \;\; \rm [K]^3$',
)
Sn_lbls['S2_dS**2'] = r'$<\delta S^2> \;\; \rm [m/s]^2$'
Sn_lbls['S2_duL**2'] = Sn_lbls['S2']
Sn_lbls['S2_duT**2'] = Sn_lbls['S2_duT']
Sn_lbls['S2_dT**2'] = Sn_lbls['S2_dT']
Sn_lbls['S2_duLT**2'] = Sn_lbls['S2_duLT']
Sn_lbls['S3_duLduLduL'] = Sn_lbls['S3']
Sn_lbls['S3_duTduTduT'] = Sn_lbls['S3_duT']
Sn_lbls['S3_duLTduLTduLT'] = Sn_lbls['S3_duLT']
Sn_lbls['S3_duLdSdS'] = r'$<\delta u_L \delta S^2> \;\; \rm [m/s]^2$'
Sn_lbls['S3_dTdTdT'] = Sn_lbls['S3_dT']


def set_fontsize(ax, fsz):
    """
    Set the fontsize throughout an Axis

    Args:
        ax (Matplotlib Axis):
        fsz (float): Font size

    Returns:

    """
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fsz)
