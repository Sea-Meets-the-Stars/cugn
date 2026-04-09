

def qg_output_file(x0, y0, dx:int=None, dtime='5years'):    
    
    # Build it
    if dx == 200:
        output_file = f'../Analysis/Output/SF_region_x{int(x0)}_y{int(y0)}_200km_{dtime}.nc' 
    elif dx == 300:
        output_file = f'../Analysis/Output/SF_region_x{int(x0)}_y{int(y0)}_300km_{dtime}.nc' 
    else:
        output_file = f'../Analysis/Output/SF_region_x{int(x0)}_y{int(y0)}_{dtime}.nc' 
    
    # Return output_file
    return output_file