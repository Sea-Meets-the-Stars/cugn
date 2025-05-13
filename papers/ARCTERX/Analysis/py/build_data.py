
from load_profilers import load_by_asset
from profiler import profilers_io

all_assets = ['Alto', 'Flip', 'Slocum', 'Spray', 'Solo', 
              'EMApex', 'VMP', 'Triaxus', 'Seaglider']

def main():
    profilers = load_by_asset(all_assets)
    profilers_io.write_profilers(profilers, 
        'ARCTERX-IOP2025-Leg2.json')

if __name__ == '__main__':
    main()