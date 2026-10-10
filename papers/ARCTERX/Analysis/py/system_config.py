import platform


def get_paths():
    if platform.system() == 'Darwin':
        paths = {
            'cruiseshare': '/Volumes/cruiseshare',
            'tgt-data': '/Volumes/tgt-data/TN441B',
            'Shore': '/Volumes/Shore',
            'Ship': '/Volumes/Ship'
        }
    elif platform.system() == 'Windows':
        paths = {
            'cruiseshare': 'C:/cruiseshare',
            'tgt-data': 'C:/tgt_data/TN441B',
            'Shore': 'C:/Shore',
            'Ship': 'C:/Ship'
        }
    elif platform.system() == 'Linux':
        paths = {
            'cruiseshare': '/run/user/1000/gvfs/smb-share:server=10.43.20.20,share=cruiseshare/',
            'tgt-data': 'None',
        }
    else:
        raise Exception(f'Unsupported platform {platform.system()}')
    return paths


def get_path(name):
    paths = get_paths()
    return paths[name]
