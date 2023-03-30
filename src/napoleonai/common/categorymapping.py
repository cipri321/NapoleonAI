from typing import Tuple, Dict, Optional


def get_category_mapping_from_file(filepath: str) -> Optional[Tuple[Dict, int]]:
    """
    Returns the mapping encoded saved in the file located at filepath
    The file should have the structure:
    cat0 0
    cat1 1
    ...
    catn n

    :param filepath: str, location of the mapping file
    :return: dictionary containing the category mapping and number of categories
    """
    with open(filepath, 'rt') as f:
        no_cats = 0
        cat_map = dict()
        while line := f.readline():
            parsed_line = line.split()
            original_category = parsed_line[0]
            mapped_category = int(parsed_line[1])
            cat_map[original_category] = mapped_category
            no_cats = max(no_cats, mapped_category)
        return cat_map, len(cat_map.keys())


def write_category_mapping_to_file(filepath: str, cat_map: Dict[str, int]) -> bool:
    """
    Writes the category mapping contained in cat_map in file located at filepath with the following format:
    cat0 0
    cat1 1
    ...
    catn n

    :param filepath: str
    :param cat_map: Dict[str, int]
    :return: true, if the file could be written correctly, false, otherwise
    """
    with open(filepath, 'wt') as f:
        for key, val in cat_map.items():
            f.write(f'{key} {val}\n')
        return True
    return False
