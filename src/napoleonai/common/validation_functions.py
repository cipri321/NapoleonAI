from typing import Dict, Union, Tuple

MandatoryData = Dict[str, Union[type, Tuple[type, str]]]


def validate_configuration(configuration: Dict, mandatory_data: MandatoryData):
    for md in mandatory_data.keys():
        if md not in configuration:
            raise ValueError(
                f'{md} should be in data configuration' +
                ('' if not isinstance(mandatory_data[md], tuple) else mandatory_data[md][1])
            )
