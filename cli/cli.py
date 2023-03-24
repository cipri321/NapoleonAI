from cli.configuration import parser
from controller.create_model import create_model
from configuration.get_configuration import get_configuration


def run():
    args = parser.parse_args()
    create_model(get_configuration(args.configuration[0].split('=')[1]))
