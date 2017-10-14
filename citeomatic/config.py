import argparse
import logging
import os
import pickle
import sys
import time
import typing
from ast import literal_eval

import numpy
import pandas
import traitlets
from traitlets.config import Configurable

from citeomatic import traits, file_util
from .file_util import read_json, read_pickle, write_file, write_json, write_pickle

# These properties will be ignored for argument parsing.
IGNORED_TRAITS = {'parent', 'config'}


def generic_parser(trait, v):
    if v.startswith('@json:'):
        try:
            return read_json(v[6:])
        except Exception as e:
            raise argparse.ArgumentTypeError('Failed to parse JSON', e)

    if v.startswith('@eval:'):
        try:
            return eval(v[6:])
        except Exception as e:
            raise argparse.ArgumentTypeError('Failed to evaluate argument', e)

    if v.startswith('@pickle:'):
        try:
            return read_pickle(v[8:])
        except Exception as e:
            raise argparse.ArgumentTypeError(
                'Failed to read pickle file %s' % v[8:], e
            )

    if v.startswith('@csv:'):
        try:
            return pandas.read_csv(v[5:])
        except Exception as e:
            raise argparse.ArgumentTypeError(
                'Failed to read CSV file %s' % v[5:], e
            )

    if v.startswith('@call:'):
        try:
            import importlib
            fqn = v[6:]
            module_parts = fqn.split('.')
            module_name = '.'.join(module_parts[:-1])
            fn = module_parts[-1]
            mod = importlib.import_module(module_name)
            return getattr(mod, fn)()
        except Exception as e:
            raise argparse.ArgumentTypeError(
                'Failed to invoke method: %s: %s' % (v, e)
            )

    if isinstance(trait, (traitlets.Unicode, traitlets.Enum)):
        return v

    if isinstance(trait, traitlets.Int):
        return int(v)

    if isinstance(trait, traitlets.Bool):
        try:
            iv = int(v)
            return bool(iv)
        except ValueError as _:
            pass

        if v.lower() == 'true':
            return True
        if v.lower() == 'false':
            return False
        raise argparse.ArgumentTypeError(
            '"%s" could not be parsed as a boolean'
        )

    return literal_eval(v)


def parser_for_trait(trait: traitlets.TraitType) -> object:
    def _trait_parser(v):
        return generic_parser(trait, v)

    _trait_parser.__name__ = trait.__class__.__name__
    return _trait_parser


def setup_default_logging(level=logging.INFO):
    pandas.options.display.width = 200
    for h in list(logging.root.handlers):
        logging.root.removeHandler(h)

    logging.basicConfig(
        format=
        '%(levelname).1s%(asctime)-15s %(filename)s:%(lineno)d %(message)s',
        level=level,
        stream=sys.stderr
    )

    logging.getLogger('elasticsearch').setLevel(logging.WARN)


class Config(Configurable):
    """
    Basic configuration management for research experiments.

    A configuration has a base directory where models can be stored and a version tag.  Helper
    methods are supplied to access the version specific directory and read files.  The
    configuration itself, (along with any code versioning information available) is automatically
    serialized to the output directory on first use.  This makes it easy to resume an experiment
    given only the output directory.

    In practice, a new version should be created for each significant change to allow for
    tracking of progress.
    """
    version = traits.Unicode(default_value='', allow_none=True)
    base_dir = traits.Unicode(default_value='.')
    existing_config_pickle = traits.Unicode(allow_none=True)
    log_level = traits.Unicode(required=False, default_value='info')
    description = None

    rest_args = []
    _configured = False

    def __init__(self, **kw):
        Configurable.__init__(self, **kw)
        self._run_id = time.strftime('%Y-%m-%d-%H-%M-%S')
        self.reset(kw)
        assert self.base_dir, 'Must specify a base directory to write to.'
        self._configured = True

        self._finished_init()

    def _finished_init(self):
        """
        Called after initial configuration is completed (attributes assigned, etc.)
        :return:
        """
        pass

    def reset(self, kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def tmp_dir(self):
        return os.path.join(self.output_dir(), 'tmp-%s' % self._run_id)

    def dump_configuration(self, output_dir=None):
        """
        Write the configuration object to the given output directory.

        Always emits a pickle file, but if YAML is available, or the configuration
        values are JSON serializable, those formats will be emitted as well.

        :param output_dir: Directory to write configuration to.
        :return:
        """
        if output_dir is None:
            output_dir = self.output_dir()

        logging.info('Writing configuration to %s', output_dir)
        write_pickle(
            os.path.join(output_dir, 'config.pickle'), self._trait_values
        )
        try:
            import yaml
            write_file(
                os.path.join(output_dir, 'config.yaml'),
                yaml.dump(self._trait_values)
            )
        except ImportError as _:
            logging.warning('Failed to import YAML')

        try:
            write_json(
                os.path.join(output_dir, 'config.json'), self._trait_values
            )
        except:
            pass

    def output_dir_path(self):
        if not self.version:
            path = self.base_dir
        else:
            path = os.path.join(self.base_dir, self.version)
        return path

    def output_dir(self):
        path = self.output_dir_path()

        if not os.path.exists(path):
            os.system('mkdir -p "%s"' % path)

        return path

    def output_file(self, name) -> str:
        return os.path.join(self.output_dir(), name)

    def output_fd(self, name, mode) -> 'file':
        return open(self.output_file(name), mode)

    def write_file(self, name, data):
        write_file(self.output_file(name), data)

    def write_json(self, name, data, indent=2, sort_keys=True):
        if not name.endswith('.json'):
            name += '.json'

        write_json(
            self.output_file(name), data, indent=indent, sort_keys=sort_keys
        )

    def write_pickle(self, name, data):
        if not name.endswith('.pickle'):
            name += '.pickle'

        write_pickle(self.output_file(name), data)

    def read_pickle(self, name):
        if os.path.exists(self.output_file(name)):
            return read_pickle(self.output_file(name))
        elif os.path.exists(self.output_file(name + '.pickle')):
            return read_pickle(self.output_file(name) + '.pickle')
        else:
            assert False, 'Failed to find pickle file: "%s"' % self.output_file(
                name
            )

    def setup_logging(self):
        """
        Initialize logging for this configuration.

        Output will be written to stderr, and appended to the appropriate
        log files in the output directory for this config.
        :return:
        """
        log_level = getattr(logging, self.log_level.upper())
        setup_default_logging(log_level)
        logger = logging.getLogger()
        handler = logging.StreamHandler(
            file_util.open(self.output_file('LOG'), 'a')
        )
        handler.setFormatter(
            logging.Formatter(
                '%(levelname).1s%(asctime)-15s %(filename)s:%(lineno)d %(message)s',
            )
        )
        logger.addHandler(handler)

        logging.info('Initialized configuration (%s)', self.__class__.__name__)
        logging.info('Writing to: %s', self.output_dir())

    def __repr__(self):
        return self.__class__.__name__
        # return json.dumps(
        #     {name: str(getattr(self, name)) for name in self.traits().keys()}
        # )

    @classmethod
    def parse_command_line(
        cls, argv, add_help, defaults=None, allow_unknown_args=True
    ):
        if defaults is None:
            defaults = {}

        if argv is None:
            argv = sys.argv

        parser = argparse.ArgumentParser(
            add_help=add_help,
            description=cls.description,
            formatter_class=argparse.MetavarTypeHelpFormatter
        )

        def _help_for(name, trait):
            if name in defaults:
                return str(defaults[name])[:50]
            if trait.default_value == traitlets.Undefined:
                return None
            return str(trait.default_value)[:50]

        for name, trait in sorted(cls.class_traits().items()):
            if name in IGNORED_TRAITS:
                continue
            if isinstance(trait, traitlets.List):
                nargs = '*'
                type_parser = parser_for_trait(trait._trait)
            else:
                nargs = None
                type_parser = parser_for_trait(trait)

            parser.add_argument(
                '--%s' % name,
                type=type_parser,
                nargs=nargs,
                help=trait.help or '%s' % _help_for(name, trait),
                required=trait.metadata.get('required', False)
            )

        # print('Argv: %s', argv)
        if allow_unknown_args:
            parsed, rest = parser.parse_known_args(args=argv)
        else:
            parsed = parser.parse_args(args=argv)
            rest = []

        parsed = {k: v for (k, v) in parsed.__dict__.items() if v is not None}
        return parsed, rest

    @classmethod
    def parse_environment(cls):
        env_values = {}
        for name, trait in cls.class_traits().items():
            if name.upper() in os.environ:
                env_values[name] = generic_parser(
                    trait, os.environ[name.upper()]
                )
        return env_values

    @classmethod
    def initialize(cls, argv=None, add_help=True, **kw):
        setup_default_logging()

        parsed, rest = cls.parse_command_line(
            argv, defaults=kw, add_help=add_help
        )
        env_parsed = cls.parse_environment()

        if 'existing_config_pickle' in env_parsed:
            logging.info(
                'Restoring from existing configuration pickle: %s',
                env_parsed['existing_config_pickle']
            )
            return cls.load_from_pickle(env_parsed['existing_config_pickle'])

        kw = dict(kw)
        kw.update(parsed)
        kw.update(env_parsed)

        config = cls(**kw)
        config.setup_logging()

        cls.rest_args = rest[1:]

        Config.v = config
        return config

    @classmethod
    def load_from_pickle(cls, filename):
        """Restore an existing configuration object from the given data directory."""
        with open(filename, 'rb') as f:
            config = pickle.loads(f.read())
            # reset the base directory to be wherever our pickle file came from
            config.base_dir = os.path.dirname(
                os.path.dirname(os.path.abspath(filename))
            )
            logging.info('Config: %s', config.base_dir)
            return config


class App(Config):
    defaults = {}

    def main(self, args):
        pass

    @classmethod
    def run(cls, module_name):
        if module_name == '__main__':
            instance = cls.initialize(**cls.defaults)
            logging.info('Running: %s', ' '.join(sys.argv))
            instance.main(instance.rest_args)


JsonData = typing.Union[list, dict, str, int, float]


class JsonSerializable(traitlets.HasTraits):
    def to_dict(self) -> dict:
        """Recursively convert objects to dicts to allow json serialization."""
        return {
            JsonSerializable.serialize(k): JsonSerializable.serialize(v)
            for (k, v) in self._trait_values.items()
        }

    @staticmethod
    def serialize(obj: typing.Union['JsonSerializable', JsonData]):
        if isinstance(obj, JsonSerializable):
            return obj.to_dict()
        elif isinstance(obj, list):
            return [JsonSerializable.serialize(v) for v in obj]
        elif isinstance(obj, dict):
            res_dict = dict()
            for (key, value) in obj.items():
                assert type(key) == str
                res_dict[key] = JsonSerializable.serialize(value)
            return res_dict
        else:
            return obj

    @classmethod
    def from_dict(cls, json_data: dict):
        assert (type(json_data) == dict)
        args = {}
        for (k, v) in cls.class_traits().items():
            args[k] = JsonSerializable.deserialize(v, json_data[k])
        return cls(**args)

    @staticmethod
    def deserialize(target_trait: traitlets.TraitType, json_data: JsonData):
        """
        N.B. Using this function on complex objects is not advised; prefer to use an explicit serialization scheme.
        """
        # Note: calling importlib.reload on this file breaks issubclass (http://stackoverflow.com/a/11461574/6174778)
        if isinstance(target_trait, traitlets.Instance
                     ) and issubclass(target_trait.klass, JsonSerializable):
            return target_trait.klass.from_dict(json_data)
        elif isinstance(target_trait, traitlets.List):
            assert isinstance(json_data, list)
            return [
                JsonSerializable.deserialize(target_trait._trait, element)
                for element in json_data
            ]
        elif isinstance(target_trait, traitlets.Dict):
            # Assume all dictionary keys are strings
            assert isinstance(json_data, dict)
            res_dict = dict()
            for (key, value) in json_data.items():
                assert type(key) == str
                res_dict[key] = JsonSerializable.deserialize(
                    target_trait._trait, value
                )
            return res_dict
        else:
            return json_data

    def __repr__(self):
        traits_list = [
            '%s=%s' % (k, repr(v)) for (k, v) in self._trait_values.items()
        ]
        return type(self).__name__ + '(' + ', '.join(traits_list) + ')'