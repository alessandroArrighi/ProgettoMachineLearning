import json
import jsonschema
from pathlib import Path
from types import SimpleNamespace
import sys

import jsonschema.exceptions

def ConfigChecker(conf: str, conf_schema: str) -> object:
    conf_file = Path(conf)
    schema_file = Path(conf_schema)

    if not conf_file.is_file():
        print("Conf file not exist")
        sys.exit(-1)

    if not conf_file.suffix == '.json':
        print("Conf file must be .json")
        sys.exit(-1)

    
    if not schema_file.is_file():
        print("Schema file not exist")
        sys.exit(-1)

    if not schema_file.suffix == '.json':
        print("Schema file must be .json")
        sys.exit(-1)

    with open(conf_file) as c:
        with open(schema_file) as s:
            conf = json.load(c)
            schema = json.load(s)

            try:
                jsonschema.validate(instance = conf, schema = schema)
            except jsonschema.exceptions.ValidationError:
                print("Config file is not following schema rules")
                sys.exit(-1)
            except jsonschema.exceptions.SchemaError:
                print("Schema is not valid")
                sys.exit(-1)

    conf = None
    with open(Path(conf_file)) as c:
        conf = json.loads(c.read(), object_hook = lambda c: SimpleNamespace(**c))

    return conf

if __name__ == "__main__":
    ConfigChecker('./conf/conf.json', './conf/conf_schema.json')