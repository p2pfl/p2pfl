#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2022 Pedro Guijas Bravo.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

"""
Generate python code from proto file using grpc_tools.protoc.
"""

import grpc_tools  # type: ignore
from grpc_tools import protoc

if __name__ == "__main__":
    GRPC_PATH = grpc_tools.__path__[0]  # type: ignore
    PROTO_PATH = "p2pfl/communication/protocols/protobuff/proto"
    PROTO_FILE = "node.proto"

    command = [
        "grpc_tools.protoc",
        f"--proto_path={PROTO_PATH}",  # User proto files
        f"--proto_path={GRPC_PATH}/_proto",  # Google proto files
        f"--python_out={PROTO_PATH}",
        f"--grpc_python_out={PROTO_PATH}",
        f"--mypy_out={PROTO_PATH}",
        f"--mypy_grpc_out={PROTO_PATH}",
        f"{PROTO_PATH}/{PROTO_FILE}",
    ]

    print(f"Generating python code from proto file {PROTO_FILE} using grpc_tools.protoc...")

    if protoc.main(command) != 0:
        raise Exception("Error generating python code from proto file")

    # Fix imports (from {PROTO_PATH} import ...)
    print("Fixing imports in generated python code...")

    package = ".".join(PROTO_PATH.split("/"))

    # node_pb2_grpc.py  -> import node_pb2 as node__pb2
    with open(f"{PROTO_PATH}/node_pb2_grpc.py", "r") as f:
        text = f.read()
        # replace
        text = text.replace("import node_pb2 as node__pb2", f"from {package} import node_pb2 as node__pb2")
        # write
        with open(f"{PROTO_PATH}/node_pb2_grpc.py", "w") as f:
            f.write(text)

    # node_pb2_grpc.pyi -> import node_pb2
    # ignore grpc.aio error
    with open(f"{PROTO_PATH}/node_pb2_grpc.pyi", "r") as f:
        text = f.read()
        # replace
        text = text.replace("import node_pb2", f"from {package} import node_pb2")
        text = text.replace("import grpc.aio", "import grpc.aio # type: ignore")
        # write
        with open(f"{PROTO_PATH}/node_pb2_grpc.pyi", "w") as f:
            f.write(text)
