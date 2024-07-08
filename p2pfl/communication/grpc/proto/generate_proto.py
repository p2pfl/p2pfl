import grpc_tools
from grpc_tools import protoc

if __name__ == "__main__":
    # python -m grpc_tools.protoc -I=p2pfl/communication/grpc/proto --python_out=p2pfl/communication/grpc/proto --grpc_python_out=p2pfl/communication/grpc/proto --mypy_out=p2pfl/communication/grpc/proto p2pfl/communication/grpc/proto/node.proto

    GRPC_PATH = grpc_tools.__path__[0]
    PROTO_PATH = "p2pfl/communication/grpc/proto"
    PROTO_FILE = "node.proto"

    command = [
        "grpc_tools.protoc",
        f"--proto_path={GRPC_PATH}/_proto",
        f"--proto_path={PROTO_PATH}",
        f"--python_out={PROTO_PATH}",
        f"--grpc_python_out={PROTO_PATH}",
        f"--mypy_out={PROTO_PATH}",
        f"--mypy_grpc_out={PROTO_PATH}",
        f"{PROTO_PATH}/{PROTO_FILE}",
    ]

    if protoc.main(command) != 0:
        raise Exception("Error generating python code from proto file")
