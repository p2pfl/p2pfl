#
# This file is part of the federated_learning_p2p (p2pfl) distribution (see https://github.com/pguijas/federated_learning_p2p).
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

"""Address parser."""

import os
import socket
from ipaddress import ip_address
from typing import Optional


class AddressParser:
    """
    Address parser. Determines if the address is a Unix domain address or an IP address (IPv4 or IPv6).

    Args:
        address: The address to parse.

    """

    def __init__(self, address: str):
        """Initialize the address parser."""
        self.host: Optional[str] = None
        self.port: Optional[int] = None
        self.is_v6: Optional[bool] = None
        self.unix_domain = False
        self.__parse_address(address)

    def __parse_address(self, address: str) -> None:
        if self.__is_unix_domain_address(address):
            self.unix_domain = True
            self.host = address
        else:
            try:
                raw_host, _, raw_port = address.rpartition(":")

                if raw_host != "":
                    self.port = int(raw_port)

                    if self.port > 65535 or self.port < 1:
                        raise ValueError("Port number is invalid.")

                else:
                    raw_host = raw_port

                    # Random port
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.bind(("", 0))
                        self.port = s.getsockname()[1]

                self.host = raw_host.translate({ord(i): None for i in "[]"})
                self.is_v6 = ip_address(self.host).version == 6

            except ValueError:
                self.host = None
                self.port = None
                self.is_v6 = None

    def __is_unix_domain_address(self, address: str) -> bool:
        """Check if the given address is a Unix domain address."""
        # Ensure the URL is in the correct format
        if not address.startswith("unix://"):
            return False

        # Extract the path from the URL
        socket_path = address[len("unix://") :]

        return os.path.isabs(socket_path)

    def get_parsed_address(self) -> str:
        """
        Get the parsed address.

        Returns:
            The parsed address.

        """
        if self.unix_domain:
            if self.host is None:
                raise ValueError("Unix domain address is invalid.")
            return self.host
        elif self.host is not None:
            return f"[{self.host}]:{self.port}" if self.is_v6 else f"{self.host}:{self.port}"
        else:
            raise ValueError("The address is invalid.")


if __name__ == "__main__":
    # Example usage:
    parser = AddressParser("127.0.0.1:8080")
    print(parser.get_parsed_address())  # Output: ('127.0.0.1', 8080, False)

    parser = AddressParser("127.0.0.1")
    print(parser.get_parsed_address())  # Output: ('127.0.0.1', 8080, False)

    parser = AddressParser("[::1]:8080")
    print(parser.get_parsed_address())  # Output: ('::1', 8080, True)

    parser = AddressParser("unix:///var/run/socket")
    print(parser.get_parsed_address())  # Output: ('unix:///var/run/socket', None, None)
