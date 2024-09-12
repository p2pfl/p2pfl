#!/bin/bash

#
# This file is part of the federated_learning_p2p (p2pfl) distribution (see https://github.com/pguijas/p2pfl).
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
#
#   """Bash script to generate the required certificates for ssl (example)."""
#

CA_PASS="supersecretpassword"

# Generar CA
openssl genpkey -algorithm RSA -out ca.key -aes256 -pass pass:${CA_PASS}
openssl req -x509 -new -nodes -key ca.key -sha256 -days 1024 -out ca.crt -config openssl.cnf -passin pass:${CA_PASS}

# Generar certificado del servidor
openssl genpkey -algorithm RSA -out server.key -pass pass:${CA_PASS}
openssl req -new -key server.key -out server.csr -config openssl.cnf
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out server.crt -days 500 -sha256 -extfile server_ext.cnf -extensions v3_req -passin pass:${CA_PASS}

# Generar certificado del cliente
openssl genpkey -algorithm RSA -out client.key -pass pass:${CA_PASS}
openssl req -new -key client.key -out client.csr -config openssl.cnf
openssl x509 -req -in client.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out client.crt -days 500 -sha256 -extfile client_ext.cnf -extensions v3_req -passin pass:${CA_PASS}
