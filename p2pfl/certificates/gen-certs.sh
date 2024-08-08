#!/bin/bash

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
#
#   """Bash script to generate the required certificates for ssl (example)."""
#


IPV4_1="127.0.0.1"
IPV6_1="::1"

openssl genpkey -algorithm RSA -out server.key

cat > server.cnf <<EOF
[ req ]
default_bits       = 2048
distinguished_name = req_distinguished_name
req_extensions     = req_ext
x509_extensions    = v3_ca
prompt             = no

[ req_distinguished_name ]
CN = localhost

[ req_ext ]
subjectAltName = @alt_names

[ v3_ca ]
subjectAltName = @alt_names

[ alt_names ]
IP.1 = $IPV4_1
IP.3 = $IPV6_1
EOF

openssl req -new -key server.key -out server.csr -config server.cnf

openssl x509 -req -days 365 -in server.csr -signkey server.key -out server.crt -extensions v3_ca -extfile server.cnf