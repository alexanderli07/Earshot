#!/usr/bin/env bash
# One-command local MongoDB for Earshot accounts. Downloads the community
# server to ~/.local/mongodb (no Homebrew/Docker/admin needed), starts it on
# localhost:27017, and prints how to run the backend with auth enabled.
set -euo pipefail

MONGO_HOME="$HOME/.local/mongodb"
VERSION="8.0.12"

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64)  TGZ="mongodb-macos-arm64-$VERSION.tgz"  URL_OS="osx" ;;
  Darwin-x86_64) TGZ="mongodb-macos-x86_64-$VERSION.tgz" URL_OS="osx" ;;
  Linux-x86_64)  TGZ="mongodb-linux-x86_64-ubuntu2204-$VERSION.tgz" URL_OS="linux" ;;
  *) echo "Unsupported platform $(uname -s)-$(uname -m); install MongoDB manually." >&2; exit 1 ;;
esac

if [ ! -x "$MONGO_HOME/bin/mongod" ]; then
  echo "Downloading MongoDB $VERSION..."
  mkdir -p "$MONGO_HOME"
  curl -fL "https://fastdl.mongodb.org/$URL_OS/$TGZ" | tar xz -C "$MONGO_HOME" --strip-components=1
fi

mkdir -p "$MONGO_HOME/data" "$MONGO_HOME/log"
if pgrep -x mongod >/dev/null; then
  echo "mongod already running."
else
  "$MONGO_HOME/bin/mongod" --dbpath "$MONGO_HOME/data" \
    --logpath "$MONGO_HOME/log/mongod.log" --bind_ip 127.0.0.1 --fork
fi

echo
echo "MongoDB is up on localhost:27017. Start the backend with accounts enabled:"
echo
echo "  export EARSHOT_MONGO_URI=mongodb://localhost:27017"
echo "  python -m app.main"
