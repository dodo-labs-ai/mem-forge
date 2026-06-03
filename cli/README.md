# memforge interactive CLI

A small Node script that wraps the canonical Python `memforge` commands behind a
[Clack](https://github.com/bombshell-dev/clack) menu. It is loaded
automatically when `memforge` is invoked with no subcommand.

## Setup

```sh
cd cli
npm install
```

## Usage

Once installed, just run the Python entrypoint with no arguments:

```sh
memforge
```

The Python launcher spawns this script and the menu opens. Every action calls
the underlying scriptable `memforge` subcommand with `MEMFORGE_NO_INTERACTIVE=1`
in its environment, so there is no recursion.

## Development

```sh
node index.mjs        # run the menu directly
npm test              # dependency wiring + menu shape
```
