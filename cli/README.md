# memforge interactive CLI

A small Node script that wraps the canonical Python `memforge` commands behind a
[Clack](https://github.com/bombshell-dev/clack) menu. It is loaded
automatically when `memforge` is invoked with no subcommand.

## Setup

The installed Python package owns this UI. When `memforge` is invoked with no
subcommand, the Python launcher copies the packaged files into a versioned user
cache and runs `npm ci --omit=dev` there on first use. Users should not run
`npm install` in this directory as part of normal MemForge setup.

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
npm ci
node index.mjs        # run the menu directly
npm test              # dependency wiring + menu shape
```
