# RawSignal — Production Data Video Factory v2

This version turns generated raw data into evolving procedural video **and data-derived audio**, then sends the finished MP4 to Telegram.

## 23 data families

Decimal, Binary, Hexadecimal, Octal, Alphabet, Alphanumeric, Roman Numerals, ASCII Characters, Morse Code, DNA Sequences, Coordinates, Timestamps, Prime Numbers, Mathematical Constants, Color Codes, Random Walks, Mathematical Expressions, Frequency Data, Hashes, UUIDs, Regular Expressions, JSON Structures, Graph Structures.

## 230 base combinations

23 data families × 10 visual engines = **230 generator combinations**.

IDs include `D01`, `H08`, `M04`, `DNA02`, `HASH08`, `JS10`, `G09`, etc.

## Visual engines

Particle Universe, Flow Field, Wave Ocean, Cellular Evolution, Geometry Machine, Fractal World, Data Sculpture, Digital Firestorm, Network Organism, Data Program.

Each render gets a unique seed, parameter profile and a variable duration between the configured minimum and maximum. The animation evolves through time instead of displaying a static visualization.

## Audio

The audio is synthesized from the same data values used by the visual system. Pitch, modulation and pulse activity are data-derived. There is no stock/copyrighted background track.

## GitHub Actions

Set repository secrets:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

Daily schedule remains 18:30 UTC (00:00 IST). Manual dispatch accepts a generator ID or can be left blank for random selection.

## Production defaults

- 1280×720
- 30 FPS
- 20–180 seconds, selected independently for every video
- H.264 MP4
- AAC audio
- Telegram delivery
- manual YouTube upload

GitHub-hosted runners have finite compute quotas. If long 720p renders approach the Actions timeout, lower resolution/FPS before increasing the duration ceiling.
