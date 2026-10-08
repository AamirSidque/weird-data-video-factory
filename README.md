# Weird Data Video Factory

A GitHub Actions-based generative video system that turns exactly six raw-data types into procedural videos and sends each finished MP4 to Telegram.

## Fixed data types

1. Decimal
2. Binary
3. Hexadecimal
4. Octal
5. Alphabet
6. Alphanumeric

## 60 production generators

### Decimal
D01 Particle Galaxy  
D02 Flow Field  
D03 Wave Ocean  
D04 Cellular Life  
D05 Geometry Machine  
D06 Fractal World  
D07 Data Sculpture  
D08 Fire/Rain  
D09 Network Organism  
D10 Data Program

### Binary
B01 Particle Galaxy  
B02 Flow Field  
B03 Wave Ocean  
B04 Cellular Life  
B05 Geometry Machine  
B06 Fractal World  
B07 Data Sculpture  
B08 Fire/Rain  
B09 Network Organism  
B10 Data Program

### Hexadecimal
H01 Particle Galaxy  
H02 Flow Field  
H03 Wave Ocean  
H04 Cellular Life  
H05 Geometry Machine  
H06 Fractal World  
H07 Data Sculpture  
H08 Fire/Rain  
H09 Network Organism  
H10 Data Program

### Octal
O01 Particle Galaxy  
O02 Flow Field  
O03 Wave Ocean  
O04 Cellular Life  
O05 Geometry Machine  
O06 Fractal World  
O07 Data Sculpture  
O08 Fire/Rain  
O09 Network Organism  
O10 Data Program

### Alphabet
A01 Particle Galaxy  
A02 Flow Field  
A03 Wave Ocean  
A04 Cellular Life  
A05 Geometry Machine  
A06 Fractal World  
A07 Data Sculpture  
A08 Fire/Rain  
A09 Network Organism  
A10 Data Program

### Alphanumeric
X01 Particle Galaxy  
X02 Flow Field  
X03 Wave Ocean  
X04 Cellular Life  
X05 Geometry Machine  
X06 Fractal World  
X07 Data Sculpture  
X08 Fire/Rain  
X09 Network Organism  
X10 Data Program

The visual mapping changes with the data type, so H06 is not the same visual treatment as D06.

## Telegram setup

1. Open Telegram and talk to `@BotFather`.
2. Create a bot with `/newbot`.
3. Copy the bot token.
4. Send `/start` to your new bot.
5. Obtain your chat ID. One simple method is to send a message to the bot and inspect:
   `https://api.telegram.org/botYOUR_TOKEN/getUpdates`
6. In your GitHub repository, go to:
   `Settings -> Secrets and variables -> Actions`
7. Create:
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`

Never put either secret in source code.

## Run manually

GitHub:
`Actions -> Generate Daily Weird Data Video -> Run workflow`

To force a particular generator, enter for example:

`H06`

Other examples:

`D01`
`B04`
`O05`
`A03`
`X10`

## Change production settings

Edit `config.json`:

- `width`
- `height`
- `fps`
- `duration_seconds`
- `crf`
- `preset`

Default is intentionally modest:

- 640x360
- 24 FPS
- 20 seconds
- H.264

This keeps daily GitHub Actions rendering practical.

## Local test

Requires Python 3.11+ and FFmpeg.

```bash
pip install -r requirements.txt
python -m src.main --generator H06 --seed 123456
```

For local Telegram delivery:

Windows PowerShell:

```powershell
$env:TELEGRAM_BOT_TOKEN="YOUR_TOKEN"
$env:TELEGRAM_CHAT_ID="YOUR_CHAT_ID"
python -m src.main --generator H06
```

## Daily behavior

The scheduled workflow:

1. chooses one of 60 generators;
2. creates a unique seed;
3. creates the raw data;
4. renders the video with Pillow/NumPy;
5. encodes MP4 using FFmpeg;
6. creates a deterministic title;
7. sends the MP4 and title to Telegram;
8. stores the generated file temporarily as a GitHub Actions artifact.

You manually upload the received video to YouTube.

## Important

This project deliberately does not automatically upload to YouTube. The output is delivered to Telegram for manual review/upload.

The generators are procedural and deterministic: the same generator + seed produces the same video.
