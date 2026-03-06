READme.md

# MonoSLAM

## Dataset

ETH3D SLAM (cables_2_mono)

This repo can be run on the ETH3D SLAM datasets. Below is the minimal, reproducible way to download the `cables_2_mono` sequence into `data/`.

### Download `cables_2_mono` into `data/eth3d/`

From the repo root:

```bash
mkdir -p data/eth3d
cd data/eth3d

SEQ=cables_2_mono
URL="https://www.eth3d.net/data/slam/datasets/${SEQ}.zip"

Optional: verify the URL responds
curl -I "$URL" | head

Download and unzip
wget -O "${SEQ}.zip" "$URL"
unzip -q "${SEQ}.zip" -d "${SEQ}"

