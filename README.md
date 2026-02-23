# foels
This is a 4D reconstruction project of Masahiro Ogawa.

# Setup
## Using docker
This is the easiest way.
Just clone this repositoy as;
```bash
git clone git@github.com:MasahiroOgawa/foels.git
```
And start devcontainer by opening vscode and selecting "reopen in devcontainer".

## local setup
clone
```bash
git clone git@github.com:MasahiroOgawa/foels.git
git submodule update --init --recursive
```
Install uv for virtual environment management
```
curl -LsSf https://astral.sh/uv/install.sh | sh

```

Set up virtual environment
```
uv sync
source .venv/bin/activate
```

**Notice:** InternImage, and unsupervised_detection will be dropped in the future, so it still use old conda environment.
```
source ${HOME}/anaconda3/bin/activate
conda init
conda env create -f foels/ext/InternImage/environment.yml
conda env create -f foels/ext/unsupervised_detection/environment.yml
```

# Run demo
To run the demo with sample data:
```
./script/run_foels.sh
```
This will process the sample data in `data/sample/` directory.

# Run on your own data
There are two ways to specify your input data:

## Method 1: Edit configuration file (recommended for repeated use)
Edit `script/foels_param.yaml` and set your input directory:
```yaml
MovingObjectExtractor:
  input_dir: "path/to/your/data"  # can be relative or absolute path
```
Then run without arguments:
```
./script/run_foels.sh
```

## Method 2: Command line arguments (quick one-time use)
Pass input directly as arguments:
```
./script/run_foels.sh [input_video_or_images] [result_directory]
```
Examples:
```bash
# Process images in a directory
./script/run_foels.sh /path/to/image/directory

# Process a video file
./script/run_foels.sh /path/to/video.mp4

# Specify both input and output directories
./script/run_foels.sh /path/to/input /path/to/output
```

**Note:** Command line arguments will override the configuration file settings.

# Debug 
set LOG_LEVEL=5 in script/foels_param.yaml, then
```
./script/run_foels.sh
```
(It will stop at some state.)
Then in left pane, push debug mark, choose "Attach to Python script in run_foels.sh" configuration, push run mark next to configuration mark. Then, the debugging will start.

# Update code
## Everyday update for users
```
git pull
git submodule update --init --recursive
```

## Everyday update for developers
```
(in case you change submodule,
 commit in ext/* directory, and
 cd ../../..
)
git add .
git commit -m "<any comments>"
git push --recurse-submodules=on-demand
(for the first time push, you need to run below;
git push --recurse-submodules=on-demand --set-upstream origin <your branch name>
)
```

## To import submodule upstream update
```
cd ext/<some submodule>
git fetch upstream
git switch main
git merge upstream/main
```

# License
[License](LICENSE) is in this repository.
But if you want to use this for your other needs, e.g. commertially, please consult me.