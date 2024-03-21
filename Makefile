.PHONY: clean-temp clean-rust run-notebook run-script docker-build docker-run setup-env cargo-build commit-push

PROJECT_NAME="threshold"

clean-temp:
    find . -type f -name '*.pyc' -delete
    find . -type d -name '__pycache__' -delete
    find . -type f -name '*.tmp' -delete
    find ./results/temp -type f \( -name '*.pickle' -o -name '*.json' \) -delete
    find ./data/temp -type f \( -name '*.pickle' -o -name '*.json' \) -delete


clean-rust:
    cargo clean

docker-build:
    docker build -t $(PROJECT_NAME) .

docker-run:
    docker run -p 8000:8000 $(PROJECT_NAME)

setup-env:
    test -d .$(PROJECT_NAME) || python3 -m venv .$(PROJECT_NAME)
    source .$(PROJECT_NAME)/bin/activate; pip install -r requirements.txt

cargo-build:
    cargo build --release

commit-push:
    git add .
    git commit -m "Automated commit"
    git push
