name := Fluorescence predictor

version := 0.1.0


srcs := ./src/fluorescencepredictor.py
form := src/ui_form.py


ifeq ($(OS),Windows_NT)
    SHELL := powershell.exe
    .SHELLFLAGS := -NoProfile -Command
endif

all: build

build: gen_ui
ifeq ($(OS),Windows_NT)
	./.venv/Scripts/pyinstaller --windowed --add-data "src/model;model" --add-data "src/brain.ico;." --name '$(name)' --icon ./src/brain.ico $(srcs)
else
	./.venv/Scripts/pyinstaller --windowed --add-data "src/model:model" --add-data "src/brain.ico:." --name '$(name)' $(srcs)
endif

gen_ui: install_requirements
	./.venv/Scripts/pyside6-uic src/form.ui -o $(form)
	
install_requirements: create_venv
	./.venv/Scripts/pip install -r ./requirements.txt
		
create_venv:
	python -m venv .venv
	
clean: clean_venv clean_build

clean_venv:
	rm -r .venv
	
clean_build:
	rm -r dist
	rm -r build