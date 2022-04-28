export SUITESPARSE_PATH = /nobackup/owhsu/sparse-datasets/suitesparse/
export FROSTT_PATH = /nobackup/owhsu/sparse-datasets/frostt/

.PHONY: formats
formats:
	./scripts/generate_suitesparse_formats.py

.PHONY: environment
environment:
	conda env export > environment.yml
	pip list --format=freeze > requirements.txt

sam: taco/build 
	 cd compiler && bash -x ./sam-kernels.sh

taco/build: submodules
	 mkdir -p compiler/taco/build 
	 cd compiler/taco/build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j8

#.PHONY: results
#results:
#	 mkdir -p $
#	 mkdir -p results/numpy

.PHONY: submodules
submodules:
	@if git submodule status | egrep -q '^[-]|^[+]' ; then \
		echo "INFO: Need to reinitialize git submodules"; \
		git submodule update --init; \
	fi

.PHONY: clean 
clean:
	cd compiler && rm -rf sam-outputs
	cd compiler/taco/build && make clean
