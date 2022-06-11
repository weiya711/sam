ifeq ($(NEVA),ON)
	export SUITESPARSE_PATH=/nobackup/owhsu/sparse-datasets/suitesparse/
	export FROSTT_PATH=/nobackup/owhsu/sparse-datasets/frostt/
	export SUITESPARSE_FORMATTED_PATH=/nobackup/owhsu/sparse-datasets/suitesparse-formatted
	export FROSTT_FORMATTED_TACO_PATH=/nobackup/owhsu/sparse-datasets/frostt-formatted/taco-tensor
	export FROSTT_FORMATTED_PATH=/nobackup/owhsu/sparse-datasets/frostt-formatted
endif 

guard-%:
	@ if [ "${${*}}" = "" ]; then \
        	echo "Environment variable $* not set"; \
        	exit 1; \
	fi

csv: 
	scripts/pytest_suitesparse_with_benchmarks.sh

tests: sam 
	python scripts/test_generating_code.py
	make run

run: submodules
	./scripts/pytest_suitesparse.sh

.PHONY: suitesparse-formats
suitesparse-formats: guard-SUITESPARSE_FORMATTED_PATH guard-SUITESPARSE_PATH
	rm -rf ${SUITESPARSE_FORMATTED_PATH}/*
	set -e && ./scripts/generate_suitesparse_formats.sh

.PHONY: env
env:
	export SUITESPARSE_PATH=/nobackup/owhsu/sparse-datasets/suitesparse/
	export FROSTT_PATH=/nobackup/owhsu/sparse-datasets/frostt/
	export SUITESPARSE_FORMATTED_PATH=/nobackup/owhsu/sparse-datasets/suitesparse-formatted

.PHONY: pydepends
pydepends:
	conda env export > environment.yml
	pip list --format=freeze > requirements.txt

sam: taco/build 
	 cd compiler && bash -xe ./sam-kernels.sh

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
