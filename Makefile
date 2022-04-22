export SUITESPARSE_PATH = /nobackup/owhsu/sparse-datasets/suitesparse/
export FROSTT_PATH = /nobackup/owhsu/sparse-datasets/frostt/


generate-sam: taco/build 
	 cd compiler
	 ./sam-kernels.sh

taco/build: check-and-reinit-submodules
	 mkdir -p compiler/taco/build 
	 cd compiler/taco/build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j8

#.PHONY: results
#results:
#	 mkdir -p $
#	 mkdir -p results/numpy

.PHONY: check-and-reinit-submodules
check-and-reinit-submodules:
	@if git submodule status | egrep -q '^[-]|^[+]' ; then \
		echo "INFO: Need to reinitialize git submodules"; \
		git submodule update --init; \
	fi
