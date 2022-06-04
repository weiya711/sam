BENCHES := ""

# Set OPENMP=ON if compiling TACO with OpenMP support.
ifeq ($(OPENMP),)
OPENMP := "OFF"
endif
# Set NEVA=ON if compiling on the Stanford cluster (Neva or Kiwi).
ifeq ($(NEVA),)
NEVA := "OFF"
endif
# Set GEN=ON if you would like to generate "other" tensors for performance into a file
ifeq ($(GEN),)
GEN := "OFF"
endif


ifeq ("$(NEVA)","ON")
CMD := OMP_PROC_BIND=true LD_LIBRARY_PATH=compiler/build/lib/:$(LD_LIBRARY_PATH) numactl -C 0,2,4,6,8,10,12 -m 0 compiler/build/taco-bench $(BENCHFLAGS)
else
CMD := LD_LIBRARY_PATH=compiler/build/lib/:$(LD_LIBRARY_PATH) compiler/build/taco-bench $(BENCHFLAGS)
endif

export SUITESPARSE_PATH = /nobackup/owhsu/sparse-datasets/suitesparse/
export FROSTT_PATH = /nobackup/owhsu/sparse-datasets/frostt/
export SUITESPARSE_FORMATTED_PATH=/nobackup/owhsu/sparse-datasets/suitesparse-formatted
export TACO_TENSOR_PATH=/nobackup/owhsu/sparse-datasets/

csv: 
	scripts/pytest_suitesparse_with_benchmarks.sh

tests: sam 
	python scripts/test_generating_code.py
	make run

run: submodules
	./scripts/pytest_suitesparse.sh

.PHONY: formats
formats:
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

taco-bench: compiler/build/taco-bench
ifeq ($(BENCHES),"")
	$(CMD) --benchmark_out_format="csv" --benchmark_out="$(TACO_OUT)" --benchmark_repetitions=10 --benchmark_counters_tabular=true
else
	$(CMD) --benchmark_filter="$(BENCHES)" --benchmark_out_format="csv" --benchmark_out="$(TACO_OUT)" --benchmark_repetitions=10 --benchmark_counters_tabular=true
endif

compiler/build/taco-bench: submodules compiler/benchmark/googletest
	mkdir -p compiler/build/ && cd compiler/build/ && cmake -DOPENMP=$(OPENMP) -DNEVA=$(NEVA) -DGEN=$(GEN) ../ && $(MAKE) taco-bench

compiler/benchmark/googletest: submodules
	if [ ! -d "compiler/benchmark/googletest" ] ; then git clone https://github.com/google/googletest compiler/benchmark/googletest; fi
	
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
